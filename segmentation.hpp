#ifndef __SEGMENTATION_HPP
#define __SEGMENTATION_HPP

// For INT_MAX
#include <climits>
#include "cuda_utils.hpp"
#include "limits.hpp" // for device side numeric_limits min() and max()

using namespace cudahack; // for device side numeric_limits

/* The following kernel function is used to do data reduction of unimodal queries
   (e.g. nanopore 3000 or 4000Hz raw current data to dwell segments, a.k.a. "events" in a "squiggle")
   with the Bellman k-segmentation (dynamic programming) algorithm and even weights for sample error.
   The minimized measure is the mean squared error of the segmental regression.
   The segmentation is "adaptive" insofar as a maximum number of expected segments is passed in, but if 
   <min_segment_size data points are creating their own segments we assume that they are noise and we should try 
   segmenting again with a smaller number of expected segments. 

   Process a bunch of different data series in parallel on the GPU. This is an intense dynamic programming process that has some parallelizable steps. Also,
   segmentation is a semi-local optimization problem in practice, so if you have a really long query, we need to split it up into chunks for segmentation
   to keep things moving along at a reasonable pace with shared data in each kernel's L1 cache. An 'epsilon' heuristic is used to merge last/first segments of  
   neighbouring segmentation subtask results.

   The distance calculation and usage are in heavily used loops, so to keep it in L1 cache we need to downaverage (take avg value of non-overlapping blocks of signal) 
   the original data at least a bit to fit.  This also has the effect of minimizing the impact of errant individual data points in the stream.

   We also dynamically set a maximum size for a segment (as a multiple of the average size), in order to reduce the search space and memory requirement.
*/
// MAX_DP_SAMPLES should not exceed 256. Otherwise, sums and squares accumulators used in the segmentation kernel (short and int respectively) could overflow in edge cases of extremely noisy, high dynamic range signal.
#define MAX_DP_SAMPLES 256
// For the macros below: l = left index, r = right index, n = number of values stores per left index (i.e. the extent of the dynamic programming choices for any value of l)
#define SUMS(l,r,n) sums[(r)-(l)+(n)*(l)]
#define SQUARES(l,r,n) squares[(r)-(l)+(n)*(l)]
#define MEANSQUARE(l,r,n) SUMS(l,r,n)*SUMS(l,r,n)/((r)-(l)+1)
// The following are 1-based, whereas SUMS, SQUARES and MEAN are 0-based
#define DIST(l,r,n) SQUARES((l)-1,(r)-1,n)-MEANSQUARE((l)-1,(r)-1,n)
#define K_SEG_DIST(p,n,l) k_seg_dist[(l)*((p)%2)+(n)]
#define K_SEG_PATH(p,n,l) k_seg_path[(l)*((p)-1)+(n)]
// The k_seg_path_working_buffer arg is working buffer for the backtracking part of dynamic programing process (picking the best segmentation path out of all the paths calculated).  
// Many mallocs inside the kernel is way slower as it forces serialization of threads,
// so expecting a wrapper function to do this en masse for us and we just use a non-overlapping slice of it.

// Fancy math here to ensure that a variable is memory aligned for the size of T (you can't use __align__ when x is a dynamic pointer, like in threadblock shared memory)
// Note that we've explicitly limited ourselves to 64 bit systems here, as uintptr_t type requires C++11, which is *still* flakey in some compilers shipped with modern OSes
#define LOCATION(x) (reinterpret_cast<unsigned long long>(x))
#define SHARED_MEM_ALIGN(T,x) x = (T *) (sizeof(T)*DIV_ROUNDUP(LOCATION(x),sizeof(T)))

template<typename T>
__global__ void adaptive_device_segmentation(T **all_series, size_t *all_series_lengths, int raw_samples_per_threadblock, short max_expected_k, 
                                             int min_segment_size, int sharedMemSize, unsigned short *k_seg_path_working_buffer, 
                                             T *output_segmental_medians){

	// To facilitate the segmentation happening over multiple GPUs but writing back to a unified result space, we have put sentinel
	// values of NULL into the all_series array to indicate that this device should not process this member of the series, 
	// so we can just return without doing any work.
	const T *series = all_series[blockIdx.x];
	if(series == 0){
		return;
	}

	// Recomputed rather than passed in to save SM registers, must correspond to same line in the host function further below.
	short downsample_width = DIV_ROUNDUP(min_segment_size,3);
	int orig_N = all_series_lengths[blockIdx.x];
	if(blockIdx.y*raw_samples_per_threadblock >= orig_N){ // no data to process
		return; // this is safe as all threads in this block will take this short-circuit branch and not cause hangs in the __syncthreads() calls below
	}

	// "*_before_us" vars are used to determine the offset of this threadblock's required global memory intermediate-working and output values.
	// As the input all_series is a ragged array, we need to figure out where to write the values for this query
	// by summing up the lengths of the raw and segmented data before this block.
	size_t datapoints_before_us = 0; // this is intermediate values, local to the device
	size_t segments_before_us = 0; // this is output, global across devices
	for(int query_num = 0; query_num < blockIdx.x; query_num++){
		// K segments guaranteed per full threadblock. Count all full thread blocks for this query, plus the last partial block.
		int query_length = all_series_lengths[query_num];
		if(all_series[query_num] != 0){ // we're determining a location in a per device buffer here, only count the seqs that were assigned to this GPU
			datapoints_before_us += DIV_ROUNDUP(query_length,downsample_width);
		}
		segments_before_us += DIV_ROUNDUP(query_length,raw_samples_per_threadblock)*max_expected_k;
	}
	// Account for offset within this seq
	datapoints_before_us += raw_samples_per_threadblock/downsample_width*blockIdx.y;
	segments_before_us += blockIdx.y*max_expected_k;

	// Let's switch to local mode for the indexing, N is the threadblock size or the remainder when it's the last block for this query
	int N = ((blockIdx.y+1)*raw_samples_per_threadblock > orig_N) ? orig_N%raw_samples_per_threadblock : raw_samples_per_threadblock;

	// If this is the last thread block for this query, pro-rate the expected_k passed in (which is for a full threadblock)
	int expected_k = max_expected_k;
	if((blockIdx.y+1)*raw_samples_per_threadblock > orig_N){
		expected_k = DIV_ROUNDUP((orig_N%raw_samples_per_threadblock),DIV_ROUNDUP(raw_samples_per_threadblock, max_expected_k));
	}

        int N_ds = N/downsample_width;

        // Must allocate shared memory in bytes before it can be cast to the template variable
        unsigned short *breakpoints = shared_memory_proxy<unsigned short>();   // size = max_expected_k+1
        unsigned int *options = reinterpret_cast<unsigned int *>(&breakpoints[max_expected_k+1]); // size = N_ds+1
	SHARED_MEM_ALIGN(unsigned int, options);

        // Keep a matrix of the total segmentation costs for any p-segmentation of a subsequence series[1:n] where 1<=p<=k and 1<=n<=N. The 0th column at
        // the beginning encodes that a (k-1)-segmentation is penalized compared to a whole-segment average.
        // In the segmentation dynamic programming, we only ever need to be storing the current value of K_SEG_DIST for p, and reading the optimal solutions for p-1.
        // This means that in practice we only need to store 2*(N+1) values of K_SEG_DIST in an alternating storage pattern, which is what the K_SEG_DIST macro defined earlier does.
        unsigned int *k_seg_dist = &options[N_ds+1];  

	// This is the start of threadblock shared memory space that will be clobbered later. Don't declare anything except ephemera in L1 past this point.
        T *downsample_qtype = reinterpret_cast<T *>(&k_seg_dist[2*(N_ds+1)]); // size = N_ds
	SHARED_MEM_ALIGN(T, downsample_qtype);

        if(threadIdx.x*downsample_width <= N-downsample_width){
                downsample_qtype[threadIdx.x] = 0;
		int num_downsamples = 0;
                for(int i = 0; i < downsample_width && downsample_width*threadIdx.x+threadIdx.x+i < N; i++){
			// Warp fetches should coalesce to slurp up adjacent global memory fairly quickly.
                        downsample_qtype[threadIdx.x] += series[downsample_width*threadIdx.x+i+blockIdx.y*raw_samples_per_threadblock]; 
			num_downsamples++;
                }
                downsample_qtype[threadIdx.x] /= num_downsamples; // Correct averaged values are in 0, downsample_width, 2*downsample_width, etc.
        }

	// We will rewrite the data to be segmented as unsigned characters (256 levels) so we can cram as much into L1 cache as possible. To
	// do this we need to find the downsampled min and max values, then scale everything to that so we lose as little resolution as possible.
	volatile T *T_max = &downsample_qtype[N_ds];   // size = N_ds/CUDA_WARP_WIDTH during map, single value after reduce
	volatile T *T_min = &T_max[N_ds/CUDA_WARP_WIDTH];  // size = N_ds/CUDA_WARP_WIDTH during map, single value after reduce
	T warp_max = threadIdx.x < N_ds ? downsample_qtype[threadIdx.x] : numeric_limits<T>::min();
        T warp_min = threadIdx.x < N_ds ? downsample_qtype[threadIdx.x] : numeric_limits<T>::max();
	__syncwarp();
	warp_min = warpReduceMin<T>(warp_min); // across the warp
	warp_max = warpReduceMax<T>(warp_max); // across the warp
	int lane = threadIdx.x % CUDA_WARP_WIDTH;
	int wid = threadIdx.x / CUDA_WARP_WIDTH;
	if(!lane && wid < N_ds/CUDA_WARP_WIDTH){
		T_max[wid] = warp_max;
		T_min[wid] = warp_min;
	}
         __syncthreads();
        // Get in-bounds values only for final threadblock reduction, calculated by the first warp's threads (threadblock may not be full).
        if(!wid){
		warp_max = (threadIdx.x < N_ds / CUDA_WARP_WIDTH) ? T_max[lane] : numeric_limits<T>::min();
		warp_max = warpReduceMax<T>(warp_max); // across all threads in the block
                warp_min = (threadIdx.x < N_ds / CUDA_WARP_WIDTH) ? T_min[lane] : numeric_limits<T>::max();
                warp_min = warpReduceMin<T>(warp_min); // across all threads in the block
		if(!lane){ // first, master thread only
			T_max[0] = warp_max;
			T_max[1] = warp_min; // collapse answers in the L1 cache space
			T_min = &T_max[1];
		}
        }
        __syncthreads();
                
        // Note that options, K_SEG_* and DIST are all indexed starting at 1, not 0 for logical simplicity.
        // An index array allowing for the reconstruction of the regression with the lowest cost.
        unsigned short *k_seg_path = &k_seg_path_working_buffer[max_expected_k*datapoints_before_us];

	// Following variable clobbers the warp min/maxes arrays that we don't need anymore. Need the +2 because we are retaining the singular threadblock max and min.
        volatile unsigned int *squares = reinterpret_cast<volatile unsigned int *>(&downsample_qtype[N_ds+2]);  // size = N_ds*e, dynamic based on determination of 'e' below
	SHARED_MEM_ALIGN(unsigned int, squares);

       	// To save memory and split homopolymers in nanopore data, cap the size of any given segment. This assumes some kind of Poisson like distribution for segment size.
       	// The extent (e) of the DP search for any given downsamples data index, just a short name for readability of SUMS(), SQUARES(), etc. macro calls below
       	int e = (sharedMemSize-LOCATION(squares)+LOCATION(breakpoints))/(N_ds*(sizeof(unsigned short)+(sizeof(unsigned int)))); // bytes available/needed_per_datapoint
       	volatile unsigned short *sums = reinterpret_cast<volatile unsigned short *>(&squares[N_ds*e]); // size = N_ds*e (dynamic based on 'e')
	// No need to align sums' pointer since the preceeding variable has a larger width (int vs short).

	// Populate the base case of the diagonal for each matrix.
        if(threadIdx.x < N_ds){
            	SUMS(threadIdx.x,threadIdx.x,e) = (unsigned char) (256.0*((downsample_qtype[threadIdx.x]-(*T_min))/((*T_max)-(*T_min)+1))); // +1 to avoid "div by 0" errors in edge case of absolutely no variance
                SQUARES(threadIdx.x,threadIdx.x,e) = SUMS(threadIdx.x,threadIdx.x,e)*SUMS(threadIdx.x,threadIdx.x,e);
        }
        __syncthreads(); // Make sure we've loaded all the data before starting the compute.

        // In parallel, fill the cumulative sums and squares matrices over the range of allowed segment lengths (in downsampled units).
        // For space efficiency, in the SUMS amd SQUARES macros we're storing as a square indexed on x:segment start location and y:possible segment length, not as a diagonal matrix.
        // Note that we've constrained the maximum length of a segment to e due to memory considerations
        // and assumption that a Poisson-like distribution governs segment length. This means that very long segments will be split
        // automatically, e.g. homopolymers in nanopore data.
        if(threadIdx.x < N_ds){
                for (int s=1; s < e; s++){ // s is the segment span (i.e. base 0)
                // Incrementally update every partial sum (unless we're past the end of the data in this thread)
                        if(threadIdx.x + s < N_ds){
                                int idx = threadIdx.x;
                                SUMS(idx,idx+s,e) = SUMS(idx,idx+s-1,e) + SUMS(idx+s,idx+s,e);
                                SQUARES(idx,idx+s,e) = SQUARES(idx,idx+s-1,e) + SQUARES(idx+s,idx+s,e);
                        }
                }
        }
        __syncthreads();

        bool exit = false;
	// Variables for the binary search for maximum possible value of K that doesn't generate tiny noise segments.
	short smallest_noisy_k_found = expected_k+1;
	short test_expected_k = DIV_ROUNDUP(expected_k,2); 
	short delta = test_expected_k/2;

        while(!exit) {

                if(threadIdx.x < N_ds){
                        // Initialize regression distances for the case k=1 (a single segment) directly from the precomputed starting-at-zero cumulative distance calculations.
                        if(threadIdx.x+1 <= e) {
                                K_SEG_DIST(1,threadIdx.x+1,N_ds) = DIST(1,threadIdx.x+1,e); 
                        } else {
                                K_SEG_DIST(1,threadIdx.x+1,N_ds) = INT_MAX/2; // div by 2 to avoid overflow when adding later
                        }
                        // Initialize the path for the trivial case where we have p segments through p datapoints.
                        // The right boundary for the p case is p by the pigeonhole principle.
                        if(threadIdx.x < test_expected_k){
                                K_SEG_PATH(threadIdx.x+1,threadIdx.x,N_ds) = (unsigned short) test_expected_k-1;
                        }
                }
                __syncthreads();

                // Now incrementally calculate the optimal p-segmentation solution in series for each 1 < p <= k.
                for(int p=2; p <= test_expected_k; p++){

                        for(int n=p; n <= N_ds; n++){ // start at p segments being created from p+1 data points, the first non-trivial case
                                // Pick the dividing point with the lowest cumulative Mean Square Error distance measure.
                                // Note that only a subset of the N options will be calculated.
                                // Only generate (in parallel) options with segment length <= e.
                                if(threadIdx.x+1 >= p && threadIdx.x+1 <= n && n-threadIdx.x-1 < e) {
                                        options[threadIdx.x+1] = K_SEG_DIST(p-1,threadIdx.x,N_ds) + DIST(threadIdx.x+1,n,e); // two rules for DIST(l,r,e): l <= r && r-l < e
                                }
                                __syncthreads();

                                // Single thread calculating the minimum cost regression option to get p segments by chopping up n+e
                                // datapoints (or N if we're near the right edge already), with the last segment starting at least at position n.
                                if(! threadIdx.x){
                                        int minval = INT_MAX;
                                        int minidx = -1;
                                        for(int idx = (n-e+1 > p) ? n-e+1 : p, start = idx; idx-start < e && idx <= n; idx++){
                                                if(options[idx] < minval){
                                                        minval = options[idx];
                                                        minidx = idx;
                                                }
                                        }

                                        // Memoize the solution value for the next value of p.
                                        K_SEG_DIST(p,n,N_ds) = minval;

                                        // Store the locations of the solutions too so we can use the solution to generate segment stats later.
                                        K_SEG_PATH(p,n-1,N_ds) = (unsigned short) minidx-1;
                                }
                                __syncthreads();
                        }
                }

                // Single thread, backtrack from the right end of the dataset, which has a known optimal k-segment solution right boundary
                // (the length of the input list), to that last segment's left border, taking the
                // k-1 solution from that point, etc. until we get to the single segment which necessarily starts at the left edge (first input item).
                if(! threadIdx.x){
                        breakpoints[test_expected_k] = N_ds; 
                        for (int p = test_expected_k-1; p >= 1; p--){
                                breakpoints[p] = K_SEG_PATH(p+1,breakpoints[p+1]-1,N_ds);
                        }
                        breakpoints[0] = 0; // Left exclusive boundary is always 0 (one-based indexing) so we start at data point 1 for any segmentation.
                }
                __syncthreads();

                // Check if any ranges are unusually small data segments (i.e. <min_segment_size/downsample_width), which means that we're letting the noise in the data still create segments where they shouldn't.
                // If this is the case, we want to lower the expected number of segments (hence "adaptive" in the function name) and try again.
		bool noisy = false;
                for(int i = 1; i <= test_expected_k; i++) {
                        int left = breakpoints[i-1];
                        int right = breakpoints[i];
                        // Redundant computation amongst threads here is as a single computation followed by __syncthreads() because of unsynchronized median calculation below.
                        if(right-left < min_segment_size/downsample_width){ 
				noisy = true;
                        }
                }
		if(delta == 0 || test_expected_k + delta > max_expected_k){
			exit = true;
		}
		else if(test_expected_k != 1 && noisy){
			smallest_noisy_k_found = test_expected_k;
               		test_expected_k -= delta;
		}
		else{
			if(test_expected_k == 1 || smallest_noisy_k_found == test_expected_k + 1){
				// Exit condition found, a wee tiny irriducible segment or the biggest possible k value that's non-noisy.
				exit = true;
			}
			else{
				test_expected_k += delta;
			}
		}
		delta = DIV_ROUNDUP(delta,2);
                if(exit){
                	// At this point we can clobber all downaveraged and cumulative cost data in L1 cache variables as we are done with them.
                	// Let's load the original data series now so the calculations below here will be less affected by latency in most threads.
                	// TODO: check if the original data is bigger than the L1 space we have available (i.e. a *huge* downaveraging width was applied), and downsample accordingly.
                	volatile T *orig_data_copy = reinterpret_cast<volatile T *>(downsample_qtype);
                	if(threadIdx.x*downsample_width+blockIdx.y*raw_samples_per_threadblock < orig_N){
                        	for(int i = 0; i < downsample_width && threadIdx.x*downsample_width + i < raw_samples_per_threadblock && threadIdx.x*downsample_width + blockIdx.y*raw_samples_per_threadblock + i < orig_N; i++){
                                	orig_data_copy[threadIdx.x*downsample_width+i] = series[threadIdx.x*downsample_width+blockIdx.y*raw_samples_per_threadblock+i];
                        	}
			}
			__syncthreads(); // so all the data for any given segment is in place before we sort

			if(threadIdx.x > test_expected_k && threadIdx.x <= max_expected_k) {
				// Sentinel answer for unused result slots (K is smaller than the expect value passed in), since the amount of space for answers was preallocated.
                        	output_segmental_medians[segments_before_us+threadIdx.x-1] = numeric_limits<T>::max();
			}

                	// In parallel, get the median from each segment and set its value in the constant cache query value.
                	// We are using the median rather than the mean as it is less affected by outlier datapoints caused by noise.
                	else if(threadIdx.x > 0 && threadIdx.x <= test_expected_k){
                        	int right_boundary = breakpoints[threadIdx.x]*downsample_width;
                        	// Bounds check as last coordinate may represent a partial downsample since the original data length is not necessarily a multiple of downsample_width.
                        	if(right_boundary > N){
                                	right_boundary = N;
                        	}
				
                        	// Bubble sort the L1 cache values for each segment in place so we can run this quickly and easily get the median for each segment.
                        	int left_boundary = (breakpoints[threadIdx.x-1])*downsample_width;
                        	for(int i = left_boundary; i < right_boundary-1; ++i){ // +1 as starting point as left boundary is non-inclusive.
                                	for(int j = i+1; j < right_boundary; ++j){
                                        	if(orig_data_copy[i] > orig_data_copy[j]){
                                                	T tmp = orig_data_copy[i];
                                                	orig_data_copy[i] = orig_data_copy[j];
                                                	orig_data_copy[j] = tmp;
                                        	}
                                	}
				}

                        	// Write the segment median to global memory. *DO NOT* rely on the updated value in other parts of this kernel...
                        	// write cache is not coherent on all GPUs, and is only flushed on threadblock termination.
                        	if((left_boundary-right_boundary)%2){
                        		output_segmental_medians[segments_before_us+threadIdx.x-1] = orig_data_copy[(left_boundary+right_boundary)/2];
				}
				else{
					output_segmental_medians[segments_before_us+threadIdx.x-1] = (orig_data_copy[(left_boundary+right_boundary)/2]+orig_data_copy[(left_boundary+right_boundary)/2+1])/2;
				}
                	}
                }
                __syncthreads(); // Don't proceed to the next smaller expected K until all the threads have come to the same conclusion.
        }
}

/* The results of the segmentation go into T** segmented_sequences and size_t *segmented_seq_lengths, which are arrays that get allocated here (you should free them later). */
template<typename T>
__host__ void
adaptive_segmentation(T **sequences, size_t *seq_lengths, int num_seqs, int min_segment_length,
                      T ***segmented_sequences, size_t **segmented_seq_lengths, int prefix_length_to_skip, cudaStream_t stream = 0) {

	// If a real sequence segment was split over two sample averaging windows, we need to ensure that the window is 1/3 (or less) of the segment length so
	// as to get a representative median of that segment in at least one window.
	int downaverage_width = DIV_ROUNDUP(min_segment_length,3);
	short threads_per_block = CUDA_THREADBLOCK_MAX_THREADS;
	if(threads_per_block > MAX_DP_SAMPLES){	// any more threads than samples assigned per threadblock would cause neeedless spinning of the wheels
		threads_per_block = MAX_DP_SAMPLES;
	}
	int samples_per_block = threads_per_block*downaverage_width;
	int maximum_k_per_subtask = DIV_ROUNDUP(threads_per_block,((float) min_segment_length)/downaverage_width);

	// TODO: maybe do a number of grids and use multiple devices if present rather than doing all the computation in one grid (and the associated memory requirement of that)
	int deviceCount;
	cudaGetDeviceCount(&deviceCount); CUERR("Getting GPU device count in segmentation setup method");
	
	// Suss out the total queries size so we can allocate the right amount of working buffers and results arrays.
	long *all_seqs_downaverage_length; // breaking it down into what's need per device based on assigned seqs for each device
	cudaMallocHost(&all_seqs_downaverage_length, sizeof(long)*deviceCount); CUERR("Allocating CPU memory for array of downaveraged seq length totals");
	for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		all_seqs_downaverage_length[currDevice] = 0; // poor man's memset()
	}
	long total_expected_segments = 0;
	int longest_query = 0;
	T ***gpu_rawseqs;
	cudaMallocHost(&gpu_rawseqs, sizeof(T **)*deviceCount); CUERR("Allocating CPU memory for array of segmenting raw query starts");
	for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		cudaSetDevice(currDevice);
		cudaMalloc(&gpu_rawseqs[currDevice], sizeof(T *)*num_seqs);   CUERR("Allocating GPU memory for segmenting raw query starts");
	}
	T ***rawseq_ptrs;
	cudaMallocHost(&rawseq_ptrs, sizeof(T **)*deviceCount); CUERR("Allocating CPU memory for array of segmenting raw queries");
	for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		cudaSetDevice(currDevice);
		cudaMallocHost(&rawseq_ptrs[currDevice], sizeof(T *)*num_seqs);   CUERR("Allocating CPU memory for segmenting raw queries");
	}
	T **padded_segmented_sequences;
        cudaMallocHost(&padded_segmented_sequences, sizeof(T *)*num_seqs); CUERR("Allocating CPU memory for the padded segmented sequence pointers");
        cudaMallocManaged(segmented_sequences, sizeof(T *)*num_seqs); CUERR("Allocating managed memory for the segmented sequence pointers");
	cudaMallocManaged(segmented_seq_lengths, sizeof(size_t)*num_seqs); CUERR("Allocating managed memory for the segmented sequence lengths");
	cudaStream_t *dev_stream;
        cudaMallocHost(&dev_stream, sizeof(cudaStream_t)*deviceCount); CUERR("Allocating CPU memory for sequence segmentation streams");
	for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		cudaSetDevice(currDevice);
		cudaStreamCreate(&dev_stream[currDevice]); CUERR("Creating device stream for sequence segmentation");
	}
	for(int i = 0; i < num_seqs; ++i){
		if(seq_lengths[i] > longest_query){
			longest_query = seq_lengths[i];
		}
		// Keep a tally of the max number of segments that can be generated (we need to allocate memory for this later)
		(*segmented_seq_lengths)[i] = maximum_k_per_subtask*DIV_ROUNDUP(seq_lengths[i],samples_per_block);
        	total_expected_segments += (*segmented_seq_lengths)[i];

    		// Asynchronously slurp each query into device memory for maximum PCIe bus transfer rate efficiency from CPU to the GPU via the Copy Engine, or lazy copy in managed memory. 
		for(int currDevice = 0; currDevice < deviceCount; currDevice++){
			cudaSetDevice(currDevice);
			T **rawseq_ptr = rawseq_ptrs[currDevice]; // splitting the data up amongst the GPUs available
			if(i%deviceCount == currDevice){ // Not for use in this GPU, set the sequence pointer to null so it'll be skipped in the seg kernel
    				cudaMalloc(&rawseq_ptr[i], sizeof(T)*seq_lengths[i]);   CUERR("Allocating GPU memory for segmenting raw input sequence");
    				cudaMemcpyAsync(rawseq_ptr[i], sequences[i], sizeof(T)*seq_lengths[i], cudaMemcpyHostToDevice, dev_stream[currDevice]);          CUERR("Launching raw query copy to managed memory for segmentation");
        			all_seqs_downaverage_length[currDevice] += DIV_ROUNDUP(seq_lengths[i], downaverage_width);
			}
			else{
				rawseq_ptr[i] = (T *) 0;
			}
		}
	}
	for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		cudaSetDevice(currDevice);
		cudaMemcpyAsync(gpu_rawseqs[currDevice], rawseq_ptrs[currDevice], sizeof(T *)*num_seqs, cudaMemcpyHostToDevice, dev_stream[currDevice]); CUERR("Launching raw query pointer array copy from CPU to GPU for segmentation");
	}

	// Allocate all of the memory required to store the segmentation results in one go, then do the pointer math so that segmented_sequences
	// points to the start of the results slice corresponding to each input sequence.
	T *all_segmentation_results = 0;
	cudaMallocManaged(&all_segmentation_results, sizeof(T)*total_expected_segments);     CUERR("Allocating managed memory for segmentation results");
	long cursor = 0;
        for(int i = 0; i < num_seqs; ++i){
		padded_segmented_sequences[i] = &all_segmentation_results[cursor];

		cursor += maximum_k_per_subtask*DIV_ROUNDUP(seq_lengths[i],samples_per_block);
	}

	size_t **gpu_rawseq_lengths = 0;
	cudaMallocHost(&gpu_rawseq_lengths, sizeof(size_t *)*deviceCount); CUERR("Allocating CPU memory for array of raw input query lengths for segmentation");
	for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		cudaSetDevice(currDevice);
    		cudaMalloc(&gpu_rawseq_lengths[currDevice], sizeof(size_t)*num_seqs);   CUERR("Allocating GPU memory for raw input query lengths for segmentation");
		cudaMemcpyAsync(gpu_rawseq_lengths[currDevice], seq_lengths, sizeof(size_t)*num_seqs, cudaMemcpyHostToDevice, dev_stream[currDevice]); CUERR("Launching raw query lengths copy from CPU to GPU for segmentation");
	}

	// Divvy up the work into a kernel grid based on the longest input query.
	int max_req_block_in_a_query = DIV_ROUNDUP(longest_query,samples_per_block);

        // Working memory for the segmentation that will happen in the kernel to follow.
	// It's too big to fit in L1 cache, so use global memory, or host if required via Managed Memory :-P
	unsigned short **k_seg_path_working_buffer;
	cudaMallocHost(&k_seg_path_working_buffer, sizeof(unsigned short *)*deviceCount); CUERR("Allocating CPU memory for array of GPU segmentation buffer pointers");
	// Invoke the segmentation kernel once all the async memory copies are finished.
	cudaStreamSynchronize(stream);    CUERR("Synchronizing stream after raw query transfer to GPU for segmentation"); 
	for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		cudaSetDevice(currDevice);
        	size_t k_seg_path_size = sizeof(unsigned short)*all_seqs_downaverage_length[currDevice]*maximum_k_per_subtask;
       		cudaMalloc(&k_seg_path_working_buffer[currDevice], k_seg_path_size);
        	if(cudaGetLastError() != cudaSuccess) {
                	std::cerr << "Not enough GPU memory to do segmentation completely on device, using CUDA managed memory instead." << std::endl;
                	cudaMallocManaged(&k_seg_path_working_buffer, k_seg_path_size);
	        	std::cerr << "K seg buffer size is " << k_seg_path_size << " at " << k_seg_path_working_buffer << std::endl;
        	}
        	CUERR("Allocating GPU memory for segmentation paths");

		dim3 raw_grid(num_seqs, max_req_block_in_a_query, 1);
		//std::cerr << "Processing " << samples_per_block << " samples per threadblock with " << threads_per_block << " threads, max " << 
                //     maximum_k_per_subtask << " segments, grid (" << num_seqs << ", " << max_req_block_in_a_query << ",1)" << std::endl;
		int maxSharedMemoryPerBlockOptin = 48*1024; // default is 48K for device backward compatibility
		cudaDeviceGetAttribute(&maxSharedMemoryPerBlockOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, currDevice);
		cudaFuncSetAttribute(reinterpret_cast<void*>(adaptive_device_segmentation<T>), cudaFuncAttributeMaxDynamicSharedMemorySize, maxSharedMemoryPerBlockOptin); // Maximize explicit use of L1 cache
        	adaptive_device_segmentation<T><<<raw_grid,threads_per_block,maxSharedMemoryPerBlockOptin,dev_stream[currDevice]>>>(gpu_rawseqs[currDevice],
                               gpu_rawseq_lengths[currDevice], samples_per_block, maximum_k_per_subtask, min_segment_length, maxSharedMemoryPerBlockOptin, k_seg_path_working_buffer[currDevice], 
                               all_segmentation_results);  CUERR("Launching sequence segmentation");
	}
	int dotsPrinted = 0; // gimpy here as granularity is # devices not actual jobs, but it's better than nothing?
	for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		cudaStreamSynchronize(dev_stream[currDevice]); CUERR("Synchronizing CUDA device after sequence segmentation");
		dotsPrinted = updatePercentageComplete(currDevice+1, deviceCount, dotsPrinted);
		cudaStreamDestroy(dev_stream[currDevice]); CUERR("Destroying now-redundant CUDA device stream that was used for sequence segmentation");
		for(int i = 0; i < num_seqs; i++){
			if(rawseq_ptrs[currDevice][i] != 0){
				cudaFree(rawseq_ptrs[currDevice][i]); CUERR("Freeing GPU memory for a raw query");
			}
		}
		cudaFreeHost(rawseq_ptrs[currDevice]);			    CUERR("Freeing CPU memory for raw query pointers");
		cudaFree(gpu_rawseqs[currDevice]);                          CUERR("Freeing GPU memory for raw queries");
		cudaFree(gpu_rawseq_lengths[currDevice]);                   CUERR("Freeing GPU memory for raw query lengths");
        	cudaFree(k_seg_path_working_buffer[currDevice]);            CUERR("Freeing GPU memory for segmentation paths");
	}
	cudaFreeHost(rawseq_ptrs);                          CUERR("Freeing CPU memory for array of device raw query pointers");
	cudaFreeHost(gpu_rawseqs);                          CUERR("Freeing CPU memory for array of device raw queries");
        cudaFreeHost(k_seg_path_working_buffer);            CUERR("Freeing CPU memory for array of device segmentation path buffers");
	cudaFreeHost(all_seqs_downaverage_length);	    CUERR("Freeing CPU memory for array of downaverage lengths");
	cudaStreamSynchronize(stream);                  CUERR("Synchronizing stream after sequence segmentation");

	// See if the segments at the edge of each segmentation block need to be merged (i.e. a segment was artificially split across two CUDA kernel grid tasks).
	// The criterion is that the segments' medians differ by less than the proportion 'epsilon', which is automaticaly determined as the minimum difference between 
	// neighbouring elements *within* the segmentation blocks for a given segmented sequence.  *NOTA BENE: This assumes no change in the dynamic range of the signal over time.*
	// At the same time, we allocated enough memory for segmentation results where the actual K in each subtask (kernel call grid element) was the expected K.
	// If the adaptive segmentation actually found that there was less than K segments in the subtask, the remaining unused results slots will have a sentinel
	// value, numeric_limits<T>::max(), that we must eliminate before passing the answer back to the caller.
	for(int i = 0; i < num_seqs; ++i){
		T epsilon = std::numeric_limits<T>::max(); // N.B.: it's critical to use the std:: qualifier otherwise you're accessing device side limits from the imported cudahack
		T *segmented_sequence = padded_segmented_sequences[i];
		T previous_value = segmented_sequence[0];
		size_t segmented_seq_length = (*segmented_seq_lengths)[i]; // this is the preallocated max possible length of results, in reality it has a lot of undefined values probably
		// Find the minimum signal value change between segments that were generate together (i.e. no intervening sentinel (max) values).
		for(int j = 1; j < segmented_seq_length; ++j){
			T current_value = segmented_sequence[j];
			if(current_value != std::numeric_limits<T>::max() &&
                           previous_value != std::numeric_limits<T>::max() && 
                           (current_value <= previous_value && previous_value - current_value < epsilon || 
			    current_value > previous_value && current_value - previous_value < epsilon)){ // unused slots in the segmentation answer are max valued, so will not beat epsilon 
				// Take into account the fact that we could have neighbouring two segments with the same value 
				// because the segmentation algorithm uses a byte (0-255) scaled averaging of input sequence bins (as opposed to a sliding window) 
				// to calculate the residual sums of squares, but then the median value from adjacent segment member bins is returned, which could be the same.
				// In this case the epsilon is zero, and we will ignore that, skipping over these artifacts and taking the smallest non-zero epsilon.
				if(current_value-previous_value == 0){
					for(int k = j; k < segmented_seq_length; k++){
						// Shift down the values as part of the segment value deduplication
						segmented_sequence[k-1] = segmented_sequence[k];
						// Shortcircuit: end of the contiguous non-sentinel values
						if(segmented_sequence[k] == std::numeric_limits<T>::max()){
							break;
						}

					}
				}
				else{
					// Not using abs function because needs supported type hack (cast and recast) for short, etc.
					epsilon = current_value < previous_value ? (previous_value-current_value) : (current_value-previous_value);
				}
			}
			previous_value = current_value; 

		}
		bool prev_seg_val_undefined = false;
		int cursor = 0;
		for(int j = 0; j < segmented_seq_length; ++j){
			if(segmented_sequence[j] == std::numeric_limits<T>::max()){
				if(!prev_seg_val_undefined){
					prev_seg_val_undefined = true;
				}
				continue;
			}
			else{
                       		// Candidate for edge merge
                       		// i.e. at the start of a new subtask range
				if(prev_seg_val_undefined){
					prev_seg_val_undefined = false;
                           		if(segmented_sequence[cursor-1] < segmented_sequence[j]+epsilon && // i.e. very similar
                                   	   segmented_sequence[cursor-1] > segmented_sequence[j]-epsilon){
                               			segmented_sequence[cursor-1] = (segmented_sequence[cursor-1]+segmented_sequence[j])/2; // i.e. take the avg
					}
					else{ // No merge
					}
				}
                                if(cursor != j){ // We've skipped something already, so all subsequent segment values need to shift left
                                                 // Copy to results as-is.
                                       segmented_sequence[cursor] = segmented_sequence[j];
                                }
                               	cursor++;
                       }

		}
		// Set the reported segments total for the seq to reflect the adaptive segmentation results.
		// In rare instances with a tiny amount of final block index data, you can end up with sqrt(max) avg that we should ignore.
		if(segmented_sequence[cursor-1] >= std::numeric_limits<T>::max()/2){
			cursor--;
		}

		int prefix_length_skipped = 0;
		if(prefix_length_to_skip){
			if(cursor <= prefix_length_to_skip){ // We've been asked to skip more sequence elements than exist, provide the bare minimum (should be removed/ignored by caller).
				prefix_length_skipped = cursor - 1;
				cursor = 1;
			}
			else{
				prefix_length_skipped = prefix_length_to_skip;
				cursor -= prefix_length_to_skip;
			}
		}
		(*segmented_seq_lengths)[i] = cursor;

		// Now that we know the real length, allocate the final memory for the sequence (so later we can free up the big block we wrote to in bulk)
		cudaMallocManaged(&(*segmented_sequences)[i], sizeof(T)*cursor); CUERR("Allocating managed memory for a segmented sequence");
		cudaMemcpyAsync((*segmented_sequences)[i], &segmented_sequence[prefix_length_skipped], sizeof(T)*cursor, cudaMemcpyHostToHost, stream); CUERR("Copying a segmented sequence to managed memory");
	}
	cudaStreamSynchronize(stream); CUERR("Synchronizing stream after segemented sequence copy to managed memory");// ensure all the copying had finished before freeing the original results
	cudaFreeHost(padded_segmented_sequences); CUERR("Freeing managed memory for segmented sequence pointers");
	// No need to free this as the first pointer in padded_segmented_sequences is the same address.
	//cudaFreeHost(all_segmentation_results);  CUERR("Freeing managed memory for segmented sequences buffer");
}


#endif
