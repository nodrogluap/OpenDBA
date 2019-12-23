#ifndef __SEGMENTATION_HPP
#define __SEGMENTATION_HPP

// For INT_MAX
#include <climits>
// For FLT_MAX etc.
#include <cfloat>
#include "cuda_utils.hpp"
#include "limits.hpp" // for device side numeric_limits min() and max()

using namespace cudahack; // for device side numeric_limits

/* The following kernel function is used to do data reduction of unimodal queries
   (e.g. nanopore 4000Hz raw current data to dwell segments, a.k.a. "events" in a "squiggle")
   with the Bellman k-segmentation (dynamic programming) algorithm and even weights for sample error.
   The minimized measure is the mean squared error of the segmental regression.
   The segmentation is "adaptive" insofar as a maximum number of expected segments is passed in, but if 
   <min_segment_size data points are creating their own segments we assume that they are noise and we should try 
   segmentaing again with a smaller number of expected segments. 

   Process a bunch of different data series in parallel on the GPU. This is an intense dynamic programming process that has some parallelizable steps. Also,
   segmentation is a semi-local optimization problem in practice, so if you have a really long query, we need to split it up into at most chunks of MAX_SEGMENT_SRC_LENGTH for segmentation
   to keep things moving along at a reasonable pace with shared data in each kernel's L1 cache, with a (MAX_SEGMENT_SRC_LENGTH-avg_segment_length)/MAX_SEGMENT_SRC_LENGTH chance that we falsely
   merge two segments at the border between threadblock results. That's assuming...

   define MAX_SEGMENT_SRC_LENGTH MAX_DP_SAMPLES*MAX_DOWNSAMPLING

   The distance calculation and usage are in heavily used loops, so to keep it in L1 cache we need to downsample the original data at least a bit to fit.
   This also has the effect of minimizing the impact of errant data points in the stream.

   We also set a maximum size for a segment (as a multiple of the average size), in order to reduce the search space and memory requirement.
*/
// MAX_DP_SAMPLES should not exceed 256. Otherwise, sums and squares accumulators used in the segmentation kernel could overflow in edge cases of extremely noisy, high dynamic range signal.
#define MAX_DP_SAMPLES 256
#define MIN_DOWNAVERAGING 2
#define MAX_DOWNAVERAGING 16
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

template<typename T>
__global__ void adaptive_device_segmentation(T **all_series, unsigned int *all_series_lengths, int raw_samples_per_threadblock, short expected_k, 
                                             short downsample_width, float max_segment_avg_multiple, int min_segment_size, T *sums, T *squares, unsigned short *k_seg_path_working_buffer, 
                                             T *output_segmental_medians){

	// If a series length is 0, there's nothing to do... obviously
	int orig_N = all_series_lengths[blockIdx.x];
	if(blockIdx.y*raw_samples_per_threadblock > orig_N){
		return; // this is safe as all threads in this block will take this short-circuit branch and not cause hangs in the __syncthreads() calls below
	}
	T *series = all_series[blockIdx.x];

	// "*_before_us" vars are used to determine the offset of this threadblock's required global memory intermediate-working and output values.
	// As the input all_series is a ragged array, we need to figure out where to write the values for this query
	// by summing up the lengths of the raw and segmented data before this block.
	int datapoints_before_us = 0;
	int segments_before_us = 0;
	for(int query_num = 0; query_num < blockIdx.x; query_num++){
		// K segments guaranteed per full threadblock. Count all full thread blocks for this query, plus the last partial block.
		int query_length = all_series_lengths[query_num];
		datapoints_before_us += query_length/downsample_width;
		segments_before_us += query_length/raw_samples_per_threadblock*expected_k + DIV_ROUNDUP((query_length%raw_samples_per_threadblock),(raw_samples_per_threadblock/expected_k));
	}
	datapoints_before_us += raw_samples_per_threadblock/downsample_width*blockIdx.y;
	segments_before_us += blockIdx.y*expected_k;

	// Let's switch to local mode for the indexing, N is the threadblock size or the remainder when it's the last block for this query
	int N = ((blockIdx.y+1)*raw_samples_per_threadblock > orig_N) ? orig_N%raw_samples_per_threadblock : raw_samples_per_threadblock;

	// If this is the last thread block for this query, pro-rate the expected_k passed in (which is for a full threadblock)
	if((blockIdx.y+1)*raw_samples_per_threadblock > orig_N){
		expected_k = DIV_ROUNDUP((orig_N%raw_samples_per_threadblock),DIV_ROUNDUP(raw_samples_per_threadblock, expected_k));
	}

        // Must allocate shared memory in bytes before it can be cast to the template variable
        extern __shared__ __align__(sizeof(T)) unsigned char my_shared_mem[];
        T *downsample_qtype = reinterpret_cast<T *>(my_shared_mem);
        // extern __shared__ T downsample_qtype

        // Note that options, K_SEG_* and DIST are all indexed starting at 1, not 0 for logical simplicity.
        // An index array allowing for the reconstruction of the regression with the lowest cost.
        unsigned short *k_seg_path = &k_seg_path_working_buffer[expected_k*datapoints_before_us];
        // To save memory and split homopolymers in nanopore data, cap the size of any given segment. This assumes some kind of Poisson like distribution for segment size.
        // The extent (e) of the DP search for any given downsamples data index, just a short name for readability of SUMS(), SQUARES(), etc. macro calls below
        short e = short(N/downsample_width/expected_k*max_segment_avg_multiple);
        sums = &sums[raw_samples_per_threadblock/downsample_width*e*blockIdx.y];
        squares = &squares[raw_samples_per_threadblock/downsample_width*e*blockIdx.y];
        short initial_expected_k = expected_k;

        if(threadIdx.x*downsample_width <= N-downsample_width){ // Added -1 so we don't run past the length of series
                downsample_qtype[threadIdx.x] = 0;
                for(int i = 0; i < downsample_width; i++){
                        downsample_qtype[threadIdx.x] += series[downsample_width*threadIdx.x+i+blockIdx.y*raw_samples_per_threadblock]; // warp fetches should coalesce to slurp up adjacent global memory fairly quickly
                }
                downsample_qtype[threadIdx.x] /= downsample_width; // correct averaged values are in 0, downsample_width, 2*downsample_width, etc.
        }

        int N_ds = N/downsample_width;

        bool exit = false;

        while(!exit) {

                // No rescaling necessary, just cast the type in a massively parallel manner
                if(sizeof(T) == 1){
                        // Squeeze the copying into the first few warps (low threadIdx.x values) for improved speed
                        if(threadIdx.x < N_ds){
                                SUMS(threadIdx.x,threadIdx.x,e) = (T) downsample_qtype[threadIdx.x];
                                SQUARES(threadIdx.x,threadIdx.x,e) = SUMS(threadIdx.x,threadIdx.x,e)*SUMS(threadIdx.x,threadIdx.x,e);
                        }
                }
                else{
                        // Rewrite the data to be segmented as unsigned characters (256 levels) so we can cram as much into L1 cache as possible. To
                        // do this we need to find the downsampled min and max values, then scale everything to that so we lose as little resolution as possible.
                        T *warp_maxs = (T *) &downsample_qtype[N_ds]; // double duty shared memory, won't need sums until finished with maxs/mins
                        T *warp_mins = &warp_maxs[blockDim.x/CUDA_WARP_WIDTH-1]; // map/reduce temp storage for each warp min except first in threadblock

                        __syncwarp();
                        T warp_max = sizeof(T) == 2 ? SHRT_MIN : FLT_MIN;
                        T warp_min = sizeof(T) == 2 ? SHRT_MAX : FLT_MAX;
                        if(threadIdx.x%CUDA_WARP_WIDTH == 0){ // find range for the warp
                                for(int i = 0; i < CUDA_WARP_WIDTH; i++){
                                        int pos = threadIdx.x+i;
                                        if(threadIdx.x+i < N_ds){
                                                if(downsample_qtype[pos] > warp_max) warp_max = downsample_qtype[pos];
                                                if(downsample_qtype[pos] < warp_min) warp_min = downsample_qtype[pos];
                                        }
                                }
                                if(threadIdx.x && threadIdx.x < N_ds){ // stick min & max within-warp results into shared memory for all but the first warp
                                        warp_maxs[threadIdx.x/CUDA_WARP_WIDTH-1] = warp_max;
                                        warp_mins[threadIdx.x/CUDA_WARP_WIDTH-1] = warp_min;
                                }
                        }
                        __syncthreads();
                        if(! threadIdx.x){ // find range for the thread block
                                for(int i = 0; i < N_ds/CUDA_WARP_WIDTH; i++){
                                        if(warp_maxs[i] > warp_max) warp_max = warp_maxs[i];
                                        if(warp_mins[i] < warp_min) warp_min = warp_mins[i];
                                } //note: don't use warp_maxs or warp_mins after this point as SUMS() calls clobber them
                                // rescale values according to this range and safely clobber the downsample_qtype array (by stride, in one thread only) with the new column-major sums array at the same time.
                                // Could sync threads and parallelized rescaling if we didn't clobber, but don't think it'd net save any cycles and every shared memory slot is precious.
                                for(int i = 0; i < N_ds; i++){
                                        SUMS(i,i,e) = (unsigned char) (256.0*((downsample_qtype[i]-warp_min)/(warp_max-warp_min+1))); // +1 to avoid "div by 0" errors in edge case of absolutely no variance
                                }
                                for(int i = 0; i < N_ds; i++){
                                        SQUARES(i,i,e) = SUMS(i,i,e)*SUMS(i,i,e);
                                }
                        }
                }
                __syncthreads(); // make sure we've loaded all the data before starting the compute

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

                unsigned int *options = (unsigned int *) &downsample_qtype[N_ds]; // another double duty shared memory spot as we don't need the downsampled values any more

                // Keep a matrix of the total segmentation costs for any p-segmentation of a subsequence series[1:n] where 1<=p<=k and 1<=n<=N. The 0th column at
                // the beginning encodes that a (k-1)-segmentation is penalized compared to a whole-segment average.
                // In the segmentation dynamic programming, we only ever need to be storing the current value of K_SEG_DIST for p, and reading the optimal solutions for p-1.
                // This means that in practice we only need to store 2*(N+1) values of K_SEG_DIST in an alternating storage pattern, which is what the K_SEG_DIST macro defined earlier does.
                unsigned int *k_seg_dist = options + sizeof(unsigned int)*(N_ds+1);

                if(threadIdx.x < N_ds){
                        // Initialize regression distances for the case k=1 (a single segment) directly from the precomputed starting-at-zero cumulative distance calculations.
                        if(threadIdx.x+1 <= e) {
                                K_SEG_DIST(1,threadIdx.x+1,N_ds) = DIST(1,threadIdx.x+1,e); // TODO indexing for standard, change to save memory
                        } else {
                                K_SEG_DIST(1,threadIdx.x+1,N_ds) = INT_MAX/2; // div by 2 to avoid overflow when adding later
                        }
                        // Initialize the path for the trivial case where we have p segments through p datapoints.
                        // The right boundary for the p case is p by the pigeonhole principle.
                        if(threadIdx.x < expected_k){
                                K_SEG_PATH(threadIdx.x+1,threadIdx.x,N_ds) = (unsigned short) expected_k-1;
                        }
                }
                __syncthreads();

                // Now incrementally calculate the optimal p-segmentation solution in series for each 1 < p <= k.
                for(int p=2; p <= expected_k; p++){

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

                // At this point we can clobber the cumulative distance L1 cache variables as we are done with them.
                // Let's load the original data series now so the calculations below here will be less affected by latency in most threads.
                T *orig_data_copy = (T *) &downsample_qtype[N_ds];
                if(threadIdx.x*downsample_width+blockIdx.y*raw_samples_per_threadblock < orig_N){
                        for(int i = 0; i < downsample_width && threadIdx.x*downsample_width + i < raw_samples_per_threadblock && threadIdx.x*downsample_width + i < N; i++){
                                orig_data_copy[threadIdx.x*downsample_width+i] = series[threadIdx.x*downsample_width+blockIdx.y*raw_samples_per_threadblock+i];
                        }
                }

                int *breakpoints = (int *) &orig_data_copy[N]; // Another safe L1 cache var clobber, sizeof(int)*(expected_k+1).
                // Single thread, backtrack from the right end of the dataset, which has a known optimal k-segment solution right boundary
                // (the length of the input list), to that last segment's left border, taking the
                // k-1 solution from that point, etc. until we get to the single segment which necessarily starts at the left edge (first input item).
                if(! threadIdx.x){
                        breakpoints[expected_k] = N_ds; // NOT N in downsampled space... we'll check for overrun of orig_N later.
                        for (int p = expected_k-1; p >= 1; p--){
                                breakpoints[p] = K_SEG_PATH(p+1,breakpoints[p+1]-1,N_ds);
                        }
                        breakpoints[0] = 0; // Left exclusive boundary is always 0 (one-based indexing) so we start at data point 1 for any segmentation.
                }
                __syncthreads();

                // Check if any ranges are unusually small data segments (i.e. <min_segment_size/downsample_width), which means that we're letting the noise in the data still create segments where they shouldn't.
                // If this is the case, we want to lower the expected number of segments (hence "adaptive" in the function name) and try again.
                exit = true;
                for(int i = 1; i <= expected_k; i++) {
                        int left = breakpoints[i-1];
                        int right = breakpoints[i];
                        // Redundant computation amongst threads here is as a single computation followed by __syncthreads() because of unsynchronized median calculation below.
                        if(right-left < min_segment_size/downsample_width){ 
                                exit = false;
                        }
                }
                if(!exit) {
                        expected_k--;
                }
                else{
			if(threadIdx.x > expected_k && threadIdx.x <= initial_expected_k) {
				// Sentinel answer for unused result slots (K is smaller than the expect value passed in), since the amount of space for answers was preallocated.
                        	output_segmental_medians[segments_before_us+threadIdx.x-1] = numeric_limits<T>::max();
			}

                	// In parallel, get the median from each segment and set its value in the constant cache query value, merging/averaging values at each shared threadblock edge.
                	// We are using the median rather than the mean as it is less affected by outlier datapoints caused by noise.
                	else if(threadIdx.x > 0 && threadIdx.x <= expected_k){
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
					output_segmental_medians[segments_before_us+threadIdx.x-1] = (orig_data_copy[left_boundary]+orig_data_copy[right_boundary])/2;
				}
                        	datapoints_before_us = blockIdx.y*raw_samples_per_threadblock;
                	}
                }
                __syncthreads(); // Don't proceed to the next smaller expected K until all the threads have come to the same conclusion.
        }
}

/* The results of the segmentation go into T** segmented_sequences and unsigned int *segmented_seq_lengths, which are arrays that must must be preallocated. 
   The individual segmented sequences will be assigned in this method as part of one big memory block allocation, so you can simply 
   cudaFree(segmented_sequences) in the caller when you're done with them. Epsilon is the signal value proportion criterion for merging neighbouring segments 
   that may be artificial split by the parallelized divide-and-conquer approch to DP segmentation used here. */
template<typename T>
__host__ void
adaptive_segmentation(T **sequences, unsigned int *seq_lengths, int num_seqs, int expected_segment_length, float max_attenuation, float epsilon,
                      T **segmented_sequences, unsigned int *segmented_seq_lengths, cudaStream_t stream = 0) {

	// Suss out the total queries size so we can allocate the right amount of working buffers and results arrays.
	long all_seqs_total_length = 0;
	long total_expected_segments = 0;
	long max_result_segments = 0;
	int longest_query = 0;
	T **gpu_rawseqs;
	cudaMalloc(&gpu_rawseqs, sizeof(T *)*num_seqs);   CUERR("Allocating GPU memory for segmenting raw query starts");
	for(int i = 0; i < num_seqs; ++i){
        	all_seqs_total_length += seq_lengths[i];
		if(seq_lengths[i] > longest_query){
			longest_query = seq_lengths[i];
		}
		// Keep a tally of the max number of segments that can be generated (we need to allocate memory for this later)
		segmented_seq_lengths[i] = DIV_ROUNDUP(seq_lengths[i],expected_segment_length);
        	total_expected_segments += segmented_seq_lengths[i];

    		// Asynchronously slurp the queries into device memory in one request for maximum PCIe bus transfer rate efficiency from CPU to the GPU.
    		cudaMallocManaged(&gpu_rawseqs[i], sizeof(T)*seq_lengths[i]);   CUERR("Allocating GPU memory for segmenting raw input sequence");
    		cudaMemcpyAsync(&gpu_rawseqs[i], sequences[i], sizeof(T)*seq_lengths[i], cudaMemcpyHostToDevice, stream);          CUERR("Launching raw query copy from CPU to GPU for segmentation");
	}

	// Allocate all of the memory required to store the segmnentation results in one go, then do the pointer math so that segmented_sequences
	// points to the start of the results slice corresponding to each input sequence.
	T *all_segmentation_results = 0;
	cudaMallocManaged(&all_segmentation_results, sizeof(T)*total_expected_segments);     CUERR("Allocating managed memory for segmentation results");
	long cursor = 0;
        for(int i = 0; i < num_seqs; ++i){
		segmented_sequences[i] = &all_segmentation_results[cursor];
		cursor += DIV_ROUNDUP(seq_lengths[i],expected_segment_length);
	}

	unsigned int *gpu_rawseq_lengths = 0;
    	cudaMalloc(&gpu_rawseq_lengths, sizeof(unsigned int)*num_seqs);   CUERR("Allocating GPU memory for raw input query lengths for segmentation");
	cudaMemcpyAsync(&gpu_rawseq_lengths, seq_lengths, sizeof(unsigned int)*num_seqs, cudaMemcpyHostToDevice, stream); CUERR("Launching raw query lengths copy from CPU to GPU for segmentation");

	// Pick the smallest averaging window for downsampling that we can given the L1 cache needs for the given expected_segment_length.
	// Note that this does not depend on the size of type T because internally the segmentation kernel rescales the input to 256 levels and
	// primarily works internally with unsigned chars for the dynamic programming matrices of the segmentation.
	short threads_per_block = CUDA_THREADBLOCK_MAX_THREADS;
	short downaverage_width = MAX_DOWNAVERAGING;
	int downaverages_per_segment = expected_segment_length/downaverage_width;
	while(downaverage_width >= MIN_DOWNAVERAGING && downaverages_per_segment < 2){ // too low resolution for taking the median
		downaverage_width--;
		downaverages_per_segment = expected_segment_length/downaverage_width;
	}
	if(downaverage_width < MIN_DOWNAVERAGING){
		std::cerr << "Expected segment length of " << expected_segment_length << " is incompatible with minimum fixed downaveraging strategy of " <<
                   MIN_DOWNAVERAGING << " and median value characterization, aborting." << std::endl;
		exit(INSUFFICIENT_L1CACHE_FOR_EXPECTED_SEGMENT_SIZE);
	}

	// Calculate the optimal number of data elements to be processed by each kernel, based on the compute constraints of max
	// CUDA_THREADBLOCK_MAX_THREADS threads in a threadblock, MAX_DP_SAMPLES, and the expected number of segments.
	int samples_per_block = MAX_DP_SAMPLES*downaverage_width;
        if(samples_per_block > CUDA_THREADBLOCK_MAX_THREADS) {
                samples_per_block = CUDA_THREADBLOCK_MAX_THREADS;
        }
	// Divvy up the work into a kernel grid based on the longest input query.
	int max_req_block_in_a_query = DIV_ROUNDUP(longest_query,samples_per_block);

	int expected_k = DIV_ROUNDUP(samples_per_block,expected_segment_length);

        // Working memory for the segmentation that will happen in the kernel to follow.
	// It's too big to fit in L1 cache, so use global memory, or host if required via Managed Memory :-P
	int longest_allowed_segment = expected_segment_length*max_attenuation;
	unsigned short *k_seg_path_working_buffer;
        size_t k_seg_path_size = sizeof(unsigned short)*(all_seqs_total_length/downaverage_width+1)*expected_k;
	cudaMallocManaged(&k_seg_path_working_buffer, k_seg_path_size);     CUERR("Allocating managed memory for segmentation paths");
	int required_threadblock_shared_memory = 48000;

	// Invoke the segmentation kernel once all the async memory copies are finished.
	// Not neccasery if streams are working properly: cudaStreamSynchronize(stream);    CUERR("Synchronizing stream after raw query transfer to GPU for segmentation"); 
	dim3 raw_grid(num_seqs, max_req_block_in_a_query, 1);
        adaptive_device_segmentation<T><<<raw_grid,threads_per_block,required_threadblock_shared_memory,stream>>>(gpu_rawseqs,
                               gpu_rawseq_lengths, samples_per_block, expected_k, downaverage_width, max_attenuation, k_seg_path_working_buffer, 
                               all_segmentation_results);  CUERR("Launching sequence segmentation");
	cudaStreamSynchronize(stream);                  CUERR("Synchronizing stream after sequence segmentation");
        cudaFree(k_seg_path_working_buffer);            CUERR("Freeing GPU memory for segmentation paths");
	cudaFree(gpu_rawseqs);                          CUERR("Freeing GPU memory for raw queries");
	cudaFree(gpu_rawseq_lengths);                     CUERR("Freeing GPU memory for raw query lengths");

	// See if the segments at the edge of each segmentation block need to be merged (i.e. a segment was artificially split across two CUDA kernel grid tasks).
	// The criterion is that the segments' medians differ by less than the proportion 'epsilon' 
	// At the same time, we allocated enough memory for segmentation results where the actual K in each subtask (kernel call grid element) was the expected K.
	// If the adaptive segmentation actually found that there was less than K segements in the subtask, the remaining unused results slots will have a sentinel
	// value, numeric_limits::max(T), that we must eliminate before passing the answer back to the caller.
	cudaMemcpyAsync(segmented_sequences, all_segmentation_results, cudaMemcpyDeviceToHost, stream);  CUERR("Copying sequence segmentation results to host");
	cudaStreamSynchronize(stream);                  CUERR("Synchronizing stream after sequence segmentation results copy to host");
	for(int i = 0; i < num_seqs; ++i){
		int cursor = 0;
		for(int j = 0; i < segmented_seq_lengths[i]; ++i){
			if(segmented_sequences[i][j] == std::numeric_limits<T>::max()){
				continue; // Skipping this result by not incrementing the cursor
			}
			if(cursor != j){ // We've skipped something already, so all subsequent segment values need to shift left
				// Candidate for edge merge
				if(cursor != 0 && j != 0 && segmented_sequences[i][j-1] == std::numeric_limits<T>::max()  // i.e. at the start of a new subtask range
                                                         && segmented_sequences[i][cursor-1] <= segmented_sequences[i][j]*(1+epsilon) // i.e. very similar
							 && segmented_sequences[i][cursor-1] >= segmented_sequences[i][j]*(1-epsilon)){	
					segmented_sequences[i][cursor-1] = (segmented_sequences[i][cursor-1]+segmented_sequences[i][j])/2; // i.e. take the avg
				}
				else{ // Copy to results as-is.
					segmented_sequences[i][cursor] = segmented_sequences[i][j];
				}
			}
			cursor++;
		}
		// Set the reported segments total for the seq to reflect the adaptive segmentation results.
		segmented_seq_lengths[i] = cursor;
	}
}


#endif
