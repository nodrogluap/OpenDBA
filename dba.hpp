#ifndef __dba_hpp_included
#define __dba_hpp_included

//#include <chrono>
//#include <thread>
#include <thrust/sort.h>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>

#include "multithreading.h"
#include "gpu_utils.hpp"
#include "cuda_utils.hpp"
#include "dtw.hpp"
#include "limits.hpp" // for CUDA kernel comptaible max()

#define ARITH_SERIES_SUM(n) (((n)*(n+1))/2)

using namespace cudahack;

struct heterogeneous_workload {
    void *dtwCostSoFar_memptr; // we only free it, so datatype templating is not neccesary
    unsigned char *pathMatrix_memptr;
    cudaStream_t stream;
};

__host__
CUT_THREADPROC dtwStreamCleanup(void *void_arg){
	heterogeneous_workload *workload = (heterogeneous_workload *) void_arg;
	// ... GPU is done with processing, continue on new CPU thread...

	// Free dynamically allocated resources that were associated with data processing done in the stream.
	//std::cerr << "Freeing memory" << std::endl;
	if(workload->dtwCostSoFar_memptr != 0){
		cudaFree(workload->dtwCostSoFar_memptr); CUERR("Freeing DTW intermediate cost values");
	}
	if(workload->pathMatrix_memptr != 0){
		cudaFree(workload->pathMatrix_memptr); CUERR("Freeing DTW path matrix");
	}
	cudaStreamDestroy(workload->stream); CUERR("Removing a CUDA stream after completion");

	CUT_THREADEND;
}

__host__
void CUDART_CB dtwStreamCleanupLaunch(cudaStream_t stream, cudaError_t status, void *streamResources) {
	// Check status of GPU after stream operations are done. Die if there was a failure.
	CUERR("On callback after DTWDistance calculations completed.");

	// Spawn new CPU worker thread and perform stream resource cleanup on the CPU (since calling the CUDA API from within this callback is not allowed according to the docs).
	cutStartThread(dtwStreamCleanup, streamResources);
}

template<typename T>
__host__ int approximateMedoidIndex(T **gpu_sequences, size_t maxSeqLength, size_t num_sequences, size_t *sequence_lengths, char **sequence_names, size_t **gpu_sequence_lengths, int use_open_start, int use_open_end, char *output_prefix, cudaStream_t stream) {
	int deviceCount;
 	cudaGetDeviceCount(&deviceCount); CUERR("Getting GPU device count");

	unsigned int *maxThreads;
	cudaMallocHost(&maxThreads, sizeof(unsigned int)*deviceCount); CUERR("Allocating CPU memory for CUDA device properties");
	cudaDeviceProp deviceProp;
	for(int i = 0; i < deviceCount; i++){
		cudaGetDeviceProperties(&deviceProp, i); CUERR("Getting GPU device properties");
		maxThreads[i] = deviceProp.maxThreadsPerBlock;
	}
	//std::cerr << "Maximum of " << maxThreads << " threads per block on this device" << std::endl;

	T **gpu_dtwPairwiseDistances = 0;
	cudaMallocHost(&gpu_dtwPairwiseDistances,sizeof(T *)*deviceCount);  CUERR("Allocating CPU memory for GPU DTW pairwise distances' pointers");

	size_t numPairwiseDistances = (num_sequences-1)*num_sequences/2; // arithmetic series of 1..(n-1)
	for(int i = 0; i < deviceCount; i++){
		cudaSetDevice(i);
		cudaMalloc(&gpu_dtwPairwiseDistances[i], sizeof(T)*numPairwiseDistances); CUERR("Allocating GPU memory for DTW pairwise distances");
	}
	T *cpu_dtwPairwiseDistances = 0;
	cudaMallocHost(&cpu_dtwPairwiseDistances, sizeof(T)*numPairwiseDistances); CUERR("Allocating page locked CPU memory for DTW pairwise distances");

	int priority_high, priority_low, descendingPriority;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
	descendingPriority = priority_high;
	// To save on space while still calculating all possible DTW paths, we process all DTWs for one sequence at the same time.
        // So allocate space for the dtwCost to get to each point on the border between grid vertical swaths of the total cost matrix.
	int dotsPrinted = 0;
	std::cerr << "Step 2 of 3: Finding medoid" << std::endl;
	std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
	char spinner[4] = { '|', '/', '-', '\\'};
	for(size_t seq_index = 0; seq_index < num_sequences-1; seq_index++){
		int currDevice = seq_index%deviceCount;
		cudaSetDevice(currDevice);
        	dim3 threadblockDim(maxThreads[currDevice], 1, 1);
        	dim3 gridDim((num_sequences-seq_index-1), 1, 1);
		size_t current_seq_length = sequence_lengths[seq_index];
		// We are allocating each time rather than just once at the start because if the sequences have a large
                // range of lengths and we sort them from shortest to longest we will be allocating the minimum amount of
		// memory necessary.
		size_t dtwCostSoFarSize = sizeof(T)*current_seq_length*gridDim.x;
		size_t freeGPUMem;
		size_t totalGPUMem;
		cudaMemGetInfo(&freeGPUMem, &totalGPUMem);	
		while(freeGPUMem < dtwCostSoFarSize){
			//std::this_thread::sleep_for(std::chrono::seconds(1));
			usleep(1000);
			//std::cerr << "Waiting for memory to be freed before launching " << std::endl;
			cudaMemGetInfo(&freeGPUMem, &totalGPUMem);
		}
		int newDotTotal = 100*((float) seq_index/(num_sequences-2));
		if(newDotTotal > dotsPrinted){
			for(; dotsPrinted < newDotTotal; dotsPrinted++){
				std::cerr << "\b.|";
			}
		}
		else{
			std::cerr << "\b" << spinner[seq_index%4];
		}
		T *dtwCostSoFar = 0;
		T *cpu_dtwCostSoFar = 0;
		cudaMalloc(&dtwCostSoFar, dtwCostSoFarSize);  CUERR("Allocating GPU memory for DTW pairwise distance intermediate values");
		cudaMallocHost(&cpu_dtwCostSoFar, dtwCostSoFarSize);  CUERR("Allocating CPU memory for DTW pairwise distance intermediate values");

		// Make calls to DTWDistance serial within each seq, but allow multiple seqs on the GPU at once.
		cudaStream_t seq_stream; 
		cudaStreamCreateWithPriority(&seq_stream, cudaStreamNonBlocking, descendingPriority);
		if(descendingPriority < priority_low){
			descendingPriority++;
		}
		for(size_t offset_within_seq = 0; offset_within_seq < maxSeqLength; offset_within_seq += threadblockDim.x){
			// We have a circular buffer in shared memory of three diagonals for minimal proper DTW calculation.
        		int shared_memory_required = threadblockDim.x*sizeof(T)*3;
			// Null char pointer arg below means we aren't storing the path for each alignment right now.
			DTWDistance<<<gridDim,threadblockDim,shared_memory_required,seq_stream>>>((T *) 0, (size_t) 0, seq_index, offset_within_seq, gpu_sequences[currDevice], maxSeqLength,
										num_sequences, gpu_sequence_lengths[currDevice], dtwCostSoFar, (unsigned char *) 0, (size_t) 0, gpu_dtwPairwiseDistances[currDevice], use_open_start, use_open_end); CUERR("DTW vertical swath calculation with path storage");
		}
		// Will cause memory to be freed in callback after seq DTW completion, so the sleep_for() polling above can 
		// eventually release to launch more kernels as free memory increases (if it's not already limited by the kernel grid block queue).
		heterogeneous_workload *cleanup_workload = 0;
		cudaMallocHost(&cleanup_workload, sizeof(heterogeneous_workload)); CUERR("Allocating page locked CPU memory for DTW stream callback data");
		cleanup_workload->dtwCostSoFar_memptr = dtwCostSoFar;
		cleanup_workload->pathMatrix_memptr = 0;
		cleanup_workload->stream = seq_stream;
		cudaStreamAddCallback(seq_stream, dtwStreamCleanupLaunch, cleanup_workload, 0);
	}
	// TODO: use a fancy cleanup thread barrier here so that multiple DBAs could be running on the same device and not interfere with each other at this step.
	for(int i = 0; i < deviceCount; i++){
                cudaSetDevice(i);
		cudaDeviceSynchronize(); CUERR("Synchronizing CUDA device after all DTW calculations");
	}

	T *dtwSoS;
	// Technically dtsSoS does not need to be page locked as it doesn't get copied to the GPU, but we're futureproofing it and it's going 
	// to be in an existing page most likely anyway, given all the cudaMallocHost() calls before this.
	cudaMallocHost(&dtwSoS, sizeof(T)*num_sequences); CUERR("Allocating CPU memory for DTW pairwise distance sums of squares");
	std::memset(dtwSoS, 0, sizeof(T)*num_sequences);
        // Reassemble the whole pair matrix (upper right only) from the rows that each device processed.
	for(int i = 0; i < deviceCount; i++){
		cudaSetDevice(i);
		for(int j = i; j < num_sequences-1; j+= deviceCount){
			size_t offset = ARITH_SERIES_SUM(num_sequences-1)-ARITH_SERIES_SUM(num_sequences-j-1);
			cudaMemcpy(cpu_dtwPairwiseDistances + offset, 
                                   gpu_dtwPairwiseDistances[i] + offset, 
				   sizeof(T)*(num_sequences-j-1), cudaMemcpyDeviceToHost); CUERR("Copying DTW pairwise distances to CPU");
		}
	}
	size_t index_offset = 0;
	std::ofstream mats((std::string(output_prefix)+std::string(".pair_dists.txt")).c_str());
	for(size_t seq_index = 0; seq_index < num_sequences-1; seq_index++){
		mats << sequence_names[seq_index];
		for(size_t pad = 0; pad < seq_index; ++pad){
			mats << "\t";
		}
		mats << "\t0"; //self-distance
		for(size_t paired_seq_index = seq_index + 1; paired_seq_index < num_sequences; ++paired_seq_index){
			T dtwPairwiseDistanceSquared = cpu_dtwPairwiseDistances[index_offset+paired_seq_index-seq_index-1];
			mats << "\t" << dtwPairwiseDistanceSquared;
			dtwPairwiseDistanceSquared *= dtwPairwiseDistanceSquared;
			dtwSoS[seq_index] += dtwPairwiseDistanceSquared;
			dtwSoS[paired_seq_index] += dtwPairwiseDistanceSquared;
			//std::cerr << "gpu_dtwPairwiseDistances for (" << seq_index << "," << paired_seq_index << ") is " << dtwPairwiseDistanceSquared << std::endl;
		}
		index_offset += num_sequences-seq_index-1;
		mats << std::endl;
	}
	// Last line is pro forma as all pair distances have already been printed
	mats << sequence_names[num_sequences-1];
	for(size_t pad = 0; pad < num_sequences; ++pad){
                mats << "\t";
        }
	mats << "0" << std::endl;

	// Pick the smallest squared distance across all the sequences.
	int medoidIndex = -1;
	T lowestSoS = std::numeric_limits<T>::max();
	for(size_t i = 0; i < num_sequences-1; ++i){
		if (dtwSoS[i] < lowestSoS) {
			medoidIndex = i;
			lowestSoS = dtwSoS[i];
		}
	}

	cudaFreeHost(dtwSoS); CUERR("Freeing CPU memory for DTW pairwise distance sum of squares");
	cudaFreeHost(cpu_dtwPairwiseDistances); CUERR("Freeing page locked CPU memory for DTW pairwise distances");
	for(int i = 0; i < deviceCount; i++){
		cudaSetDevice(i);
		cudaFree(gpu_dtwPairwiseDistances[i]); CUERR("Freeing GPU memory for DTW pairwise distances");
	}
	cudaFreeHost(gpu_dtwPairwiseDistances); CUERR("Freeing CPU memory for GPU DTW pairwise distancesa' pointers");
	mats.close();
	return medoidIndex;
}

/**
 * Employ the backtracking algorithm through the path matrix to find the optimal DTW path for 
 * sequence (indexed by i) vs. centroid (indexed by j), accumulating the sequence value at each centroid element 
 * for eventual averaging on the host side once all sequences have been through this same process on the GPU
 * (there is no point in in doing the whole thing host side since copying all the path matrices back to the CPU would be slower, oocupy more CPU memory,
 * and this kernel execution can be interleaved with memory waiting DTW calculation kernels to reduce turnaround time).
 */
template<typename T>
__global__
void updateCentroid(T *seq, T *centroidElementSums, unsigned int *nElementsForMean, unsigned char *pathMatrix, size_t pathColumns, size_t pathRows, size_t pathMemPitch){
	// Backtrack from the end of both sequences to the start to get the optimal path.
	int j = pathColumns - 1;
	int i = pathRows - 1;

	unsigned char move = pathMatrix[pitchedCoord(j,i,pathMemPitch)];
	while (move != NIL && move != NIL_OPEN_RIGHT) {
		// Don't count open end moves as contributing to the consensus.
		if(move != OPEN_RIGHT){ 
			atomicAdd(&centroidElementSums[j], seq[i]);
			atomicInc(&nElementsForMean[j], numeric_limits<unsigned int>::max());
		}
		i += moveI[move];
		j += moveJ[move];
		move = pathMatrix[pitchedCoord(j,i,pathMemPitch)];
	}
	// If the path matrix & moveI & moveJ are sane, we will necessarily be at i == 0, j == 0 when the backtracking finishes.
	if(i != 0 || j != 0){
		// Executing a PTX assembly language trap is the closest we get to throwing an exception in a CUDA kernel. 
		// The next call to CUERR() will report "an illegal instruction was encountered".
		asm("trap;"); 
	}

	if(move != NIL_OPEN_RIGHT) {
		atomicAdd(&centroidElementSums[j], seq[i]);
		atomicInc(&nElementsForMean[j], numeric_limits<unsigned int>::max());
	}
}

/**
 * Returns the delta (max movement of a single point in the centroid) after update.
 *
 * @param C a gpu-side centroid sequence array
 *
 * @param updatedMean a cpu-side location for the result of the DBAUpdate to the centroid sequence
 */
template<typename T>
__host__ double 
DBAUpdate(T *C, size_t centerLength, T *sequences, size_t maxSeqLength, size_t num_sequences, size_t *sequence_lengths, int use_open_start, int use_open_end, T *updatedMean, cudaStream_t stream) {
	T *gpu_centroidAlignmentSums;
	cudaMalloc(&gpu_centroidAlignmentSums, sizeof(T)*centerLength); CUERR("Allocating GPU memory for barycenter update sequence element sums");
	cudaMemsetAsync(gpu_centroidAlignmentSums, 0, sizeof(T)*centerLength, stream); CUERR("Initialzing GPU memory for barycenter update sequence element sums to zero");

	T *cpu_centroid;
        cudaMallocHost(&cpu_centroid, sizeof(T)*centerLength); CUERR("Allocating CPU memory for incoming centroid");
	cudaMemcpyAsync(cpu_centroid, C, sizeof(T)*centerLength, cudaMemcpyDeviceToHost, stream); CUERR("Copying incoming GPU centroid to CPU");

        int dev = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev); CUERR("Getting GPU device properties");

        unsigned int maxThreads = deviceProp.maxThreadsPerBlock;

	unsigned int *nElementsForMean, *cpu_nElementsForMean;
	cudaMalloc(&nElementsForMean, sizeof(unsigned int)*centerLength); CUERR("Allocating GPU memory for barycenter update sequence pileup");
	cudaMallocHost(&cpu_nElementsForMean, sizeof(unsigned int)*centerLength); CUERR("Allocating CPU memory for barycenter sequence pileup");

        int priority_high, priority_low, descendingPriority;
        cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
        descendingPriority = priority_high;

        // Allocate space for the dtwCost to get to each point on the border between grid vertical swaths of the total cost matrix against the consensus C.
	// Generate the path matrix though for each sequence relative to the centroid, and update the centroid means accordingly.
	int dotsPrinted = 0;
	std::cerr << "0%        10%       20%       30%       40%       50%       60%       70%       80%       90%       100%" << std::endl;
	char spinner[4] = { '|', '/', '-', '\\'};
        for(size_t seq_index = 0; seq_index <= num_sequences-1; seq_index++){
                dim3 threadblockDim(maxThreads, 1, 1);
                size_t current_seq_length = sequence_lengths[seq_index];
                // We are allocating each time rather than just once at the start because if the sequences have a large
                // range of lengths and we sort them from shortest to longest we will be allocating the minimum amount of
                // memory necessary.
                size_t dtwCostSoFarSize = sizeof(T)*current_seq_length;
                size_t pathMatrixSize = sizeof(unsigned char)*current_seq_length*centerLength;
                size_t freeGPUMem;
                size_t totalGPUMem;
                cudaMemGetInfo(&freeGPUMem, &totalGPUMem);
                while(freeGPUMem < dtwCostSoFarSize+pathMatrixSize*1.05){ // assume pitching could add up to 5%
                        //std::this_thread::sleep_for(std::chrono::seconds(1));
                        usleep(1000);
                        //std::cerr << "Waiting for memory to be freed before launching " << std::endl;
                        cudaMemGetInfo(&freeGPUMem, &totalGPUMem);
                }

                int newDotTotal = 100*((float) seq_index/(num_sequences-1));
                if(newDotTotal > dotsPrinted){
                        for(; dotsPrinted < newDotTotal; dotsPrinted++){
                                std::cerr << "\b.|";
                        }
                }
                else{
                        std::cerr << "\b" << spinner[seq_index%4];
                }

		T *dtwCostSoFar = 0;
                cudaMalloc(&dtwCostSoFar, dtwCostSoFarSize);  CUERR("Allocating GPU memory for DTW pairwise distance intermediate values");

        	size_t pathPitch;
        	unsigned char *pathMatrix = 0;
		// Column major allocation x-axis is 2nd seq
        	cudaMallocPitch(&pathMatrix, &pathPitch, centerLength, current_seq_length); CUERR("Allocating pitched GPU memory for sequence:centroid path matrix");

                // Make calls to DTWDistance serial within each seq, but allow multiple seqs on the GPU at once.
                cudaStream_t seq_stream;
                cudaStreamCreateWithPriority(&seq_stream, cudaStreamNonBlocking, descendingPriority);
                if(descendingPriority < priority_low){
                        descendingPriority++;
                }
                for(size_t offset_within_seq = 0; offset_within_seq < centerLength; offset_within_seq += threadblockDim.x){
                        // We have a circular buffer in shared memory of three diagonals for minimal proper DTW calculation.
                        int shared_memory_required = threadblockDim.x*sizeof(T)*3;
			// 0 arg here means that we are not storing the pairwise distance (total cost) between the sequences back out to global memory.
                        DTWDistance<<<1,threadblockDim,shared_memory_required,seq_stream>>>(C, centerLength, seq_index, offset_within_seq, sequences, maxSeqLength,
                                                 num_sequences, sequence_lengths, dtwCostSoFar, pathMatrix, pathPitch, (T *) 0, use_open_start, use_open_end); CUERR("DTW vertical swath calculation with cost storage");
			
                }
		updateCentroid<<<1,1,0,seq_stream>>>(sequences+maxSeqLength*seq_index, gpu_centroidAlignmentSums, nElementsForMean, pathMatrix, centerLength, current_seq_length, pathPitch);
                // Will cause memory to be freed in callback after seq DTW completion, so the sleep_for() polling above can
                // eventually release to launch more kernels as free memory increases (if it's not already limited by the kernel grid block queue).
                heterogeneous_workload *cleanup_workload = 0;
                cudaMallocHost(&cleanup_workload, sizeof(heterogeneous_workload)); CUERR("Allocating page locked CPU memory for DTW stream callback data");
                cleanup_workload->dtwCostSoFar_memptr = dtwCostSoFar;
                cleanup_workload->pathMatrix_memptr = pathMatrix;
                cleanup_workload->stream = seq_stream;
                cudaStreamAddCallback(seq_stream, dtwStreamCleanupLaunch, cleanup_workload, 0);
        }
	// TODO: use a fancy cleanup thread barrier here so that multiple DBAs could be running on the same device and not interfere with each other at this step.
        cudaDeviceSynchronize(); CUERR("Synchronizing CUDA device after all DTW calculations");

	cudaMemcpy(cpu_nElementsForMean, nElementsForMean, sizeof(T)*centerLength, cudaMemcpyDeviceToHost); CUERR("Copying barycenter update sequence pileup from GPU to CPU");
	cudaMemcpy(updatedMean, gpu_centroidAlignmentSums, sizeof(T)*centerLength, cudaMemcpyDeviceToHost); CUERR("Copying barycenter update sequence element sums from GPU to CPU");
	cudaStreamSynchronize(stream);  CUERR("Synchronizing CUDA stream before computing centroid mean");
	for (int t = 0; t < centerLength; t++) {
		updatedMean[t] /= cpu_nElementsForMean[t];
	}
	cudaFree(gpu_centroidAlignmentSums); CUERR("Freeing GPU memory for the barycenter update sequence element sums");
	cudaFree(nElementsForMean); CUERR("Freeing GPU memory for the barycenter update sequence pileup");
	cudaFreeHost(cpu_nElementsForMean);  CUERR("Freeing CPU memory for the barycenter update sequence pileup");

	// Calculate the difference between the old and new barycenter.
	// Convergence is defined as when all points in the old and new differ by less than a 
	// given delta (relative to std dev since every sequence is Z-normalized), so return the max point delta.
	double max_delta = 0.0;
	for(int t = 0; t < centerLength; t++) {
		double delta = std::abs((double) (cpu_centroid[t]-updatedMean[t]));
		if(delta > max_delta){
			max_delta = delta;
		}
	}
	cudaFreeHost(cpu_centroid); CUERR("Freeing CPU memory for the incoming centroid");
	return max_delta;

}

/**
 * Performs the DBA averaging by first finding the median over a sample,
 * then doing iterations of the update until  the convergence condition is met.
 * 
 * @param sequences
 *                ragged 2D array of numeric sequences (of type T) to be averaged
 * @param num_sequences
 *                the number of sequences to be run through the algorithm
 * @param sequence_lengths
 *                the length of each member of the ragged array
 * @param convergence_delta
 *                the convergence stop criterium as proportion in range [0,1) of change in medoid distance between medoid update rounds
 * @param barycenter
 *                pointer to the resulting sequence barycenter array. Array will be allocated by this function, so must be freed by caller with cudaFreeHost(barycenter) 
 */
template <typename T>
__host__ void performDBA(T **sequences, int num_sequences, size_t *sequence_lengths, char **sequence_names, double convergence_delta, int use_open_start, int use_open_end, char *output_prefix, T **barycenter, size_t *barycenter_length, cudaStream_t stream=0) {

	// Sort the sequences by length for memory efficiency in computation later on.
	size_t *sequence_lengths_copy;
	cudaMallocHost(&sequence_lengths_copy, sizeof(size_t)*num_sequences); CUERR("Allocating CPU memory for sortable copy of sequence lengths");
	if(memcpy(sequence_lengths_copy, sequence_lengths, sizeof(size_t)*num_sequences) != sequence_lengths_copy){
		std::cerr << "Running memcpy to populate sequence_lengths_copy failed";
		exit(1);
	}
	thrust::sort_by_key(sequence_lengths_copy, sequence_lengths_copy + num_sequences, sequences); CUERR("Sorting sequences by length");
	thrust::sort_by_key(sequence_lengths, sequence_lengths + num_sequences, sequence_names); CUERR("Sorting sequence names by length");
	size_t maxLength = sequence_lengths[num_sequences-1];

	// Send the sequence metadata and data out to all the devices being used.
        int deviceCount;
        cudaGetDeviceCount(&deviceCount); CUERR("Getting GPU device count");
        std::cerr << "Devices found: " << deviceCount << std::endl;

	size_t **gpu_sequence_lengths = 0;
	cudaMallocHost(&gpu_sequence_lengths, sizeof(size_t **)*deviceCount); CUERR("Allocating GPU memory for array of sequence lengths");
	for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		cudaSetDevice(currDevice);
		cudaMalloc(&gpu_sequence_lengths[currDevice], sizeof(size_t)*num_sequences); CUERR("Allocating GPU memory for array of sequence length pointers");
        	cudaMemcpyAsync(gpu_sequence_lengths[currDevice], sequence_lengths, sizeof(size_t)*num_sequences, cudaMemcpyHostToDevice, stream); CUERR("Copying sequence lengths to GPU memory");
	}

	T **gpu_sequences = 0;
	cudaMallocHost(&gpu_sequences, sizeof(T**)*deviceCount); CUERR("Allocating GPU memory for array of sequences");
        for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		cudaSetDevice(currDevice);
		cudaMalloc(&gpu_sequences[currDevice], sizeof(T)*num_sequences*maxLength); CUERR("Allocating GPU memory for array of sequences");
		// Make a GPU copy of the input ragged 2D array as an evenly spaced 1D array for performance
		for (int i = 0; i < num_sequences; i++) {
        		cudaMemcpyAsync(gpu_sequences[currDevice]+i*maxLength, sequences[i], sequence_lengths[i]*sizeof(T), cudaMemcpyHostToDevice, stream); CUERR("Copying sequence to GPU memory");
		}
	}

	cudaStreamSynchronize(stream); CUERR("Synchronizing the CUDA stream after sequences' copy to GPU");
	//std::cerr << "Normalizing " << num_sequences << " input streams (longest is " << maxLength << ")" << std::endl;
	normalizeSequences(gpu_sequences, maxLength, num_sequences, gpu_sequence_lengths, -1, maxLength, stream);

        // Pick a seed sequence from the original input, with the smallest L2 norm.
	size_t medoidIndex = approximateMedoidIndex(gpu_sequences, maxLength, num_sequences, sequence_lengths, sequence_names, gpu_sequence_lengths, use_open_start, use_open_end, output_prefix, stream);
        size_t medoidLength = sequence_lengths[medoidIndex];
	std::cerr << std::endl << "Initial medoid " << sequence_names[medoidIndex] << " has length " << medoidLength << std::endl;
	T *gpu_barycenter = 0;
	cudaSetDevice(0);
	cudaMalloc(&gpu_barycenter, sizeof(T)*medoidLength); CUERR("Allocating GPU memory for DBA result");
        cudaMemcpyAsync(gpu_barycenter, gpu_sequences[0]+maxLength*medoidIndex, medoidLength*sizeof(T), cudaMemcpyDeviceToDevice, stream);  CUERR("Copying medoid seed to GPU memory");

	// Z-normalize the sequences in parallel on the GPU, once all the async memcpy calls are done.
        // Refine the alignment iteratively.
	T *new_barycenter = 0;
	cudaMallocHost(&new_barycenter, sizeof(T)*medoidLength); CUERR("Allocating GPU memory for DBA update result");
	int maxRounds = 1000;
	cudaSetDevice(0);
	for (int i = 0; i < maxRounds; i++) {
		std::cerr << std::endl << "Step 3 of 3 (round " << i << " of max " << maxRounds << " to find delta < " << convergence_delta << "): Converging centroid" << std::endl;
		double delta = DBAUpdate(gpu_barycenter, medoidLength, gpu_sequences[0], maxLength, num_sequences, sequence_lengths, use_open_start, use_open_end, new_barycenter, stream);
		std::cerr << std::endl << "New delta is " << delta << std::endl;
		if(delta < convergence_delta){
			break;
		}
		cudaMemcpy(gpu_barycenter, new_barycenter, sizeof(T)*medoidLength, cudaMemcpyHostToDevice);
	}

	double medoidAvg = 0.0;
	double medoidStdDev = 0.0;
	T *medoidSequence = sequences[medoidIndex];
	for(int i = 0; i < medoidLength; i++){
		medoidAvg += medoidSequence[i];
	}
	medoidAvg /= medoidLength;
        for(int i = 0; i < medoidLength; i++){
                medoidStdDev = (medoidAvg - medoidSequence[i])*(medoidAvg - medoidSequence[i]);
        }
	medoidStdDev = sqrt(medoidStdDev/medoidLength);
	for(int i = 0; i < medoidLength; i++){
		new_barycenter[i] = (T) (medoidAvg+new_barycenter[i]*medoidStdDev);
	}

	// Send the result back to the CPU.
	*barycenter = new_barycenter;

	// Clean up the GPU memory we don't need any more.
        cudaFree(gpu_barycenter); CUERR("Freeing GPU memory for barycenter");
	for(int i = 0; i < deviceCount; i++){
		cudaFree(gpu_sequences[i]); CUERR("Freeing GPU memory for sequence");
	} 
	cudaFreeHost(gpu_sequences); CUERR("Freeing CPU memory for GPU sequence pointer array");
	for(int i = 0; i < deviceCount; i++){
		cudaFree(gpu_sequence_lengths[i]); CUERR("Freeing GPU memory for sequence lengths");
	} 
	cudaFreeHost(gpu_sequence_lengths); CUERR("Freeing GPU memory for the sequence lengths pointer array");

	*barycenter_length = medoidLength;
}

#endif
