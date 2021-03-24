#ifndef __dba_hpp_included
#define __dba_hpp_included

#include <thrust/sort.h>
#include <iostream>
#include <fstream>
#include <string>
#if defined(_WIN32)
	#include <Windows.h>
	extern "C"{
		#include "getopt.h"
	}
#else
	#include <unistd.h>
#endif

#include "exit_codes.hpp"
#include "gpu_utils.hpp"
#include "io_utils.hpp"
#include "cuda_utils.hpp"
#include "dtw.hpp"
#include "limits.hpp" // for CUDA kernel compatible max()

#define ARITH_SERIES_SUM(n) (((n)*(n+1))/2)

using namespace cudahack; // for device-side numeric limits

template<typename T>
__host__ int approximateMedoidIndex(T **gpu_sequences, size_t maxSeqLength, size_t num_sequences, size_t *sequence_lengths, char **sequence_names, size_t **gpu_sequence_lengths, int use_open_start, int use_open_end, char *output_prefix, cudaStream_t stream) {
	int deviceCount;
 	cudaGetDeviceCount(&deviceCount); CUERR("Getting GPU device count in medoid approximation method");

	unsigned int *maxThreads = getMaxThreadsPerDevice(deviceCount); // from cuda_utils.hpp

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
			#ifdef _WIN32
				Sleep(1);
			#else
				usleep(1000);
			#endif
			//std::cerr << "Waiting for memory to be freed before launching " << std::endl;
			cudaMemGetInfo(&freeGPUMem, &totalGPUMem);
		}
		dotsPrinted = updatePercentageComplete(seq_index+1, num_sequences-1, dotsPrinted);
		T *dtwCostSoFar = 0;
		T *newDtwCostSoFar = 0;
		cudaMallocManaged(&dtwCostSoFar, dtwCostSoFarSize);  CUERR("Allocating GPU memory for DTW pairwise distance intermediate values");
		cudaMallocManaged(&newDtwCostSoFar, dtwCostSoFarSize); CUERR("Allocating GPU memory for new DTW pairwise distance intermediate values");

		// Make calls to DTWDistance serial within each seq, but allow multiple seqs on the GPU at once.
		cudaStream_t seq_stream; 
		cudaStreamCreateWithPriority(&seq_stream, cudaStreamNonBlocking, descendingPriority);
		if(descendingPriority < priority_low){
			descendingPriority++;
		}
		for(size_t offset_within_seq = 0; offset_within_seq < maxSeqLength; offset_within_seq += threadblockDim.x){
			// We have a circular buffer in shared memory of three diagonals for minimal proper DTW calculation, and an array for an inline findMin()
        		int shared_memory_required = threadblockDim.x*3*sizeof(T);
			// Null unsigned char pointer arg below means we aren't storing the path for each alignment right now.
			// And (T *) 0, (size_t) 0, (T *) 0, (size_t) 0, means that the sequences to be compared will be defined by seq_index (1st, Y axis seq) and the block x index (2nd, X axis seq)
			DTWDistance<<<gridDim,threadblockDim,shared_memory_required,seq_stream>>>((T *) 0, (size_t) 0, (T *) 0, (size_t) 0, seq_index, offset_within_seq, gpu_sequences[currDevice], maxSeqLength,
										num_sequences, gpu_sequence_lengths[currDevice], dtwCostSoFar, newDtwCostSoFar, (unsigned char *) 0, (size_t) 0, gpu_dtwPairwiseDistances[currDevice], use_open_start, use_open_end); CUERR("DTW vertical swath calculation with cost storage");
			cudaMemcpyAsync(dtwCostSoFar, newDtwCostSoFar, dtwCostSoFarSize, cudaMemcpyDeviceToDevice, seq_stream); CUERR("Copying DTW pairwise distance intermediate values");
		}
		// Will cause memory to be freed in callback after seq DTW completion, so the sleep_for() polling above can 
		// eventually release to launch more kernels as free memory increases (if it's not already limited by the kernel grid block queue).
		addStreamCleanupCallback(dtwCostSoFar, newDtwCostSoFar, 0, seq_stream);
	}
        cudaFreeHost(maxThreads); CUERR("Freeing CPU memory for device thread properties");
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
		}
		index_offset += num_sequences - seq_index - 1;
		mats << std::endl;
	}
	// Last line is pro forma as all pair distances have already been printed
	mats << sequence_names[num_sequences-1];
	for(size_t pad = 0; pad < num_sequences; ++pad){
                mats << "\t";
        }
	mats << "0" << std::endl;

	int medoidIndex = -1;
	// Pick the smallest squared distance across all the sequences.
	if(num_sequences > 2){
		T lowestSoS = std::numeric_limits<T>::max();
		for(size_t i = 0; i < num_sequences; ++i){
			if (dtwSoS[i] < lowestSoS) {
				medoidIndex = i;
				lowestSoS = dtwSoS[i];
			}
		}
	} else{	// Pick the longest sequence that contributed to the cumulative distance if we only have 2 sequences
		medoidIndex = sequence_lengths[0] > sequence_lengths[1] ? 0 : 1;
	}

	cudaFreeHost(dtwSoS); CUERR("Freeing CPU memory for DTW pairwise distance sum of squares");
	cudaFreeHost(cpu_dtwPairwiseDistances); CUERR("Freeing page locked CPU memory for DTW pairwise distances");
	for(int i = 0; i < deviceCount; i++){
		cudaSetDevice(i);
		cudaFree(gpu_dtwPairwiseDistances[i]); CUERR("Freeing GPU memory for DTW pairwise distances");
	}
	cudaFreeHost(gpu_dtwPairwiseDistances); CUERR("Freeing CPU memory for GPU DTW pairwise distances' pointers");
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
void updateCentroid(T *seq, T *centroidElementSums, unsigned int *nElementsForMean, unsigned char *pathMatrix, size_t pathColumns, size_t pathRows, size_t pathMemPitch, int flip_seq_order){
	// Backtrack from the end of both sequences to the start to get the optimal path.
	size_t j = pathColumns - 1;
	size_t i = pathRows - 1;
	if(flip_seq_order){
		size_t tmp = i;
		i = j;
		j = tmp;
	}

	unsigned char move = pathMatrix[pitchedCoord(j,i,pathMemPitch)];
	while (move != NIL && move != NIL_OPEN_RIGHT) {
		// Don't count open end moves as contributing to the consensus.
		if(move != OPEN_RIGHT){ 
			// flip_seq_order indicates that the consensus is on the Y axis for this path matrix rather than the X axis.
			atomicAdd(&centroidElementSums[flip_seq_order ? i : j], seq[flip_seq_order ? j : i]);
			atomicInc(&nElementsForMean[flip_seq_order ? i : j], numeric_limits<unsigned int>::max());
		}
		// moveI and moveJ are defined device-side in dtw.hpp
		i += (size_t) moveI[move];
		j += (size_t) moveJ[move];
		move = pathMatrix[pitchedCoord(j,i,pathMemPitch)];
	}
	// If the path matrix & moveI & moveJ are sane, we will necessarily be at i == 0, j == 0 when the backtracking finishes.
	if(i != 0 || j != 0){
		// Executing a PTX assembly language trap is the closest we get to throwing an exception in a CUDA kernel. 
		// The next call to CUERR() will report "an illegal instruction was encountered".
		asm("trap;"); 
	}

	if(move != NIL_OPEN_RIGHT) {
		atomicAdd(&centroidElementSums[0], seq[0]);
		atomicInc(&nElementsForMean[0], numeric_limits<unsigned int>::max());
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
DBAUpdate(T *C, size_t centerLength, T *sequences, size_t maxSeqLength, size_t num_sequences, size_t *sequence_lengths, size_t *gpu_sequence_lengths, int use_open_start, int use_open_end, T *updatedMean, cudaStream_t stream) {
	T *gpu_centroidAlignmentSums;
	cudaMallocManaged(&gpu_centroidAlignmentSums, sizeof(T)*centerLength); CUERR("Allocating GPU memory for barycenter update sequence element sums");
	cudaMemset(gpu_centroidAlignmentSums, 0, sizeof(T)*centerLength); CUERR("Initialzing GPU memory for barycenter update sequence element sums to zero");

	T *cpu_centroid;
        cudaMallocHost(&cpu_centroid, sizeof(T)*centerLength); CUERR("Allocating CPU memory for incoming centroid");
	cudaMemcpy(cpu_centroid, C, sizeof(T)*centerLength, cudaMemcpyDeviceToHost); CUERR("Copying incoming GPU centroid to CPU");

	int deviceCount;
        cudaGetDeviceCount(&deviceCount); CUERR("Getting GPU device count in DBA update function");
        unsigned int *maxThreads = getMaxThreadsPerDevice(deviceCount);

	unsigned int *nElementsForMean, *cpu_nElementsForMean;
	cudaMallocManaged(&nElementsForMean, sizeof(unsigned int)*centerLength); CUERR("Allocating GPU memory for barycenter update sequence pileup");
	cudaMemset(nElementsForMean, 0, sizeof(unsigned int)*centerLength); CUERR("Initialzing GPU memory for barycenter update sequence pileup to zero");
	cudaMallocHost(&cpu_nElementsForMean, sizeof(unsigned int)*centerLength); CUERR("Allocating CPU memory for barycenter sequence pileup");

        int priority_high, priority_low, descendingPriority;
        cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
        descendingPriority = priority_high;

        // Allocate space for the dtwCost to get to each point on the border between grid vertical swaths of the total cost matrix against the consensus C.
	// Generate the path matrix though for each sequence relative to the centroid, and update the centroid means accordingly.
	int dotsPrinted = 0;
        for(size_t seq_index = 0; seq_index <= num_sequences-1; seq_index++){
                dim3 threadblockDim(maxThreads[0], 1, 1); //TODO: parallelize across devices
                size_t current_seq_length = sequence_lengths[seq_index];
                // We are allocating each time rather than just once at the start because if the sequences have a large
                // range of lengths and we sort them from shortest to longest we will be allocating the minimum amount of
                // memory necessary.
                size_t pathMatrixSize = sizeof(unsigned char)*current_seq_length*centerLength;
                size_t freeGPUMem;
                size_t totalGPUMem;
                size_t dtwCostSoFarSize = sizeof(T)*current_seq_length;
		int flip_seq_order = 0;
		if(use_open_end && centerLength < current_seq_length){
			flip_seq_order = 1;
			dtwCostSoFarSize = sizeof(T)*centerLength;
		}
                cudaMemGetInfo(&freeGPUMem, &totalGPUMem);
                while(freeGPUMem < dtwCostSoFarSize+pathMatrixSize*1.05){ // assume pitching could add up to 5%
                        #ifdef _WIN32
							Sleep(1);
						#else
							usleep(1000);
						#endif
                        //std::cerr << "Waiting for memory to be freed before launching " << std::endl;
                        cudaMemGetInfo(&freeGPUMem, &totalGPUMem);
                }

		dotsPrinted = updatePercentageComplete(seq_index+1, num_sequences, dotsPrinted);

		T *dtwCostSoFar = 0;
		T *newDtwCostSoFar = 0;
                cudaMallocManaged(&dtwCostSoFar, dtwCostSoFarSize);  CUERR("Allocating GPU memory for DTW pairwise distance intermediate values");
                cudaMallocManaged(&newDtwCostSoFar, dtwCostSoFarSize);  CUERR("Allocating GPU memory for new DTW pairwise distance intermediate values");

		// Under the assumption that long sequences have the same or more information than the centroid, flip the DTW comparison so the centroid has an open end.
		// Otherwise you're cramming extra sequence data into the wrong spot and the DTW will give up and choose an all-up then all-open right path instead of a diagonal,
		// which messes with the consensus building.
        	size_t pathPitch;
        	unsigned char *pathMatrix = 0;
		// Column major allocation x-axis is 2nd seq
        	if(flip_seq_order){
			cudaMallocPitch(&pathMatrix, &pathPitch, current_seq_length, centerLength); CUERR("Allocating pitched GPU memory for centroid:sequence path matrix");
		}
		else{
			cudaMallocPitch(&pathMatrix, &pathPitch, centerLength, current_seq_length); CUERR("Allocating pitched GPU memory for sequence:centroid path matrix");
		}

                // Make calls to DTWDistance serial within each seq, but allow multiple seqs on the GPU at once.
                cudaStream_t seq_stream;
                cudaStreamCreateWithPriority(&seq_stream, cudaStreamNonBlocking, descendingPriority);
                if(descendingPriority < priority_low){
                        descendingPriority++;
                }

		int dtw_limit = flip_seq_order ? current_seq_length : centerLength;
#if DEBUG == 1
		std::string cost_filename = std::string("costmatrix")+std::to_string(seq_index);
		std::ofstream cost(cost_filename);
		if(!cost.is_open()){
			std::cerr << "Cannot write to " << cost_filename << std::endl;
			return CANNOT_WRITE_DTW_PATH_MATRIX;
		}	
#endif
		//std::cerr << "length of dtw limit is " << dtw_limit << std::endl;
                for(size_t offset_within_seq = 0; offset_within_seq < dtw_limit; offset_within_seq += threadblockDim.x){
                        // We have a circular buffer in shared memory of three diagonals for minimal proper DTW calculation.
                        int shared_memory_required = threadblockDim.x*3*sizeof(T);
			// 0 arg here means that we are not storing the pairwise distance (total cost) between the sequences back out to global memory.
                        if(flip_seq_order){
				// Specify both the first and second sequences explicitly (seq_inex will actually be ignored)
				DTWDistance<<<1,threadblockDim,shared_memory_required,seq_stream>>>(C, centerLength, sequences+seq_index*maxSeqLength, current_seq_length, 0, offset_within_seq, sequences, maxSeqLength,
                                                 num_sequences, gpu_sequence_lengths, dtwCostSoFar, newDtwCostSoFar, pathMatrix, pathPitch, (T *) 0, use_open_start, use_open_end); CUERR("Consensus DTW vertical swath calculation with path storage");
				cudaMemcpyAsync(dtwCostSoFar, newDtwCostSoFar, dtwCostSoFarSize, cudaMemcpyDeviceToDevice, seq_stream); CUERR("Copying DTW pairwise distance intermediate values with flipped sequence order");
			}
			else{
				// (T *) 0, (size_t) 0 indicates that the sequences should be grabbed from sequences[] using the index seq_index
				DTWDistance<<<1,threadblockDim,shared_memory_required,seq_stream>>>((T *) 0, (size_t) 0, C, centerLength, seq_index, offset_within_seq, sequences, maxSeqLength,
                                                 num_sequences, gpu_sequence_lengths, dtwCostSoFar, newDtwCostSoFar, pathMatrix, pathPitch, (T *) 0, use_open_start, use_open_end); CUERR("Sequence DTW vertical swath calculation with path storage");
				cudaMemcpyAsync(dtwCostSoFar, newDtwCostSoFar, dtwCostSoFarSize, cudaMemcpyDeviceToDevice, seq_stream); CUERR("Copying DTW pairwise distance intermediate values without flipped sequence order");
			}
			cudaStreamSynchronize(seq_stream);
#if DEBUG == 1
			for(int i = 0; i < dtw_limit; i++){
				cost << dtwCostSoFar[i] << ", ";
			}
			cost << std::endl;
#endif
                }
#if DEBUG == 1
		cost.close();
#endif
		/* Start of debugging code, which saves the DTW path for each sequence vs. consensus. Requires C++11 compatibility. */
		cudaStreamSynchronize(seq_stream);

		updateCentroid<<<1,1,0,seq_stream>>>(sequences+maxSeqLength*seq_index, gpu_centroidAlignmentSums, nElementsForMean, pathMatrix, centerLength, current_seq_length, pathPitch, flip_seq_order);
                // Will cause memory to be freed in callback after seq DTW completion, so the sleep_for() polling above can
                // eventually release to launch more kernels as free memory increases (if it's not already limited by the kernel grid block queue).
		addStreamCleanupCallback(dtwCostSoFar, newDtwCostSoFar, pathMatrix, seq_stream);
		
		int num_columns = centerLength;
		int num_rows = current_seq_length;
		if(flip_seq_order){int tmp = num_rows; num_rows = num_columns; num_columns = tmp;}
		
		unsigned char *cpu_stepMatrix = 0;
		
	        cudaMallocHost(&cpu_stepMatrix, sizeof(unsigned char)*pathPitch*num_rows); CUERR("Allocating CPU memory for step matrix");
        	cudaMemcpy(cpu_stepMatrix, pathMatrix, sizeof(unsigned char)*pathPitch*num_rows, cudaMemcpyDeviceToHost);  CUERR("Copying GPU to CPU memory for step matrix");

#if DEBUG == 1
		std::string step_filename = std::string("stepmatrix")+std::to_string(seq_index);
		writeDTWPathMatrix<T>(cpu_stepMatrix, step_filename.c_str(), num_columns, num_rows, pathPitch);
#endif
		
		std::string path_filename = std::string("path")+std::to_string(seq_index);
		writeDTWPath(cpu_stepMatrix, path_filename.c_str(), sequences+seq_index*maxSeqLength, current_seq_length, cpu_centroid, centerLength, num_columns, num_rows, pathPitch, flip_seq_order);
		cudaFreeHost(cpu_stepMatrix);
		/* end of debugging code */

        }
	cudaFreeHost(maxThreads); CUERR("Freeing CPU memory for device thread properties");
	// TODO: use a fancy cleanup thread barrier here so that multiple DBAs could be running on the same device and not interfere with each other at this step.
        cudaDeviceSynchronize(); CUERR("Synchronizing CUDA device after all DTW calculations");

	cudaMemcpy(cpu_nElementsForMean, nElementsForMean, sizeof(T)*centerLength, cudaMemcpyDeviceToHost); CUERR("Copying barycenter update sequence pileup from GPU to CPU");
	cudaMemcpy(updatedMean, gpu_centroidAlignmentSums, sizeof(T)*centerLength, cudaMemcpyDeviceToHost); CUERR("Copying barycenter update sequence element sums from GPU to CPU");
	cudaStreamSynchronize(stream);  CUERR("Synchronizing CUDA stream before computing centroid mean");
	for (int t = 0; t < centerLength; t++) {
		//std::cout << t << "\t" << updatedMean[t] << "\t" << cpu_nElementsForMean[t] << std::endl;
		updatedMean[t] /= cpu_nElementsForMean[t];
	}
	cudaFree(gpu_centroidAlignmentSums); CUERR("Freeing GPU memory for the barycenter update sequence element sums");
	cudaFree(nElementsForMean); CUERR("Freeing GPU memory for the barycenter update sequence pileup");
	cudaFreeHost(cpu_nElementsForMean);  CUERR("Freeing CPU memory for the barycenter update sequence pileup");

	// Calculate the difference between the old and new barycenter.
	// Convergence is defined as when all points in the old and new differ by less than a 
	// given delta (relative to std dev since every sequence is Z-normalized), so return the max point delta.
	double max_delta = (double) 0.0f;
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
__host__ void performDBA(T **sequences, int num_sequences, size_t *sequence_lengths, char **sequence_names, int use_open_start, int use_open_end, char *output_prefix, T **barycenter, size_t *barycenter_length, int norm_sequences, cudaStream_t stream=0) {

	// Sort the sequences by length for memory efficiency in computation later on.
	size_t *sequence_lengths_copy;
	cudaMallocHost(&sequence_lengths_copy, sizeof(size_t)*num_sequences); CUERR("Allocating CPU memory for sortable copy of sequence lengths");
	if(memcpy(sequence_lengths_copy, sequence_lengths, sizeof(size_t)*num_sequences) != sequence_lengths_copy){
		std::cerr << "Running memcpy to populate sequence_lengths_copy failed";
		exit(1);
	}
	thrust::sort_by_key(sequence_lengths_copy, sequence_lengths_copy + num_sequences, sequences); CUERR("Sorting sequences by length");
	thrust::sort_by_key(sequence_lengths, sequence_lengths + num_sequences, sequence_names); CUERR("Sorting sequence names by length");
	cudaFreeHost(sequence_lengths_copy); CUERR("Freeing CPU memory for sortable copy of sequence lengths");
	size_t maxLength = sequence_lengths[num_sequences-1];

	// Send the sequence metadata and data out to all the devices being used.
        int deviceCount;
        cudaGetDeviceCount(&deviceCount); CUERR("Getting GPU device count in DBA setup method");
#if DEBUG == 1
        std::cerr << "Devices found: " << deviceCount << std::endl;
#endif

	size_t **gpu_sequence_lengths = 0;
	cudaMallocHost(&gpu_sequence_lengths, sizeof(size_t **)*deviceCount); CUERR("Allocating GPU memory for array of sequence lengths");
	for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		cudaSetDevice(currDevice);
		cudaMallocManaged(&gpu_sequence_lengths[currDevice], sizeof(size_t)*num_sequences); CUERR("Allocating GPU memory for array of sequence length pointers");
        	cudaMemcpyAsync(gpu_sequence_lengths[currDevice], sequence_lengths, sizeof(size_t)*num_sequences, cudaMemcpyHostToDevice, stream); CUERR("Copying sequence lengths to GPU memory");
	}

	T **gpu_sequences = 0;
	cudaMallocHost(&gpu_sequences, sizeof(T**)*deviceCount); CUERR("Allocating GPU memory for array of sequences");
        for(int currDevice = 0; currDevice < deviceCount; currDevice++){
		cudaSetDevice(currDevice);
		cudaMallocManaged(&gpu_sequences[currDevice], sizeof(T)*num_sequences*maxLength); CUERR("Allocating GPU memory for array of sequences");
		// Make a GPU copy of the input ragged 2D array as an evenly spaced 1D array for performance
		for (int i = 0; i < num_sequences; i++) {
        		cudaMemcpyAsync(gpu_sequences[currDevice]+i*maxLength, sequences[i], sequence_lengths[i]*sizeof(T), cudaMemcpyHostToDevice, stream); CUERR("Copying sequence to GPU memory");
		}
	}

	cudaStreamSynchronize(stream); CUERR("Synchronizing the CUDA stream after sequences' copy to GPU");
#if DEBUG == 1
	std::cerr << "Normalizing " << num_sequences << " input streams (longest is " << maxLength << ")" << std::endl;
#endif
	if(norm_sequences) normalizeSequences(gpu_sequences, maxLength, num_sequences, gpu_sequence_lengths, -1, stream);

        // Pick a seed sequence from the original input, with the smallest L2 norm.
	setupPercentageDisplay("Step 2 of 3: Finding initial medoid");
	size_t medoidIndex = approximateMedoidIndex(gpu_sequences, maxLength, num_sequences, sequence_lengths, sequence_names, gpu_sequence_lengths, use_open_start, use_open_end, output_prefix, stream);
	teardownPercentageDisplay();	
        size_t medoidLength = sequence_lengths[medoidIndex];
	std::cerr << std::endl << "Initial medoid " << sequence_names[medoidIndex] << " has length " << medoidLength << std::endl;
	T *gpu_barycenter = 0;
	cudaMallocManaged(&gpu_barycenter, sizeof(T)*medoidLength); CUERR("Allocating GPU memory for DBA result");
        cudaMemcpyAsync(gpu_barycenter, gpu_sequences[0]+maxLength*medoidIndex, medoidLength*sizeof(T), cudaMemcpyDeviceToDevice, stream);  CUERR("Copying medoid seed to GPU memory");

	// Z-normalize the sequences in parallel on the GPU, once all the async memcpy calls are done.
        // Refine the alignment iteratively.
	T *new_barycenter = 0;
	cudaMallocHost(&new_barycenter, sizeof(T)*medoidLength); CUERR("Allocating GPU memory for DBA update result");
#if DEBUG == 1
	int maxRounds = 1;
#else
	int maxRounds = 1; // PG: tmp
#endif
	cudaSetDevice(0);
	for (int i = 0; i < maxRounds; i++) {
		setupPercentageDisplay("Step 3 of 3 (round " + std::to_string(i+1) +  " of max " + std::to_string(maxRounds) + " to achieve delta 0): Converging centroid");
		double delta = DBAUpdate(gpu_barycenter, medoidLength, gpu_sequences[0], maxLength, num_sequences, sequence_lengths, gpu_sequence_lengths[0], use_open_start, use_open_end, new_barycenter, stream);
		teardownPercentageDisplay();
		std::cerr << std::endl << "New delta is " << delta << std::endl;
		if(delta == 0){
			break;
		}
		cudaMemcpy(gpu_barycenter, new_barycenter, sizeof(T)*medoidLength, cudaMemcpyHostToDevice);  CUERR("Copying updated DBA medoid to GPU");
	}

	if(norm_sequences) {
		/* Rescale the average to the centroid's value range. */
		double medoidAvg = (double) 0.0f;
		double medoidStdDev = (double) 0.0f;
		T *medoidSequence = sequences[medoidIndex];
		for(int i = 0; i < medoidLength; i++){
			medoidAvg += medoidSequence[i];
		}
		medoidAvg /= medoidLength;
			for(int i = 0; i < medoidLength; i++){
					medoidStdDev += (medoidAvg - medoidSequence[i])*(medoidAvg - medoidSequence[i]);
			}
		medoidStdDev = sqrt(medoidStdDev/medoidLength);
		//std::cout << "Rescaling centroid to medoid's mean and std dev: " << medoidAvg << ", " << medoidStdDev << std::endl;
		for(int i = 0; i < medoidLength; i++){
			new_barycenter[i] = (T) (medoidAvg+new_barycenter[i]*medoidStdDev);
		}
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

/* Note that this method may adjust the total number of sequences, so that zero length sequences (after prefix chopping) do not go into the DBA later on. */
template <typename T>
__host__ void chopPrefixFromSequences(T *sequence_prefix, size_t sequence_prefix_length, T ***cpu_sequences, int *num_sequences, size_t *sequence_lengths, char **sequence_names, char *output_prefix, int norm_sequences, cudaStream_t stream=0){

        // Send the sequence metadata and data out to all the devices being used.
        int deviceCount;
        cudaGetDeviceCount(&deviceCount); CUERR("Getting GPU device count in prefix chop method");
	int dotsPrinted = 0;

        size_t **gpu_sequence_lengths = 0;
        cudaMallocHost(&gpu_sequence_lengths, sizeof(size_t **)*deviceCount); CUERR("Allocating GPU memory for array of sequence lengths for chopping");
        for(int currDevice = 0; currDevice < deviceCount; currDevice++){
                cudaSetDevice(currDevice);
                cudaMallocManaged(&gpu_sequence_lengths[currDevice], sizeof(size_t)*(*num_sequences)); CUERR("Allocating GPU memory for array of sequence length pointers");
                cudaMemcpyAsync(gpu_sequence_lengths[currDevice], sequence_lengths, sizeof(size_t)*(*num_sequences), cudaMemcpyHostToDevice, stream); CUERR("Copying sequence lengths to GPU memory for prefix chopping");
        }

	size_t maxLength = 0;
	for(int i = 0; i < (*num_sequences); i++){
		if(sequence_lengths[i] > maxLength){
			maxLength = sequence_lengths[i];
		}
	}

        T **gpu_sequences = 0;
	// Not strictly necessary at the moment, but futureproofing for when we have within-GPU parallelism.
        cudaMallocHost(&gpu_sequences, sizeof(T*)*deviceCount); CUERR("Allocating CPU memory for array of device-side sequence pointers");
        for(int currDevice = 0; currDevice < deviceCount; currDevice++){
                cudaSetDevice(currDevice);
                cudaMallocManaged(&gpu_sequences[currDevice], sizeof(T)*(*num_sequences)*maxLength); CUERR("Allocating GPU memory for array of sequences");
                // Make a GPU copy of the input ragged 2D array as an evenly spaced 1D array for performance
                for (int i = 0; i < *num_sequences; i++) {
                        cudaMemcpyAsync(gpu_sequences[currDevice]+i*maxLength, (*cpu_sequences)[i], sequence_lengths[i]*sizeof(T), cudaMemcpyHostToDevice, stream); CUERR("Copying sequence to GPU memory for prefix chopping");
                }
        }
        T **gpu_sequence_prefixs = 0;
        cudaMallocHost(&gpu_sequence_prefixs, sizeof(T*)*deviceCount); CUERR("Allocating CPU memory for array of device-side sequence prefix pointers");
        for(int currDevice = 0; currDevice < deviceCount; currDevice++){
                cudaSetDevice(currDevice);
                cudaMalloc(&gpu_sequence_prefixs[currDevice], sizeof(T)*sequence_prefix_length); CUERR("Allocating GPU memory for array of sequence prefixes");
                cudaMemcpyAsync(gpu_sequence_prefixs[currDevice], sequence_prefix, sizeof(T)*sequence_prefix_length, cudaMemcpyHostToDevice, stream); CUERR("Copying sequence prefix to GPU memory for prefix chopping");
	}
	for(int i = 0; i < deviceCount; i++){
                cudaSetDevice(i);
                cudaDeviceSynchronize(); CUERR("Synchronizing CUDA device after sequence copy to GPU for chopping");
        }
    	if(norm_sequences){
	       	normalizeSequences(gpu_sequences, maxLength, *num_sequences, gpu_sequence_lengths, -1, stream); CUERR("Normalizing input sequences for prefix chopping");
		normalizeSequence(gpu_sequence_prefixs, sequence_prefix_length, stream); CUERR("Normalizing sequence prefix for chopping");
	}
	size_t *chopPositions = 0;
	cudaMallocHost(&chopPositions, sizeof(size_t)*(*num_sequences)); CUERR("Allocating CPU memory for sequence prefix chopping locations");

        unsigned int *maxThreads = getMaxThreadsPerDevice(deviceCount);

        // Declared sentinels to add semantics to DTWDistance call params.
        // A lot of DTW kernel parameters are ignored because we are launching without a real grid, so vars to infer 
        // kernel instance job divisions and result locations are not needed like they are in the medoid finding.
     	int DONT_USE_OPEN_START = 0; 
        int USE_OPEN_END = 1;
	int IGNORED_SEQ_INDEX_FROM_GRID = 0;
	size_t *IGNORED_GPU_SEQ_LENGTHS = 0;
	int IGNORED_GPU_SEQ_MAX_LENGTH = 0;
	T *IGNORED_SEQ_PTRS = 0;
	int IGNORED_NUM_SEQS = 0;
	T *NO_FINAL_COST_PAIR_MATRIX = 0;
        cudaStream_t *seq_streams;
	cudaMallocHost(&seq_streams, sizeof(cudaStream_t)*deviceCount); CUERR("Allocating CPU memory for sequence processing streams");
        T **dtwCostSoFars = 0;
        T **newDtwCostSoFars = 0;
	cudaMallocHost(&dtwCostSoFars, sizeof(T *)*deviceCount); CUERR("Allocating CPU memory for GPU DTW cost memory pointers");
	cudaMallocHost(&newDtwCostSoFars, sizeof(T *)*deviceCount); CUERR("Allocating CPU memory for GPU new DTW cost memory pointers");
	unsigned char **pathMatrixs = 0;
	cudaMallocHost(&pathMatrixs, sizeof(unsigned char *)*deviceCount); CUERR("Allocating CPU memory for GPU DTW path matrix pointers");
	// Record how many hits there are to each position in the leader in each input sequence.
        int **leaderPathHistograms = 0;
	cudaMallocHost(&leaderPathHistograms, sizeof(int **)*(*num_sequences)); CUERR("Allocating CPU memory for leader path histogram pointers");
	for(int i = 0; i < *num_sequences; i++){	 
		cudaMallocHost(&leaderPathHistograms[i], sizeof(int)*sequence_prefix_length); CUERR("Allocating CPU memory for a leader path histogram");
	}
        for(size_t seq_swath_start = 0; seq_swath_start < *num_sequences; seq_swath_start += deviceCount){

		for(int currDevice = 0; currDevice < deviceCount; currDevice++){
			size_t seq_index = seq_swath_start + currDevice;
			if(seq_index >= *num_sequences){
				break;
			}
			cudaSetDevice(currDevice);
                	size_t current_seq_length = sequence_lengths[seq_index];
       			// Need to run an open end DTW to find where the end of the prefix is in the input sequence based on the path
			// TODO: parallelize within each GPU (see memory alloc note below).
			size_t pathPitch = ((current_seq_length/512)+1)*512; // Have to pitch ourselves as no managed API for this exists

                	size_t dtwCostSoFarSize = sizeof(T)*sequence_prefix_length;
                	// This is small potatoes, we're in real trouble if we can't allocate this.
                	cudaMalloc(&dtwCostSoFars[currDevice], dtwCostSoFarSize);  CUERR("Allocating GPU memory for prefix chopping DTW pairwise distance intermediate values");
                	cudaMalloc(&newDtwCostSoFars[currDevice], dtwCostSoFarSize);  CUERR("Allocating GPU memory for prefix chopping new DTW pairwise distance intermediate values");
                
                        cudaStreamCreate(&seq_streams[currDevice]);
                
			// This is the potentially big matrix if either the prefix or the sequences are long, hence why we are not parallelizing with GPU for the moment.
                	cudaMallocManaged(&pathMatrixs[currDevice], pathPitch*sequence_prefix_length*sizeof(unsigned char)); CUERR("Allocating pitched GPU memory for prefix:sequence path matrix for prefix chopping");

       			dim3 threadblockDim(maxThreads[currDevice], 1, 1);
			int shared_memory_required = threadblockDim.x*3*sizeof(T);
			for(size_t offset_within_seq = 0; offset_within_seq < current_seq_length; offset_within_seq += threadblockDim.x){
        			DTWDistance<<<1,threadblockDim,shared_memory_required,seq_streams[currDevice]>>>(gpu_sequence_prefixs[currDevice], sequence_prefix_length, 
												      	     gpu_sequences[currDevice]+seq_index*maxLength, current_seq_length, 
													     IGNORED_SEQ_INDEX_FROM_GRID, offset_within_seq, 
													     IGNORED_SEQ_PTRS, IGNORED_GPU_SEQ_MAX_LENGTH,
                                                 							     IGNORED_NUM_SEQS, IGNORED_GPU_SEQ_LENGTHS, 
													     dtwCostSoFars[currDevice], 
													     newDtwCostSoFars[currDevice],
													     pathMatrixs[currDevice], pathPitch, NO_FINAL_COST_PAIR_MATRIX, 
											 	 	     DONT_USE_OPEN_START, USE_OPEN_END); 
				CUERR("Launching DTW match of sequences to the sequence prefix");
				cudaMemcpyAsync(dtwCostSoFars[currDevice], newDtwCostSoFars[currDevice], dtwCostSoFarSize, cudaMemcpyDeviceToDevice, seq_streams[currDevice]); CUERR("Copying DTW sequence prefix costs between kernel calls");
			}
			dotsPrinted = updatePercentageComplete(seq_index+1, *num_sequences, dotsPrinted);
		}
       	        for(int currDevice = 0; currDevice < deviceCount; currDevice++){
			size_t seq_index = seq_swath_start + currDevice;
			if(seq_index >= *num_sequences){
				break;
			}
			cudaSetDevice(currDevice); CUERR("Setting active device for DTW path matrix results");
                	cudaStreamSynchronize(seq_streams[currDevice]); CUERR("Synchronizing CUDA device after sequence prefix swath calculation");
			cudaStreamDestroy(seq_streams[currDevice]); CUERR("Destroying now-redundant CUDA device stream");
			cudaFree(dtwCostSoFars[currDevice]);
			cudaFree(newDtwCostSoFars[currDevice]);

       			// Need to run an open end DTW to find where the end of the prefix is in the input sequence based on the path
			// TODO: parallelize within each GPU (see memory alloc note below).
                	size_t current_seq_length = sequence_lengths[seq_index];
			size_t pathPitch = ((current_seq_length/512)+1)*512; // Have to pitch ourselves as no managed API for this exists

			unsigned char *cpu_pathMatrix = 0;
                	size_t columnLimit = current_seq_length - 1;
                	size_t rowLimit = sequence_prefix_length - 1;
                	cudaMallocHost(&cpu_pathMatrix, sizeof(unsigned char)*pathPitch*sequence_prefix_length); CUERR("Allocating host memory for prefix DTW path matrix copy");
                	cudaMemcpy(cpu_pathMatrix, pathMatrixs[currDevice], sizeof(unsigned char)*pathPitch*sequence_prefix_length, cudaMemcpyDeviceToHost); CUERR("Copying prefix DTW path matrix from device to host");
#if DEBUG == 1
			//writeDTWPathMatrix(pathMatrixs[currDevice], (std::string("prefixchop_costmatrix")+std::to_string(seq_index)).c_str(), columnLimit+1, rowLimit+1, pathPitch);
#endif
			cudaFree(pathMatrixs[currDevice]);

                	int moveI[] = { -1, -1, 0, -1, 0 };
                	int moveJ[] = { -1, -1, -1, 0, -1 };
                	int j = columnLimit;
                	int i = rowLimit;
                	unsigned char move = cpu_pathMatrix[pitchedCoord(j,i,pathPitch)];
                	while (move == OPEN_RIGHT) {
                        	i += moveI[move];
                        	j += moveJ[move];
                        	move = cpu_pathMatrix[pitchedCoord(j,i,pathPitch)];
                	}
			chopPositions[seq_index] = j;
			// Now record how many positions in the query correspond to each position in the leader.
			int *leaderPathHistogram = leaderPathHistograms[seq_index];
			leaderPathHistogram[i] = 1;
			while (move != NIL) {
                                i += moveI[move];
                                j += moveJ[move];
				leaderPathHistogram[i]++;
                                move = cpu_pathMatrix[pitchedCoord(j,i,pathPitch)];
                        }
                	cudaFreeHost(cpu_pathMatrix);
        	}
	}
	cudaFreeHost(maxThreads); CUERR("Freeing CPU memory for device thread properties");
	cudaFreeHost(seq_streams); CUERR("Freeing CPU memory for prefix chopping CUDA streams");
	cudaFreeHost(dtwCostSoFars); CUERR("Freeing CPU memory for prefix chopping DTW cost intermediate values");
	cudaFreeHost(pathMatrixs); CUERR("Freeing CPU memory for prefix chopping DTW path matrices");
        for(int currDevice = 0; currDevice < deviceCount; currDevice++){
                cudaSetDevice(currDevice);
                cudaFree(gpu_sequences[currDevice]); CUERR("Freeing GPU memory for a chopping device sequence array");
		cudaFree(gpu_sequence_prefixs[currDevice]); CUERR("Freeing GPU memory for a chopping device sequence prefix");
	}
	cudaFreeHost(gpu_sequences); CUERR("Freeing CPU memory for chopping sequence array pointers");
	cudaFreeHost(gpu_sequence_prefixs); CUERR("Freeing CPU memory for chopping sequence prefix pointers");
        for(int i = 0; i < deviceCount; i++){
                cudaFree(gpu_sequence_lengths[i]); CUERR("Freeing GPU memory for sequence lengths in prefix search");
        }
        cudaFreeHost(gpu_sequence_lengths); CUERR("Freeing GPU memory for the sequence lengths pointer array in prefix search");


	// We're going to have to free the incoming sequences once we've chopped them down and made a new more compact copy.
	std::ofstream chop((std::string(output_prefix)+std::string(".prefix_chop.txt")).c_str());
	int num_zero_length_sequences_skipped = 0;
	for(int i = 0; i < *num_sequences; i++){
		chop << sequence_names[i] << "\t" << chopPositions[i+num_zero_length_sequences_skipped] << "\t" << sequence_lengths[i];
		int *leaderPathHistogram = leaderPathHistograms[i+num_zero_length_sequences_skipped];
		for(int j = 0; j < sequence_prefix_length; j++){
			chop << "\t" << leaderPathHistogram[j];
		}
		chop << std::endl;

		size_t chopped_seq_length = sequence_lengths[i] - chopPositions[i+num_zero_length_sequences_skipped];

		// Remove from the inputs entirely as there is nothing left.
		if(chopped_seq_length == 0){
			std::cerr << "Skipping " << sequence_names[i] << " due to zero-length after prefix chopping" << std::endl;
			cudaFreeHost(leaderPathHistograms[i+num_zero_length_sequences_skipped]); CUERR("Freeing a leader path histogram array on host for zer-length sequence after prefix chop");
			num_zero_length_sequences_skipped++;
			for(int j = i+1; j < *num_sequences; j++){
				sequence_names[j-1] = sequence_names[j];
				cpu_sequences[j-1] = cpu_sequences[j];
				sequence_lengths[j-1] = sequence_lengths[j];
			}
			(*num_sequences)--;
			i--;
			continue;
		}
		T *new_seq = 0;
		cudaMallocHost(&new_seq, sizeof(T)*chopped_seq_length); CUERR("Allocating host memory for chopped sequence pointers");
		T *chopped_seq_start = (*cpu_sequences)[i]+chopPositions[i+num_zero_length_sequences_skipped];
		if(memcpy(new_seq, chopped_seq_start, sizeof(T)*chopped_seq_length) != new_seq){
                	std::cerr << "Running memcpy to copy prefix chopped sequence failed";
                	exit(CANNOT_COPY_PREFIX_CHOPPED_SEQ);
        	}
		cudaFreeHost((*cpu_sequences)[i]); CUERR("Freeing original sequence pointer on host");
		(*cpu_sequences)[i] = new_seq;
		sequence_lengths[i] = chopped_seq_length;
		cudaFreeHost(leaderPathHistograms[i+num_zero_length_sequences_skipped]); CUERR("Freeing a leader path histogram array on host");
	}
	chop.close();

	// TODO: normalize the signal based on the leader match

	cudaFreeHost(chopPositions); CUERR("Freeing chop position records on host");
	cudaFreeHost(leaderPathHistograms); CUERR("Freeing leader path histogram pointer array on host");
}

#endif
