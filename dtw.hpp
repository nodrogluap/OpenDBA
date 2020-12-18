#ifndef __dtw_hpp_included
#define __dtw_hpp_included

#include "cuda_utils.hpp"
#include "limits.hpp" // for device side numeric_limits min() and max()

using namespace cudahack; // for device side numeric_limits

// sentinel value for the start of the DTW alignment, the stop condition for backtracking (ergo has no corresponding moveI or moveJ)
#define NIL 255 
#define DIAGONAL 1
#define RIGHT 2
#define UP 3
// Special move designations that do not affect backtracking algorithm per se, but does affect cost (open=no accumulation of cost for rightward move). 
#define OPEN_RIGHT 4 
#define NIL_OPEN_RIGHT 254 

// For two series I & J, encode that the cost matrix DTW path (i,j) backtracking index decrement options for the DTW steps declared above are:
// unset (0) => (-1, -1), DIAGONAL => (-1,-1), RIGHT => (0,-1), UP => (-1,0), OPEN_RIGHT => (0,-1)
__device__ __constant__ short moveI[] = { -1, -1, 0, -1, 0 };
__device__ __constant__ short moveJ[] = { -1, -1, -1, 0, -1 };

// How to find the 1D index of (X,Y) in the pitched (i.e. coalescing memory access aligned) memory for the DTW path matrix
#define pitchedCoord(Column,Row,mem_pitch) ((size_t) ((Row)*(mem_pitch))+(Column))

#define ARITH_SERIES_SUM(n) (((n)*(n+1))/2)

// Need this because you cannot template dynamically allocated kernel memory in CUDA, as per https://stackoverflow.com/questions/27570552/templated-cuda-kernel-with-dynamic-shared-memory
template <typename T>
__device__ T* shared_memory_proxy() {
    extern __shared__ unsigned char memory[];
    return reinterpret_cast<T*>(memory);
}

/**
 * Compute the distance between a given pair of sequences along every White-Neely step pattern option, for the given vertical swath of the cost matrix.
 */
template<typename T>
__global__ void DTWDistance(const T *first_seq_input, const size_t first_seq_input_length, const T *second_seq_input, const size_t second_seq_input_length, const size_t first_seq_index, 
                            const size_t offset_within_second_seq, const T *gpu_sequences, const size_t maxSeqLength, const size_t num_sequences, const size_t *gpu_sequence_lengths, 
                            T *dtwCostSoFar, T *newDtwCostSoFar, unsigned char *pathMatrix, const size_t pathMemPitch, T *dtwPairwiseDistances, const int use_open_start, const int use_open_end){
	// We need temporary storage for three diagonals of the wavefront calculation of the cost matrix to calculate the optimal path steps as a diagonal "wavefront" until we iterate 
	// through every position of the first sequence.
	T *costs = shared_memory_proxy<T>();

	// Which two are we comparing in this threadblock?
	// See if there is anything to process in this thread block 
	const size_t second_seq_length = second_seq_input ? second_seq_input_length : gpu_sequence_lengths[first_seq_index+blockIdx.x+1];
	if(offset_within_second_seq >= second_seq_length){
		return; // all threads in the threadblock will return
	}

	const size_t first_seq_length = first_seq_input ? first_seq_input_length : gpu_sequence_lengths[first_seq_index];
	const T *first_seq = first_seq_input ? first_seq_input : &gpu_sequences[first_seq_index*maxSeqLength];
	const T *second_seq = second_seq_input ? second_seq_input : &gpu_sequences[(first_seq_index+blockIdx.x+1)*maxSeqLength];

	// Point to the correct spot in global memory where the costs are being stored.
	dtwCostSoFar = &dtwCostSoFar[first_seq_length*blockIdx.x];
	newDtwCostSoFar = &newDtwCostSoFar[first_seq_length*blockIdx.x];

	// Each thread will be using the same second sequence value throughout the rest of the kernel, so store it as a local variable for efficiency.
	const T second_seq_thread_val = offset_within_second_seq+threadIdx.x >= second_seq_length ? 0 : second_seq[offset_within_second_seq+threadIdx.x];
	// printf("offset_within_second_seq: %i, second_seq_thread_val: %f\n", offset_within_second_seq, second_seq_thread_val);

	// Possible shortcut: If we are allowing open right moves only at the end of the alignment, 
	// and the top row of the matrix is in that state, and it's the lowest cost option in this column
	// of the DTW cost matrix, there is no need to continue computing further grid X elements. Why? 
	// Because any other path will just get more expensive, while the open right move can continue
	// eating up the subject to get to the upper right corner of the DTW matrix without any additional cost.
	// This will speed up prefix searches in particular, where the prefix (1st seq) length is a small proportion of the 2nd's.
	// We don't even need to be keeping the pathMatrix to know we are in that state, because it's the overriding move choice (see used_open_right_end_cost below)
	// when you're at the top of the matrix in open end mode.
	if(offset_within_second_seq > first_seq_length && use_open_end && !use_open_start){
		// Check if the search has already been abrogated by a previous kernel call (further left in the DTW matrix calculation) 
		if(dtwCostSoFar[0] == numeric_limits<T>::max()){
			if(pathMatrix != 0 && offset_within_second_seq+threadIdx.x < second_seq_length){
                                pathMatrix[pitchedCoord(offset_within_second_seq+threadIdx.x,first_seq_length-1,pathMemPitch)] = OPEN_RIGHT;
                        }
			return;
		}
		
		// Otherwise map/reduce within this kernel to pretty efficiently find the minimum value across the 1D dtwCostSoFar array without variable length threadblock shared memory.
		T minval = threadIdx.x < first_seq_length ? dtwCostSoFar[threadIdx.x] : numeric_limits<T>::max(); 
		for(int i = 1; i*blockDim.x < first_seq_length; i++){ 
			// Assign each thread to find the minimum values strided (by # threads doing work) across the length of the first sequence. 
			if(i*blockDim.x+threadIdx.x < first_seq_length && minval > dtwCostSoFar[i*blockDim.x+threadIdx.x]){
				// Hopefully mostly coalesced memory access
				minval = dtwCostSoFar[i*blockDim.x+threadIdx.x];
			}
		}
		minval = warpReduceMin<T>(minval); // across the warp
		int lane = threadIdx.x % CUDA_WARP_WIDTH;
    		int wid = threadIdx.x / CUDA_WARP_WIDTH;
		T *warp_minvals = costs; // threadblock shared memory space is the same as the costs pseudo 2D array since we won't need it outside of this block 
		if(!lane) warp_minvals[wid] = minval;
      		__syncthreads();  
      		// Get in-bounds values only for final threadblock reduction, calculated by the first warp's threads (threadblock may not be full).
      		if(!wid){
			minval = (threadIdx.x < blockDim.x / CUDA_WARP_WIDTH) ? warp_minvals[lane] : numeric_limits<T>::max();
			warp_minvals[0] = warpReduceMin<T>(minval); // across all threads in the block
		}
      		__syncthreads();  

		// Top row value is the lowest for this column, only need to populate the open_right move for correct backtracking and cumulative cost calcs
		if(dtwCostSoFar[first_seq_length-1] == warp_minvals[0]){
			if(pathMatrix != 0 && offset_within_second_seq+threadIdx.x < second_seq_length){
				pathMatrix[pitchedCoord(offset_within_second_seq+threadIdx.x,first_seq_length-1,pathMemPitch)] = OPEN_RIGHT;
			}
			// Make sure bottom row's DTW cost so far is set to the max possible. 
			// This is how we indicate that we've decided to abrogated the rest of the search.
			if(threadIdx.x == 0){
				dtwCostSoFar[0] = numeric_limits<T>::max();
			}
			return;
	  	}
	}

	if(threadIdx.x == 0){
		// Populate the bottom row of the vertical swath on every kernel invocation, this can't be done in parallel.
		const T first_seq_start_val = first_seq[0];
		if(offset_within_second_seq == 0){
			costs[0] = 0;
			if(pathMatrix != 0){
				pathMatrix[pitchedCoord(0,0,pathMemPitch)] = use_open_start ? NIL_OPEN_RIGHT : NIL; // sentinel for path backtracking algorithm termination
			}
		}
		else{
			costs[0] = dtwCostSoFar[0];
			if(pathMatrix != 0){
				pathMatrix[pitchedCoord(offset_within_second_seq,0,pathMemPitch)] = use_open_start ? OPEN_RIGHT : RIGHT;
			}
		}
		costs[0] += use_open_start ? 0 : (first_seq_start_val-second_seq_thread_val)*(first_seq_start_val-second_seq_thread_val);
		int col;
		for(col = 1; col < blockDim.x && offset_within_second_seq+col < second_seq_length; col++){
			T diff = use_open_start ? 0 : first_seq_start_val-second_seq[offset_within_second_seq+col];
			costs[col+blockDim.x*(col%3)] = costs[(col-1)+blockDim.x*((col-1)%3)]+diff*diff;
			if(pathMatrix != 0){
				pathMatrix[pitchedCoord(offset_within_second_seq+col,0,pathMemPitch)] = use_open_start ? OPEN_RIGHT : RIGHT;
			}
		}
		newDtwCostSoFar[0] = costs[(col-1)];
	}
	
	int i; // Indicates the ordinal of the diagonal of the wave front cost values being calculated
	for(i = 1; i < first_seq_length+blockDim.x; i++){

		if(offset_within_second_seq+threadIdx.x < second_seq_length && // We're within the sequence bounds?
		   threadIdx.x < i &&                                          // The diff still corresponds to a spot in the cost matrix?
		   i-threadIdx.x < first_seq_length){
			volatile T up_cost = numeric_limits<T>::max();
			volatile T diag_cost = numeric_limits<T>::max();
			volatile T right_cost = numeric_limits<T>::max();
			volatile T diff = first_seq[i-threadIdx.x]-second_seq_thread_val;

			// The left edge of cost matrix vertical swath is a special case as we need to 
			// access previously global mem stored intermediate costs.
			int used_open_right_end_cost = 0;
			if(threadIdx.x == 0){
				// Straight up is always an option
				up_cost = costs[blockDim.x*((i-1)%3)] + diff*diff;
				if(offset_within_second_seq != 0){
					// All three steps are possible, two drawn from previous intermediate results
					right_cost = dtwCostSoFar[i];
					diag_cost = dtwCostSoFar[i-1];
					if(i-threadIdx.x < first_seq_length-1 || !use_open_end){
						right_cost += diff*diff;
						diag_cost += diff*diff;
					}
					else{
						used_open_right_end_cost = 1;
					}
				}
			}
			// For all other threads all the input data is stored locally in costs[].
			else{
				up_cost = costs[threadIdx.x+blockDim.x*((i-1)%3)] + diff*diff;
				right_cost = costs[(threadIdx.x-1)+blockDim.x*((i-1)%3)] + diff*diff;
				diag_cost = costs[(threadIdx.x-1)+blockDim.x*((i-2)%3)] + diff*diff;
			}

			// Use the White-Neely step pattern (a diagonal move is preferred to right-up or up-right if costs are equivalent).
			if(use_open_end && i-threadIdx.x == first_seq_length-1 && threadIdx.x != 0){
				// No extra cost to consume a sequence element from the first sequence, just copy it over from the previous column.
				right_cost = costs[(threadIdx.x-1)+blockDim.x*((i-1)%3)];
				used_open_right_end_cost = 1;
			}
			// char move;
			if(diag_cost > up_cost){
				if(up_cost > right_cost){
					costs[threadIdx.x+blockDim.x*(i%3)] = right_cost;
					if(pathMatrix != 0){pathMatrix[pitchedCoord(offset_within_second_seq+threadIdx.x,i-threadIdx.x,pathMemPitch)] = used_open_right_end_cost ? OPEN_RIGHT : RIGHT;}
					// move = 'R';
				}
				else{
					costs[threadIdx.x+blockDim.x*(i%3)] = up_cost;
					if(pathMatrix != 0){pathMatrix[pitchedCoord(offset_within_second_seq+threadIdx.x,i-threadIdx.x,pathMemPitch)] = UP;}
					// move = 'U';
				}
			}
			else{
				if(diag_cost > right_cost){
					costs[threadIdx.x+blockDim.x*(i%3)] = right_cost;
                                        if(pathMatrix != 0){pathMatrix[pitchedCoord(offset_within_second_seq+threadIdx.x,i-threadIdx.x,pathMemPitch)] = used_open_right_end_cost ? OPEN_RIGHT : RIGHT;}
					// move = 'R';
				}
				else{
					costs[threadIdx.x+blockDim.x*(i%3)] = diag_cost;
					if(pathMatrix != 0){pathMatrix[pitchedCoord(offset_within_second_seq+threadIdx.x,i-threadIdx.x,pathMemPitch)] = DIAGONAL;}
					// move = 'D';
				}
			}
			// if(first_seq_index == 0) printf("0, %i, %hi, %f, %f, %f, %f, %c\n", i, threadIdx.x, up_cost, right_cost, diag_cost, diff*diff, move);
			// if(first_seq_index == 1) printf("1, %i, %hi, %f, %f, %f, %f, %c\n", i, threadIdx.x, up_cost, right_cost, diag_cost, diff*diff, move);
			// Right edge is a special case as we need to store back out intermediate result to global mem
			// for the use of the next kernel call with a larger offset_within_second_seq.
			if(threadIdx.x == blockDim.x-1 || offset_within_second_seq+threadIdx.x == second_seq_length - 1){
				newDtwCostSoFar[i-threadIdx.x] = costs[threadIdx.x+blockDim.x*(i%3)];
			}
		}

		// To ensure all required previous costs from neighbouring threads are calculated and available for the next iteration.
		__syncthreads();
	}
	// If this is the end of the second sequence, we now know the total cost of the alignment and can populate 
	// global var dtwPairwiseDistances. This is more efficient than doing a round trip on the PCI bus to the CPU for the same purpose.
	if(offset_within_second_seq+blockDim.x >= second_seq_length){
		if(dtwPairwiseDistances != 0 && threadIdx.x == 0){
			// 1D index for row into distances upper left pairs triangle is the total size of the triangle, minus all those that haven't been processed yet. 
			dtwPairwiseDistances[ARITH_SERIES_SUM(num_sequences-1)-ARITH_SERIES_SUM(num_sequences-first_seq_index-1)+blockIdx.x] = (T) sqrtf(newDtwCostSoFar[first_seq_length-1]);
		}
	}
	
}

#endif
