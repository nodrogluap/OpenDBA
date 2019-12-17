#ifndef __dba_gpu_utils_included
#define __dba_gpu_utils_included

#include <cmath>

#include "cuda_utils.hpp"

#if DOUBLE_UNSUPPORTED == 1
#define ACCUMULATOR_PRIMITIVE_TYPE float
#else
#define ACCUMULATOR_PRIMITIVE_TYPE double
#endif

template<typename T>
__global__ void calc_sums(T *sequences, size_t maxSeqLength, size_t num_sequences, size_t *sequence_lengths, T *sequence_sums){
        __shared__ float warp_sums[CUDA_WARP_WIDTH];

        // grid X index is the sequence to be processed, grid Y is the chunk of that sequence to process (each chunk is thread block sized)
        size_t seq_pos = blockIdx.y*blockDim.x+threadIdx.x;
        T warp_sum = 0;
        if(seq_pos < sequence_lengths[blockIdx.x]){ // Coalesced global mem reads
                warp_sum = sequences[maxSeqLength*blockIdx.x+seq_pos];
        }
        __syncwarp();

        // Reduce the warp
        for (int offset = CUDA_WARP_WIDTH/2; offset > 0; offset /= 2){
                warp_sum += __shfl_down_sync(FULL_MASK, warp_sum, offset);
        }
	// Memoize the warp's total
        if(threadIdx.x%CUDA_WARP_WIDTH == 0){
                warp_sums[threadIdx.x/CUDA_WARP_WIDTH] = warp_sum;
        }
        __syncthreads();

        int warp_limit = (int) ceilf(blockDim.x/((double) CUDA_WARP_WIDTH));
        // Reduce the whole threadblock
        if(threadIdx.x < CUDA_WARP_WIDTH){
                warp_sum = threadIdx.x < warp_limit ? warp_sums[threadIdx.x] : 0;
                __syncwarp();
                for (int offset = CUDA_WARP_WIDTH/2; offset > 0; offset /= 2){
                        warp_sum += __shfl_down_sync(FULL_MASK, warp_sum, offset);
                }
                // Add to the total for this sequence
                if(threadIdx.x == 0){
                        atomicAdd(&sequence_sums[blockIdx.x], warp_sum);
                }
        }
}

template<typename T>
__global__ void calc_sum_of_squares(T *sequences, size_t maxSeqLength, size_t num_sequences, size_t *sequence_lengths, T *sequence_sums, ACCUMULATOR_PRIMITIVE_TYPE *sequence_sum_of_squares){
        __shared__ float warp_sums_of_squares[CUDA_WARP_WIDTH];

        // grid X index is the sequence to be processed, grid Y is the chunk of that sequence to process (each chunk is thread block sized)
        size_t seq_pos = blockIdx.y*blockDim.x+threadIdx.x;
        double warp_sum = 0;
        if(seq_pos < sequence_lengths[blockIdx.x]){ // Coalesced global mem reads
                warp_sum = sequences[maxSeqLength*blockIdx.x+seq_pos]-(sequence_sums[blockIdx.x]/((double) sequence_lengths[blockIdx.x]));
                warp_sum *= warp_sum;
        }
        __syncwarp();

        // Reduce the warp
        for (int offset = CUDA_WARP_WIDTH/2; offset > 0; offset /= 2){
                warp_sum += __shfl_down_sync(FULL_MASK, warp_sum, offset);
        }
        if(threadIdx.x%CUDA_WARP_WIDTH == 0){
                warp_sums_of_squares[threadIdx.x/CUDA_WARP_WIDTH] = warp_sum;
        }
        __syncthreads();

        int warp_limit = (int) ceilf(blockDim.x/((double) CUDA_WARP_WIDTH));
        // Reduce the whole threadblock
        if(threadIdx.x < CUDA_WARP_WIDTH){
                warp_sum = threadIdx.x < warp_limit ? warp_sums_of_squares[threadIdx.x] : 0;
                __syncwarp();
                for (int offset = CUDA_WARP_WIDTH/2; offset > 0; offset /= 2){
                        warp_sum += __shfl_down_sync(FULL_MASK, warp_sum, offset);
                }
                // Add to the total for this sequence
                if(threadIdx.x == 0){
                        atomicAdd(&sequence_sum_of_squares[blockIdx.x], warp_sum);
                }
        }
}

// Z-norm
template<typename T>
__global__ void rescale_sequences(T *sequences, size_t maxSeqLength, size_t num_sequences, size_t *sequence_lengths, T *sequence_sums, ACCUMULATOR_PRIMITIVE_TYPE *sequence_sum_of_squares, double target_mean, double target_stddev){
        // grid X index is the sequence to be processed, grid Y is the chunk of that sequence to process (each chunk is thread block sized)
        size_t seq_pos = blockIdx.y*blockDim.x+threadIdx.x;
        size_t seq_length = sequence_lengths[blockIdx.x];
        if(seq_pos < seq_length){ // Coalesced global mem reads
		double seq_mean = sequence_sums[blockIdx.x]/((double) seq_length);
		double seq_stddev = sqrt(sequence_sum_of_squares[blockIdx.x]/((double) seq_length));
                // If a data series was present that was all constant values, the std dev will be zero and cause problems (division by zero). 
                // Since in this case each sequence value is the seq_mean, we can just set seq_stddev to any non zero value and it will evaluate properly, since   
                // the numerator will be zero anyways.
                if(seq_stddev == 0){
			seq_stddev = 1;
		}
                sequences[maxSeqLength*blockIdx.x+seq_pos] = (T) (target_mean+((sequences[maxSeqLength*blockIdx.x+seq_pos]-seq_mean)/seq_stddev)*target_stddev);
        }
}

// Convert the data on each device into a normalized form.
template<typename T>
__host__ void normalizeSequences(T **sequences, size_t maxSeqLength, size_t num_sequences, size_t **sequence_lengths, int refSequenceIndex, cudaStream_t stream){
        // grid X index is the sequence to be processed, grid Y is the chunk of that sequence to process (each chunk is thread block sized)
        dim3 threadblockDim(CUDA_WARP_WIDTH*CUDA_WARP_WIDTH, 1, 1);
        dim3 gridDim(num_sequences, (int) ceilf(maxSeqLength/((double) threadblockDim.x)), 1);
        int shared_memory_required = sizeof(T)*CUDA_WARP_WIDTH;

	// Most efficient way to do this (power and memory usage wise) is to just use the first device to compute the normalization, then copy the data between the devices.
	cudaSetDevice(0); CUERR("Setting device for calculation of sequences' normalization");
        T *sequence_sums;
        cudaMalloc(&sequence_sums, sizeof(T)*num_sequences);  CUERR("Allocating GPU memory for sequence means");
        calc_sums<<<gridDim,threadblockDim,shared_memory_required,stream>>>(sequences[0], maxSeqLength, num_sequences, sequence_lengths[0], sequence_sums); CUERR("Calculating sequence sums");
        ACCUMULATOR_PRIMITIVE_TYPE *sequence_sum_of_squares;
        cudaMalloc(&sequence_sum_of_squares, sizeof(double)*num_sequences);  CUERR("Allocating GPU memory for sequence residuals' sum of squares");
        calc_sum_of_squares<<<gridDim,threadblockDim,shared_memory_required,stream>>>(sequences[0], maxSeqLength, num_sequences, sequence_lengths[0], sequence_sums, sequence_sum_of_squares); CUERR("Calculating sequences' sum of squares");

	// Not a valid index, so rescale each sequence to have a mean of 0 and a standard deviation of 1 (i.e. Z-normalize)
	if(refSequenceIndex < 0 || refSequenceIndex >= num_sequences){
        	rescale_sequences<<<gridDim,threadblockDim,0,stream>>>(sequences[0], maxSeqLength, num_sequences, sequence_lengths[0], sequence_sums, sequence_sum_of_squares, 0, 1); CUERR("Z-normalizing sequences");
	}
	else{
		T target_mean, target_std_dev;
		size_t seq_length;
		cudaMemcpy(&target_mean, sequence_sums+refSequenceIndex, sizeof(T), cudaMemcpyDeviceToHost); CUERR("Copying reference sequence sums to host");
		cudaMemcpy(&seq_length, sequence_lengths+refSequenceIndex, sizeof(size_t), cudaMemcpyDeviceToHost); CUERR("Copying reference sequence's length to host");
		target_mean /= seq_length;
		cudaMemcpy(&target_std_dev, sequence_sum_of_squares+refSequenceIndex, sizeof(T), cudaMemcpyDeviceToHost); CUERR("Copying reference sequence's sum of squares to host");
		target_std_dev = sqrt(target_std_dev);
        	rescale_sequences<<<gridDim,threadblockDim,0,stream>>>(sequences[0], maxSeqLength, num_sequences, sequence_lengths[0], sequence_sums, sequence_sum_of_squares, target_mean, target_std_dev); CUERR("Rescaling sequences to target mean and std dev");
	}

        int deviceCount;
        cudaGetDeviceCount(&deviceCount); CUERR("Getting GPU device count during normalization");
	for(int currDevice = 1; currDevice < deviceCount; currDevice++){
		// Efficient if using TCC-enabled CUDA driver (e.g. Telsa and Quadro), otherwise transparently does copy via CPU buffer (e.g. GTX cards)
		cudaMemcpyPeer(sequences[currDevice], currDevice, sequences[0], 0, sizeof(T)*maxSeqLength*num_sequences);
 	}

}

template<typename T>
__host__ void normalizeSequence(T **gpu_sequence_copies, size_t seqLength, cudaStream_t stream){
	// A bit of gymnastics to get a pointer on the GPU for the incoming sequence length
	size_t *gpu_sequence_length = 0;
        cudaMalloc(&gpu_sequence_length, sizeof(size_t)); CUERR("Allocating GPU memory for sequence length for normalization");
	cudaMemcpy(gpu_sequence_length, &seqLength, sizeof(size_t), cudaMemcpyHostToDevice); CUERR("Copying sequence length to GPU for normalization");
	normalizeSequences<T>(gpu_sequence_copies, seqLength, 1, &gpu_sequence_length, -1, stream);
}

#endif
