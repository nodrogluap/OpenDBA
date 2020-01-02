#ifndef __dba_cuda_utils_included
#define __dba_cuda_utils_included

#define CUDA_THREADBLOCK_MAX_L1CACHE 48000
// Note that you should not change this to >1028 unless you carefully review all the code for reduction steps that imply 32x32 map-reduce!
#define CUDA_THREADBLOCK_MAX_THREADS 1024
#define CUDA_WARP_WIDTH 32
#define CUERR(MSG) { cudaError_t err; \
    if ((err = cudaGetLastError()) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << MSG << ")" << std::endl; exit((int) err);}}
#define FULL_MASK 0xffffffff

#define DIV_ROUNDUP(numerator, denominator) (((numerator) + (denominator) - 1)/(denominator))

// Find the smallest value for a local variable within a warp 
template<typename T>
__inline__ __device__ T warpReduceMin(T val){
    for (int offset = CUDA_WARP_WIDTH/2; offset > 0; offset /= 2){
        T tmpVal = __shfl_down_sync(FULL_MASK, val, offset);
        if (tmpVal < val){
            val = tmpVal;
        }
    }
    return val;
}

template<typename T>
__inline__ __device__ T warpReduceMax(T val){
    for (int offset = CUDA_WARP_WIDTH/2; offset > 0; offset /= 2){
        T tmpVal = __shfl_down_sync(FULL_MASK, val, offset);
        if (tmpVal > val){
            val = tmpVal;
        }
    }
    return val;
}

#endif
