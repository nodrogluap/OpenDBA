#ifndef __dba_cuda_utils_included
#define __dba_cuda_utils_included

#include "multithreading.h"

#define CUDA_THREADBLOCK_MAX_L1CACHE 48000
// Note that you should not change this to >1028 unless you carefully review all the code for reduction steps that imply 32x32 map-reduce!
#ifndef CUDA_THREADBLOCK_MAX_THREADS
#define CUDA_THREADBLOCK_MAX_THREADS 1024
#endif
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

unsigned int * getMaxThreadsPerDevice(int deviceCount){
        unsigned int *maxThreads;
        cudaMallocHost(&maxThreads, sizeof(unsigned int)*deviceCount); CUERR("Allocating CPU memory for CUDA device properties");
        cudaDeviceProp deviceProp;
        for(int i = 0; i < deviceCount; i++){
                cudaGetDeviceProperties(&deviceProp, i); CUERR("Getting GPU device properties");
                // When debugging there are too many registers in the DTW kernel (due to number of local variables to
                // track without optimization) and you get failure to launch when using a full thread count complement.
#if DEBUG == 1
                maxThreads[i] = deviceProp.maxThreadsPerBlock/4;
#else
                maxThreads[i] = deviceProp.maxThreadsPerBlock;
#endif
        }
        return maxThreads;
}

// Methods below free resources after done using asynchronously called DTW kernels
struct heterogeneous_workload {
    void *dtwCostSoFar_memptr; // we only free it, so datatype templating is not neccesary
    void *newDtwCostSoFar_memptr; // we only free it, so datatype templating is not neccesary
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
        if(workload->newDtwCostSoFar_memptr != 0){
                cudaFree(workload->newDtwCostSoFar_memptr); CUERR("Freeing new DTW intermediate cost values");
        }
        if(workload->pathMatrix_memptr != 0){
                cudaFree(workload->pathMatrix_memptr); CUERR("Freeing DTW path matrix");
        }
        cudaStreamDestroy(workload->stream); CUERR("Removing a CUDA stream after completion");
        cudaFreeHost(workload); CUERR("Freeing host memory for dtwStreamCleanup");

        CUT_THREADEND;
}

__host__
void CUDART_CB dtwStreamCleanupLaunch(cudaStream_t stream, cudaError_t status, void *streamResources) {
        // Check status of GPU after stream operations are done. Die if there was a failure.
        CUERR("On callback after DTWDistance calculations completed.");

        // Spawn new CPU worker thread and perform stream resource cleanup on the CPU (since calling the CUDA API from within this callback is not allowed according to the docs).
        cutStartThread(dtwStreamCleanup, streamResources);
}

void addStreamCleanupCallback(void *dtwCostSoFar, void *newDtwCostSoFar, unsigned char *pathMatrix, cudaStream_t stream){
        heterogeneous_workload *cleanup_workload = 0;
        cudaMallocHost(&cleanup_workload, sizeof(heterogeneous_workload)); CUERR("Allocating page locked CPU memory for DTW stream callback data");
        cleanup_workload->dtwCostSoFar_memptr = dtwCostSoFar;
        cleanup_workload->newDtwCostSoFar_memptr = newDtwCostSoFar;
        cleanup_workload->pathMatrix_memptr = pathMatrix;
        cleanup_workload->stream = stream;
        cudaStreamAddCallback(stream, dtwStreamCleanupLaunch, cleanup_workload, 0);
}
#endif
