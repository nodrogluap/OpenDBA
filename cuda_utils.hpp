#ifndef __dba_cuda_utils_included
#define __dba_cuda_utils_included

#define CUDA_THREADBLOCK_MAX_L1CACHE 48000
#define CUDA_WARP_WIDTH 32
#define CUERR(MSG) { cudaError_t err; \
    if ((err = cudaGetLastError()) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " (" << MSG << ")" << std::endl; exit((int) err);}}
#define FULL_MASK 0xffffffff

#endif
