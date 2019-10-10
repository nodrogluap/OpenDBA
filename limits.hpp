#ifndef __CUDA_LIMITS_HPP
#define __CUDA_LIMITS_HPP

#if defined(_WIN32)
	typedef unsigned short ushort;
	typedef unsigned int uint;
#endif

#include <limits.h>

namespace cudahack {

	template <class T> struct numeric_limits;

	template <> struct numeric_limits<short> {
    		__device__ __forceinline__ static short min() { return SHRT_MIN; }
    		__device__ __forceinline__ static short max() { return SHRT_MAX; }
	};

	template <> struct numeric_limits<ushort> {
    		__device__ __forceinline__ static ushort min() { return 0; }
    		__device__ __forceinline__ static ushort max() { return USHRT_MAX; }
	};

	template <> struct numeric_limits<int> {
    		__device__ __forceinline__ static int min() { return INT_MIN; }
    		__device__ __forceinline__ static int max() { return INT_MAX; }
	};

	template <> struct numeric_limits<uint> {
    		__device__ __forceinline__ static uint min() { return 0; }
    		__device__ __forceinline__ static uint max() { return UINT_MAX; }
	};

	template <> struct numeric_limits<unsigned long long int> {
    		__device__ __forceinline__ static unsigned long long int min() { return 0; }
    		__device__ __forceinline__ static unsigned long long int max() { return ULLONG_MAX; }
	};

	template <> struct numeric_limits<float> {
    		__device__ __forceinline__ static float min() { return FLT_MIN; }
    		__device__ __forceinline__ static float max() { return FLT_MAX; }
	};

	template <> struct numeric_limits<double> {
    		__device__ __forceinline__ static double min() { return DBL_MIN; }
    		__device__ __forceinline__ static double max() { return DBL_MAX; }
	};
}

#endif
