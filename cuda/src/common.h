///////////////////////////////////////////////////////////////////////////////
// Header for common includes and utility functions
///////////////////////////////////////////////////////////////////////////////

#ifndef COMMON_H
#define COMMON_H


///////////////////////////////////////////////////////////////////////////////
// Common includes
///////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <memory.h>
#include <math.h>

#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cufft.h>
#include <cuComplex.h>

///////////////////////////////////////////////////////////////////////////////
// Error checking functions
///////////////////////////////////////////////////////////////////////////////
//static const char *_cudaGetErrorEnum(cufftResult error)
//{
//    switch (error)
//    {
//        case CUFFT_SUCCESS:
//            return "CUFFT_SUCCESS";

//        case CUFFT_INVALID_PLAN:
//            return "CUFFT_INVALID_PLAN";

//        case CUFFT_ALLOC_FAILED:
//            return "CUFFT_ALLOC_FAILED";

//        case CUFFT_INVALID_TYPE:
//            return "CUFFT_INVALID_TYPE";

//        case CUFFT_INVALID_VALUE:
//            return "CUFFT_INVALID_VALUE";

//        case CUFFT_INTERNAL_ERROR:
//            return "CUFFT_INTERNAL_ERROR";

//        case CUFFT_EXEC_FAILED:
//            return "CUFFT_EXEC_FAILED";

//        case CUFFT_SETUP_FAILED:
//            return "CUFFT_SETUP_FAILED";

//        case CUFFT_INVALID_SIZE:
//            return "CUFFT_INVALID_SIZE";

//        case CUFFT_UNALIGNED_DATA:
//            return "CUFFT_UNALIGNED_DATA";
//    }

//    return "<unknown>";
//}

//#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)
//inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
//{
//	if( CUFFT_SUCCESS != err) {
//		fprintf(stderr, "CUFFT error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, \
//			_cudaGetErrorEnum(err)); \
//			cudaDeviceReset(); assert(0); \
//	}
//}


///////////////////////////////////////////////////////////////////////////////
// Common constants
///////////////////////////////////////////////////////////////////////////////
const int StrideAlignment = 32;

// #ifdef DOUBLE_PRECISION
//     typedef double real;
//     typedef cufftDoubleComplex complex;
// #else
//     typedef float real;
//     typedef cufftComplex complex;
// #endif

typedef cufftComplex complex;

#define PI 3.1415926535897932384626433832795028841971693993751

// block size for shared memory
#define BLOCK_X  32
#define BLOCK_Y  32


///////////////////////////////////////////////////////////////////////////////
// Common functions
///////////////////////////////////////////////////////////////////////////////

// A GPU timer
struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
  	}

  	~GpuTimer()
  	{
    	cudaEventDestroy(start);
    	cudaEventDestroy(stop);
  	}

	void Start()
  	{
    	cudaEventRecord(start, 0);
  	}

  	void Stop()
  	{
    	cudaEventRecord(stop, 0);
  	}

  	float Elapsed()
  	{
    	float elapsed;
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&elapsed, start, stop);
    	return elapsed;
  	}
};

// Align up n to the nearest multiple of m
inline int iAlignUp(int n, int m = StrideAlignment)
{
    int mod = n % m;

    if (mod)
        return n + m - mod;
    else
        return n;
}

// round up n/m
inline int iDivUp(int n, int m)
{
    return (n + m - 1) / m;
}

// swap two values
template<typename T>
inline void Swap(T &a, T &b)
{
    T t = a;
    a = b;
    b = t;
}

// Wrap to [-0.5 0.5], then add 0.5 to [0 1] for final phase show
__host__
__device__
inline float wrap(float x)
{
    return x - floor(x + 0.5f) + 0.5f;
}

// nearest integer of power of 2
__host__
inline int nearest_power_of_two(int x)
{
	return 2 << (static_cast<int>(std::log2(x) + 1.0) - 1);
}
#endif
