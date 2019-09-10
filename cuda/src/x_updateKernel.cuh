#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

#include "common.h"

template <class T, class W>
__global__
void dct_ReorderEvenKernel(T *phi, int N_width, int N_height, W *y)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix >= N_width || iy >= N_height) return;
    
    const int pos = ix + iy * N_width;
    
	y[pos].y = 0.0f;
    if (iy < N_height/2){
        y[pos].x = phi[ix + 2*iy*N_width];
    }
    else{
        y[pos].x = phi[ix + (2*(N_height-iy)-1)*N_width];
    }
}

template <class T, class W>
__global__
void dct_MultiplyFFTWeightsKernel(int N_width, int N_height, W *y, T *b, const W *ww)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= N_width || iy >= N_height) return;
	
	const int pos = ix + iy * N_width;
	const int pos_tran = iy + ix * N_height; // transpose on the output b
    
	b[pos_tran] = (ww[iy].x * y[pos].x + ww[iy].y * y[pos].y) / (T)N_height;
	if (iy == 0)
		b[pos_tran] /= sqrtf(2);
}

template <class T>
__global__
void divide_mat_x_hatKernel(T *phi, const T *mat_x_hat, int N_width, int N_height)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= N_width || iy >= N_height) return;
	
	const int pos = ix + iy * N_width;
	
	phi[pos] /= mat_x_hat[pos];
}

template <class T, class W>
__global__
void idct_MultiplyFFTWeightsKernel(int N_width, int N_height, T *b, W *y, const W *ww)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= N_width || iy >= N_height) return;
	
	const int pos = ix + iy * N_width;
    
	y[pos].x = ww[iy].x * b[pos];
	y[pos].y = ww[iy].y * b[pos];
	if (iy == 0){
		y[pos].x /= sqrtf(2);
		y[pos].y /= sqrtf(2);
	}
}

template <class T, class W>
__global__
void idct_ReorderEvenKernel(W *y, int N_width, int N_height, T *phi)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix >= N_width || iy >= N_height) return;
    
//    const int pos = ix + iy * N_width;
    const int pos = iy + ix * N_height;
    
    if ((iy & 1) == 0){ // iy is even
        phi[pos] = y[ix + iy/2*N_width].x / (T) N_height;
    }
    else{ // iy is odd
        phi[pos] = y[ix + (N_height-(iy+1)/2)*N_width].x / (T) N_height;
    }
}


template <class T, class W>
void x_update(T *phi, W *y, const T *mat_x_hat, 
              const W *ww_1, const W *ww_2,
         int N_width, int N_height, cufftHandle plan_dct_1, cufftHandle plan_dct_2)
{
    dim3 threads(32, 32);
    dim3 blocks_1(iDivUp(N_width,  threads.x), iDivUp(N_height, threads.y));
    dim3 blocks_2(iDivUp(N_height, threads.x), iDivUp(N_width,  threads.y));

    // first DCT
    dct_ReorderEvenKernel<T,W><<<blocks_1, threads>>>(phi, N_width, N_height, y);
//    cufftSafeCall(cufftExecC2C(plan_dct_1, y, y, CUFFT_FORWARD));
    cufftExecC2C(plan_dct_1, y, y, CUFFT_FORWARD);
    dct_MultiplyFFTWeightsKernel<T,W><<<blocks_1, threads>>>(N_width, N_height, y, phi, ww_1);
    
    // second DCT
    dct_ReorderEvenKernel<T,W><<<blocks_2, threads>>>(phi, N_height, N_width, y);
    cufftExecC2C(plan_dct_2, y, y, CUFFT_FORWARD);
    dct_MultiplyFFTWeightsKernel<T,W><<<blocks_2, threads>>>(N_height, N_width, y, phi, ww_2);
    
    // divided by mat_x_hat
    divide_mat_x_hatKernel<<<blocks_1, threads>>>(phi, mat_x_hat, N_width, N_height);
    
    // first IDCT
    idct_MultiplyFFTWeightsKernel<T,W><<<blocks_1, threads>>>(N_width, N_height, phi, y, ww_1);
    cufftExecC2C(plan_dct_1, y, y, CUFFT_INVERSE);
    idct_ReorderEvenKernel<T,W><<<blocks_1, threads>>>(y, N_width, N_height, phi);
    
    // second IDCT
    idct_MultiplyFFTWeightsKernel<T,W><<<blocks_2, threads>>>(N_height, N_width, phi, y, ww_2);
    cufftExecC2C(plan_dct_2, y, y, CUFFT_INVERSE);
    idct_ReorderEvenKernel<T,W><<<blocks_2, threads>>>(y, N_height, N_width, phi);
}
