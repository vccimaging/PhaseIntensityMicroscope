#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <helper_cuda.h>

typedef cufftComplex complex;

#ifndef PREPARE_CUFFT_WARMUP_H
#define PREPARE_CUFFT_WARMUP_H

void cufft_warper(complex *h_in, int n, int m, cufftHandle plan, complex *h_out);

void cufft_prepare(int nLevels, int *pH_N, int *pW_N,
	cufftHandle *plan_dct_1, cufftHandle *plan_dct_2,
	cufftHandle *plan_dct_3, cufftHandle *plan_dct_4,
	cufftHandle *plan_dct_5, cufftHandle *plan_dct_6,
	cufftHandle *plan_dct_7, cufftHandle *plan_dct_8);

#endif
