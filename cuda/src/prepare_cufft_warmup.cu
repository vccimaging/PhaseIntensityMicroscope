#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <helper_cuda.h>
#include "assert.h"

typedef cufftComplex complex;

void cufft_warper(complex *h_in, int n, int m, cufftHandle plan, complex *h_out)
{
    const int data_size = n*m*sizeof(complex);

    // device memory allocation
    complex *d_temp;
    checkCudaErrors(cudaMalloc(&d_temp,  data_size));

    // transfer data from host to device
    checkCudaErrors(cudaMemcpy(d_temp, h_in, data_size, cudaMemcpyHostToDevice));

	// Compute the FFT
	cufftExecC2C(plan, d_temp, d_temp, CUFFT_FORWARD);

    // transfer result from device to host
    checkCudaErrors(cudaMemcpy(h_out, d_temp, data_size, cudaMemcpyDeviceToHost));

    // cleanup
    checkCudaErrors(cudaFree(d_temp));
}

void cufft_prepare(int nLevels, int *pH_N, int *pW_N,
	cufftHandle *plan_dct_1, cufftHandle *plan_dct_2,
	cufftHandle *plan_dct_3, cufftHandle *plan_dct_4,
	cufftHandle *plan_dct_5, cufftHandle *plan_dct_6,
	cufftHandle *plan_dct_7, cufftHandle *plan_dct_8)
{
	// prepare cufft plans & warmup
	printf("Preparing CuFFT plans and warmups ...  ");
	
	int Length1[1], Length2[1];
	if (nLevels >= 1)
	{
		Length1[0] = pH_N[0]; // for each FFT, the Length1 is N_height
		Length2[0] = pW_N[0];  // for each FFT, the Length2 is N_width
		cufftPlanMany(plan_dct_1, 1, Length1,
			Length1, pW_N[0], 1,
			Length1, pW_N[0], 1,
			CUFFT_C2C, pW_N[0]);
		cufftPlanMany(plan_dct_2, 1, Length2,
			Length2, pH_N[0], 1,
			Length2, pH_N[0], 1,
			CUFFT_C2C, pH_N[0]);
	}
	else
	{
		printf("No CuFFT plans prepared; out ... \n");
	}

	if (nLevels >= 2)
	{
		Length1[0] = pH_N[1]; // for each FFT, the Length1 is N_height
		Length2[0] = pW_N[1];  // for each FFT, the Length2 is N_width
		cufftPlanMany(plan_dct_3, 1, Length1,
			Length1, pW_N[1], 1,
			Length1, pW_N[1], 1,
			CUFFT_C2C, pW_N[1]);
		cufftPlanMany(plan_dct_4, 1, Length2,
			Length2, pH_N[1], 1,
			Length2, pH_N[1], 1,
			CUFFT_C2C, pH_N[1]);
	}

	if (nLevels >= 3)
	{
		Length1[0] = pH_N[2]; // for each FFT, the Length1 is N_height
		Length2[0] = pW_N[2];  // for each FFT, the Length2 is N_width				
		cufftPlanMany(plan_dct_5, 1, Length1,
			Length1, pW_N[2], 1,
			Length1, pW_N[2], 1,
			CUFFT_C2C, pW_N[2]);
		cufftPlanMany(plan_dct_6, 1, Length2,
			Length2, pH_N[2], 1,
			Length2, pH_N[2], 1,
			CUFFT_C2C, pH_N[2]);
	}

	if (nLevels >= 4)
	{
		Length1[0] = pH_N[3]; // for each FFT, the Length1 is N_height
		Length2[0] = pW_N[3];  // for each FFT, the Length2 is N_width			
		cufftPlanMany(plan_dct_7, 1, Length2,
			Length1, pW_N[3], 1,
			Length1, pW_N[3], 1,
			CUFFT_C2C, pW_N[3]);
		cufftPlanMany(plan_dct_8, 1, Length2,
			Length2, pH_N[3], 1,
			Length2, pH_N[3], 1,
			CUFFT_C2C, pH_N[3]);
	}

	// cufft warmup
	int N_width = pW_N[0];
	int N_height = pH_N[0];
	complex *h_warmup_in = new complex[N_width * N_height];
	complex *h_warmup_out = new complex[N_width * N_height];
	cufft_warper(h_warmup_in, N_width, N_height, *plan_dct_1, h_warmup_out);
	cufft_warper(h_warmup_in, N_width, N_height, *plan_dct_2, h_warmup_out);
	delete[] h_warmup_in;
	delete[] h_warmup_out;
	printf("Done.\n");
}