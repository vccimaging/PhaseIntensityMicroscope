// Please keep this include order to make sure a successful compilation!
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <cuda_profiler_api.h> // for profiling
#include <helper_cuda.h>

// OpenCV
#include <opencv2/highgui.hpp>

// include kernels
#include "common.h"
#include "prepare_cufft_warmup.h"
#include "computeDCTweightsKernel.cuh"
#include "x_updateKernel.cuh"
#include "prox_gKernel.cuh"
#include "addKernel.cuh"
#include "medianfilteringKernel.cuh"

// thrust headers
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <cmath>

#include "cws_A_phi.h"


// =============================================================
// thrust definitions
// =============================================================
// thrust operators
template <typename T>
struct absolute{
    __host__ __device__
    T operator()(const T& x) const {
        return fabs(x);
    }
};
template <typename T>
struct square{
    __host__ __device__
    T operator()(const T& x) const {
        return x * x;
    }
};

// define reduction pointer
thrust::device_ptr<real> thrust_ptr;

// thrust setup arguments
absolute<real>     u_op_abs;
square<real>       u_op_sq;
thrust::plus<real> b_op;
// =============================================================


// =============================================================
// norm & mean functions
// =============================================================
template <int N>
real norm(real *d_x, int n)
{
    switch (N)
    {
        case 1:
        {
            thrust_ptr = thrust::device_pointer_cast(d_x);
            return thrust::transform_reduce(thrust_ptr, thrust_ptr + n, u_op_abs, 0.0, b_op);
        }
        case 2:
        {
            thrust_ptr = thrust::device_pointer_cast(d_x);
            return thrust::transform_reduce(thrust_ptr, thrust_ptr + n, u_op_sq, 0.0, b_op);
        }
        default:
        {
            printf("Undefined norm!\n");
            return 0;
        }
    }
}
template <int N>
real mean(real *d_x, int n)
{
    switch (N)
    {
        case 1:
            return norm<1>(d_x, n) / static_cast<real>(n);
        case 2:
            return norm<2>(d_x, n) / static_cast<real>(n);
        default:
        {
            printf("Undefined mean!\n");
            return 0;
        }
    }
}
// =============================================================


// =============================================================
// helper functions
// =============================================================
// ----- set to ones -----
__global__
void setonesKernel(real *I, cv::Size N)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= N.width || iy >= N.height) return;

    I[ix + iy * N.width] = 1.0f;
}
void setasones(real *I, cv::Size N)
{
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(N.width, threads.x), iDivUp(N.height, threads.y));
    setonesKernel<<<blocks, threads>>>(I, N);
}
// =============================================================


// =============================================================
// constant memory cache
// =============================================================
__constant__ real shrinkage_value[2];
__constant__ int L_SIZE[2];
// =============================================================


// =============================================================
// pre-computation: DCT basis function
// =============================================================
__device__
double DCT_kernel(const int ix, const int iy, int w, int h)
{
    return 4.0 - 2*cos(PI*(double)ix/(double)w) - 2*cos(PI*(double)iy/(double)h);
}
__global__
void mat_A_hatKernel(opt_A opt_A_t, cv::Size M, real *mat_A_hat)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = ix + iy * M.width;

    if (ix >= M.width || iy >= M.height) return;

    // \nabla^2
    double lap_mat = DCT_kernel(ix, iy, M.width, M.height);

    // \nabla^4 + \nabla^2
    double K_mat = lap_mat * (lap_mat + 1.0);

    // get mat_A_hat
    mat_A_hat[pos] = (double) ( 1.0 + (opt_A_t.tau_new + opt_A_t.mu) * K_mat );
}
__global__
void mat_phi_hatKernel(opt_phi opt_phi_t, real beta, cv::Size N, real *mat_phi_hat)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = ix + iy * N.width;

    if (ix >= N.width || iy >= N.height) return;

    // \nabla^2
    double lap_mat = DCT_kernel(ix, iy, N.width, N.height);

    // get mat_phi_hat (pre-divide mu so there is no need for re-scaling phi in ADMM)
    if (pos == 0){
        mat_phi_hat[pos] = 1.0;
    }
    else{
        mat_phi_hat[pos] = (double) ( 
            (opt_phi_t.mu + beta + beta * lap_mat) * lap_mat );
    }
}
void compute_inverse_mat(opt_A opt_A_t, opt_phi opt_phi_t, real beta,
    cv::Size M, cv::Size N, real *mat_A_hat, real *mat_phi_hat)
{
    dim3 threads(16, 16);
    dim3 blocks_M(iDivUp(M.width, threads.x), iDivUp(M.height, threads.y));
    dim3 blocks_N(iDivUp(N.width, threads.x), iDivUp(N.height, threads.y));
    mat_A_hatKernel<<<blocks_M, threads>>>(opt_A_t, M, mat_A_hat);
    mat_phi_hatKernel<<<blocks_N, threads>>>(opt_phi_t, beta, N, mat_phi_hat);
}
// =============================================================


// =============================================================
// operators
// =============================================================
// ----- nabla -----
__global__
void nablaKernel(const real *__restrict__ I, cv::Size N, real *Ix, real *Iy)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = ix + iy * N.width;

    if (ix >= N.width || iy >= N.height) return;

    // replicate/symmetric boundary condition
    if(ix == 0){
        Ix[pos] = 0.0f;
    }
    else{
        Ix[pos] = I[pos] - I[(ix-1) + iy * N.width];
    }
    if (iy == 0){
        Iy[pos] = 0.0f;
    }
    else{
        Iy[pos] = I[pos] - I[ix     + (iy-1) * N.width];
    }
}
void nabla(const real *I, cv::Size N, real *nabla_x, real *nabla_y)
{
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(N.width, threads.x), iDivUp(N.height, threads.y));
    nablaKernel<<<blocks, threads>>>(I, N, nabla_x, nabla_y);
}
// -----------------

// ----- nablaT -----
__global__
void nablaTKernel(const real *__restrict__ Ix, const real *__restrict__ Iy, cv::Size N, real *div)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = ix + iy * N.width;

    real val1, val2;
    if (ix >= N.width || iy >= N.height) return;

    // replicate/symmetric boundary condition
    if(ix == N.width-1){
        val1 = 0.0f;
    }
    else{
        val1 = Ix[pos] - Ix[ix+1  + iy     * N.width];
    }
    if (iy == N.height-1){
        val2 = 0.0f;
    }
    else{
        val2 = Iy[pos] - Iy[ix    + (iy+1) * N.width];
    }
    div[pos] = val1 + val2;
}
void nablaT(const real *__restrict__ Ix, const real *__restrict__ Iy, cv::Size N, real *div)
{
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(N.width, threads.x), iDivUp(N.height, threads.y));
    nablaTKernel<<<blocks, threads>>>(Ix, Iy, N, div);
}
// -----------------

// ----- nabla2 -----
__global__
void nabla2Kernel(const real *__restrict__ I, cv::Size N, real *L)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int py = iy * N.width;
    const int pos = ix + py;

    if (ix >= N.width || iy >= N.height) return;

    // replicate/symmetric boundary condition (3x3 stencil)
    const int x_min = max(ix-1, 0);
    const int x_max = min(ix+1, N.width-1);
    const int y_min = max(iy-1, 0);
    const int y_max = min(iy+1, N.height-1);

    L[pos] = -4 * I[pos] + I[x_min + py] + I[x_max + py] + I[ix + y_min*N.width] + I[ix + y_max*N.width];
}
void nabla2(const real *I, cv::Size N, real *L)
{
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(N.width, threads.x), iDivUp(N.height, threads.y));
    nabla2Kernel<<<blocks, threads>>>(I, N, L);
}
// -----------------

// ----- K & KT -----
void K(const real *I, cv::Size N, real *grad_x, real *grad_y, real *L)
{
    nabla(I, N, grad_x, grad_y);
    nabla2(I, N, L);
}
__global__
void KTKernel(const real *__restrict__ Ix, const real *__restrict__ Iy, 
              const real *__restrict__ L, cv::Size N, real *I)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int py = iy * N.width;
    const int pos = ix + py;

    if (ix >= N.width || iy >= N.height) return;

    // replicate/symmetric boundary condition (3x3 stencil)
    const int x_min = max(ix-1, 0);
    const int x_max = min(ix+1, N.width-1);
    const int y_min = max(iy-1, 0);
    const int y_max = min(iy+1, N.height-1);

    real Div = Ix[pos] - Ix[x_max + iy*N.width]  +  Iy[pos] - Iy[ix + y_max*N.width];
    real Lap = -4 * L[pos] + L[x_min + py] + L[x_max + py] + L[ix + y_min*N.width] + L[ix + y_max*N.width];
    I[pos] = Div + Lap;
}
void KT(const real *grad_x, const real *grad_y, const real *L, cv::Size N, real *I)
{
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(N.width, threads.x), iDivUp(N.height, threads.y));
    KTKernel<<<blocks, threads>>>(grad_x, grad_y, L, N, I);
}
__global__
void MKernel(const real *in, cv::Size M, cv::Size N, real *out)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos_N = ix + iy * N.width;
	const int pos_M = ix - L_SIZE[0] + (iy - L_SIZE[1]) * M.width;

	if (ix >= N.width || iy >= N.height) return;
	else if (ix >= L_SIZE[0] && ix < N.width  - L_SIZE[0] &&
		     iy >= L_SIZE[1] && iy < N.height - L_SIZE[1]) {
        out[pos_M] = in[pos_N];
    }
    else return;
}
void Mask_func(const real *in, cv::Size M, cv::Size N, real *out)
{
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(N.width, threads.x), iDivUp(N.height, threads.y));
    MKernel<<<blocks, threads>>>(in, M, N, out);
}
// -----------------
// =============================================================


// =============================================================
// objective functions
// =============================================================
__global__
void res1Kernel(const real *A, const real *b, const real *I0, cv::Size N, real *tmp)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = ix + iy * N.width;

    if (ix >= N.width || iy >= N.height) return;
    tmp[pos] = A[pos] - b[pos]/I0[pos];
}
__global__
void res2Kernel(const real *A, const real *b, const real *I0, cv::Size M, real *tmp)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = ix + iy * M.width;

    if (ix >= M.width || iy >= M.height) return;
    tmp[pos] = I0[pos] * A[pos] - b[pos];
}
__global__
void res3Kernel(real *gx, real *gy, real *gt, real *wx, real *wy,
                cv::Size M, cv::Size N, real *out, bool isM = true)
{
    // This function will be used by:
    // - 1. Cost function of phi (function: obj_phi)
    // - 2. Update I_warp (function: update_I_warp)
    // To fulfill both goals, size of out is designed to be either M or N.
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos_N = ix + iy * N.width;
	const int pos_M = ix - L_SIZE[0] + (iy - L_SIZE[1]) * M.width;
    const int pos_tmp = isM ? pos_M : pos_N;

	if (ix >= N.width || iy >= N.height) return;
	else if (ix >= L_SIZE[0] && ix < N.width  - L_SIZE[0] &&
		     iy >= L_SIZE[1] && iy < N.height - L_SIZE[1]) {
        out[pos_tmp] = gx[pos_M] * wx[pos_N] + gy[pos_M] * wy[pos_N] + gt[pos_M];
    }
    else if (!isM) {
        out[pos_tmp] = 0.0f; // outside area be zeros; in this case tmp is of size N
    }
    else return;
}
real obj_A(const real *A, const real *b, const real *I0, opt_A opt_A_t, cv::Size N,
            real *tmp1, real *tmp2, real *tmp3)
{
    // compute A - b/I0, and data term
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(N.width, threads.x), iDivUp(N.height, threads.y));
    res1Kernel<<<blocks, threads>>>(A, b, I0, N, tmp1);
    real data_term = norm<2>(tmp1, N.area());

    // compute KA, and priors
    K(A, N, tmp1, tmp2, tmp3);
    real L1_term = opt_A_t.gamma_new * ( norm<1>(tmp1, N.area()) + norm<1>(tmp2, N.area()) + norm<1>(tmp3, N.area()) );
    real L2_term = opt_A_t.tau_new   * ( norm<2>(tmp1, N.area()) + norm<2>(tmp2, N.area()) + norm<2>(tmp3, N.area()) );

    return data_term + L1_term + L2_term;
}
real obj_phi(const real *phi, real *gx, real *gy, real *gt,
              real alpha, real beta, cv::Size M, cv::Size N,
              real *tmp1_N, real *tmp2_N, real *tmp3_N)
{
    // compute data term
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(N.width, threads.x), iDivUp(N.height, threads.y));
    nabla(phi, N, tmp1_N, tmp2_N);
    res3Kernel<<<blocks, threads>>>(gx, gy, gt, tmp1_N, tmp2_N, M, N, tmp1_N, false);
    real data_term = norm<2>(tmp1_N, N.area());

    // compute nabla phi, and phi priors
    K(phi, N, tmp1_N, tmp2_N, tmp3_N);
    real L1_term = alpha * ( norm<1>(tmp1_N, N.area()) + norm<1>(tmp2_N, N.area()) );
    real L2_term = beta  * ( norm<2>(tmp1_N, N.area()) + norm<2>(tmp2_N, N.area()) + norm<2>(tmp3_N, N.area()) );

    return data_term + L1_term + L2_term;
}
real obj_total(const real *A, const real *phi, const real *b, const real *I0,
                real alpha, real beta, real gamma, real tau, cv::Size M, cv::Size N,
                real *tmp1_M, real *tmp2_M, real *tmp3_M,
                real *tmp1_N, real *tmp2_N, real *tmp3_N)
{
    // compute I0*A - b, and data term
    dim3 threads(16, 16);
    dim3 blocks(iDivUp(M.width, threads.x), iDivUp(M.height, threads.y));
    res2Kernel<<<blocks, threads>>>(A, b, I0, M, tmp1_M);
    real data_term = norm<2>(tmp1_M, M.area());

    // compute nabla phi, and phi priors
    K(phi, N, tmp1_N, tmp2_N, tmp3_N);
    real phi_L1_term = alpha * ( norm<1>(tmp1_N, N.area()) + norm<1>(tmp2_N, N.area()) );
    real phi_L2_term = beta  * ( norm<2>(tmp1_N, N.area()) + norm<2>(tmp2_N, N.area()) + norm<2>(tmp3_N, N.area()) );

    // compute KA, and A priors
    K(A, M, tmp1_M, tmp2_M, tmp3_M);
    real A_L1_term = gamma * ( norm<1>(tmp1_M, M.area()) + norm<1>(tmp2_M, M.area()) + norm<1>(tmp3_M, M.area()) );
    real A_L2_term = tau   * ( norm<2>(tmp1_M, M.area()) + norm<2>(tmp2_M, M.area()) + norm<2>(tmp3_M, M.area()) );

    return data_term + phi_L1_term + phi_L2_term + A_L1_term + A_L2_term;
}
// =============================================================


// =============================================================
// sub functions for updates
// =============================================================
__global__
void ADMM_AKernel(real *zeta_x, real *zeta_y, real *zeta_z,
                  real *tmp_x, real *tmp_y, real *tmp_z, cv::Size N)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int pos = ix + iy * N.width;

    if (ix >= N.width || iy >= N.height) return;

    // compute u = KA + zeta
    real u_x = tmp_x[pos] + zeta_x[pos];
    real u_y = tmp_y[pos] + zeta_y[pos];
    real u_z = tmp_z[pos] + zeta_z[pos];

    // B-update
    real B_x = prox_l1(u_x, shrinkage_value[0]);
    real B_y = prox_l1(u_y, shrinkage_value[0]);
    real B_z = prox_l1(u_z, shrinkage_value[0]);

    // zeta-update
    zeta_x[pos] = u_x - B_x;
    zeta_y[pos] = u_y - B_y;
    zeta_z[pos] = u_z - B_z;

    // store B-zeta
    tmp_x[pos] = 2*B_x - u_x;
    tmp_y[pos] = 2*B_y - u_y;
    tmp_z[pos] = 2*B_z - u_z;
}
__global__
void ADMM_precomputeAKernel(real *I_warp, real *I0, real *KT_res, real mu, cv::Size N, real *A)
{
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    const int py = iy * N.width;
    const int pos = ix + py;

    if (ix >= N.width || iy >= N.height) return;

    // pre-compute for A-update (I_warp/I0 can be pre-computed)
    A[pos] = I_warp[pos] / I0[pos] + mu * KT_res[pos];
}
void A_update(real *A, real *I_warp, real *I0,
              real *zeta_x, real *zeta_y, real *zeta_z,
              real *grad_x_M, real *grad_y_M, real *lap_M,
              complex *tmp_dct_M, real *mat_A_hat,
              complex *ww_1_M, complex *ww_2_M, cufftHandle plan_dct_1_M, cufftHandle plan_dct_2_M,
              opt_A opt_A_t, cv::Size M)
{
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(M.width, threads.x), iDivUp(M.height, threads.y));

    // initial objective
    real objval;
    if (opt_A_t.isverbose)
    {
        printf("-- A update\n");
        objval = obj_A(A, I_warp, I0, opt_A_t, M, grad_x_M, grad_y_M, lap_M);
        printf("---- init, obj = %.4e \n", objval);    
    }

    // initialization
    checkCudaErrors(cudaMemset(zeta_x,   0, M.area()*sizeof(real)));
    checkCudaErrors(cudaMemset(zeta_y,   0, M.area()*sizeof(real)));
    checkCudaErrors(cudaMemset(zeta_z,   0, M.area()*sizeof(real)));
    checkCudaErrors(cudaMemset(grad_x_M, 0, M.area()*sizeof(real)));
    checkCudaErrors(cudaMemset(grad_y_M, 0, M.area()*sizeof(real)));
    checkCudaErrors(cudaMemset(lap_M,    0, M.area()*sizeof(real)));

    // ADMM loop
    for (signed int i = 1; i <= opt_A_t.iter; ++i)
    {
        // compute KT(B - zeta)
        KT(grad_x_M, grad_y_M, lap_M, M, A);

        // A-update inversion
        ADMM_precomputeAKernel<<<blocks, threads>>>(I_warp, I0, A, opt_A_t.mu, M, A);
        x_update(A, tmp_dct_M, mat_A_hat, ww_1_M, ww_2_M, M.width, M.height, plan_dct_1_M, plan_dct_2_M);

        // B-update & zeta-update
        K(A, M, grad_x_M, grad_y_M, lap_M);
        ADMM_AKernel<<<blocks, threads>>>(zeta_x, zeta_y, zeta_z, grad_x_M, grad_y_M, lap_M, M);
    }

    // show final objective
    if (opt_A_t.isverbose)
    {
        objval = obj_A(A, I_warp, I0, opt_A_t, M, grad_x_M, grad_y_M, lap_M);
        printf("---- iter = %d, obj = %.4e \n", opt_A_t.iter, objval);
    }
}
__global__
void I_warp_updateKernel(real *A, real *I0, cv::Size M, real *I_warp)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * M.width;

    if (ix >= M.width || iy >= M.height) return;
    I_warp[pos] = A[pos] * I0[pos];
}
void I_warp_update(real *A, real *I0, cv::Size M, real *I_warp)
{
	dim3 threads(32, 32);
	dim3 blocks(iDivUp(M.width, threads.x), iDivUp(M.height, threads.y));
    I_warp_updateKernel<<<blocks, threads>>>(A, I0, M, I_warp);
}
void update_I_warp(real *I_warp, real *phi, real *gx, real *gy, 
                   real *tmp1, real *tmp2, cv::Size M, cv::Size N)
{
	dim3 threads(32, 32);
	dim3 blocks(iDivUp(N.width, threads.x), iDivUp(N.height, threads.y));
    nabla(phi, N, tmp1, tmp2);
    res3Kernel<<<blocks, threads>>>(gx, gy, I_warp, tmp1, tmp2, M, N, I_warp, true);
}
texture<real, 2, cudaReadModeElementType> texTarget;
template<int mods> // template global kernel to handle all cases
__global__
void ComputeDerivativesL1Kernel(const real *__restrict__ I0, cv::Size M, real mu,
	real *A11, real *A12, real *A22, real *a, real *b, real *c, const real *__restrict__ I = NULL)
{
	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;
	const int pos = ix + iy * M.width;

	if (ix >= M.width || iy >= M.height) return;

    // compute the derivatives
    real gx, gy, gt;
    switch (mods)
    {
        case 0: // image size is multiple of 32; use texture memory
        {
            real dx = 1.0f / (real)M.width;
            real dy = 1.0f / (real)M.height;
        
            real x = ((real)ix + 0.5f) * dx;
            real y = ((real)iy + 0.5f) * dy;
        
            // x derivative
            gx  = tex2D(texTarget, x - 2.0f * dx, y);
            gx -= tex2D(texTarget, x - 1.0f * dx, y) * 8.0f;
            gx += tex2D(texTarget, x + 1.0f * dx, y) * 8.0f;
            gx -= tex2D(texTarget, x + 2.0f * dx, y);
            gx /= 12.0f;
        
            // t derivative
            gt  = tex2D(texTarget, x, y) - I0[pos];
        
            // y derivative
            gy  = tex2D(texTarget, x, y - 2.0f * dy);
            gy -= tex2D(texTarget, x, y - 1.0f * dy) * 8.0f;
            gy += tex2D(texTarget, x, y + 1.0f * dy) * 8.0f;
            gy -= tex2D(texTarget, x, y + 2.0f * dy);
            gy /= 12.0f;
        }
        break;

        case 1: // image size is not multiple of 32; use shared memory
        {
            const int tx = threadIdx.x + 2;
            const int ty = threadIdx.y + 2;
        
            if (1)
            {
                // save to shared memory
                __shared__ real smem[BLOCK_X+4][BLOCK_Y+4];
                smem[tx][ty] = I[pos];
                set_bd_shared_memory<5>(smem, I, tx, ty, ix, iy, M.width, M.height);
                __syncthreads();

                // x derivative
                gx  = smem[tx-2][ty];
                gx -= smem[tx-1][ty] * 8.0f;
                gx += smem[tx+1][ty] * 8.0f;
                gx -= smem[tx+2][ty];
                gx /= 12.0f;

                // t derivative
                gt  = smem[tx][ty] - I0[pos];
            
                // y derivative
                gy  = smem[tx][ty-2];
                gy -= smem[tx][ty-1] * 8.0f;
                gy += smem[tx][ty+1] * 8.0f;
                gy -= smem[tx][ty+2];
                gy /= 12.0f;
            }
            else
            {
                // x derivative
                gx  = I[max(0,ix-2) + iy * M.width];
                gx -= I[max(0,ix-1) + iy * M.width] * 8.0f;
                gx += I[min(M.width-1,ix+1) + iy * M.width] * 8.0f;
                gx -= I[min(M.width-1,ix+2) + iy * M.width];
                gx /= 12.0f;
    
                // t derivative
                gt  = I[pos] - I0[pos];
            
                // y derivative
                gy  = I[ix + max(0,iy-2) * M.width];
                gy -= I[ix + max(0,iy-1) * M.width] * 8.0f;
                gy += I[ix + min(M.height-1,iy+1) * M.width] * 8.0f;
                gy -= I[ix + min(M.height-1,iy+2) * M.width];
                gy /= 12.0f;
            }
        }
        break;
    }

    // pre-caching
	real gxx = gx*gx;
	real gxy = gx*gy;
	real gyy = gy*gy;
	real denom = 1.0f / (mu * (gxx + gyy + mu));

	// L1 + L2
	A11[pos] = denom * (gyy + mu); // A11
	A12[pos] = -gxy * denom;       // A12
	A22[pos] = denom * (gxx + mu); // A22
	a[pos] = gx;
	b[pos] = gy;
	c[pos] = gt;
}
void ComputeDerivativesL1(const real *__restrict__ I0, const real *__restrict__ I1,
	cv::Size M, int s, real mu, 
	real *A11, real *A12, real *A22, real *a, real *b, real *c)
{
    int mods = (((M.width % 32) == 0) && ((M.height % 32) == 0)) ? 0 : 1;
    switch (mods)
    {
        case 0: // image size is multiple of 32; use texture memory
        {
            dim3 threads(32, 32);
            dim3 blocks(iDivUp(M.width, threads.x), iDivUp(M.height, threads.y));
        
            // replicate if a coordinate value is out-of-range
            texTarget.addressMode[0] = cudaAddressModeClamp;
            texTarget.addressMode[1] = cudaAddressModeClamp;
            texTarget.filterMode = cudaFilterModeLinear;
            texTarget.normalized = true;
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<real>();
            cudaBindTexture2D(0, texTarget, I1, M.width, M.height, s*sizeof(real));

            ComputeDerivativesL1Kernel<0><<<blocks, threads>>>(I0, M, mu, A11, A12, A22, a, b, c);
        }
        break;

        case 1: // image size is not multiple of 32; use shared memory
        {
            dim3 threads(BLOCK_X, BLOCK_Y);
            dim3 blocks(iDivUp(M.width, threads.x), iDivUp(M.height, threads.y));
            ComputeDerivativesL1Kernel<1><<<blocks, threads>>>(I0, M, mu, A11, A12, A22, a, b, c, I1);
        }
        break;
    }
}
void phi_update(real *phi, real *I_warp, real *I0,
    real alpha, real beta,
    real *zeta_x, real *zeta_y, real *grad_x_N, real *grad_y_N,
    real *A11, real *A12, real *A22, real *gx, real *gy, real *gt,
    complex *tmp_dct_N, real *mat_phi_hat,
    complex *ww_1_N, complex *ww_2_N, cufftHandle plan_dct_1_N, cufftHandle plan_dct_2_N,
    opt_phi opt_phi_t, cv::Size M, cv::Size N)
{
    dim3 threads(32, 32);
    dim3 blocks(iDivUp(N.width, threads.x), iDivUp(N.height, threads.y));

    // compute derivatives and constants
    ComputeDerivativesL1(I0, I_warp, M, M.width, opt_phi_t.mu, 
                         A11, A12, A22, gx, gy, gt);

    // initial objective
    real objval;
    if (opt_phi_t.isverbose)
    {
        printf("-- phi update\n");
        objval = obj_phi(phi, gx, gy, gt, alpha, beta, M, N, grad_x_N, grad_y_N, zeta_x);
        printf("---- init, obj = %.4e \n", objval);
    }

    // initialization
    checkCudaErrors(cudaMemset(zeta_x,   0, N.area()*sizeof(real)));
    checkCudaErrors(cudaMemset(zeta_y,   0, N.area()*sizeof(real)));
    checkCudaErrors(cudaMemset(grad_x_N, 0, N.area()*sizeof(real)));
    checkCudaErrors(cudaMemset(grad_y_N, 0, N.area()*sizeof(real)));

    // ADMM loop
    for (signed int i = 1; i <= opt_phi_t.iter; ++i)
    {
        // compute nablaT(w - zeta)
        nablaT(grad_x_N, grad_y_N, N, phi);

        // phi-update inversion
        x_update(phi, tmp_dct_N, mat_phi_hat, ww_1_N, ww_2_N, N.width, N.height, plan_dct_1_N, plan_dct_2_N);

        // w-update & zeta-update
        nabla(phi, N, grad_x_N, grad_y_N);
        prox_gL1Kernel<<<blocks, threads>>>(zeta_x, zeta_y, grad_x_N, grad_y_N, opt_phi_t.mu, alpha, 
                       A11, A12, A22, gx, gy, gt, N.width, N.height, M.width, M.height);
    }

    // show final objective
    if (opt_phi_t.isverbose)
    {
        objval = obj_phi(phi, gx, gy, gt, alpha, beta, M, N, grad_x_N, grad_y_N, zeta_x);
        printf("---- iter = %d, obj = %.4e \n", opt_phi_t.iter, objval);
    }

    // update I_warp as: I_warp += [gx; gy] * \nabla\phi
    update_I_warp(I_warp, phi, gx, gy, grad_x_N, grad_y_N, M, N);
}
// =============================================================


CWS_A_Phi::CWS_A_Phi(cv::Mat I0, cv::Mat &h_A, cv::Mat &h_phi, opt_algo opt_algo_out)
{
    opt_algo_t = opt_algo_out;

    // is verbose
    opt_A_t.isverbose   = opt_algo_t.isverbose_sub;
    opt_phi_t.isverbose = opt_algo_t.isverbose_sub;

    // ADMM parameters
    opt_A_t.mu   = opt_algo_t.mu_A;
    opt_phi_t.mu = opt_algo_t.mu_phi;

    // total number of alternation iterations
    opt_A_t.iter   = opt_algo_t.A_iter;
    opt_phi_t.iter = opt_algo_t.phi_iter;

    // determine sizes
    M = I0.size();

    // validate L
    if (opt_algo_t.L.width < 2 || opt_algo_t.L.height < 2)
    {
        printf("L unspecified or wrong; will use nearest power of two of M; L \in [2, 32].\n");
        opt_algo_t.L.width  = min(max(2, (nearest_power_of_two(M.width)  - M.width) /2), 32);
        opt_algo_t.L.height = min(max(2, (nearest_power_of_two(M.height) - M.height)/2), 32);
    }

    // set L
    opt_phi_t.L.width  = opt_algo_t.L.width;
    opt_phi_t.L.height = opt_algo_t.L.height;

    // set N
    N = M + opt_algo_t.L + opt_algo_t.L;
    printf("M = [%d, %d], N = [%d, %d] \n", M.width, M.height, N.width, N.height);

    // allocate host cv::Mat containers
    h_A   = cv::Mat::zeros(M, CV_32F);
    h_phi = cv::Mat::zeros(N, CV_32F);

    // allocate device pointers
    checkCudaErrors(cudaMalloc(&d_I0,     M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&d_I,      M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&d_I0_tmp, M.area()*sizeof(real)));

    // copy from host to device
    checkCudaErrors(cudaMemcpy(d_I0, I0.ptr<real>(0), M.area()*sizeof(real), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_I0_tmp, d_I0, M.area()*sizeof(real), cudaMemcpyDeviceToDevice));

    // compute A-update parameters
    opt_A_t.gamma_new = opt_algo_t.gamma / mean<2>(d_I0, M.area());
    opt_A_t.tau_new   = opt_algo_t.tau   / mean<2>(d_I0, M.area());

    // set shrinkage value & cache to constant memory
    real shrinkage_value_array [2] = { opt_A_t.gamma_new/(2.0f*opt_algo_t.mu_A), opt_algo_t.alpha/(2.0f*opt_algo_t.mu_A) };
    cudaMemcpyToSymbol(shrinkage_value, shrinkage_value_array, 2*sizeof(real));

    // cache L to constant memory
    int L_size_array [2] = { opt_algo_t.L.width, opt_algo_t.L.height };
    cudaMemcpyToSymbol(L_SIZE, L_size_array, 2*sizeof(int));

    // allocate variable arrays
    // -- main variables
    checkCudaErrors(cudaMalloc(&A,         M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&phi,       N.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&Delta_phi, N.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&I_warp,    M.area()*sizeof(real)));

    // -- update variables
    checkCudaErrors(cudaMalloc(&zeta_x_M, M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&zeta_y_M, M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&zeta_z_M, M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&zeta_x_N, N.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&zeta_y_N, N.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&zeta_z_N, N.area()*sizeof(real)));

    // -- temp arrays
    checkCudaErrors(cudaMalloc(&grad_x_M, M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&grad_y_M, M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&lap_M,    M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&grad_x_N, N.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&grad_y_N, N.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&lap_N,    N.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&tmp_dct_M,M.area()*sizeof(complex)));
    checkCudaErrors(cudaMalloc(&tmp_dct_N,N.area()*sizeof(complex)));

    // -- temp variables for constant coefficients caching
    checkCudaErrors(cudaMalloc(&A11, M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&A12, M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&A22, M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&a,   M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&b,   M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&c,   M.area()*sizeof(real)));

    // prepare FFT plans
    int pH_M[1] = {M.height}, pW_M[1] = {M.width};
    int pH_N[1] = {N.height}, pW_N[1] = {N.width};
    cufft_prepare(1, pH_M, pW_M, &plan_dct_1_M, &plan_dct_2_M, NULL, NULL, NULL, NULL, NULL, NULL);
    cufft_prepare(1, pH_N, pW_N, &plan_dct_1_N, &plan_dct_2_N, NULL, NULL, NULL, NULL, NULL, NULL);

    // prepare DCT coefficients
    checkCudaErrors(cudaMalloc(&ww_1_M, M.height*sizeof(complex)));
    checkCudaErrors(cudaMalloc(&ww_2_M, M.width *sizeof(complex)));
    checkCudaErrors(cudaMalloc(&ww_1_N, N.height*sizeof(complex)));
    checkCudaErrors(cudaMalloc(&ww_2_N, N.width *sizeof(complex)));
    computeDCTweights(M.width, M.height, ww_1_M, ww_2_M);
    computeDCTweights(N.width, N.height, ww_1_N, ww_2_N);

    // prepare inversion matrices
    checkCudaErrors(cudaMalloc(&mat_A_hat,   M.area()*sizeof(real)));
    checkCudaErrors(cudaMalloc(&mat_phi_hat, N.area()*sizeof(real)));
    compute_inverse_mat(opt_A_t, opt_phi_t, opt_algo_t.beta, M, N, mat_A_hat, mat_phi_hat);
}

CWS_A_Phi::~CWS_A_Phi()
{
    checkCudaErrors(cudaFree(d_I0));
    checkCudaErrors(cudaFree(d_I));
    checkCudaErrors(cudaFree(d_I0_tmp));

    // result variables -------------
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(phi));
    checkCudaErrors(cudaFree(Delta_phi));
    checkCudaErrors(cudaFree(I_warp));

    // update variables -------------
    checkCudaErrors(cudaFree(zeta_x_M));
    checkCudaErrors(cudaFree(zeta_y_M));
    checkCudaErrors(cudaFree(zeta_z_M));
    checkCudaErrors(cudaFree(zeta_x_N));
    checkCudaErrors(cudaFree(zeta_y_N));
    checkCudaErrors(cudaFree(zeta_z_N));

    // temp variables -------------
    checkCudaErrors(cudaFree(grad_x_M));
    checkCudaErrors(cudaFree(grad_y_M));
    checkCudaErrors(cudaFree(lap_M));
    checkCudaErrors(cudaFree(grad_x_N));
    checkCudaErrors(cudaFree(grad_y_N));
    checkCudaErrors(cudaFree(lap_N));
    checkCudaErrors(cudaFree(tmp_dct_M));
    checkCudaErrors(cudaFree(tmp_dct_N));
    checkCudaErrors(cudaFree(A11));
    checkCudaErrors(cudaFree(A12));
    checkCudaErrors(cudaFree(A22));
    checkCudaErrors(cudaFree(a));
    checkCudaErrors(cudaFree(b));
    checkCudaErrors(cudaFree(c));

    // FFT plans -------------
    cufftDestroy(plan_dct_1_M);
    cufftDestroy(plan_dct_2_M);
    cufftDestroy(plan_dct_1_N);
    cufftDestroy(plan_dct_2_N);

    // DCT coefficients -------------
    checkCudaErrors(cudaFree(ww_1_M));
    checkCudaErrors(cudaFree(ww_2_M));
    checkCudaErrors(cudaFree(ww_1_N));
    checkCudaErrors(cudaFree(ww_2_N));

    // inversion matrices -------------
    checkCudaErrors(cudaFree(mat_A_hat));
    checkCudaErrors(cudaFree(mat_phi_hat));
}

void CWS_A_Phi::setParas(opt_algo opt_algo_out)
{
    opt_algo_t = opt_algo_out;
}

// =============================================================
// main algorithm
// =============================================================
void CWS_A_Phi::solver(cv::Mat I)
{
    // CWS_A_PHI Simutanous intensity and wavefront recovery. Solve for:
    // min            || i(x+\nabla phi) - A i_0(x) ||_2^2            + ...
    //A,phi   alpha   || \nabla phi ||_1                              + ...
    //        beta  ( || \nabla phi ||_2^2 + || \nabla^2 phi ||_2^2 ) + ...
    //        gamma ( || \nabla A ||_1     + || \nabla^2 A ||_1 )     + ...
    //        tau   ( || \nabla A ||_2^2   + || \nabla^2 A ||_2^2 )

    // copy from host to device
    checkCudaErrors(cudaMemcpy(d_I, I.ptr<real>(0), M.area()*sizeof(real), cudaMemcpyHostToDevice));

    // record variables
    real objval, mean_Delta_phi;

    // initialization
    setasones(A, M);
    checkCudaErrors(cudaMemset(phi,       0, N.area()*sizeof(real)));
    checkCudaErrors(cudaMemset(Delta_phi, 0, N.area()*sizeof(real)));
    checkCudaErrors(cudaMemcpy(I_warp,  d_I, M.area()*sizeof(real), cudaMemcpyDeviceToDevice));

    // initial objective
    if (opt_algo_t.isverbose)
    {
        objval = obj_total(A, phi, I_warp, d_I0, opt_algo_t.alpha, opt_algo_t.beta, opt_algo_t.gamma, opt_algo_t.tau, M, N,
                            grad_x_M, grad_y_M, lap_M, grad_x_N, grad_y_N, lap_N);
        printf("iter = %d, obj = %.4e\n", 0, objval);
    }

    // the loop
    real obj_min = 0x7f800000; // = Inf in float
    for (signed int outer_loop = 1; outer_loop <= opt_algo_t.iter; ++outer_loop)
    {
        // === A-update ===
        A_update(A, I_warp, d_I0,
            zeta_x_M, zeta_y_M, zeta_z_M,
            grad_x_M, grad_y_M, lap_M,
            tmp_dct_M, mat_A_hat, ww_1_M, ww_2_M, plan_dct_1_M, plan_dct_2_M, opt_A_t, M);

        // median filtering A
        medfilt2(A, M.width, M.height, A);

        // update d_I0_tmp
        I_warp_update(A, d_I0, M, d_I0_tmp);

        // === phi-update ===
        phi_update(Delta_phi, I_warp, d_I0_tmp, opt_algo_t.alpha, opt_algo_t.beta, zeta_x_N, zeta_y_N, grad_x_N, grad_y_N,
            A11, A12, A22, a, b, c, tmp_dct_N, mat_phi_hat,
            ww_1_N, ww_2_N, plan_dct_1_N, plan_dct_2_N, opt_phi_t, M, N);

        // medfilt2(Delta_phi, N.width, N.height, Delta_phi);

        // === accmulate phi ===
        mean_Delta_phi = mean<1>(Delta_phi, N.area());
        if (mean_Delta_phi < opt_algo_t.phi_tol)
        {
            printf("-- mean(|Delta phi|) = %.4e < %.4e = tol; Quit. \n", mean_Delta_phi, opt_algo_t.phi_tol);
            break;
        }
        else 
        {
            printf("-- mean(|Delta phi|) = %.4e \n", mean_Delta_phi);
            Add(phi, Delta_phi, N.area(), phi);
        }

        // === records ===
        if (opt_algo_t.isverbose)
        {
            objval = obj_total(A, phi, I_warp, d_I0, opt_algo_t.alpha, opt_algo_t.beta, opt_algo_t.gamma, opt_algo_t.tau, M, N,
                                grad_x_M, grad_y_M, lap_M, grad_x_N, grad_y_N, lap_N);
            printf("iter = %d, obj = %.4e\n", outer_loop, objval);
            if (objval > obj_min)
            {
                printf("Obj increasing; Quit. \n");
                break;
            }
            else
                obj_min = objval;
        }
    }

    // median filtering phi
    medfilt2(phi, N.width, N.height, phi);
}


// download results from GPU to host, and crop phi
void CWS_A_Phi::download(cv::Mat &h_A, cv::Mat &h_phi)
{
    // dummy variable
    cv::Mat phi_tmp = cv::Mat::zeros(N, CV_32F);

    // transfer result from device to host
    checkCudaErrors(cudaMemcpy(h_A.ptr<real>(0), A, M.area()*sizeof(real), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(phi_tmp.ptr<real>(0), phi, N.area()*sizeof(real), cudaMemcpyDeviceToHost));

    // crop phi to preserve only the center part
    h_phi = phi_tmp(cv::Rect(opt_algo_t.L.width, opt_algo_t.L.height, M.width, M.height)).clone();
}

// wrapper function for cws
void cws_A_phi(cv::Mat I0, cv::Mat I, cv::Mat &h_A, cv::Mat &h_phi, opt_algo para_algo)
{
    // define cws object
    CWS_A_Phi cws_obj(I0, h_A, h_phi, para_algo);

    // record variables
    real fps;
    GpuTimer timer;

    // timing started
    timer.Start();

    // the loop
    int rep_times = 1;
    for (signed int i = 0; i < rep_times; ++i)
    {
        // run cws solver
        cws_obj.solver(I);
        cws_obj.download(h_A, h_phi);

        // timing ended
        timer.Stop();
        fps = 1000/timer.Elapsed()*rep_times;
        printf("Mean elapsed time: %.4f ms. Mean frame rate: %.4f fps. \n", timer.Elapsed()/rep_times, fps);
    }
}
