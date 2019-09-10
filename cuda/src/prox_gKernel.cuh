#include "common.h"

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//////////////////////////////   L1+L2 Flow    ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__
__device__
float sign(T val)
{
    return val > T(0) ? 1.0f : -1.0f; // 0 is seldom
}
// proximal operator for L1 operator
__host__
__device__
float prox_l1(float u, float tau)
{
    return sign<float>(u) * max(fabsf(u) - tau, 0.0f);
}

// LASSO in R2 (simple)
__device__
void lassoR2_simple(float A11, float A12, float A22, 
			 float a, float b, float c, 
			 float ux, float uy, float mu, float alpha,
			 float& ours_x, float& ours_y)
{
	// 1. get (x0, y0)
	// -- temp
	float tx = mu*ux - a*c;
	float ty = mu*uy - b*c;

	// -- l2 minimum
	float x0_x = A11*tx + A12*ty;
	float x0_y = A12*tx + A22*ty;

	// 2. get the sign of optimal
	float sign_x = sign<float>(x0_x);
	float sign_y = sign<float>(x0_y);

	// 3. get the optimal
	ours_x = x0_x - alpha/2.0f * (A11 * sign_x + A12 * sign_y);
	ours_y = x0_y - alpha/2.0f * (A12 * sign_x + A22 * sign_y);

	// 4. check sign and map to the range
	ours_x = sign(ours_x) == sign_x ? ours_x : 0.0f;
	ours_y = sign(ours_y) == sign_y ? ours_y : 0.0f;
}

// LASSO in R2 (complete)
__device__
void lassoR2_complete(float A11, float A12, float A22, 
			 float a, float b, float c, 
			 float ux, float uy, float mu, float alpha,
			 float& ours_x, float& ours_y)
{
	// temp
	float tx = mu*ux - a*c;
	float ty = mu*uy - b*c;

	// l2 minimum
	float x0_x = A11*tx + A12*ty;
	float x0_y = A12*tx + A22*ty;

	// x1 (R^n) [x1 0] & x2 (R^n) [0 x2]
	float x1 = (tx > 0) ? ((tx - alpha/2.0f) / (a*a + mu)) : ((tx + alpha/2.0f) / (a*a + mu));
	float x2 = (ty > 0) ? ((ty - alpha/2.0f) / (b*b + mu)) : ((ty + alpha/2.0f) / (b*b + mu));

	// x3 (R^2n)
	tx = alpha/2.0f * (A11 + A12);
	ty = alpha/2.0f * (A12 + A22);
	bool tb = (tx*x0_x + ty*x0_y) > 0;
	float x3_x = tb ? (x0_x - tx) : (x0_x + tx);
	float x3_y = tb ? (x0_y - ty) : (x0_y + ty);
	
	// x4 (R^2n)
	tx = alpha/2.0f * (A11 - A12);
	ty = alpha/2.0f * (A12 - A22);
	tb = (tx*x0_x + ty*x0_y) > 0;
	float x4_x = tb ? (x0_x - tx) : (x0_x + tx);
	float x4_y = tb ? (x0_y - ty) : (x0_y + ty);

	// cost functions
	float cost[4];
	
	// temp
	float uxx = ux*ux;
	float uyy = uy*uy;

	// cost function of x1
	float t1 = a*x1 + c;
	tx = x1 - ux;
	cost[0] = t1*t1 + mu*(tx*tx + uyy) + alpha*fabsf(x1);
	
	// cost function of x2
	t1 = b*x2 + c;
	ty = x2 - uy;
	cost[1] = t1*t1 + mu*(uxx + ty*ty) + alpha*fabsf(x2);
	
	// cost function of x3
	t1 = a*x3_x + b*x3_y + c;
	tx = x3_x - ux;
	ty = x3_y - uy;
	cost[2] = t1*t1 + mu*(tx*tx + ty*ty) + alpha*(fabsf(x3_x) + fabsf(x3_y));
	
	// cost function of x4
	t1 = a*x4_x + b*x4_y + c;
	tx = x4_x - ux;
	ty = x4_y - uy;
	cost[3] = t1*t1 + mu*(tx*tx + ty*ty) + alpha*(fabsf(x4_x) + fabsf(x4_y));
	
	// cost function of x5
	float cost_min = c*c + mu*(uxx + uyy); // [0 0] solution

	// find minimum
	signed int I = 5;
	for (signed int i = 0; i < 4; ++i)
	{
		if (cost[i] < cost_min)
		{
			cost_min = cost[i];
			I = i + 1;
		}
	}

	// set final solution
	switch (I)
	{
		case 1: 
			ours_x = x1;
			ours_y = 0.0f;
			break;
		case 2:
			ours_x = 0.0f;
			ours_y = x2;
			break;
		case 3:
			ours_x = x3_x;
			ours_y = x3_y;
			break;
		case 4:
			ours_x = x4_x;
			ours_y = x4_y;
			break;
		case 5:
			ours_x = 0.0f;
			ours_y = 0.0f;
			break;
	}
}


__global__ void prox_gL1Kernel(float *zeta_x, float *zeta_y,
	float *temp_x, float *temp_y, float mu, float alpha,
	float *A11, float *A12, float *A22, float *a, float *b, float *c,
	int N_width, int N_height, int M_width, int M_height)
{
	const int L_width = (N_width - M_width) / 2;
	const int L_height = (N_height - M_height) / 2;

	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	const int pos_N = ix + iy * N_width;
	const int pos_M = ix - L_width + (iy - L_height) * M_width;

	float temp_w_x = temp_x[pos_N] + zeta_x[pos_N];
	float temp_w_y = temp_y[pos_N] + zeta_y[pos_N];
	float val_x, val_y;

	// w-update
	if (ix >= N_width || iy >= N_height) return;
	else if (ix >= L_width  && ix < N_width  - L_width &&
		     iy >= L_height && iy < N_height - L_height)
	{ // update interior flow
		// lassoR2_simple(A11[pos_M], A12[pos_M], A22[pos_M], 
		// 	    		   a[pos_M], b[pos_M], c[pos_M], 
		// 	    		temp_w_x, temp_w_y, mu, alpha, val_x, val_y);
		lassoR2_complete(A11[pos_M], A12[pos_M], A22[pos_M], 
			    		   a[pos_M], b[pos_M], c[pos_M], 
			    		temp_w_x, temp_w_y, mu, alpha, val_x, val_y);
	}
	else { // keep exterior flow unchanged
		val_x = temp_w_x;
		val_y = temp_w_y;
	}

	// zeta-update
	zeta_x[pos_N] = temp_w_x - val_x;
	zeta_y[pos_N] = temp_w_y - val_y;

	// pre-store value of (w - zeta)
	temp_x[pos_N] = mu * (2 * val_x - temp_w_x);
	temp_y[pos_N] = mu * (2 * val_y - temp_w_y);
}

static
void prox_gL1(float *zeta_x, float *zeta_y,
	float *temp_x, float *temp_y, float mu, float alpha,
	float *A11, float *A12, float *A22, 
	float *a, float *b, float *c,
	int N_width, int N_height, int M_width, int M_height)
{
	dim3 threads(32, 32);
	dim3 blocks(iDivUp(N_width, threads.x), iDivUp(N_height, threads.y));

	prox_gL1Kernel<<<blocks, threads>>>(zeta_x, zeta_y, temp_x, temp_y,
		mu, alpha, A11, A12, A22, a, b, c,
		N_width, N_height, M_width, M_height);
}


///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//////////////////////////////     L2 Flow     ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// \brief compute proximal operator of g
///
/// \param[in/out]  w_x       flow along x
/// \param[in/out]  w_y       flow along y
/// \param[in/out]  zeta_x    dual variable zeta along x
/// \param[in/out]  zeta_y    dual variable zeta along y
/// \param[in/out]  temp_x    temp variable (either for nabla(x) or w-zeta) along x
/// \param[in/out]  temp_y    temp variable (either for nabla(x) or w-zeta) along y
/// \param[in]      mu        proximal parameter
/// \param[in]      w11, w12_or_w22, w13, w21, w23   
///                           pre-computed weights
/// \param[in]      N_width   unknown width
/// \param[in]      N_height  unknown height
/// \param[in]      M_width   image width
/// \param[in]      M_height  image height
///////////////////////////////////////////////////////////////////////////////
__global__ void prox_gKernel(float *w_x, float *w_y, float *zeta_x, float *zeta_y, 
                             float *temp_x, float *temp_y, float mu,
                             const float *w11, const float *w12_or_w22, 
                             const float *w13, const float *w21, const float *w23, 
                             int N_width, int N_height, 
                             int M_width, int M_height)
{
    const int L_width  = (N_width  - M_width)/2;
    const int L_height = (N_height - M_height)/2;
    
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos_N = ix + iy * N_width;
    const int pos_M  = (ix-L_width) + (iy-L_height) * M_width;
    
    float temp_w_x = temp_x[pos_N] + zeta_x[pos_N];
    float temp_w_y = temp_y[pos_N] + zeta_y[pos_N];
    float val_x, val_y;
    
    // w-update
    if (ix >= N_width || iy >= N_height) return;
    else if (ix >= L_width  && ix < N_width -L_width && 
             iy >= L_height && iy < N_height-L_height){ // update interior flow
        val_x = w11[pos_M] * temp_w_x + w12_or_w22[pos_M] * temp_w_y + w13[pos_M];
        val_y = w21[pos_M] * temp_w_y + w12_or_w22[pos_M] * temp_w_x + w23[pos_M];
    }
    else { // keep exterior flow unchanged
        val_x = temp_w_x;
        val_y = temp_w_y;
    }
    w_x[pos_N] = val_x;
    w_y[pos_N] = val_y;
        
    // zeta-update
    zeta_x[pos_N] = temp_w_x - val_x;
    zeta_y[pos_N] = temp_w_y - val_y;
    
    // pre-store value of (w - zeta)
    temp_x[pos_N] = mu * (2*val_x - temp_w_x);
    temp_y[pos_N] = mu * (2*val_y - temp_w_y);
}



///////////////////////////////////////////////////////////////////////////////
/// \brief compute proximal operator of g
///
/// \param[in/out]  w_x       flow along x
/// \param[in/out]  w_y       flow along y
/// \param[in/out]  zeta_x    dual variable zeta along x
/// \param[in/out]  zeta_y    dual variable zeta along y
/// \param[in/out]  temp_x    temp variable (either for nabla(x) or w-zeta) along x
/// \param[in/out]  temp_y    temp variable (either for nabla(x) or w-zeta) along y
/// \param[in]      mu        proximal parameter
/// \param[in]      w11, w12_or_w22, w13, w21, w23   
///                           pre-computed weights
/// \param[in]      N_width   unknown width
/// \param[in]      N_height  unknown height
/// \param[in]      M_width   image width
/// \param[in]      M_height  image height
///////////////////////////////////////////////////////////////////////////////
static
void prox_g(float *w_x, float *w_y, float *zeta_x, float *zeta_y, 
            float *temp_x, float *temp_y, float mu,
            const float *w11, const float *w12_or_w22, 
            const float *w13, const float *w21, const float *w23, 
            int N_width, int N_height,  int M_width, int M_height)
{
    dim3 threads(32, 6);
    dim3 blocks(iDivUp(N_width, threads.x), iDivUp(N_height, threads.y));

    prox_gKernel<<<blocks, threads>>>(w_x, w_y, zeta_x, zeta_y, temp_x, temp_y,
                                       mu, w11, w12_or_w22, w13, w21, w23, 
                                      N_width, N_height, M_width, M_height);
}
