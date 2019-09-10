#ifndef CWS_A_PHI_H
#define CWS_A_PHI_H

#include <cufft.h>

// accuracy
#ifdef USE_DOUBLES
typedef double real;
typedef cufftDoubleComplex complex;
#else
typedef float real;
typedef cufftComplex complex;
#endif

// default parameters
struct opt_algo {
    bool  isverbose_sub = false;
    bool  isverbose = false;
    int   iter     = 3;
    int   A_iter   = 20;
    int   phi_iter = 20;
    float alpha = 0.1f;
    float beta  = 0.1f;
    float gamma = 100.0f;
	float tau   = 5.0f;
	float phi_tol = 0.05f;
    float mu_A   = 0.1f;
    float mu_phi = 100.0f;
    cv::Size L = cv::Size(1,1); 
    // (1,1) for initializers; if unspecified, will be updated in CWS_A_Phi::CWS_A_Phi as L = cv::Size(2,2)
};


// =============================================================
// algorithm parameter structures
// =============================================================
struct opt_A {
    bool isverbose = false;
    int  iter = 10;
    real mu = 0.1;
    real gamma_new;
    real tau_new;
};

struct opt_phi {
    bool isverbose = false;
    int  iter = 10;
    real mu = 100.0;
    cv::Size L = cv::Size(2,2);
};
// =============================================================


class CWS_A_Phi
{
    // algorithm parameters
    opt_A opt_A_t;
    opt_phi opt_phi_t;

    // device pointers
    real *d_I0, *d_I, *d_I0_tmp;

    // update variables -------------
    real *zeta_x_M, *zeta_y_M, *zeta_z_M;
    real *zeta_x_N, *zeta_y_N, *zeta_z_N;

    // temp variables -------------
    real *grad_x_M, *grad_y_M, *lap_M;
    real *grad_x_N, *grad_y_N, *lap_N;
    complex *tmp_dct_M, *tmp_dct_N;
    real *A11, *A12, *A22, *a, *b, *c;

    // FFT plans -------------
    cufftHandle plan_dct_1_M, plan_dct_2_M;
    cufftHandle plan_dct_1_N, plan_dct_2_N;

    // DCT coefficients -------------
    complex *ww_1_M, *ww_2_M, *ww_1_N, *ww_2_N;

    // inversion matrices -------------
    real *mat_A_hat, *mat_phi_hat;

public:
	// dimensions
	cv::Size M, N;

    // algorithm parameters
    opt_algo opt_algo_t;

	// result variables -------------
	real *A, *phi, *Delta_phi, *I_warp;

    CWS_A_Phi(cv::Mat I0, cv::Mat &h_A, cv::Mat &h_phi, opt_algo opt_algo_out);
    ~CWS_A_Phi();
    void setParas(opt_algo opt_algo_out);
    void solver(cv::Mat I);
    void download(cv::Mat &h_A, cv::Mat &h_phi);
};

void cws_A_phi(cv::Mat I0, cv::Mat I, cv::Mat &A, cv::Mat &phi, opt_algo para_algo);

#endif
