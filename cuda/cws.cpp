#include <opencv2/highgui.hpp>
#include "argtable3/argtable3.h"
#include "common.h"
#include "cws_A_phi.h"
#include "IO_helper_functions.h"

// default: disable demo
bool isdemo = false;

bool read_rep_images(const char *image_name, cv::Mat &img, int *size = NULL, double max = 255.0, double min = 0.0)
{
	cv::Mat temp = cv::imread(image_name, CV_LOAD_IMAGE_GRAYSCALE);
	if (temp.empty())
	{
		printf(image_name, "%s is not found; Program exits.\n");
		return true;
	}

	// set region of interest
	if (size != NULL)
	{
		int L_w = (temp.cols - size[0]) / 2;
		int L_h = (temp.rows - size[1]) / 2;
		temp = temp(cv::Rect(L_w, L_h, size[0], size[1]));
	}
	temp.convertTo(img, CV_32F);

	// convert to approximately [0 255]
	img.convertTo(img, CV_32F, 255.0f / (max-min));

	return false;
}


bool is_size_invalid(int s, int s_ref)
{
	if (0 < s && s <= s_ref && (s % 2 == 0)) return false;
	else return true;
}


int cws(double *priors, int *iter, double *mu, double phi_tol, int v,
		int *out_size, int nsize, int *L, int Lsize, const char *outfile, const char **infiles, int ninfiles)
{
	// remove CUDA timing latency
	cudaFree(0);
	cudaSetDevice(0);

	// images
	cv::Mat img_ref, img_cap;
	char image_name[80];
	
	// read images
	if (read_rep_images(infiles[0], img_ref, out_size)) return -1;
	if (read_rep_images(infiles[1], img_cap, out_size)) return -1;

	// output size check
	if (is_size_invalid(out_size[0], img_ref.cols) || is_size_invalid(out_size[1], img_ref.rows))
	{
		out_size[0] = img_ref.cols;
		out_size[1] = img_ref.rows;
		printf("out_size wrong or unspecified; Set as input image size = [%d, %d].\n\n", out_size[0], out_size[1]);
	}

	// define contatiner variables
	cv::Mat A, phi;
	opt_algo para_algo;

	// tradeoff parameters
    para_algo.alpha = priors[0];
    para_algo.beta  = priors[1];
    para_algo.gamma = priors[2];
    para_algo.tau   = priors[3];

	// if verbose for sub-update energy report
	para_algo.isverbose = v > 0 ? true : false;

	// alternating iterations
	para_algo.iter     = iter[0];
	para_algo.A_iter   = iter[1];
	para_algo.phi_iter = iter[2];

	// ADMM parameters
	para_algo.mu_A   = mu[0];
	para_algo.mu_phi = mu[1];

    // tolerance for incremental phase
    para_algo.phi_tol = phi_tol;

	// set L
	para_algo.L.width  = L[0];
	para_algo.L.height = L[1];

	// run the solver
	cws_A_phi(img_ref, img_cap, A, phi, para_algo);

	// save result
	WriteFloFile(outfile, img_ref.cols, img_ref.rows, A.ptr<float>(0), phi.ptr<float>(0));

	// return
	cudaDeviceReset();
	return 0;
}


void print_solver_info(char *progname)
{
	printf("%s version 1.0 (Nov 2018) \n", progname);
	printf("Simultaneous intensity and wavefront recovery GPU solver for the coded wavefront sensor. Solve for:\n\n");
    printf(" min            || i(x+\\nabla phi) - A i_0(x) ||_2^2            +\n");
    printf("A,phi   alpha   || \\nabla phi ||_1                              +\n");
    printf("        beta  ( || \\nabla phi ||_2^2 + || \\nabla^2 phi ||_2^2 ) +\n");
    printf("        gamma ( || \\nabla A ||_1     + || \\nabla^2 A ||_1 )     +\n");
    printf("        tau   ( || \\nabla A ||_2^2   + || \\nabla^2 A ||_2^2 ).\n\n");
	printf("Inputs : i_0 (reference), i (measure).\n");
	printf("Outputs: A (intensity), phi (phase).\n");
	printf("\n");
	printf("by Congli Wang, VCC Imaging @ KAUST.\n");
}


int main(int argc, char **argv)
{
	// help & version info
	struct arg_lit *help     = arg_litn(NULL, "help", 0, 1, "display this help and exit");
	struct arg_lit *version  = arg_litn(NULL, "version", 0, 1, "display version info and exit");

	// model: tradeoff parameters
	struct arg_dbl *priors   = arg_dbln("p", "priors", "<double>", 0, 4, "prior weights {alpha,beta,gamma,beta} (default {0.1,0.1,100,5})");

	// algorithm: ADMM parameters
	struct arg_int *iter     = arg_intn("i", "iter", "<int>", 0, 3, "iteartions {total alternating iter, A-update iter, phi-update iter} (default {3,20,20})");
	struct arg_dbl *mu       = arg_dbln("m", "mu", "<double>", 0, 2, "ADMM parameters {mu_A,mu_phi} (default {0.1,100})");
	struct arg_dbl *phi_tol  = arg_dbl0("t", "tol_phi","<double>", "phi-update tolerance stopping criteria (default 0.05)");
	struct arg_lit *verbose  = arg_litn("v", "verbose", 0, 1, "verbose output (default 0)");
	struct arg_int *out_size = arg_intn("s", "size", "<int>", 0, 2, "output size {width,height} (default input size)");
	struct arg_int *L        = arg_intn("l", "L", "<int>", 0, 2, "padding size {pad_width,pad_height} (default nearest power of 2 of out_size, each in range [2, 32])");

	// input & output files
	struct arg_file *out  = arg_filen("o", "output", "<.flo>", 0, 1, "save output file (intensity A & wavefront phi) as *.flo file (default \"./out.flo\")");
    struct arg_file *file = arg_filen("f", "files",  "<files>", 0, 2, "input file names (reference & measurement)");
	struct arg_end  *end  = arg_end(20);

	/* the global arg_xxx structs are initialised within the argtable */
    void *argtable[] = {help, version, priors, iter, mu, phi_tol, verbose, out_size, L, out, file, end};

	int exitcode = 0;
	char progname[] = "cws";

	/* set any command line default values prior to parsing */
	priors ->dval[0] = 0.1;		// alpha
	priors ->dval[1] = 0.1;		// beta
	priors ->dval[2] = 100.0;	// gamma
	priors ->dval[3] = 5.0;		// tau
	iter   ->ival[0] = 3;		// total alternating iterations
	iter   ->ival[1] = 20;		// ADMM iterations for A-update
	iter   ->ival[2] = 20;		// ADMM iterations for phi-update
	mu     ->dval[0] = 0.1;		// ADMM parameter mu for A-update
	mu     ->dval[1] = 100.0;	// ADMM parameter mu for phi-update
	phi_tol->dval[0] = 0.05;	// tolerance stopping criteria for phi-update
	verbose->count   = 0;
	L      ->ival[0] = 1;		// L pad size for width
	L      ->ival[1] = 1;		// L pad size for height
	out->filename[0] = "./out.flo";

	// parse arguments
	int nerrors;
	nerrors = arg_parse(argc,argv,argtable);

	/* special case: '--help' takes precedence over error reporting */
	if (help->count > 0)
	{
		print_solver_info(progname);
		printf("\n");
		printf("Usage: %s", progname);
		arg_print_syntax(stdout, argtable, "\n");
		arg_print_glossary(stdout, argtable, "  %-25s %s  \n");
		exitcode = 0;
		goto exit;
	}

	/* special case: '--version' takes precedence error reporting */
	if (version->count > 0)
	{
		print_solver_info(progname);
		exitcode = 0;
		goto exit;
	}

	/* If the parser returned any errors then display them and exit */
	if (nerrors > 0)
	{
		/* Display the error details contained in the arg_end struct.*/
		arg_print_errors(stdout, end, progname);
		#if defined(_WIN32) || defined(_WIN64)
			printf("Try '%s --help' for more information.\n", progname);
		#elif defined(__unix)
			printf("Try './%s --help' for more information.\n", progname);
		#endif
		exitcode = 1;
		goto exit;
	}

	/* if no input files specified, use the test data */
	if (file->count < 2)
	{
		printf("Number of input files < 2; use test data instead; Set out_size as [992 992]\n\n");
		out_size->ival[0] = 992;
		out_size->ival[1] = 992;
		file->filename[0] = "../tests/solver_accuracy/data/img_reference.png";
		file->filename[1] = "../tests/solver_accuracy/data/img_capture.png";
		isdemo = true;
		exitcode = 1;
	}

	/* Command line parsing is complete, do the main processing */
	exitcode = cws(priors->dval, iter->ival, mu->dval, phi_tol->dval[0], verbose->count, 
				 out_size->ival, out_size->count, L->ival, L->count, out->filename[0], file->filename, file->count);

	exit:
	/* deallocate each non-null entry in argtable[] */
	arg_freetable(argtable, sizeof(argtable) / sizeof(argtable[0]));

	return exitcode;
}