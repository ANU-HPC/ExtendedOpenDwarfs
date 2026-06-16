#include <hip/hip_runtime.h>

#include <ctype.h>
#include <getopt.h>
#include <math.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <include/lsb.h>

#define AOCL_ALIGNMENT 64
#define MIN_TIME_SEC 2

#define T 1000
#define S 2
#define N 60
#define ITERATIONS 1

#define MAX_THREADS_PER_BLOCK 256
#define BLOCK_DIM 16
#define EXIT_ERROR 1

#define SDOT_BLOCK_SIZE 128
#define SDOT_BLOCK_NUM 80

#define MVMUL_BLOCK_SIZE 128
#define MVMUL_BLOCK_NUM 64

static int nstates;
static int nsymbols;
static int* obs;
static int length;
static float* scale_factors;

static float* a_d;
static float* b_d;
static float* pi_d;
static float* alpha_d;
static float* beta_d;
static float* gamma_sum_d;
static float* xi_sum_d;
static float* c_d;
static float* ones_n_d;
static float* ones_s_d;

extern "C" __global__ void init_ones_dev(float* ones_s_d, int nsymbols);
extern "C" __global__ void init_alpha_dev(float* b_d, float* pi_d, int nstates, float* alpha_d, float* ones_n_d, int obs_t);
extern "C" __global__ void calc_alpha_dev(int nstates, float* alpha_d, int offset, float* b_d, int obs_t);
extern "C" __global__ void scale_alpha_dev(int nstates, float* alpha_d, int offset, float scale);
extern "C" __global__ void init_beta_dev(int nstates, float* beta_d, int offset, float scale);
extern "C" __global__ void calc_beta_dev(float* beta_d, float* b_d, float scale_t, int nstates, int obs_t, int t);
extern "C" __global__ void calc_gamma_dev(float* gamma_sum_d, float* alpha_d, float* beta_d, int nstates, int t);
extern "C" __global__ void calc_xi_dev(float* xi_sum_d, float* a_d, float* b_d, float* alpha_d, float* beta_d, float sum_ab, int nstates, int obs_t, int t);
extern "C" __global__ void est_a_dev(float* a_d, float* alpha_d, float* beta_d, float* xi_sum_d, float* gamma_sum_d, float sum_ab, int nstates, int length);
extern "C" __global__ void scale_a_dev(float* a_d, float* c_d, int nstates);
extern "C" __global__ void acc_b_dev(float* b_d, float* alpha_d, float* beta_d, float sum_ab, int nstates, int nsymbols, int obs_t, int t);
extern "C" __global__ void est_b_dev(float* b_d, float* gamma_sum_d, int nstates, int nsymbols);
extern "C" __global__ void scale_b_dev(float* b_d, float* c_d, int nstates, int nsymbols);
extern "C" __global__ void est_pi_dev(float* pi_d, float* alpha_d, float* beta_d, float sum_ab, int nstates);
extern "C" __global__ void s_dot_kernel_naive(int n, float* paramA, int offsetA, float* paramB, int offsetB, float* partialSum_d);
extern "C" __global__ void mvm_non_kernel_naive(int m, int n, float* A, int lda, float* x, int offsetX, float* y, int offsetY);
extern "C" __global__ void mvm_trans_kernel_naive(int m, int n, float* A, int lda, float* x, int offsetX, float* y, int offsetY);

typedef struct {
	int nstates;
	int nsymbols;
	float* a;
	float* b;
	float* pi;
} Hmm;

typedef struct {
	int length;
	int* data;
} Obs;

static const char* get_lsb_name()
{
	const char* lsb_name = getenv("ODW_LSB_NAME");
	return (lsb_name == NULL || lsb_name[0] == '\0') ? "bwa_hmm" : lsb_name;
}

static void record_region_start(const char* region)
{
	LSB_Set_Rparam_string("region", region);
	LSB_Res();
}

static void record_region_end(int id)
{
	LSB_Rec(id);
}

static void hip_check(hipError_t err, const char* msg)
{
	if (err != hipSuccess) {
		fprintf(stderr, "HIP error: %s: %s\n", msg, hipGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

static void parse_pre_separator_common_args(int argc, char** argv, int* hip_device)
{
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--") == 0) {
			break;
		}

		if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
			*hip_device = atoi(argv[i + 1]);
			i++;
		} else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "-t") == 0) && i + 1 < argc) {
			i++;
		}
	}
}

static void build_app_argv(int argc, char** argv, int* app_argc, char*** app_argv)
{
	int separator = -1;

	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--") == 0) {
			separator = i;
			break;
		}
	}

	*app_argv = (char**) malloc(sizeof(char*) * (argc + 1));
	if (*app_argv == NULL) {
		fprintf(stderr, "Failed to allocate argument list\n");
		exit(EXIT_FAILURE);
	}

	(*app_argv)[0] = argv[0];
	*app_argc = 1;

	if (separator >= 0) {
		for (int i = separator + 1; i < argc; i++) {
			(*app_argv)[(*app_argc)++] = argv[i];
		}
	} else {
		for (int i = 1; i < argc; i++) {
			(*app_argv)[(*app_argc)++] = argv[i];
		}
	}

	(*app_argv)[*app_argc] = NULL;
}

static void free_vars(Hmm* hmm, Obs* obs_obj)
{
	if (hmm != NULL) {
		free(hmm->a);
		free(hmm->b);
		free(hmm->pi);
		free(hmm);
	}

	if (obs_obj != NULL) {
		free(obs_obj->data);
		free(obs_obj);
	}
}

static int timeval_subtract(struct timeval* result, struct timeval* t2, struct timeval* t1)
{
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) -
	                (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;
	return diff < 0;
}

static void tic(struct timeval* timer)
{
	gettimeofday(timer, NULL);
}

static void toc(struct timeval* timer)
{
	struct timeval tv_end;
	struct timeval tv_diff;

	gettimeofday(&tv_end, NULL);
	timeval_subtract(&tv_diff, &tv_end, timer);
	printf("%ld.%06ld\t", tv_diff.tv_sec, tv_diff.tv_usec);
}

static float dot_production(int n, float* paramA, int offsetA, float* paramB, int offsetB, hipStream_t stream)
{
	if (n <= 0) {
		return 0.0f;
	}

	int blocks = (n < SDOT_BLOCK_NUM) ? n : SDOT_BLOCK_NUM;
	int threads = SDOT_BLOCK_SIZE;

	float* partialSum = (float*) memalign(AOCL_ALIGNMENT, n * sizeof(float));
	if (partialSum == NULL) {
		fprintf(stderr, "Host allocation failed for partialSum\n");
		exit(EXIT_FAILURE);
	}
	memset(partialSum, 0, n * sizeof(float));

	record_region_start("device_side_buffer_setup");

	float* partialSum_d = NULL;
	hip_check(hipMalloc((void**) &partialSum_d, n * sizeof(float)), "failed to allocate partialSum_d");

	record_region_end(0);

	record_region_start("device_side_h2d_copy");

	hip_check(hipMemcpyAsync(partialSum_d, partialSum, n * sizeof(float), hipMemcpyHostToDevice, stream), "failed to copy partialSum_d");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize partialSum_d copy");

	record_region_end(0);

	record_region_start("setting_dot_kernel_arguments");
	record_region_end(0);

	record_region_start("dot_kernel_execution");

	hipLaunchKernelGGL(s_dot_kernel_naive, blocks, threads, 0, stream, n, paramA, offsetA, paramB, offsetB, partialSum_d);
	hip_check(hipGetLastError(), "failed to launch s_dot_kernel_naive");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize s_dot_kernel_naive");

	record_region_end(0);

	record_region_start("device_side_d2h_copy");

	hip_check(hipMemcpyAsync(partialSum, partialSum_d, n * sizeof(float), hipMemcpyDeviceToHost, stream), "failed to copy partialSum");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize partialSum read");

	record_region_end(0);

	float result = 0.0f;
	for (int i = 0; i < n; i++) {
		result += partialSum[i];
	}

	hipFree(partialSum_d);
	free(partialSum);

	return result;
}

static void mat_vec_mul(char trans, int m, int n, float* A, int lda, float* x, int offsetX, float* y, int offsetY, hipStream_t stream)
{
	if ((trans != 'n') && (trans != 'N') && (trans != 't') && (trans != 'T')) {
		return;
	}

	size_t globalWorkSize = MVMUL_BLOCK_NUM * MVMUL_BLOCK_SIZE;
	size_t localWorkSize = MVMUL_BLOCK_SIZE;

	record_region_start("setting_matvec_kernel_arguments");
	record_region_end(0);

	record_region_start("matvec_kernel_execution");

	if (trans == 't' || trans == 'T') {
		hipLaunchKernelGGL(mvm_trans_kernel_naive, globalWorkSize / localWorkSize, localWorkSize, 0, stream, 
			m, n, A, lda, x, offsetX, y, offsetY);
		hip_check(hipGetLastError(), "failed to launch mvm_trans_kernel_naive");
	} else {
		hipLaunchKernelGGL(mvm_non_kernel_naive, globalWorkSize / localWorkSize, localWorkSize, 0, stream, 
			m, n, A, lda, x, offsetX, y, offsetY);
		hip_check(hipGetLastError(), "failed to launch mvm_non_kernel_naive");
	}

	hip_check(hipStreamSynchronize(stream), "failed to synchronize matvec kernel");

	record_region_end(0);
}

static float calc_alpha(hipStream_t stream)
{
	int threads_per_block = MAX_THREADS_PER_BLOCK;
	int nblocks = (nstates + threads_per_block - 1) / threads_per_block;
	float log_lik = 0.0f;

	record_region_start("setting_alpha_kernel_arguments");
	record_region_end(0);

	record_region_start("alpha_kernel_execution");

	hipLaunchKernelGGL(init_alpha_dev, nblocks, threads_per_block, 0, stream, b_d, pi_d, nstates, alpha_d, ones_n_d, obs[0]);
	hip_check(hipGetLastError(), "failed to launch init_alpha_dev");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize init_alpha_dev");

	record_region_end(0);

	scale_factors[0] = dot_production(nstates, alpha_d, 0, ones_n_d, 0, stream);

	int tmp = 0;

	record_region_start("setting_alpha_kernel_arguments");
	record_region_end(0);

	record_region_start("alpha_kernel_execution");

	hipLaunchKernelGGL(scale_alpha_dev, nblocks, threads_per_block, 0, stream, nstates, alpha_d, tmp, scale_factors[0]);
	hip_check(hipGetLastError(), "failed to launch scale_alpha_dev");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize scale_alpha_dev");

	record_region_end(0);

	log_lik = log10(scale_factors[0]);

	for (int t = 1; t < length; t++) {
		int offset_prev = (t - 1) * nstates;
		int offset_cur = t * nstates;

		mat_vec_mul('n', nstates, nstates, a_d, nstates, alpha_d, offset_prev, alpha_d, offset_cur, stream);

		record_region_start("setting_alpha_kernel_arguments");
		record_region_end(t);

		record_region_start("alpha_kernel_execution");

		calc_alpha_dev<<<nblocks, threads_per_block, 0, stream>>>(nstates, alpha_d, offset_cur, b_d, obs[t]);
		hip_check(hipGetLastError(), "failed to launch calc_alpha_dev");
		hip_check(hipStreamSynchronize(stream), "failed to synchronize calc_alpha_dev");

		record_region_end(t);

		scale_factors[t] = dot_production(nstates, alpha_d, offset_cur, ones_n_d, 0, stream);

		record_region_start("setting_alpha_kernel_arguments");
		record_region_end(t);

		record_region_start("alpha_kernel_execution");

		hipLaunchKernelGGL(scale_alpha_dev, nblocks, threads_per_block, 0, stream, nstates, alpha_d, offset_cur, scale_factors[t]);
		hip_check(hipGetLastError(), "failed to launch scale_alpha_dev");
		hip_check(hipStreamSynchronize(stream), "failed to synchronize scale_alpha_dev");

		record_region_end(t);

		log_lik += log10(scale_factors[t]);
	}

	return log_lik;
}

static int calc_beta(hipStream_t stream)
{
	int threads_per_block = MAX_THREADS_PER_BLOCK;
	int nblocks = (nstates + threads_per_block - 1) / threads_per_block;
	int offset = (length - 1) * nstates;

	record_region_start("setting_beta_kernel_arguments");
	record_region_end(0);

	record_region_start("beta_kernel_execution");

	hipLaunchKernelGGL(init_beta_dev, nblocks, threads_per_block, 0, stream, nstates, beta_d, offset, scale_factors[length - 1]);
	hip_check(hipGetLastError(), "failed to launch init_beta_dev");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize init_beta_dev");

	record_region_end(0);

	for (int t = length - 2; t >= 0; t--) {
		record_region_start("setting_beta_kernel_arguments");
		record_region_end(t);

		record_region_start("beta_kernel_execution");

		hipLaunchKernelGGL(calc_beta_dev, nblocks, threads_per_block, 0, stream, beta_d, b_d, scale_factors[t], nstates, obs[t + 1], t);
		hip_check(hipGetLastError(), "failed to launch calc_beta_dev");
		hip_check(hipStreamSynchronize(stream), "failed to synchronize calc_beta_dev");

		record_region_end(t);

		mat_vec_mul('n', nstates, nstates, a_d, nstates, beta_d, t * nstates, c_d, 0, stream);

		record_region_start("device_side_h2d_copy");
		hip_check(
			hipMemcpyAsync(
				beta_d + (t * nstates),
				c_d,
				nstates * sizeof(float),
				hipMemcpyDeviceToDevice,
				stream),
			"failed to copy beta scratch vector");
		hip_check(hipStreamSynchronize(stream), "failed to synchronize beta scratch copy");
		record_region_end(t);
	}

	return 0;
}

static void calc_gamma_sum(hipStream_t stream)
{
	int threads_per_block = MAX_THREADS_PER_BLOCK;
	int nblocks = (nstates + threads_per_block - 1) / threads_per_block;

	float* gamma_sum_zeros = (float*) calloc(nstates, sizeof(float));

	record_region_start("device_side_h2d_copy");

	hip_check(hipMemcpyAsync(gamma_sum_d, gamma_sum_zeros, nstates * sizeof(float), hipMemcpyHostToDevice, stream), "failed to clear gamma_sum_d");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize gamma_sum_d clear");

	record_region_end(0);

	free(gamma_sum_zeros);

	for (int t = 0; t < length; t++) {
		record_region_start("setting_gamma_kernel_arguments");
		record_region_end(t);

		record_region_start("gamma_kernel_execution");

		hipLaunchKernelGGL(calc_gamma_dev, nblocks, threads_per_block, 0, stream, gamma_sum_d, alpha_d, beta_d, nstates, t);
		hip_check(hipGetLastError(), "failed to launch calc_gamma_dev");
		hip_check(hipStreamSynchronize(stream), "failed to synchronize calc_gamma_dev");

		record_region_end(t);
	}
}

static int calc_xi_sum(hipStream_t stream)
{
	float* xi_sum_zeros = (float*) calloc(nstates * nstates, sizeof(float));

	record_region_start("device_side_h2d_copy");

	hip_check(hipMemcpyAsync(xi_sum_d, xi_sum_zeros, nstates * nstates * sizeof(float), hipMemcpyHostToDevice, stream), "failed to clear xi_sum_d");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize xi_sum_d clear");

	record_region_end(0);

	free(xi_sum_zeros);

	int nblocks = (nstates + BLOCK_DIM - 1) / BLOCK_DIM;
	dim3 grid(nblocks, nblocks);
	dim3 threads(BLOCK_DIM, BLOCK_DIM);

	for (int t = 0; t < length - 1; t++) {
		float sum_ab = dot_production(nstates, alpha_d, t * nstates, beta_d, t * nstates, stream);

		record_region_start("setting_xi_kernel_arguments");
		record_region_end(t);

		record_region_start("xi_kernel_execution");

		hipLaunchKernelGGL(calc_xi_dev, grid, threads, 0, stream, xi_sum_d, a_d, b_d, alpha_d, beta_d, sum_ab, nstates, obs[t + 1], t);
		hip_check(hipGetLastError(), "failed to launch calc_xi_dev");
		hip_check(hipStreamSynchronize(stream), "failed to synchronize calc_xi_dev");

		record_region_end(t);
	}

	return 0;
}

static int estimate_a(hipStream_t stream)
{
	int nblocks = (nstates + BLOCK_DIM - 1) / BLOCK_DIM;
	dim3 grid(nblocks, nblocks);
	dim3 threads(BLOCK_DIM, BLOCK_DIM);

	float sum_ab = dot_production(nstates, alpha_d, (length - 1) * nstates, beta_d, (length - 1) * nstates, stream);

	record_region_start("setting_estimate_a_kernel_arguments");
	record_region_end(0);

	record_region_start("estimate_a_kernel_execution");

	hipLaunchKernelGGL(est_a_dev, grid, threads, 0, stream, a_d, alpha_d, beta_d, xi_sum_d, gamma_sum_d, sum_ab, nstates, length);
	hip_check(hipGetLastError(), "failed to launch est_a_dev");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize est_a_dev");

	record_region_end(0);

	mat_vec_mul('t', nstates, nstates, a_d, nstates, ones_n_d, 0, c_d, 0, stream);

	record_region_start("setting_estimate_a_kernel_arguments");
	record_region_end(0);

	record_region_start("estimate_a_kernel_execution");

	hipLaunchKernelGGL(scale_a_dev, grid, threads, 0, stream, a_d, c_d, nstates);
	hip_check(hipGetLastError(), "failed to launch scale_a_dev");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize scale_a_dev");

	record_region_end(0);

	return 0;
}

static int estimate_b(hipStream_t stream)
{
	float* b_d_zeros = (float*) calloc(nstates * nsymbols, sizeof(float));

	record_region_start("device_side_h2d_copy");

	hip_check(hipMemcpyAsync(b_d, b_d_zeros, nstates * nsymbols * sizeof(float), hipMemcpyHostToDevice, stream), "failed to clear b_d");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize b_d clear");

	record_region_end(0);

	free(b_d_zeros);

	int grid_x = (nstates + BLOCK_DIM - 1) / BLOCK_DIM;
	int grid_y = (nsymbols + BLOCK_DIM - 1) / BLOCK_DIM;
	dim3 grid(grid_x, grid_y);
	dim3 threads(BLOCK_DIM, BLOCK_DIM);

	for (int t = 0; t < length; t++) {
		float sum_ab = dot_production(nstates, alpha_d, t * nstates, beta_d, t * nstates, stream);

		record_region_start("setting_estimate_b_kernel_arguments");
		record_region_end(t);

		record_region_start("estimate_b_kernel_execution");

		hipLaunchKernelGGL(acc_b_dev, grid, threads, 0, stream, b_d, alpha_d, beta_d, sum_ab, nstates, nsymbols, obs[t], t);
		hip_check(hipGetLastError(), "failed to launch acc_b_dev");
		hip_check(hipStreamSynchronize(stream), "failed to synchronize acc_b_dev");

		record_region_end(t);
	}

	record_region_start("setting_estimate_b_kernel_arguments");
	record_region_end(0);

	record_region_start("estimate_b_kernel_execution");

	hipLaunchKernelGGL(est_b_dev, grid, threads, 0, stream, b_d, gamma_sum_d, nstates, nsymbols);
	hip_check(hipGetLastError(), "failed to launch est_b_dev");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize est_b_dev");

	record_region_end(0);

	mat_vec_mul('N', nstates, nsymbols, b_d, nstates, ones_s_d, 0, c_d, 0, stream);

	record_region_start("setting_estimate_b_kernel_arguments");
	record_region_end(0);

	record_region_start("estimate_b_kernel_execution");

	hipLaunchKernelGGL(scale_b_dev, grid, threads, 0, stream, b_d, c_d, nstates, nsymbols);
	hip_check(hipGetLastError(), "failed to launch scale_b_dev");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize scale_b_dev");

	record_region_end(0);

	return 0;
}

static int estimate_pi(hipStream_t stream)
{
	float sum_ab = dot_production(nstates, alpha_d, 0, beta_d, 0, stream);

	int threads_per_block = MAX_THREADS_PER_BLOCK;
	int nblocks = (nstates + threads_per_block - 1) / threads_per_block;

	record_region_start("setting_estimate_pi_kernel_arguments");
	record_region_end(0);

	record_region_start("estimate_pi_kernel_execution");

	hipLaunchKernelGGL(est_pi_dev, nblocks, threads_per_block, 0, stream, pi_d, alpha_d, beta_d, sum_ab, nstates);
	hip_check(hipGetLastError(), "failed to launch est_pi_dev");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize est_pi_dev");

	record_region_end(0);

	return 0;
}

static float run_hmm_bwa(Hmm* hmm, Obs* in_obs, int iterations, float threshold, hipStream_t stream)
{
	float* a = hmm->a;
	float* b = hmm->b;
	float* pi = hmm->pi;

	nsymbols = hmm->nsymbols;
	nstates = hmm->nstates;
	obs = in_obs->data;
	length = in_obs->length;

	record_region_start("host_input_setup");

	scale_factors = (float*) memalign(AOCL_ALIGNMENT, sizeof(float) * length);
	if (scale_factors == NULL) {
		fprintf(stderr, "Host allocation error: scale\n");
		exit(EXIT_FAILURE);
	}

	record_region_end(0);

	record_region_start("device_side_buffer_setup");

	hip_check(hipMalloc((void**) &a_d, sizeof(float) * nstates * nstates), "failed to allocate a_d");
	hip_check(hipMalloc((void**) &b_d, sizeof(float) * nstates * nsymbols), "failed to allocate b_d");
	hip_check(hipMalloc((void**) &pi_d, sizeof(float) * nstates), "failed to allocate pi_d");
	hip_check(hipMalloc((void**) &alpha_d, sizeof(float) * nstates * length), "failed to allocate alpha_d");
	hip_check(hipMalloc((void**) &beta_d, sizeof(float) * nstates * length), "failed to allocate beta_d");
	hip_check(hipMalloc((void**) &gamma_sum_d, sizeof(float) * nstates), "failed to allocate gamma_sum_d");
	hip_check(hipMalloc((void**) &xi_sum_d, sizeof(float) * nstates * nstates), "failed to allocate xi_sum_d");
	hip_check(hipMalloc((void**) &c_d, sizeof(float) * nstates), "failed to allocate c_d");
	hip_check(hipMalloc((void**) &ones_n_d, sizeof(float) * nstates), "failed to allocate ones_n_d");
	hip_check(hipMalloc((void**) &ones_s_d, sizeof(float) * nsymbols), "failed to allocate ones_s_d");

	record_region_end(0);

	printf("Working kernel memory: %fKiB\n",
	       ((sizeof(float) * nstates * nstates +
	         sizeof(float) * nstates * nsymbols +
	         sizeof(float) * nstates +
	         sizeof(float) * nstates * length +
	         sizeof(float) * nstates * length +
	         sizeof(float) * nstates +
	         sizeof(float) * nstates * nstates +
	         sizeof(float) * nstates +
	         sizeof(float) * nstates +
	         sizeof(float) * nsymbols) /
	        1024.0));

	int lsb_timing_repeats = 0;
	struct timeval startTime;
	struct timeval currentTime;
	struct timeval elapsedTime;

	gettimeofday(&startTime, NULL);

	float new_log_lik = 0.0f;
	float old_log_lik = 0.0f;

	do {
		LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

		record_region_start("device_side_h2d_copy");

		hip_check(hipMemcpyAsync(a_d, a, sizeof(float) * nstates * nstates, hipMemcpyHostToDevice, stream), "failed to copy a_d");
		hip_check(hipMemcpyAsync(b_d, b, sizeof(float) * nstates * nsymbols, hipMemcpyHostToDevice, stream), "failed to copy b_d");
		hip_check(hipMemcpyAsync(pi_d, pi, sizeof(float) * nstates, hipMemcpyHostToDevice, stream), "failed to copy pi_d");
		hip_check(hipStreamSynchronize(stream), "failed to synchronize H2D copies");

		record_region_end(0);

		int threads_per_block = MAX_THREADS_PER_BLOCK;
		int nblocks = (nstates + threads_per_block - 1) / threads_per_block;

		record_region_start("setting_init_kernel_arguments");
		record_region_end(0);

		record_region_start("init_kernel_execution");

		hipLaunchKernelGGL(init_ones_dev, nblocks, threads_per_block, 0, stream, ones_s_d, nsymbols);
		hip_check(hipGetLastError(), "failed to launch init_ones_dev");
		hip_check(hipStreamSynchronize(stream), "failed to synchronize init_ones_dev");

		record_region_end(0);

		for (int iter = 0; iter < iterations; iter++) {
			new_log_lik = calc_alpha(stream);

			if (calc_beta(stream) == EXIT_ERROR) {
				return EXIT_ERROR;
			}

			calc_gamma_sum(stream);

			if (calc_xi_sum(stream) == EXIT_ERROR) {
				return EXIT_ERROR;
			}

			if (estimate_a(stream) == EXIT_ERROR) {
				return EXIT_ERROR;
			}

			if (estimate_b(stream) == EXIT_ERROR) {
				return EXIT_ERROR;
			}

			if (estimate_pi(stream) == EXIT_ERROR) {
				return EXIT_ERROR;
			}

			if (threshold > 0.0f && iter > 0) {
				if (fabs(pow(10.0f, new_log_lik) - pow(10.0f, old_log_lik)) < threshold) {
					break;
				}
			}

			old_log_lik = new_log_lik;
		}

		lsb_timing_repeats++;
		gettimeofday(&currentTime, NULL);
		timersub(&currentTime, &startTime, &elapsedTime);
	} while (elapsedTime.tv_sec < MIN_TIME_SEC);

	free(scale_factors);

	record_region_start("device_side_buffer_cleanup");

	hipFree(a_d);
	hipFree(b_d);
	hipFree(pi_d);
	hipFree(alpha_d);
	hipFree(beta_d);
	hipFree(gamma_sum_d);
	hipFree(xi_sum_d);
	hipFree(c_d);
	hipFree(ones_n_d);
	hipFree(ones_s_d);

	record_region_end(0);

	return new_log_lik;
}

static struct option size_opts[] = {
	{"state number", 1, NULL, 'n'},
	{"symbol number", 1, NULL, 's'},
	{"observation number", 1, NULL, 't'},
	{"varying mode", 1, NULL, 'v'},
	{0, 0, 0, 0}
};

int main(int argc, char* argv[])
{
	int hip_device = 0;
	parse_pre_separator_common_args(argc, argv, &hip_device);

	int app_argc = 0;
	char** app_argv = NULL;
	build_app_argv(argc, argv, &app_argc, &app_argv);

	printf("Starting bwa_hmm\n");

	LSB_Init(get_lsb_name(), 0);
	LSB_Set_Rparam_int("repeats_to_two_seconds", 0);

	record_region_start("runtime_initialization");

	hip_check(hipSetDevice(hip_device), "failed to set HIP device");
	hip_check(hipFree(0), "failed to initialize HIP runtime");

	hipStream_t stream;
	hip_check(hipStreamCreate(&stream), "failed to create HIP stream");

	record_region_end(0);

	record_region_start("program_build");
	record_region_end(0);

	record_region_start("kernel_creation");
	record_region_end(0);

	int s = S;
	int t = T;
	int n = N;
	char v_model = 'n';

	int opt;
	int opt_index = 0;
	optind = 1;

	while ((opt = getopt_long(app_argc, app_argv, "n:s:t:v:", size_opts, &opt_index)) != -1) {
		switch (opt) {
			case 'v':
				v_model = optarg[0];
				break;
			case 'n':
				n = atoi(optarg);
				break;
			case 's':
				s = atoi(optarg);
				break;
			case 't':
				t = atoi(optarg);
				break;
			default:
				fprintf(stderr,
				        "Usage %s [-n state number | -s symbol number | -t observation number] [-v varying model]\n",
				        app_argv[0]);
				exit(EXIT_FAILURE);
		}
	}

	if (v_model == 'n') {
		if (n >= 8000) {
			return 0;
		}

		s = S;
		t = T;
	} else if (v_model == 's') {
		if (s >= 8000) {
			return 0;
		}

		n = N;
		t = T;
	} else if (v_model == 't') {
		if (t >= 8000) {
			return 0;
		}

		n = N;
		s = S;
	} else {
		fprintf(stderr, "Unknown varying mode '%c'; expected n, s, or t\n", v_model);
		return EXIT_FAILURE;
	}

	record_region_start("host_input_setup");

	Hmm* hmm = (Hmm*) memalign(AOCL_ALIGNMENT, sizeof(Hmm));
	Obs* obs_obj = (Obs*) memalign(AOCL_ALIGNMENT, sizeof(Obs));

	if (hmm == NULL || obs_obj == NULL) {
		fprintf(stderr, "Host allocation failed for HMM/Obs\n");
		return EXIT_FAILURE;
	}

	obs_obj->length = t;
	obs_obj->data = (int*) memalign(AOCL_ALIGNMENT, sizeof(int) * t);

	for (int i = 0; i < t; i++) {
		obs_obj->data[i] = 0;
	}

	hmm->nstates = n;
	hmm->nsymbols = s;

	hmm->a = (float*) memalign(AOCL_ALIGNMENT, sizeof(float) * n * n);
	hmm->b = (float*) memalign(AOCL_ALIGNMENT, sizeof(float) * n * s);
	hmm->pi = (float*) memalign(AOCL_ALIGNMENT, sizeof(float) * n);

	if (hmm->a == NULL || hmm->b == NULL || hmm->pi == NULL) {
		fprintf(stderr, "Host allocation failed for HMM matrices\n");
		return EXIT_FAILURE;
	}

	for (int i = 0; i < n * n; i++) {
		hmm->a[i] = 1.0f / (float) n;
	}

	for (int i = 0; i < n * s; i++) {
		hmm->b[i] = 1.0f / (float) s;
	}

	for (int i = 0; i < n; i++) {
		hmm->pi[i] = 1.0f / (float) n;
	}

	record_region_end(0);

	struct timeval timer;
	tic(&timer);

	float log_lik = run_hmm_bwa(hmm, obs_obj, ITERATIONS, 0.0f, stream);

	printf("Observations\tTime (s)\tLog_likelihood\n");
	if (v_model == 'n') {
		printf("%i\t", n);
	} else if (v_model == 's') {
		printf("%i\t", s);
	} else {
		printf("%i\t", t);
	}
	toc(&timer);
	printf("%f\n", log_lik);

	free_vars(hmm, obs_obj);

	record_region_start("runtime_finalization");

	hip_check(hipStreamDestroy(stream), "failed to destroy HIP stream");

	record_region_end(0);

	LSB_Finalize();

	free(app_argv);

	return EXIT_SUCCESS;
}
