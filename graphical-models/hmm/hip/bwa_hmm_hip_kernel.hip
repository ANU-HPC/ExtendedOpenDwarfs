#include <hip/hip_runtime.h>

#define SDOT_BLOCK_SIZE 128
#define SDOT_BLOCK_NUM 80
#define MVMUL_BLOCK_SIZE 128
#define MVMUL_BLOCK_NUM 64

extern "C" __global__
void init_ones_dev(float* ones_s_d, int nsymbols)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (unsigned int) nsymbols) {
		ones_s_d[idx] = 1.0f;
	}
}

extern "C" __global__
void init_alpha_dev(float* b_d, float* pi_d, int nstates, float* alpha_d, float* ones_n_d, int obs_t)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (unsigned int) nstates) {
		alpha_d[idx] = pi_d[idx] * b_d[(obs_t * nstates) + idx];
		ones_n_d[idx] = 1.0f;
	}
}

extern "C" __global__
void calc_alpha_dev(int nstates, float* alpha_d, int offset, float* b_d, int obs_t)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (unsigned int) nstates) {
		alpha_d[offset + idx] = alpha_d[offset + idx] * b_d[(obs_t * nstates) + idx];
	}
}

extern "C" __global__
void scale_alpha_dev(int nstates, float* alpha_d, int offset, float scale)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (unsigned int) nstates) {
		alpha_d[offset + idx] = alpha_d[offset + idx] / scale;
	}
}

extern "C" __global__
void init_beta_dev(int nstates, float* beta_d, int offset, float scale)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (unsigned int) nstates) {
		beta_d[offset + idx] = 1.0f / scale;
	}
}

extern "C" __global__
void calc_beta_dev(float* beta_d, float* b_d, float scale_t, int nstates, int obs_t, int t)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (unsigned int) nstates) {
		beta_d[(t * nstates) + idx] =
			beta_d[((t + 1) * nstates) + idx] *
			b_d[(obs_t * nstates) + idx] /
			scale_t;
	}
}

extern "C" __global__
void calc_gamma_dev(float* gamma_sum_d, float* alpha_d, float* beta_d, int nstates, int t)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (unsigned int) nstates) {
		gamma_sum_d[idx] += alpha_d[(t * nstates) + idx] * beta_d[(t * nstates) + idx];
	}
}

extern "C" __global__
void calc_xi_dev(
	float* xi_sum_d,
	float* a_d,
	float* b_d,
	float* alpha_d,
	float* beta_d,
	float sum_ab,
	int nstates,
	int obs_t,
	int t)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < (unsigned int) nstates && idy < (unsigned int) nstates) {
		xi_sum_d[(idy * nstates) + idx] +=
			alpha_d[(t * nstates) + idy] *
			a_d[(idy * nstates) + idx] *
			b_d[(obs_t * nstates) + idx] *
			beta_d[((t + 1) * nstates) + idx] /
			sum_ab;
	}
}

extern "C" __global__
void est_a_dev(
	float* a_d,
	float* alpha_d,
	float* beta_d,
	float* xi_sum_d,
	float* gamma_sum_d,
	float sum_ab,
	int nstates,
	int length)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < (unsigned int) nstates && idy < (unsigned int) nstates) {
		a_d[(idy * nstates) + idx] =
			xi_sum_d[(idy * nstates) + idx] /
			(gamma_sum_d[idy] -
			 alpha_d[((length - 1) * nstates) + idy] *
			 beta_d[((length - 1) * nstates) + idy] /
			 sum_ab);
	}
}

extern "C" __global__
void scale_a_dev(float* a_d, float* c_d, int nstates)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < (unsigned int) nstates && idy < (unsigned int) nstates) {
		a_d[(idy * nstates) + idx] = a_d[(idy * nstates) + idx] / c_d[idy];
	}
}

extern "C" __global__
void acc_b_dev(
	float* b_d,
	float* alpha_d,
	float* beta_d,
	float sum_ab,
	int nstates,
	int nsymbols,
	int obs_t,
	int t)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idy < (unsigned int) nsymbols && idx < (unsigned int) nstates && obs_t == (int) idy) {
		b_d[(idy * nstates) + idx] +=
			alpha_d[(t * nstates) + idx] *
			beta_d[(t * nstates) + idx] /
			sum_ab;
	}
}

extern "C" __global__
void est_b_dev(float* b_d, float* gamma_sum_d, int nstates, int nsymbols)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idy < (unsigned int) nsymbols && idx < (unsigned int) nstates) {
		b_d[(idy * nstates) + idx] = b_d[(idy * nstates) + idx] / gamma_sum_d[idx];
	}
}

extern "C" __global__
void scale_b_dev(float* b_d, float* c_d, int nstates, int nsymbols)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < (unsigned int) nstates && idy < (unsigned int) nsymbols) {
		if (b_d[(idy * nstates) + idx] == 0.0f) {
			b_d[(idy * nstates) + idx] = 1e-10f;
		} else {
			b_d[(idy * nstates) + idx] = b_d[(idy * nstates) + idx] / c_d[idx];
		}
	}
}

extern "C" __global__
void est_pi_dev(float* pi_d, float* alpha_d, float* beta_d, float sum_ab, int nstates)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < (unsigned int) nstates) {
		pi_d[idx] = alpha_d[idx] * beta_d[idx] / sum_ab;
	}
}

extern "C" __global__
void s_dot_kernel_naive(int n, float* paramA, int offsetA, float* paramB, int offsetB, float* partialSum_d)
{
	unsigned int tid = threadIdx.x;
	unsigned int totalThreads = gridDim.x * SDOT_BLOCK_SIZE;
	unsigned int offset = SDOT_BLOCK_SIZE * blockIdx.x;

	for (unsigned int i = offset + tid; i < (unsigned int) n; i += totalThreads) {
		partialSum_d[i] = paramA[offsetA + i] * paramB[offsetB + i];
	}
}

extern "C" __global__
void mvm_non_kernel_naive(int m, int n, float* A, int lda, float* x, int offsetX, float* y, int offsetY)
{
	unsigned int tid = threadIdx.x;
	unsigned int totalThreads = gridDim.x * MVMUL_BLOCK_SIZE;
	unsigned int offset = MVMUL_BLOCK_SIZE * blockIdx.x;

	int n_size;
	int m_size;

	if (lda == m) {
		n_size = n;
		m_size = m;
	} else {
		n_size = m;
		m_size = n;
	}

	for (unsigned int i = offset + tid; i < (unsigned int) m_size; i += totalThreads) {
		float sum = 0.0f;

		for (int j = 0; j < n_size; j++) {
			sum += A[i * n_size + j] * x[j + offsetX];
		}

		y[i + offsetY] = sum;
	}
}

extern "C" __global__
void mvm_trans_kernel_naive(int m, int n, float* A, int lda, float* x, int offsetX, float* y, int offsetY)
{
	unsigned int tid = threadIdx.x;
	unsigned int totalThreads = gridDim.x * MVMUL_BLOCK_SIZE;
	unsigned int offset = MVMUL_BLOCK_SIZE * blockIdx.x;

	int n_size;
	int m_size;

	if (lda == m) {
		n_size = n;
		m_size = m;
	} else {
		n_size = m;
		m_size = n;
	}

	for (unsigned int i = offset + tid; i < (unsigned int) m_size; i += totalThreads) {
		float sum = 0.0f;

		for (int j = 0; j < n_size; j++) {
			sum += A[j * n_size + i] * x[j + offsetX];
		}

		y[i + offsetY] = sum;
	}
}
