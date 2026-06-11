#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__device__ __forceinline__
int maximum_baseline(int a, int b, int c)
{
	int k = (a <= b) ? b : a;
	return (k <= c) ? c : k;
}

extern "C" __global__
void needle_cuda_shared_1(
	int* referrence,
	int* matrix_cuda,
	int cols,
	int penalty,
	int i,
	int block_width)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int b_index_x = bx;
	int b_index_y = i - 1 - bx;
	int index = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1);

	for (int m = 0; m < BLOCK_SIZE; m++) {
		if (tx <= m) {
			int ref_x = index + (m - tx) * cols;

			matrix_cuda[ref_x] =
				maximum_baseline(
					matrix_cuda[ref_x - (cols + 1)] + referrence[ref_x],
					matrix_cuda[ref_x - 1] - penalty,
					matrix_cuda[ref_x - cols] - penalty);
		}

		__syncthreads();
	}

	for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
		if (tx <= m) {
			int ref_x = index + (m - tx) * cols + (cols + 1) * (BLOCK_SIZE - 1 - m);

			matrix_cuda[ref_x] =
				maximum_baseline(
					matrix_cuda[ref_x - (cols + 1)] + referrence[ref_x],
					matrix_cuda[ref_x - 1] - penalty,
					matrix_cuda[ref_x - cols] - penalty);
		}

		__syncthreads();
	}
}

extern "C" __global__
void needle_cuda_shared_2(
	int* referrence,
	int* matrix_cuda,
	int cols,
	int penalty,
	int i,
	int block_width)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int b_index_x = bx + block_width - i;
	int b_index_y = block_width - bx - 1;
	int index = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1);

	for (int m = 0; m < BLOCK_SIZE; m++) {
		if (tx <= m) {
			int ref_x = index + (m - tx) * cols;

			matrix_cuda[ref_x] =
				maximum_baseline(
					matrix_cuda[ref_x - (cols + 1)] + referrence[ref_x],
					matrix_cuda[ref_x - 1] - penalty,
					matrix_cuda[ref_x - cols] - penalty);
		}

		__syncthreads();
	}

	for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
		if (tx <= m) {
			int ref_x = index + (m - tx) * cols + (cols + 1) * (BLOCK_SIZE - 1 - m);

			matrix_cuda[ref_x] =
				maximum_baseline(
					matrix_cuda[ref_x - (cols + 1)] + referrence[ref_x],
					matrix_cuda[ref_x - 1] - penalty,
					matrix_cuda[ref_x - cols] - penalty);
		}

		__syncthreads();
	}
}

extern "C" cudaError_t needle_launch_cuda_baseline_1(
	int* referrence,
	int* matrix_cuda,
	int cols,
	int penalty,
	int i,
	int block_width,
	cudaStream_t stream)
{
	needle_cuda_shared_1<<<dim3(i), dim3(BLOCK_SIZE), 0, stream>>>(
		referrence,
		matrix_cuda,
		cols,
		penalty,
		i,
		block_width);

	return cudaGetLastError();
}

extern "C" cudaError_t needle_launch_cuda_baseline_2(
	int* referrence,
	int* matrix_cuda,
	int cols,
	int penalty,
	int i,
	int block_width,
	cudaStream_t stream)
{
	needle_cuda_shared_2<<<dim3(i), dim3(BLOCK_SIZE), 0, stream>>>(
		referrence,
		matrix_cuda,
		cols,
		penalty,
		i,
		block_width);

	return cudaGetLastError();
}
