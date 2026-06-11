#include <hip/hip_runtime.h>

__global__ void lud_diagonal_hip(float* m, int matrix_dim, int offset, int block_size)
{
	int i, j;
	int tid = threadIdx.x;
	int array_offset = offset * matrix_dim + offset;

	for (i = 0; i < block_size; i++) {
		if (tid > i) {
			for (j = 0; j < i; j++) {
				m[array_offset + tid * matrix_dim + i] -=
					m[array_offset + tid * matrix_dim + j] *
					m[array_offset + j * matrix_dim + i];
			}

			m[array_offset + tid * matrix_dim + i] /=
				m[array_offset + i * matrix_dim + i];
		}

		__syncthreads();

		if (tid > i) {
			for (j = 0; j < i + 1; j++) {
				m[array_offset + (i + 1) * matrix_dim + tid] -=
					m[array_offset + (i + 1) * matrix_dim + j] *
					m[array_offset + j * matrix_dim + tid];
			}
		}

		__syncthreads();
	}
}

__global__ void lud_perimeter_hip(float* m, int matrix_dim, int offset, int block_size)
{
	int i, j;
	int array_offset;
	int array_offset_peri;
	int idx;
	int tid = threadIdx.x;
	int gid = blockIdx.x;

	if (tid < block_size) {
		idx = tid;
		array_offset = offset * matrix_dim + offset;

		for (i = 1; i < block_size; i++) {
			for (j = 0; j < i; j++) {
				m[array_offset + i * matrix_dim + (gid + 1) * block_size + idx] -=
					m[array_offset + i * matrix_dim + j] *
					m[array_offset + j * matrix_dim + (gid + 1) * block_size + idx];
			}
		}
	} else {
		idx = tid - block_size;
		array_offset_peri = (offset + (gid + 1) * block_size) * matrix_dim + offset;
		array_offset = offset * matrix_dim + offset;

		for (i = 0; i < block_size; i++) {
			for (j = 0; j < i; j++) {
				m[array_offset_peri + idx * matrix_dim + i] -=
					m[array_offset_peri + idx * matrix_dim + j] *
					m[array_offset + j * matrix_dim + i];
			}

			m[array_offset_peri + idx * matrix_dim + i] /=
				m[array_offset + i * matrix_dim + i];
		}
	}
}

__global__ void lud_internal_hip(float* m, int matrix_dim, int offset, int block_size)
{
	float sum = 0.0f;

	int global_row_id = offset + (blockIdx.y + 1) * block_size;
	int global_col_id = offset + (blockIdx.x + 1) * block_size;

	for (int i = 0; i < block_size; i++) {
		sum +=
			m[(global_row_id + threadIdx.y) * matrix_dim + offset + i] *
			m[(offset + i) * matrix_dim + global_col_id + threadIdx.x];
	}

	m[(global_row_id + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x] -= sum;
}

extern "C" hipError_t lud_launch_diagonal_hip(
	float* d_m,
	int matrix_dim,
	int offset,
	int block_size,
	hipStream_t stream)
{
	hipLaunchKernelGGL(
		lud_diagonal_hip,
		dim3(1),
		dim3(block_size),
		0,
		stream,
		d_m,
		matrix_dim,
		offset,
		block_size);

	return hipGetLastError();
}

extern "C" hipError_t lud_launch_perimeter_hip(
	float* d_m,
	int matrix_dim,
	int offset,
	int block_size,
	int blocks,
	hipStream_t stream)
{
	hipLaunchKernelGGL(
		lud_perimeter_hip,
		dim3(blocks),
		dim3(block_size * 2),
		0,
		stream,
		d_m,
		matrix_dim,
		offset,
		block_size);

	return hipGetLastError();
}

extern "C" hipError_t lud_launch_internal_hip(
	float* d_m,
	int matrix_dim,
	int offset,
	int block_size,
	int blocks,
	hipStream_t stream)
{
	hipLaunchKernelGGL(
		lud_internal_hip,
		dim3(blocks, blocks),
		dim3(block_size, block_size),
		0,
		stream,
		d_m,
		matrix_dim,
		offset,
		block_size);

	return hipGetLastError();
}
