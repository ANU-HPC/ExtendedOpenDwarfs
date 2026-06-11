#include <cuda_runtime.h>
#include <stdint.h>

__constant__ unsigned int crc32LookupDevice[8][256];

extern "C" cudaError_t crc_upload_table_cuda(const unsigned int table[8][256])
{
	return cudaMemcpyToSymbol(
		crc32LookupDevice,
		table,
		sizeof(unsigned int) * 8 * 256);
}

__global__ void crc32_slice8_cuda(
	const unsigned int* __restrict__ data,
	unsigned int length_bytes,
	unsigned int length_ints,
	unsigned int* __restrict__ res,
	size_t global_size)
{
	unsigned int crc;
	unsigned int one, two;
	unsigned int byte;
	size_t i;
	size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid >= global_size)
		return;

	crc = 0xFFFFFFFF;
	i = gid * length_ints;

	while (length_bytes >= 8)
	{
		one = data[i++] ^ crc;
		two = data[i++];

		crc = crc32LookupDevice[7][ one        & 0xFF] ^
		      crc32LookupDevice[6][(one >>  8) & 0xFF] ^
		      crc32LookupDevice[5][(one >> 16) & 0xFF] ^
		      crc32LookupDevice[4][(one >> 24)       ] ^
		      crc32LookupDevice[3][ two        & 0xFF] ^
		      crc32LookupDevice[2][(two >>  8) & 0xFF] ^
		      crc32LookupDevice[1][(two >> 16) & 0xFF] ^
		      crc32LookupDevice[0][(two >> 24)       ];

		length_bytes -= 8;
	}

	if (length_bytes)
	{
		one = data[i++];

		if (length_bytes)
		{
			byte = one & 0xFF;
			crc = (crc >> 8) ^ crc32LookupDevice[0][(crc & 0xFF) ^ byte];
			length_bytes--;
		}

		if (length_bytes)
		{
			byte = (one >> 8) & 0xFF;
			crc = (crc >> 8) ^ crc32LookupDevice[0][(crc & 0xFF) ^ byte];
			length_bytes--;
		}

		if (length_bytes)
		{
			byte = (one >> 16) & 0xFF;
			crc = (crc >> 8) ^ crc32LookupDevice[0][(crc & 0xFF) ^ byte];
			length_bytes--;
		}

		if (length_bytes)
		{
			byte = (one >> 24) & 0xFF;
			crc = (crc >> 8) ^ crc32LookupDevice[0][(crc & 0xFF) ^ byte];
			length_bytes--;
		}
	}

	res[gid] = ~crc;
}

extern "C" cudaError_t crc_launch_cuda(
	const unsigned int* d_input,
	unsigned int page_size,
	unsigned int num_words,
	unsigned int* d_output,
	size_t global_size,
	size_t local_size,
	cudaStream_t stream)
{
	dim3 block(local_size);
	dim3 grid((global_size + local_size - 1) / local_size);

	crc32_slice8_cuda<<<grid, block, 0, stream>>>(
		d_input,
		page_size,
		num_words,
		d_output,
		global_size);

	return cudaGetLastError();
}
