#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

#include <include/lsb.h>
extern "C" {
#include <include/crc_formats.h>
}

#define __constant static const
#define crc32Lookup crc32LookupHost
#include "../include/eth_crc32_lut.h"
#undef crc32Lookup
#undef __constant

#define MIN_TIME_SEC 2

extern "C" cudaError_t crc_upload_table_cuda(const unsigned int table[8][256]);

extern "C" cudaError_t crc_launch_cuda(
	const unsigned int* d_input,
	unsigned int page_size,
	unsigned int num_words,
	unsigned int* d_output,
	size_t global_size,
	size_t local_size,
	cudaStream_t stream);

unsigned char verbosity = 0;

unsigned int* num_parallel_crcs = NULL;
unsigned int page_size = 100000000;
unsigned int num_wg_sizes = 0;
unsigned int num_words;
unsigned int num_blocks;
unsigned int num_pages_last_block;
unsigned int num_block_sizes = 0;
size_t* wg_sizes = NULL;

extern "C" void check(int b, const char* msg)
{
	if (!b)
	{
		fprintf(stderr, "error: %s\n", msg);
		exit(EXIT_FAILURE);
	}
}

extern "C" void* int_new_array(const size_t N, const char* error_msg)
{
	void* ptr = malloc(N);
	check(ptr != NULL, error_msg);
	return ptr;
}

uint32_t crc32_8bytes(const void* data, size_t length)
{
	uint32_t* current = (uint32_t*) data;
	uint32_t crc = 0xFFFFFFFF;

	while (length >= 8)
	{
		uint32_t one = *current++ ^ crc;
		uint32_t two = *current++;
		crc = crc32LookupHost[7][ one        & 0xFF] ^
		      crc32LookupHost[6][(one >>  8) & 0xFF] ^
		      crc32LookupHost[5][(one >> 16) & 0xFF] ^
		      crc32LookupHost[4][ one >> 24        ] ^
		      crc32LookupHost[3][ two        & 0xFF] ^
		      crc32LookupHost[2][(two >>  8) & 0xFF] ^
		      crc32LookupHost[1][(two >> 16) & 0xFF] ^
		      crc32LookupHost[0][ two >> 24        ];
		length -= 8;
	}

	unsigned char* currentChar = (unsigned char*) current;
	while (length--)
		crc = (crc >> 8) ^ crc32LookupHost[0][(crc & 0xFF) ^ *currentChar++];

	return ~crc;
}

void usage()
{
	printf("crc -i <input_file> [hva] [-r <num_execs>] [-p <parallel_pages>] [-w <threads_per_block>]\n");
	printf("\t-h : Print this help message\n");
	printf("\t-v : Increase verbosity level\n");
	printf("\t-i : Input file name\n");
	printf("\t-a : Verify results on CPU\n");
	printf("\t-p : Number of pages to CRC in parallel, default 16\n");
	printf("\t-r : Execute program with same data <num_execs> times, default 1\n");
	printf("\t-w : CUDA threads per block, default 256\n");
	exit(0);
}

static void cuda_check(cudaError_t err, const char* msg)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void enqueueCRCDevice(
	unsigned int* h_num,
	unsigned int* h_answer,
	size_t global_size,
	size_t local_size,
	unsigned int* d_input,
	unsigned int* d_output,
	cudaStream_t stream,
	unsigned int block_id)
{
	LSB_Set_Rparam_string("region", "device_side_h2d_copy");
	LSB_Res();
	cuda_check(
		cudaMemcpyAsync(
			d_input,
			h_num,
			sizeof(char) * page_size * global_size,
			cudaMemcpyHostToDevice,
			stream),
		"failed to enqueue H2D copy");
	cuda_check(cudaStreamSynchronize(stream), "failed to synchronize H2D copy");
	LSB_Rec(block_id);

	LSB_Set_Rparam_string("region", "setting_kernel_arguments");
	LSB_Res();
	LSB_Rec(block_id);

	LSB_Set_Rparam_string("region", "kernel_execution");
	LSB_Res();
	cuda_check(
		crc_launch_cuda(
			d_input,
			page_size,
			num_words,
			d_output,
			global_size,
			local_size,
			stream),
		"failed to launch CRC kernel");
	cuda_check(cudaStreamSynchronize(stream), "failed to synchronize CRC kernel");
	LSB_Rec(block_id);

	LSB_Set_Rparam_string("region", "device_side_d2h_copy");
	LSB_Res();
	cuda_check(
		cudaMemcpyAsync(
			h_answer,
			d_output,
			sizeof(unsigned int) * global_size,
			cudaMemcpyDeviceToHost,
			stream),
		"failed to enqueue D2H copy");
	cuda_check(cudaStreamSynchronize(stream), "failed to synchronize D2H copy");
	LSB_Rec(block_id);
}

int main(int argc, char** argv)
{
	unsigned int* h_num = NULL;
	unsigned int* ocl_remainders = NULL;
	unsigned int cpu_remainder;

	unsigned int run_serial = 0;
	unsigned int h, ii, i, k;
	unsigned int num_pages = 1;
	unsigned int num_execs = 1;

	char* file = NULL;
	char* optptr = NULL;
	void* tmp = NULL;
	int c;

	cudaStream_t stream;

	const char* lsb_name = getenv("ODW_LSB_NAME");
	if (lsb_name == NULL || lsb_name[0] == '\0')
		lsb_name = "crc";

	LSB_Init(lsb_name, 0);
	LSB_Set_Rparam_int("repeats_to_two_seconds", 0);

	LSB_Set_Rparam_string("region", "runtime_initialization");
	LSB_Res();
	cuda_check(cudaFree(0), "failed to initialize CUDA runtime");
	cuda_check(cudaStreamCreate(&stream), "failed to create CUDA stream");
	LSB_Rec(0);

	while ((c = getopt(argc, argv, "avi:p:w:hr:")) != -1)
	{
		switch (c)
		{
			case 'h':
				usage();
				break;
			case 'v':
				verbosity++;
				break;
			case 'a':
				run_serial = 1;
				break;
			case 'i':
				file = optarg;
				printf("Reading Input from '%s'\n", file);
				break;
			case 'r':
				num_execs = atoi(optarg);
				printf("Executing %u times\n", num_execs);
				break;
			case 'p':
				optptr = optarg;
				num_block_sizes++;
				tmp = realloc(num_parallel_crcs, sizeof(unsigned int) * num_block_sizes);
				check(tmp != NULL, "crc.main() - Cannot allocate num_parallel_crcs");
				num_parallel_crcs = (unsigned int*) tmp;
				num_parallel_crcs[num_block_sizes - 1] = atoi(optptr);
				break;
			case 'w':
				optptr = optarg;
				num_wg_sizes++;
				tmp = realloc(wg_sizes, sizeof(size_t) * num_wg_sizes);
				check(tmp != NULL, "crc.main() - Cannot allocate wg_sizes");
				wg_sizes = (size_t*) tmp;
				wg_sizes[num_wg_sizes - 1] = atoi(optptr);
				break;
			default:
				usage();
		}
	}

	check(file != NULL, "-i option must be supplied!");

	LSB_Set_Rparam_string("region", "host_input_load");
	LSB_Res();
	h_num = read_crc(&num_pages, &page_size, file);
	LSB_Rec(0);

	LSB_Set_Rparam_int("number_of_pages", num_pages);
	LSB_Set_Rparam_int("page_size", page_size);

	if (!num_block_sizes)
	{
		num_block_sizes = 1;
		num_parallel_crcs = (unsigned int*) malloc(sizeof(unsigned int));
		num_parallel_crcs[0] = 16;
	}

	if (!num_wg_sizes)
	{
		num_wg_sizes = 1;
		wg_sizes = (size_t*) malloc(sizeof(size_t));
		wg_sizes[0] = 256;
	}

	num_words = page_size / 4;

	LSB_Set_Rparam_string("region", "kernel_creation");
	LSB_Res();
	cuda_check(crc_upload_table_cuda(crc32LookupHost), "failed to upload CRC lookup table");
	LSB_Rec(0);

	for (h = 0; h < num_block_sizes; h++)
	{
		num_blocks = num_pages / num_parallel_crcs[h];
		if (num_pages % num_parallel_crcs[h] != 0)
		{
			num_blocks++;
			num_pages_last_block = num_pages % num_parallel_crcs[h];
		}
		else
		{
			num_pages_last_block = num_parallel_crcs[h];
		}

		unsigned int** dev_input = (unsigned int**) malloc(sizeof(unsigned int*) * num_blocks);
		unsigned int** dev_output = (unsigned int**) malloc(sizeof(unsigned int*) * num_blocks);

		ocl_remainders = (unsigned int*) int_new_array(
			sizeof(unsigned int) * num_pages,
			"crc.main() - Cannot allocate output remainders");

		for (i = 0; i < num_blocks; i++)
		{
			LSB_Set_Rparam_string("region", "device_side_buffer_setup");
			LSB_Res();

			cuda_check(
				cudaMalloc(
					(void**) &dev_input[i],
					sizeof(char) * page_size * num_parallel_crcs[h]),
				"failed to allocate device input");

			cuda_check(
				cudaMalloc(
					(void**) &dev_output[i],
					sizeof(unsigned int) * num_parallel_crcs[h]),
				"failed to allocate device output");

			LSB_Rec(i);
		}

		printf("Working kernel memory: %fKiB\n",
			(sizeof(char) * page_size * num_parallel_crcs[h] +
			 sizeof(unsigned int) * num_parallel_crcs[h]) / 1024.0);

		int lsb_timing_repeats = 0;
		struct timeval startTime, currentTime, elapsedTime;
		gettimeofday(&startTime, NULL);

		do {
			LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

			for (k = 0; k < num_wg_sizes; k++)
			{
				for (ii = 0; ii < num_execs; ii++)
				{
					for (i = 0; i < num_blocks; i++)
					{
						size_t global_size;
						size_t local_size = wg_sizes[k];

						if (i == num_blocks - 1)
							global_size = num_pages_last_block;
						else
							global_size = num_parallel_crcs[h];

						enqueueCRCDevice(
							&h_num[i * num_parallel_crcs[h] * num_words],
							&ocl_remainders[i * num_parallel_crcs[h]],
							global_size,
							local_size,
							dev_input[i],
							dev_output[i],
							stream,
							i);
					}

					if (run_serial)
					{
						printf("Validating results with serial CRC...\n");
						for (i = 0; i < num_pages; i++)
						{
							cpu_remainder = crc32_8bytes(&h_num[i * num_words], page_size);
							if (cpu_remainder != ocl_remainders[i])
							{
								fprintf(
									stderr,
									"ERROR: CUDA and CPU Slice-by-8 remainders for page %u differ [CUDA: '%X', CPU: '%X']\n",
									i + 1,
									ocl_remainders[i],
									cpu_remainder);
							}
						}
					}
				}
			}

			lsb_timing_repeats++;
			gettimeofday(&currentTime, NULL);
			timersub(&currentTime, &startTime, &elapsedTime);
		} while (elapsedTime.tv_sec < MIN_TIME_SEC);

		for (i = 0; i < num_blocks; i++)
		{
			cudaFree(dev_input[i]);
			cudaFree(dev_output[i]);
		}

		free(dev_input);
		free(dev_output);
		free(ocl_remainders);
		ocl_remainders = NULL;
	}

	LSB_Finalize();

	cudaStreamDestroy(stream);

	free(h_num);
	free(num_parallel_crcs);
	free(wg_sizes);

	return 0;
}
