#include <cuda_runtime.h>

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include <include/lsb.h>
#include <include/portable_memory.h>
#include <include/rdtsc.h>

extern "C" {
#include "common.h"
}

int BLOCK_SIZE = 16;
static int do_verify = 0;

#define AOCL_ALIGNMENT 64
#define MIN_TIME_SEC 2

static struct option long_options[] = {
	{"input", 1, NULL, 'i'},
	{"device", 1, NULL, 'd'},
	{"size", 1, NULL, 's'},
	{"verify", 0, NULL, 'v'},
	{"workgroup", 1, NULL, 'w'},
	{0, 0, 0, 0}
};

extern "C" cudaError_t lud_launch_diagonal_cuda(
	float* d_m,
	int matrix_dim,
	int offset,
	int block_size,
	cudaStream_t stream);

extern "C" cudaError_t lud_launch_perimeter_cuda(
	float* d_m,
	int matrix_dim,
	int offset,
	int block_size,
	int blocks,
	cudaStream_t stream);

extern "C" cudaError_t lud_launch_internal_cuda(
	float* d_m,
	int matrix_dim,
	int offset,
	int block_size,
	int blocks,
	cudaStream_t stream);

static void cuda_check(cudaError_t err, const char* msg)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

static const char* get_lsb_name(void)
{
	const char* lsb_name = getenv("ODW_LSB_NAME");

	if (lsb_name == NULL || lsb_name[0] == '\0') {
		lsb_name = "lud";
	}

	return lsb_name;
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

static void usage(const char* argv0)
{
	fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file] [-w block_size]\n", argv0);
	exit(EXIT_FAILURE);
}

static void parse_pre_separator_common_args(int argc, char** argv, int* cuda_device)
{
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--") == 0) {
			break;
		}

		if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
			*cuda_device = atoi(argv[i + 1]);
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

int main(int argc, char** argv)
{
	int matrix_dim = 0;
	int opt, option_index = 0;
	int cuda_device = 0;
	const char* input_file = NULL;

	float* m = NULL;
	float* mm = NULL;
	float* d_m = NULL;

	cudaStream_t stream;

	parse_pre_separator_common_args(argc, argv, &cuda_device);

	int app_argc = 0;
	char** app_argv = NULL;
	build_app_argv(argc, argv, &app_argc, &app_argv);

	optind = 1;
	while ((opt = getopt_long(app_argc, app_argv, "vs:i:d:w:", long_options, &option_index)) != -1) {
		switch (opt) {
			case 'i':
				input_file = optarg;
				break;
			case 'v':
				do_verify = 1;
				break;
			case 's':
				matrix_dim = atoi(optarg);
				break;
			case 'd':
				cuda_device = atoi(optarg);
				break;
			case 'w':
				BLOCK_SIZE = atoi(optarg);
				break;
			case '?':
			case ':':
			default:
				usage(argv[0]);
		}
	}

	free(app_argv);

	LSB_Init(get_lsb_name(), 0);
	LSB_Set_Rparam_int("matrix_dimension", 0);
	LSB_Set_Rparam_int("repeats_to_two_seconds", 0);

	record_region_start("runtime_initialization");

	cuda_check(cudaSetDevice(cuda_device), "failed to select CUDA device");
	cuda_check(cudaFree(0), "failed to initialize CUDA runtime");
	cuda_check(cudaStreamCreate(&stream), "failed to create CUDA stream");

	printf("CUDA device = %d\n", cuda_device);
	printf("BLOCK_SIZE = %i\n", BLOCK_SIZE);

	record_region_end(0);

	record_region_start("host_input_setup");

	if (matrix_dim == 0 && input_file == NULL) {
		usage(argv[0]);
	}

	if (matrix_dim != 0) {
		int i, j;

		m = (float*) memalign(AOCL_ALIGNMENT, sizeof(float) * matrix_dim * matrix_dim);
		if (m == NULL) {
			fprintf(stderr, "Failed to allocate matrix\n");
			exit(EXIT_FAILURE);
		}

		for (i = 0; i < matrix_dim; i++) {
			for (j = 0; j < matrix_dim; j++) {
				m[i * matrix_dim + j] = ((float) rand() / (float) RAND_MAX);
			}
		}
	} else {
		func_ret_t ret;

		printf("Reading matrix from file %s\n", input_file);

		ret = create_matrix_from_file(&m, input_file, &matrix_dim);
		if (ret != RET_SUCCESS) {
			m = NULL;
			fprintf(stderr, "error create matrix from file %s\n", input_file);
			exit(EXIT_FAILURE);
		}
	}

	if (matrix_dim % BLOCK_SIZE != 0) {
		fprintf(stderr, "matrix dimension %d must be a multiple of BLOCK_SIZE %d\n", matrix_dim, BLOCK_SIZE);
		exit(EXIT_FAILURE);
	}

	if (do_verify) {
		printf("Before LUD\n");
		print_matrix(m, matrix_dim);
		matrix_duplicate(m, &mm, matrix_dim);
	}

	record_region_end(0);

	LSB_Set_Rparam_int("matrix_dimension", matrix_dim);

	record_region_start("kernel_creation");
	record_region_end(0);

	record_region_start("device_side_buffer_setup");

	cuda_check(
		cudaMalloc((void**) &d_m, matrix_dim * matrix_dim * sizeof(float)),
		"failed to allocate device matrix");

	record_region_end(0);

	printf("Working kernel memory: %fKiB\n",
		(matrix_dim * matrix_dim * sizeof(float)) / 1024.0);

	int lsb_timing_repeats = 0;
	struct timeval startTime, currentTime, elapsedTime;

	gettimeofday(&startTime, NULL);

	do {
		int i = 0;

		LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

		record_region_start("device_side_h2d_copy");

		cuda_check(
			cudaMemcpyAsync(
				d_m,
				m,
				matrix_dim * matrix_dim * sizeof(float),
				cudaMemcpyHostToDevice,
				stream),
			"failed to enqueue H2D matrix copy");
		cuda_check(cudaStreamSynchronize(stream), "failed to synchronize H2D copy");

		record_region_end(0);

#ifdef PROFILE_OUTER_LOOP
		record_region_start("lud_outer_loop");
#endif

		struct timeval kernel_start, kernel_end;
		gettimeofday(&kernel_start, NULL);

		for (i = 0; i < matrix_dim - BLOCK_SIZE; i += BLOCK_SIZE) {
			int blocks = (matrix_dim - i) / BLOCK_SIZE - 1;

#ifndef PROFILE_OUTER_LOOP
			record_region_start("setting_diagonal_kernel_arguments");
			record_region_end(i);
#endif

#ifndef PROFILE_OUTER_LOOP
			record_region_start("diagonal_kernel_execution");
#endif
			cuda_check(
				lud_launch_diagonal_cuda(d_m, matrix_dim, i, BLOCK_SIZE, stream),
				"failed to launch diagonal kernel");
			cuda_check(cudaStreamSynchronize(stream), "failed to synchronize diagonal kernel");
#ifndef PROFILE_OUTER_LOOP
			record_region_end(i);
#endif

#ifndef PROFILE_OUTER_LOOP
			record_region_start("setting_perimeter_kernel_arguments");
			record_region_end(i);
#endif

#ifndef PROFILE_OUTER_LOOP
			record_region_start("perimeter_kernel_execution");
#endif
			cuda_check(
				lud_launch_perimeter_cuda(d_m, matrix_dim, i, BLOCK_SIZE, blocks, stream),
				"failed to launch perimeter kernel");
			cuda_check(cudaStreamSynchronize(stream), "failed to synchronize perimeter kernel");
#ifndef PROFILE_OUTER_LOOP
			record_region_end(i);
#endif

#ifndef PROFILE_OUTER_LOOP
			record_region_start("setting_internal_kernel_arguments");
			record_region_end(i);
#endif

#ifndef PROFILE_OUTER_LOOP
			record_region_start("internal_kernel_execution");
#endif
			cuda_check(
				lud_launch_internal_cuda(d_m, matrix_dim, i, BLOCK_SIZE, blocks, stream),
				"failed to launch internal kernel");
			cuda_check(cudaStreamSynchronize(stream), "failed to synchronize internal kernel");
#ifndef PROFILE_OUTER_LOOP
			record_region_end(i);
#endif
		}

#ifdef PROFILE_OUTER_LOOP
		record_region_end(0);
#endif

		record_region_start("setting_final_diagonal_kernel_arguments");
		record_region_end(i);

		record_region_start("final_diagonal_kernel_execution");

		cuda_check(
			lud_launch_diagonal_cuda(d_m, matrix_dim, i, BLOCK_SIZE, stream),
			"failed to launch final diagonal kernel");
		cuda_check(cudaStreamSynchronize(stream), "failed to synchronize final diagonal kernel");

		record_region_end(0);

		gettimeofday(&kernel_end, NULL);
		timersub(&kernel_end, &kernel_start, &elapsedTime);

		record_region_start("device_side_d2h_copy");

		cuda_check(
			cudaMemcpyAsync(
				m,
				d_m,
				matrix_dim * matrix_dim * sizeof(float),
				cudaMemcpyDeviceToHost,
				stream),
			"failed to enqueue D2H matrix copy");
		cuda_check(cudaStreamSynchronize(stream), "failed to synchronize D2H copy");

		record_region_end(0);

		lsb_timing_repeats++;
		gettimeofday(&currentTime, NULL);
		timersub(&currentTime, &startTime, &elapsedTime);
	} while (elapsedTime.tv_sec < MIN_TIME_SEC);

	cuda_check(cudaFree(d_m), "failed to free device matrix");

	if (do_verify) {
		printf("After LUD\n");
		print_matrix(m, matrix_dim);
		printf(">>>Verify<<<<\n");
		printf("matrix_dim: %d\n", matrix_dim);
		lud_verify(mm, m, matrix_dim);
		free(mm);
	}

	cuda_check(cudaStreamDestroy(stream), "failed to destroy CUDA stream");

	LSB_Finalize();

	free(m);

	return EXIT_SUCCESS;
}
