#define LIMIT -999

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "needle.h"

#include <include/rdtsc.h>
#include <include/common_args.h>
#include <include/lsb.h>
#include <include/portable_memory.h>

#define AOCL_ALIGNMENT 64
#define MIN_TIME_SEC 2

//#define TRACE

void runTest(int argc, char** argv);

int blosum62[24][24] = {
	{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
	{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
	{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
	{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
	{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
	{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
	{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
	{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
	{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
	{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
	{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
	{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
	{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
	{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
	{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
	{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
	{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
	{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
	{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
	{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
	{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
	{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
	{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
	{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

static const char* get_lsb_name(void)
{
	const char* lsb_name = getenv("ODW_LSB_NAME");

	if (lsb_name == NULL || lsb_name[0] == '\0') {
		lsb_name = "needle";
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

double gettime(void)
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec * 1e-6;
}

int maximum(int a, int b, int c)
{
	int k;

	if (a <= b) {
		k = b;
	} else {
		k = a;
	}

	if (k <= c) {
		return c;
	}

	return k;
}

int main(int argc, char** argv)
{
	LSB_Init(get_lsb_name(), 0);
	LSB_Set_Rparam_int("max_cols", 0);
	LSB_Set_Rparam_int("max_rows", 0);
	LSB_Set_Rparam_int("penalty", 0);
	LSB_Set_Rparam_int("repeats_to_two_seconds", 0);

	record_region_start("runtime_initialization");

	ocd_init(&argc, &argv, NULL);
	ocd_initCL();

	record_region_end(0);

	runTest(argc, argv);

	record_region_start("runtime_finalization");
	ocd_finalize();
	record_region_end(0);

	LSB_Finalize();

	return EXIT_SUCCESS;
}

void usage(int argc, char** argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}

void runTest(int argc, char** argv)
{
	int max_rows, max_cols, penalty;
	int* input_itemsets;
	int* output_itemsets;
	int* referrence;
	cl_mem matrix_cuda;
	cl_mem referrence_cuda;
	int size;
	int i, j;

	cl_int errcode;

	if (argc == 3) {
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
	} else {
		usage(argc, argv);
	}

	if (atoi(argv[1]) % 16 != 0) {
		fprintf(stderr, "The dimension values must be a multiple of 16\n");
		exit(1);
	}

	printf("Start Needleman-Wunsch\n");

	record_region_start("host_input_setup");

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;

	if (_deviceType == 3) {
		referrence = (int*) memalign(AOCL_ALIGNMENT, max_rows * max_cols * sizeof(int));
		input_itemsets = (int*) memalign(AOCL_ALIGNMENT, max_rows * max_cols * sizeof(int));
		output_itemsets = (int*) memalign(AOCL_ALIGNMENT, max_rows * max_cols * sizeof(int));
	} else {
		referrence = (int*) malloc(max_rows * max_cols * sizeof(int));
		input_itemsets = (int*) malloc(max_rows * max_cols * sizeof(int));
		output_itemsets = (int*) malloc(max_rows * max_cols * sizeof(int));
	}

	if (referrence == NULL || input_itemsets == NULL || output_itemsets == NULL) {
		fprintf(stderr, "error: can not allocate memory\n");
		exit(1);
	}

	srand(7);

	for (i = 0; i < max_cols; i++) {
		for (j = 0; j < max_rows; j++) {
			input_itemsets[i * max_cols + j] = 0;
		}
	}

	for (i = 1; i < max_rows; i++) {
		input_itemsets[i * max_cols] = rand() % 10 + 1;
	}

	for (j = 1; j < max_cols; j++) {
		input_itemsets[j] = rand() % 10 + 1;
	}

	for (i = 1; i < max_cols; i++) {
		for (j = 1; j < max_rows; j++) {
			referrence[i * max_cols + j] =
				blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
		}
	}

	for (i = 1; i < max_rows; i++) {
		input_itemsets[i * max_cols] = -i * penalty;
	}

	for (j = 1; j < max_cols; j++) {
		input_itemsets[j] = -j * penalty;
	}

	record_region_end(0);

	LSB_Set_Rparam_int("max_cols", max_cols);
	LSB_Set_Rparam_int("max_rows", max_rows);
	LSB_Set_Rparam_int("penalty", penalty);

	cl_program clProgram;
	cl_kernel clKernel_nw1;
	cl_kernel clKernel_nw2;

	record_region_start("program_build");

	char kernel_files[] = "needle_kernel";
	clProgram = ocdBuildProgramFromFile(context, device_id, kernel_files, NULL);

	record_region_end(0);

	record_region_start("kernel_creation");

	clKernel_nw1 = clCreateKernel(clProgram, "needle_opencl_shared_1", &errcode);
	CHKERR(errcode, "Failed to create kernel!");

	clKernel_nw2 = clCreateKernel(clProgram, "needle_opencl_shared_2", &errcode);
	CHKERR(errcode, "Failed to create kernel!");

	record_region_end(0);

	size = max_cols * max_rows;

	record_region_start("device_side_buffer_setup");

	referrence_cuda = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		sizeof(int) * size,
		NULL,
		&errcode);
	CHKERR(errcode, "Failed to create buffer!");

	matrix_cuda = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		sizeof(int) * size,
		NULL,
		&errcode);
	CHKERR(errcode, "Failed to create buffer!");

	record_region_end(0);

	printf("Working kernel memory: %fKiB\n",
		(((float) (sizeof(int) * size + sizeof(int) * size)) / 1024.0));

	int lsb_timing_repeats = 0;
	struct timeval startTime, currentTime, elapsedTime;

	gettimeofday(&startTime, NULL);

	printf("Processing top-left matrix\n");
	printf("Processing bottom-right matrix\n");

	do {
		LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

		record_region_start("device_side_h2d_copy");

		errcode = clEnqueueWriteBuffer(
			commands,
			referrence_cuda,
			CL_TRUE,
			0,
			sizeof(int) * size,
			(void*) referrence,
			0,
			NULL,
			&ocdTempEvent);
		clFinish(commands);

		START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "NW Reference Copy", ocdTempTimer)
		END_TIMER(ocdTempTimer)

		CHKERR(errcode, "Failed to enqueue write buffer!");

		errcode = clEnqueueWriteBuffer(
			commands,
			matrix_cuda,
			CL_TRUE,
			0,
			sizeof(int) * size,
			(void*) input_itemsets,
			0,
			NULL,
			&ocdTempEvent);
		clFinish(commands);

		START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "NW Item Set Copy", ocdTempTimer)
		END_TIMER(ocdTempTimer)

		CHKERR(errcode, "Failed to enqueue write buffer!");

		record_region_end(0);

		size_t localWorkSize[2] = {BLOCK_SIZE, 1};
		size_t globalWorkSize[2];
		int block_width = (max_cols - 1) / BLOCK_SIZE;

		for (i = 1; i <= block_width; i++) {
			globalWorkSize[0] = i * localWorkSize[0];
			globalWorkSize[1] = 1 * localWorkSize[1];

			record_region_start("setting_nw1_kernel_arguments");

			errcode = clSetKernelArg(clKernel_nw1, 0, sizeof(cl_mem), (void*) &referrence_cuda);
			errcode |= clSetKernelArg(clKernel_nw1, 1, sizeof(cl_mem), (void*) &matrix_cuda);
			errcode |= clSetKernelArg(clKernel_nw1, 2, sizeof(int), (void*) &max_cols);
			errcode |= clSetKernelArg(clKernel_nw1, 3, sizeof(int), (void*) &penalty);
			errcode |= clSetKernelArg(clKernel_nw1, 4, sizeof(int), (void*) &i);
			errcode |= clSetKernelArg(clKernel_nw1, 5, sizeof(int), (void*) &block_width);

			CHKERR(errcode, "Failed to set kernel arguments!");

			record_region_end(i);

			record_region_start("nw1_kernel_execution");

			errcode = clEnqueueNDRangeKernel(
				commands,
				clKernel_nw1,
				2,
				NULL,
				globalWorkSize,
				localWorkSize,
				0,
				NULL,
				&ocdTempEvent);
			clFinish(commands);

			record_region_end(i);

			START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "NW Kernel nw1", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(errcode, "Failed to enqueue kernel!");
		}

		for (i = block_width - 1; i >= 1; i--) {
			globalWorkSize[0] = i * localWorkSize[0];
			globalWorkSize[1] = 1 * localWorkSize[1];

			record_region_start("setting_nw2_kernel_arguments");

			errcode = clSetKernelArg(clKernel_nw2, 0, sizeof(cl_mem), (void*) &referrence_cuda);
			errcode |= clSetKernelArg(clKernel_nw2, 1, sizeof(cl_mem), (void*) &matrix_cuda);
			errcode |= clSetKernelArg(clKernel_nw2, 2, sizeof(int), (void*) &max_cols);
			errcode |= clSetKernelArg(clKernel_nw2, 3, sizeof(int), (void*) &penalty);
			errcode |= clSetKernelArg(clKernel_nw2, 4, sizeof(int), (void*) &i);
			errcode |= clSetKernelArg(clKernel_nw2, 5, sizeof(int), (void*) &block_width);

			CHKERR(errcode, "Failed to set kernel arguments!");

			record_region_end(i);

			record_region_start("nw2_kernel_execution");

			errcode = clEnqueueNDRangeKernel(
				commands,
				clKernel_nw2,
				2,
				NULL,
				globalWorkSize,
				localWorkSize,
				0,
				NULL,
				&ocdTempEvent);
			clFinish(commands);

			record_region_end(i);

			START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "NW Kernel nw2", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(errcode, "Failed to enqueue kernel!");
		}

		record_region_start("device_side_d2h_copy");

		errcode = clEnqueueReadBuffer(
			commands,
			matrix_cuda,
			CL_TRUE,
			0,
			sizeof(int) * size,
			(void*) output_itemsets,
			0,
			NULL,
			&ocdTempEvent);
		clFinish(commands);

		record_region_end(0);

		START_TIMER(ocdTempEvent, OCD_TIMER_D2H, "NW Item Set Copy", ocdTempTimer)
		END_TIMER(ocdTempTimer)

		CHKERR(errcode, "Failed to enqueue read buffer!");

		lsb_timing_repeats++;

		gettimeofday(&currentTime, NULL);
		timersub(&currentTime, &startTime, &elapsedTime);
	} while (elapsedTime.tv_sec < MIN_TIME_SEC);

#ifdef TRACE
	printf("print traceback value GPU:\n");

	for (i = max_rows - 2, j = max_rows - 2; i >= 0, j >= 0;) {
		int nw, n, w, traceback;

		if (i == max_rows - 2 && j == max_rows - 2) {
			printf("%d ", output_itemsets[i * max_cols + j]);
		}

		if (i == 0 && j == 0) {
			break;
		}

		if (i > 0 && j > 0) {
			nw = output_itemsets[(i - 1) * max_cols + j - 1];
			w = output_itemsets[i * max_cols + j - 1];
			n = output_itemsets[(i - 1) * max_cols + j];
		} else if (i == 0) {
			nw = n = LIMIT;
			w = output_itemsets[i * max_cols + j - 1];
		} else if (j == 0) {
			nw = w = LIMIT;
			n = output_itemsets[(i - 1) * max_cols + j];
		}

		traceback = maximum(nw, w, n);

		printf("%d ", traceback);

		if (traceback == nw) {
			i--;
			j--;
			continue;
		} else if (traceback == w) {
			j--;
			continue;
		} else if (traceback == n) {
			i--;
			continue;
		}
	}

	printf("\n");
#endif

	record_region_start("device_side_buffer_cleanup");

	clReleaseMemObject(referrence_cuda);
	clReleaseMemObject(matrix_cuda);

	record_region_end(0);

	record_region_start("kernel_cleanup");

	clReleaseKernel(clKernel_nw1);
	clReleaseKernel(clKernel_nw2);
	clReleaseProgram(clProgram);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	record_region_end(0);

	free(referrence);
	free(input_itemsets);
	free(output_itemsets);
}
