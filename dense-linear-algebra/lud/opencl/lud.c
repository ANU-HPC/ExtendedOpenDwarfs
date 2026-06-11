#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include <include/rdtsc.h>
#include <include/common_args.h>
#include <include/lsb.h>
#include <include/portable_memory.h>

#include "common.h"

int BLOCK_SIZE = 16;
static int do_verify = 0;

#define AOCL_ALIGNMENT 64
#define MIN_TIME_SEC 2

static struct option long_options[] = {
	{"input", 1, NULL, 'i'},
	{"platform", 1, NULL, 'p'},
	{"device", 1, NULL, 'd'},
	{"size", 1, NULL, 's'},
	{"verify", 0, NULL, 'v'},
	{0, 0, 0, 0}
};

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

int main(int argc, char* argv[])
{
	int matrix_dim = 0;
	int opt, option_index = 0;
	func_ret_t ret;
	const char* input_file = NULL;
	float* m = NULL;
	float* mm = NULL;
	stopwatch sw;

	cl_program clProgram;
	cl_kernel clKernel_diagonal;
	cl_kernel clKernel_perimeter;
	cl_kernel clKernel_internal;

	cl_int errcode;
	cl_mem d_m;

	LSB_Init(get_lsb_name(), 0);
	LSB_Set_Rparam_int("matrix_dimension", 0);
	LSB_Set_Rparam_int("repeats_to_two_seconds", 0);

	record_region_start("runtime_initialization");

	ocd_init(&argc, &argv, NULL);
	ocd_initCL();

	if (ocd_get_options().workgroup_1d != 0) {
		BLOCK_SIZE = ocd_get_options().workgroup_1d;
	}

	printf("BLOCK_SIZE = %i\n", BLOCK_SIZE);

	while ((opt = getopt_long(argc, argv, "::vs:i:", long_options, &option_index)) != -1) {
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
			case '?':
				fprintf(stderr, "invalid option\n");
				break;
			case ':':
				fprintf(stderr, "missing argument\n");
				break;
			default:
				fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
				exit(EXIT_FAILURE);
		}
	}

	if ((optind < argc) || (optind == 1)) {
		fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	record_region_end(0);

	record_region_start("host_input_setup");

	if (matrix_dim == 0 && input_file == NULL) {
		fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
		exit(EXIT_FAILURE);
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
		printf("Reading matrix from file %s\n", input_file);

		ret = create_matrix_from_file(&m, input_file, &matrix_dim);
		if (ret != RET_SUCCESS) {
			m = NULL;
			fprintf(stderr, "error create matrix from file %s\n", input_file);
			exit(EXIT_FAILURE);
		}
	}

	if (do_verify) {
		printf("Before LUD\n");
		print_matrix(m, matrix_dim);
		matrix_duplicate(m, &mm, matrix_dim);
	}

	record_region_end(0);

	size_t max_worksize[3];
	errcode = clGetDeviceInfo(
		device_id,
		CL_DEVICE_MAX_WORK_ITEM_SIZES,
		sizeof(size_t) * 3,
		&max_worksize,
		NULL);
	CHKERR(errcode, "Failed to get device info!");

	LSB_Set_Rparam_int("matrix_dimension", matrix_dim);

	record_region_start("kernel_creation");

	char arg[100];
	char kernel_files[] = "lud_kernel";

	sprintf(arg, "-D BLOCK_SIZE=%d", (int) BLOCK_SIZE);

	clProgram = ocdBuildProgramFromFile(context, device_id, kernel_files, arg);

	clKernel_diagonal = clCreateKernel(clProgram, "lud_diagonal", &errcode);
	CHKERR(errcode, "Failed to create diagonal kernel!");

	clKernel_perimeter = clCreateKernel(clProgram, "lud_perimeter", &errcode);
	CHKERR(errcode, "Failed to create perimeter kernel!");

	clKernel_internal = clCreateKernel(clProgram, "lud_internal", &errcode);
	CHKERR(errcode, "Failed to create internal kernel!");

	record_region_end(0);

	record_region_start("device_side_buffer_setup");

	d_m = clCreateBuffer(
		context,
		CL_MEM_READ_WRITE,
		matrix_dim * matrix_dim * sizeof(float),
		NULL,
		&errcode);
	CHKERR(errcode, "Failed to create buffer!");

	record_region_end(0);

	printf("Working kernel memory: %fKiB\n",
		(matrix_dim * matrix_dim * sizeof(float)) / 1024.0);

	int lsb_timing_repeats = 0;
	struct timeval startTime, currentTime, elapsedTime;

	gettimeofday(&startTime, NULL);

	do {
		int i = 0;
		size_t localWorkSize[2];
		size_t globalWorkSize[2];

		LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

		stopwatch_start(&sw);

		record_region_start("device_side_h2d_copy");

		errcode = clEnqueueWriteBuffer(
			commands,
			d_m,
			CL_TRUE,
			0,
			matrix_dim * matrix_dim * sizeof(float),
			(void*) m,
			0,
			NULL,
			&ocdTempEvent);

		clFinish(commands);

		START_TIMER(ocdTempEvent, OCD_TIMER_H2D, "Matrix Copy", ocdTempTimer)
		END_TIMER(ocdTempTimer)

		CHKERR(errcode, "Failed to enqueue write buffer!");

		record_region_end(0);

#ifdef PROFILE_OUTER_LOOP
		record_region_start("lud_outer_loop");
#endif

		for (i = 0; i < matrix_dim - BLOCK_SIZE; i += BLOCK_SIZE) {
#ifndef PROFILE_OUTER_LOOP
			record_region_start("setting_diagonal_kernel_arguments");
#endif
			errcode = clSetKernelArg(clKernel_diagonal, 0, sizeof(cl_mem), (void*) &d_m);
			errcode |= clSetKernelArg(clKernel_diagonal, 1, sizeof(int), (void*) &matrix_dim);
			errcode |= clSetKernelArg(clKernel_diagonal, 2, sizeof(int), (void*) &i);
			CHKERR(errcode, "Failed to set diagonal kernel arguments!");
#ifndef PROFILE_OUTER_LOOP
			record_region_end(i);
#endif

			localWorkSize[0] = BLOCK_SIZE;
			globalWorkSize[0] = BLOCK_SIZE;

#ifndef PROFILE_OUTER_LOOP
			record_region_start("diagonal_kernel_execution");
#endif
			errcode = clEnqueueNDRangeKernel(
				commands,
				clKernel_diagonal,
				1,
				NULL,
				globalWorkSize,
				localWorkSize,
				0,
				NULL,
				&ocdTempEvent);

			clFinish(commands);
#ifndef PROFILE_OUTER_LOOP
			record_region_end(i);
#endif

			START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "Diagonal Kernel", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(errcode, "Failed to enqueue diagonal kernel!");

#ifndef PROFILE_OUTER_LOOP
			record_region_start("setting_perimeter_kernel_arguments");
#endif
			errcode = clSetKernelArg(clKernel_perimeter, 0, sizeof(cl_mem), (void*) &d_m);
			errcode |= clSetKernelArg(clKernel_perimeter, 1, sizeof(int), (void*) &matrix_dim);
			errcode |= clSetKernelArg(clKernel_perimeter, 2, sizeof(int), (void*) &i);
			CHKERR(errcode, "Failed to set perimeter kernel arguments!");
#ifndef PROFILE_OUTER_LOOP
			record_region_end(i);
#endif

			localWorkSize[0] = BLOCK_SIZE * 2;
			globalWorkSize[0] = ((matrix_dim - i) / BLOCK_SIZE - 1) * localWorkSize[0];

#ifndef PROFILE_OUTER_LOOP
			record_region_start("perimeter_kernel_execution");
#endif
			errcode = clEnqueueNDRangeKernel(
				commands,
				clKernel_perimeter,
				1,
				NULL,
				globalWorkSize,
				localWorkSize,
				0,
				NULL,
				&ocdTempEvent);

			clFinish(commands);
#ifndef PROFILE_OUTER_LOOP
			record_region_end(i);
#endif

			START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "Perimeter Kernel", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(errcode, "Failed to enqueue perimeter kernel!");

#ifndef PROFILE_OUTER_LOOP
			record_region_start("setting_internal_kernel_arguments");
#endif
			errcode = clSetKernelArg(clKernel_internal, 0, sizeof(cl_mem), (void*) &d_m);
			errcode |= clSetKernelArg(clKernel_internal, 1, sizeof(int), (void*) &matrix_dim);
			errcode |= clSetKernelArg(clKernel_internal, 2, sizeof(int), (void*) &i);
			CHKERR(errcode, "Failed to set internal kernel arguments!");
#ifndef PROFILE_OUTER_LOOP
			record_region_end(i);
#endif

			localWorkSize[0] = BLOCK_SIZE;
			localWorkSize[1] = BLOCK_SIZE;
			globalWorkSize[0] = ((matrix_dim - i) / BLOCK_SIZE - 1) * localWorkSize[0];
			globalWorkSize[1] = ((matrix_dim - i) / BLOCK_SIZE - 1) * localWorkSize[1];

#ifndef PROFILE_OUTER_LOOP
			record_region_start("internal_kernel_execution");
#endif
			errcode = clEnqueueNDRangeKernel(
				commands,
				clKernel_internal,
				2,
				NULL,
				globalWorkSize,
				localWorkSize,
				0,
				NULL,
				&ocdTempEvent);

			clFinish(commands);
#ifndef PROFILE_OUTER_LOOP
			record_region_end(i);
#endif

			START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "Internal Kernel", ocdTempTimer)
			END_TIMER(ocdTempTimer)

			CHKERR(errcode, "Failed to enqueue internal kernel!");
		}

#ifdef PROFILE_OUTER_LOOP
		record_region_end(0);
#endif

		record_region_start("setting_final_diagonal_kernel_arguments");

		errcode = clSetKernelArg(clKernel_diagonal, 0, sizeof(cl_mem), (void*) &d_m);
		errcode |= clSetKernelArg(clKernel_diagonal, 1, sizeof(int), (void*) &matrix_dim);
		errcode |= clSetKernelArg(clKernel_diagonal, 2, sizeof(int), (void*) &i);
		CHKERR(errcode, "Failed to set final diagonal kernel arguments!");

		record_region_end(i);

		localWorkSize[0] = BLOCK_SIZE;
		globalWorkSize[0] = BLOCK_SIZE;

		record_region_start("final_diagonal_kernel_execution");

		errcode = clEnqueueNDRangeKernel(
			commands,
			clKernel_diagonal,
			1,
			NULL,
			globalWorkSize,
			localWorkSize,
			0,
			NULL,
			&ocdTempEvent);

		clFinish(commands);

		record_region_end(0);

		START_TIMER(ocdTempEvent, OCD_TIMER_KERNEL, "Final Diagonal Kernel", ocdTempTimer)
		END_TIMER(ocdTempTimer)

		CHKERR(errcode, "Failed to enqueue final diagonal kernel!");

		record_region_start("device_side_d2h_copy");

		errcode = clEnqueueReadBuffer(
			commands,
			d_m,
			CL_TRUE,
			0,
			matrix_dim * matrix_dim * sizeof(float),
			(void*) m,
			0,
			NULL,
			&ocdTempEvent);

		clFinish(commands);

		record_region_end(0);

		START_TIMER(ocdTempEvent, OCD_TIMER_D2H, "Matrix Copy", ocdTempTimer)
		END_TIMER(ocdTempTimer)

		CHKERR(errcode, "Failed to enqueue read buffer!");

		stopwatch_stop(&sw);
		printf("Time consumed(ms): %lf\n", 1000 * get_interval_by_sec(&sw));

		lsb_timing_repeats++;
		gettimeofday(&currentTime, NULL);
		timersub(&currentTime, &startTime, &elapsedTime);
	} while (elapsedTime.tv_sec < MIN_TIME_SEC);

	clReleaseMemObject(d_m);

	if (do_verify) {
		printf("After LUD\n");
		print_matrix(m, matrix_dim);
		printf(">>>Verify<<<<\n");
		printf("matrix_dim: %d\n", matrix_dim);
		lud_verify(mm, m, matrix_dim);
		free(mm);
	}

	clReleaseKernel(clKernel_diagonal);
	clReleaseKernel(clKernel_perimeter);
	clReleaseKernel(clKernel_internal);
	clReleaseProgram(clProgram);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	LSB_Finalize();

	free(m);
	ocd_finalize();

	return EXIT_SUCCESS;
}
