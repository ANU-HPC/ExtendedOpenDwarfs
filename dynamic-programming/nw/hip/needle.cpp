#define LIMIT -999

#include <hip/hip_runtime.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

#include <include/lsb.h>
#include <include/portable_memory.h>

#define AOCL_ALIGNMENT 64
#define MIN_TIME_SEC 2
#define BLOCK_SIZE 16

extern "C" hipError_t needle_launch_hip_shared_1(
	int* referrence,
	int* matrix_hip,
	int cols,
	int penalty,
	int i,
	int block_width,
	hipStream_t stream);

extern "C" hipError_t needle_launch_hip_shared_2(
	int* referrence,
	int* matrix_hip,
	int cols,
	int penalty,
	int i,
	int block_width,
	hipStream_t stream);

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
	{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1},
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

static void hip_check(hipError_t err, const char* msg)
{
	if (err != hipSuccess) {
		fprintf(stderr, "HIP error: %s: %s\n", msg, hipGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

static void usage(const char* argv0)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty>\n", argv0);
	exit(EXIT_FAILURE);
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

int main(int argc, char** argv)
{
	int hip_device = 0;
	parse_pre_separator_common_args(argc, argv, &hip_device);

	int app_argc = 0;
	char** app_argv = NULL;
	build_app_argv(argc, argv, &app_argc, &app_argv);

	if (app_argc != 3) {
		usage(argv[0]);
	}

	int max_rows = atoi(app_argv[1]);
	int max_cols = atoi(app_argv[1]);
	int penalty = atoi(app_argv[2]);

	free(app_argv);

	if (max_rows % 16 != 0) {
		fprintf(stderr, "The dimension values must be a multiple of 16\n");
		exit(EXIT_FAILURE);
	}

	LSB_Init(get_lsb_name(), 0);
	LSB_Set_Rparam_int("max_cols", 0);
	LSB_Set_Rparam_int("max_rows", 0);
	LSB_Set_Rparam_int("penalty", penalty);
	LSB_Set_Rparam_int("repeats_to_two_seconds", 0);

	record_region_start("runtime_initialization");

	hip_check(hipSetDevice(hip_device), "failed to select HIP device");
	hip_check(hipFree(0), "failed to initialize HIP runtime");

	hipStream_t stream;
	hip_check(hipStreamCreate(&stream), "failed to create HIP stream");

	record_region_end(0);

	printf("Start Needleman-Wunsch\n");
	printf("HIP device = %d\n", hip_device);

	record_region_start("host_input_setup");

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;

	const int size = max_cols * max_rows;

	int* referrence = (int*) memalign(AOCL_ALIGNMENT, size * sizeof(int));
	int* input_itemsets = (int*) memalign(AOCL_ALIGNMENT, size * sizeof(int));
	int* output_itemsets = (int*) memalign(AOCL_ALIGNMENT, size * sizeof(int));

	if (referrence == NULL || input_itemsets == NULL || output_itemsets == NULL) {
		fprintf(stderr, "error: can not allocate memory\n");
		exit(EXIT_FAILURE);
	}

	srand(7);

	for (int i = 0; i < max_cols; i++) {
		for (int j = 0; j < max_rows; j++) {
			input_itemsets[i * max_cols + j] = 0;
		}
	}

	for (int i = 1; i < max_rows; i++) {
		input_itemsets[i * max_cols] = rand() % 10 + 1;
	}

	for (int j = 1; j < max_cols; j++) {
		input_itemsets[j] = rand() % 10 + 1;
	}

	for (int i = 1; i < max_cols; i++) {
		for (int j = 1; j < max_rows; j++) {
			referrence[i * max_cols + j] =
				blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
		}
	}

	for (int i = 1; i < max_rows; i++) {
		input_itemsets[i * max_cols] = -i * penalty;
	}

	for (int j = 1; j < max_cols; j++) {
		input_itemsets[j] = -j * penalty;
	}

	record_region_end(0);

	LSB_Set_Rparam_int("max_cols", max_cols);
	LSB_Set_Rparam_int("max_rows", max_rows);
	LSB_Set_Rparam_int("penalty", penalty);

	record_region_start("kernel_creation");
	record_region_end(0);

	record_region_start("device_side_buffer_setup");

	int* d_referrence = NULL;
	int* d_matrix = NULL;

	hip_check(
		hipMalloc((void**) &d_referrence, sizeof(int) * size),
		"failed to allocate reference buffer");

	hip_check(
		hipMalloc((void**) &d_matrix, sizeof(int) * size),
		"failed to allocate matrix buffer");

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

		hip_check(
			hipMemcpyAsync(
				d_referrence,
				referrence,
				sizeof(int) * size,
				hipMemcpyHostToDevice,
				stream),
			"failed to enqueue reference H2D copy");

		hip_check(
			hipMemcpyAsync(
				d_matrix,
				input_itemsets,
				sizeof(int) * size,
				hipMemcpyHostToDevice,
				stream),
			"failed to enqueue matrix H2D copy");

		hip_check(hipStreamSynchronize(stream), "failed to synchronize H2D copies");

		record_region_end(0);

		const int block_width = (max_cols - 1) / BLOCK_SIZE;

		for (int i = 1; i <= block_width; i++) {
			record_region_start("setting_nw1_kernel_arguments");
			record_region_end(i);

			record_region_start("nw1_kernel_execution");

			hip_check(
				needle_launch_hip_shared_1(
					d_referrence,
					d_matrix,
					max_cols,
					penalty,
					i,
					block_width,
					stream),
				"failed to launch NW kernel 1");

			hip_check(hipStreamSynchronize(stream), "failed to synchronize NW kernel 1");

			record_region_end(i);
		}

		for (int i = block_width - 1; i >= 1; i--) {
			record_region_start("setting_nw2_kernel_arguments");
			record_region_end(i);

			record_region_start("nw2_kernel_execution");

			hip_check(
				needle_launch_hip_shared_2(
					d_referrence,
					d_matrix,
					max_cols,
					penalty,
					i,
					block_width,
					stream),
				"failed to launch NW kernel 2");

			hip_check(hipStreamSynchronize(stream), "failed to synchronize NW kernel 2");

			record_region_end(i);
		}

		record_region_start("device_side_d2h_copy");

		hip_check(
			hipMemcpyAsync(
				output_itemsets,
				d_matrix,
				sizeof(int) * size,
				hipMemcpyDeviceToHost,
				stream),
			"failed to enqueue D2H copy");

		hip_check(hipStreamSynchronize(stream), "failed to synchronize D2H copy");

		record_region_end(0);

		lsb_timing_repeats++;

		gettimeofday(&currentTime, NULL);
		timersub(&currentTime, &startTime, &elapsedTime);
	} while (elapsedTime.tv_sec < MIN_TIME_SEC);

	record_region_start("device_side_buffer_cleanup");

	hip_check(hipFree(d_referrence), "failed to free reference buffer");
	hip_check(hipFree(d_matrix), "failed to free matrix buffer");

	record_region_end(0);

	record_region_start("runtime_finalization");

	hip_check(hipStreamDestroy(stream), "failed to destroy HIP stream");

	record_region_end(0);

	LSB_Finalize();

	free(referrence);
	free(input_itemsets);
	free(output_itemsets);

	return EXIT_SUCCESS;
}
