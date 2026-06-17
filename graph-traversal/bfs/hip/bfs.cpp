#include <hip/hip_runtime.h>

#include <errno.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "include/lsb.h"

#define AOCL_ALIGNMENT 64
#define MIN_TIME_SEC 2

struct Node
{
	int starting;
	int no_of_edges;
};

extern "C" hipError_t bfs_launch_kernel1_hip(
	const Node* d_graph_nodes,
	int* d_graph_edges,
	int* d_graph_mask,
	int* d_updating_graph_mask,
	int* d_graph_visited,
	int* d_cost,
	int no_of_nodes,
	size_t global_work_size,
	size_t local_work_size,
	hipStream_t stream);

extern "C" hipError_t bfs_launch_kernel2_hip(
	int* d_graph_mask,
	int* d_updating_graph_mask,
	int* d_graph_visited,
	int* d_over,
	int no_of_nodes,
	size_t global_work_size,
	size_t local_work_size,
	hipStream_t stream);

static const char* get_lsb_name()
{
	const char* lsb_name = getenv("ODW_LSB_NAME");

	if (lsb_name == NULL || lsb_name[0] == '\0') {
		lsb_name = "bfs";
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

static void* checked_memalign(size_t alignment, size_t size, const char* name)
{
	void* ptr = memalign(alignment, size);

	if (ptr == NULL) {
		fprintf(stderr, "Host allocation failed for %s (%zu bytes)\n", name, size);
		exit(EXIT_FAILURE);
	}

	return ptr;
}

static int checked_scanf(FILE* fp, const char* fmt, void* out, const char* what)
{
	int rc = fscanf(fp, fmt, out);

	if (rc != 1) {
		fprintf(stderr, "Failed reading %s\n", what);
		exit(EXIT_FAILURE);
	}

	return rc;
}

static void read_graph(
	const char* filename,
	Node** h_graph_nodes_out,
	int** h_graph_edges_out,
	int** h_graph_mask_out,
	int** h_updating_graph_mask_out,
	int** h_graph_visited_out,
	int** h_cost_out,
	unsigned int* no_of_nodes_out,
	unsigned int* edge_list_size_out,
	int* source_out)
{
	FILE* fp = fopen(filename, "r");

	if (fp == NULL) {
		fprintf(stderr, "Error reading graph file '%s': %s\n", filename, strerror(errno));
		exit(EXIT_FAILURE);
	}

	unsigned int no_of_nodes = 0;
	unsigned int edge_list_size = 0;
	int source = 0;

	checked_scanf(fp, "%u", &no_of_nodes, "node count");

	Node* h_graph_nodes = (Node*) checked_memalign(AOCL_ALIGNMENT, sizeof(Node) * no_of_nodes, "h_graph_nodes");
	int* h_graph_mask = (int*) checked_memalign(AOCL_ALIGNMENT, sizeof(int) * no_of_nodes, "h_graph_mask");
	int* h_updating_graph_mask = (int*) checked_memalign(AOCL_ALIGNMENT, sizeof(int) * no_of_nodes, "h_updating_graph_mask");
	int* h_graph_visited = (int*) checked_memalign(AOCL_ALIGNMENT, sizeof(int) * no_of_nodes, "h_graph_visited");

	for (unsigned int i = 0; i < no_of_nodes; i++) {
		int start = 0;
		int edgeno = 0;

		if (fscanf(fp, "%d %d", &start, &edgeno) != 2) {
			fprintf(stderr, "Failed reading node %u\n", i);
			exit(EXIT_FAILURE);
		}

		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i] = 0;
		h_updating_graph_mask[i] = 0;
		h_graph_visited[i] = 0;
	}

	checked_scanf(fp, "%d", &source, "source node");
	source = 0;

	h_graph_mask[source] = 1;
	h_graph_visited[source] = 1;

	checked_scanf(fp, "%u", &edge_list_size, "edge list size");

	int* h_graph_edges = (int*) checked_memalign(AOCL_ALIGNMENT, sizeof(int) * edge_list_size, "h_graph_edges");

	for (unsigned int i = 0; i < edge_list_size; i++) {
		int id = 0;
		int cost = 0;

		if (fscanf(fp, "%d", &id) != 1 || fscanf(fp, "%d", &cost) != 1) {
			fprintf(stderr, "Failed reading edge %u\n", i);
			exit(EXIT_FAILURE);
		}

		h_graph_edges[i] = id;
	}

	fclose(fp);

	int* h_cost = (int*) checked_memalign(AOCL_ALIGNMENT, sizeof(int) * no_of_nodes, "h_cost");

	for (unsigned int i = 0; i < no_of_nodes; i++) {
		h_cost[i] = -1;
	}
	h_cost[source] = 0;

	*h_graph_nodes_out = h_graph_nodes;
	*h_graph_edges_out = h_graph_edges;
	*h_graph_mask_out = h_graph_mask;
	*h_updating_graph_mask_out = h_updating_graph_mask;
	*h_graph_visited_out = h_graph_visited;
	*h_cost_out = h_cost;
	*no_of_nodes_out = no_of_nodes;
	*edge_list_size_out = edge_list_size;
	*source_out = source;
}

int main(int argc, char** argv)
{
	int hip_device = 0;
	parse_pre_separator_common_args(argc, argv, &hip_device);

	int app_argc = 0;
	char** app_argv = NULL;
	build_app_argv(argc, argv, &app_argc, &app_argv);

	LSB_Init(get_lsb_name(), 0);
	LSB_Set_Rparam_int("repeats_to_two_seconds", 0);

	record_region_start("runtime_initialization");

	hip_check(hipSetDevice(hip_device), "failed to select HIP device");
	hip_check(hipFree(0), "failed to initialize HIP runtime");

	hipStream_t stream;
	hip_check(hipStreamCreate(&stream), "failed to create HIP stream");

	record_region_end(0);

	if (app_argc < 2) {
		fprintf(stderr, "Usage: bfs <graph-file>\n");

		record_region_start("runtime_finalization");
		hip_check(hipStreamDestroy(stream), "failed to destroy HIP stream");
		record_region_end(0);

		LSB_Finalize();
		free(app_argv);
		return EXIT_FAILURE;
	}

	record_region_start("program_build");
	record_region_end(0);

	record_region_start("kernel_creation");
	record_region_end(0);

	printf("Reading File\n");

	Node* h_graph_nodes = NULL;
	int* h_graph_edges = NULL;
	int* h_graph_mask = NULL;
	int* h_updating_graph_mask = NULL;
	int* h_graph_visited = NULL;
	int* h_cost = NULL;

	unsigned int no_of_nodes = 0;
	unsigned int edge_list_size = 0;
	int source = 0;

	record_region_start("host_input_setup");

	read_graph(
		app_argv[1],
		&h_graph_nodes,
		&h_graph_edges,
		&h_graph_mask,
		&h_updating_graph_mask,
		&h_graph_visited,
		&h_cost,
		&no_of_nodes,
		&edge_list_size,
		&source);

	record_region_end(0);

	printf("Read File\n");

	record_region_start("device_side_buffer_setup");

	Node* d_graph_nodes = NULL;
	int* d_graph_edges = NULL;
	int* d_graph_mask = NULL;
	int* d_updating_graph_mask = NULL;
	int* d_graph_visited = NULL;
	int* d_cost = NULL;
	int* d_over = NULL;

	hip_check(hipMalloc((void**) &d_graph_nodes, sizeof(Node) * no_of_nodes), "failed to allocate d_graph_nodes");
	hip_check(hipMalloc((void**) &d_graph_edges, sizeof(int) * edge_list_size), "failed to allocate d_graph_edges");
	hip_check(hipMalloc((void**) &d_graph_mask, sizeof(int) * no_of_nodes), "failed to allocate d_graph_mask");
	hip_check(hipMalloc((void**) &d_updating_graph_mask, sizeof(int) * no_of_nodes), "failed to allocate d_updating_graph_mask");
	hip_check(hipMalloc((void**) &d_graph_visited, sizeof(int) * no_of_nodes), "failed to allocate d_graph_visited");
	hip_check(hipMalloc((void**) &d_cost, sizeof(int) * no_of_nodes), "failed to allocate d_cost");
	hip_check(hipMalloc((void**) &d_over, sizeof(int)), "failed to allocate d_over");

	record_region_end(0);

	printf("Working kernel memory: %fKiB\n",
		(sizeof(Node) * no_of_nodes +
		 sizeof(int) * edge_list_size +
		 sizeof(int) * no_of_nodes +
		 sizeof(int) * no_of_nodes +
		 sizeof(int) * no_of_nodes +
		 sizeof(int) * no_of_nodes +
		 sizeof(int)) / 1024.0);

	size_t local_work_size = no_of_nodes < 256 ? no_of_nodes : 256;
	size_t global_work_size =
		(no_of_nodes / local_work_size) * local_work_size +
		((no_of_nodes % local_work_size) == 0 ? 0 : local_work_size);

	printf("global_work_size=%zu local_work_size=%zu\n", global_work_size, local_work_size);

	int k = 0;
	int lsb_timing_repeats = 0;

	struct timeval start_time;
	struct timeval current_time;
	struct timeval elapsed_time;

	gettimeofday(&start_time, NULL);

	do {
		LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

		record_region_start("device_side_h2d_copy");

		hip_check(hipMemcpyAsync(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes, hipMemcpyHostToDevice, stream), "failed to copy d_graph_nodes");
		hip_check(hipMemcpyAsync(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size, hipMemcpyHostToDevice, stream), "failed to copy d_graph_edges");
		hip_check(hipMemcpyAsync(d_graph_mask, h_graph_mask, sizeof(int) * no_of_nodes, hipMemcpyHostToDevice, stream), "failed to copy d_graph_mask");
		hip_check(hipMemcpyAsync(d_updating_graph_mask, h_updating_graph_mask, sizeof(int) * no_of_nodes, hipMemcpyHostToDevice, stream), "failed to copy d_updating_graph_mask");
		hip_check(hipMemcpyAsync(d_graph_visited, h_graph_visited, sizeof(int) * no_of_nodes, hipMemcpyHostToDevice, stream), "failed to copy d_graph_visited");
		hip_check(hipMemcpyAsync(d_cost, h_cost, sizeof(int) * no_of_nodes, hipMemcpyHostToDevice, stream), "failed to copy d_cost");
		hip_check(hipStreamSynchronize(stream), "failed to synchronize initial H2D copies");

		record_region_end(lsb_timing_repeats);

		int stop = 0;

		do {
			stop = 0;

			record_region_start("device_side_h2d_copy");

			hip_check(hipMemcpyAsync(d_over, &stop, sizeof(int), hipMemcpyHostToDevice, stream), "failed to copy stop flag");
			hip_check(hipStreamSynchronize(stream), "failed to synchronize stop flag copy");

			record_region_end(k);

			record_region_start("kernel1_execution");

			hip_check(
				bfs_launch_kernel1_hip(
					d_graph_nodes,
					d_graph_edges,
					d_graph_mask,
					d_updating_graph_mask,
					d_graph_visited,
					d_cost,
					no_of_nodes,
					global_work_size,
					local_work_size,
					stream),
				"failed to launch kernel1");

			hip_check(hipStreamSynchronize(stream), "failed to synchronize kernel1");

			record_region_end(k);

			record_region_start("kernel2_execution");

			hip_check(
				bfs_launch_kernel2_hip(
					d_graph_mask,
					d_updating_graph_mask,
					d_graph_visited,
					d_over,
					no_of_nodes,
					global_work_size,
					local_work_size,
					stream),
				"failed to launch kernel2");

			hip_check(hipStreamSynchronize(stream), "failed to synchronize kernel2");

			record_region_end(k);

			record_region_start("device_side_d2h_copy");

			hip_check(hipMemcpyAsync(&stop, d_over, sizeof(int), hipMemcpyDeviceToHost, stream), "failed to read stop flag");
			hip_check(hipStreamSynchronize(stream), "failed to synchronize stop flag read");

			record_region_end(k);

			k++;
		} while (stop == 1);

		lsb_timing_repeats++;
		gettimeofday(&current_time, NULL);
		timersub(&current_time, &start_time, &elapsed_time);
	} while (elapsed_time.tv_sec < MIN_TIME_SEC);

	printf("Kernel Executed %d times\n", k);

	record_region_start("device_side_d2h_copy");

	hip_check(hipMemcpyAsync(h_cost, d_cost, sizeof(int) * no_of_nodes, hipMemcpyDeviceToHost, stream), "failed to read d_cost");
	hip_check(hipStreamSynchronize(stream), "failed to synchronize d_cost read");

	record_region_end(0);

	FILE* fpo = fopen("bfs_result.txt", "w");
	if (fpo != NULL) {
		for (unsigned int i = 0; i < no_of_nodes; i++) {
			fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
		}
		fclose(fpo);
		printf("Result stored in bfs_result.txt\n");
	}

	record_region_start("device_side_buffer_cleanup");

	hipFree(d_graph_nodes);
	hipFree(d_graph_edges);
	hipFree(d_graph_mask);
	hipFree(d_updating_graph_mask);
	hipFree(d_graph_visited);
	hipFree(d_cost);
	hipFree(d_over);

	record_region_end(0);

	record_region_start("runtime_finalization");

	hip_check(hipStreamDestroy(stream), "failed to destroy HIP stream");

	record_region_end(0);

	LSB_Finalize();

	free(h_graph_nodes);
	free(h_graph_edges);
	free(h_graph_mask);
	free(h_updating_graph_mask);
	free(h_graph_visited);
	free(h_cost);
	free(app_argv);

	return EXIT_SUCCESS;
}
