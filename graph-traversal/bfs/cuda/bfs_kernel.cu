#include <cuda_runtime.h>

struct Node
{
	int starting;
	int no_of_edges;
};

extern "C" __global__
void bfs_kernel1(
	const Node* g_graph_nodes,
	int* g_graph_edges,
	int* g_graph_mask,
	int* g_updating_graph_mask,
	int* g_graph_visited,
	int* g_cost,
	int no_of_nodes)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < (unsigned int) no_of_nodes && g_graph_mask[tid] != 0) {
		g_graph_mask[tid] = 0;

		int max_edge = g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting;

		for (int i = g_graph_nodes[tid].starting; i < max_edge; i++) {
			int id = g_graph_edges[i];

			if (!g_graph_visited[id]) {
				g_cost[id] = g_cost[tid] + 1;
				g_updating_graph_mask[id] = 1;
			}
		}
	}
}

extern "C" __global__
void bfs_kernel2(
	int* g_graph_mask,
	int* g_updating_graph_mask,
	int* g_graph_visited,
	int* g_over,
	int no_of_nodes)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < (unsigned int) no_of_nodes && g_updating_graph_mask[tid] == 1) {
		g_graph_mask[tid] = 1;
		g_graph_visited[tid] = 1;
		*g_over = 1;
		g_updating_graph_mask[tid] = 0;
	}
}

extern "C" cudaError_t bfs_launch_kernel1_cuda(
	const Node* d_graph_nodes,
	int* d_graph_edges,
	int* d_graph_mask,
	int* d_updating_graph_mask,
	int* d_graph_visited,
	int* d_cost,
	int no_of_nodes,
	size_t global_work_size,
	size_t local_work_size,
	cudaStream_t stream)
{
	bfs_kernel1<<<
		dim3(global_work_size / local_work_size),
		dim3(local_work_size),
		0,
		stream>>>(
			d_graph_nodes,
			d_graph_edges,
			d_graph_mask,
			d_updating_graph_mask,
			d_graph_visited,
			d_cost,
			no_of_nodes);

	return cudaGetLastError();
}

extern "C" cudaError_t bfs_launch_kernel2_cuda(
	int* d_graph_mask,
	int* d_updating_graph_mask,
	int* d_graph_visited,
	int* d_over,
	int no_of_nodes,
	size_t global_work_size,
	size_t local_work_size,
	cudaStream_t stream)
{
	bfs_kernel2<<<
		dim3(global_work_size / local_work_size),
		dim3(local_work_size),
		0,
		stream>>>(
			d_graph_mask,
			d_updating_graph_mask,
			d_graph_visited,
			d_over,
			no_of_nodes);

	return cudaGetLastError();
}
