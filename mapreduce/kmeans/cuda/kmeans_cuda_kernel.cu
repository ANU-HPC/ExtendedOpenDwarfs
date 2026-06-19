#include <float.h>

extern "C" __global__
void invert_mapping_kernel(
    const float* input,
    float* output,
    int npoints,
    int nfeatures)
{
    int point_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (point_id < npoints) {
        for (int i = 0; i < nfeatures; i++) {
            output[point_id + npoints * i] = input[point_id * nfeatures + i];
        }
    }
}

extern "C" __global__
void kmeans_point_kernel(
    const float* features,
    int nfeatures,
    int npoints,
    int nclusters,
    int* membership,
    const float* clusters)
{
    int point_id = blockIdx.x * blockDim.x + threadIdx.x;
    int index = -1;

    if (point_id < npoints) {
        float min_dist = FLT_MAX;

        for (int i = 0; i < nclusters; i++) {
            int cluster_base_index = i * nfeatures;
            float dist = 0.0f;

            for (int j = 0; j < nfeatures; j++) {
                int addr = point_id + j * npoints;
                float diff = features[addr] - clusters[cluster_base_index + j];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                index = i;
            }
        }

        membership[point_id] = index;
    }
}
