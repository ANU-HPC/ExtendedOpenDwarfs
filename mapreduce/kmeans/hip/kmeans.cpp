#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "../../include/lsb.h"
}

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err__ = (call);                                               \
        if (err__ != hipSuccess) {                                               \
            fprintf(stderr, "HIP error %s:%d: %s\n",                            \
                    __FILE__, __LINE__, hipGetErrorString(err__));              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                       \
    } while (0)

extern "C" int setup(int argc, char** argv);

extern "C" __global__
void invert_mapping_kernel(
    const float* input,
    float* output,
    int npoints,
    int nfeatures);

extern "C" __global__
void kmeans_point_kernel(
    const float* features,
    int nfeatures,
    int npoints,
    int nclusters,
    int* membership,
    const float* clusters);

static size_t working_kernel_memory = 0;
static size_t cluster_invokations = 0;

static int localWorkSize = 256;
static int globalWorkSize = 0;

static int* membership_new = nullptr;
static float* block_new_centers = nullptr;

static float* feature_d = nullptr;
static float* feature_flipped_d = nullptr;
static int* membership_d = nullptr;
static float* clusters_d = nullptr;

extern "C"
void initCL()
{
    /* Preserve the legacy KMeans host interface name. */
    HIP_CHECK(hipFree(nullptr));
}

extern "C"
void allocateMemory(int npoints, int nfeatures, int nclusters, float** features)
{
    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();

    globalWorkSize = ((npoints + localWorkSize - 1) / localWorkSize) * localWorkSize;

    membership_new = (int*) malloc((size_t)npoints * sizeof(int));
    if (membership_new == nullptr) {
        fprintf(stderr, "Failed to allocate membership_new\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < npoints; i++) {
        membership_new[i] = -1;
    }

    block_new_centers = (float*) malloc((size_t)nclusters * nfeatures * sizeof(float));
    if (block_new_centers == nullptr) {
        fprintf(stderr, "Failed to allocate block_new_centers\n");
        exit(EXIT_FAILURE);
    }

    HIP_CHECK(hipMalloc((void**)&feature_flipped_d, (size_t)npoints * nfeatures * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&feature_d,         (size_t)npoints * nfeatures * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&membership_d,      (size_t)npoints * sizeof(int)));
    HIP_CHECK(hipMalloc((void**)&clusters_d,        (size_t)nclusters * nfeatures * sizeof(float)));

    working_kernel_memory += (size_t)npoints * nfeatures * sizeof(float);
    working_kernel_memory += (size_t)npoints * nfeatures * sizeof(float);
    working_kernel_memory += (size_t)npoints * sizeof(int);
    working_kernel_memory += (size_t)nclusters * nfeatures * sizeof(float);

    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "device_side_h2d_copy");
    LSB_Res();
    HIP_CHECK(hipMemcpy(
        feature_flipped_d,
        features[0],
        (size_t)npoints * nfeatures * sizeof(float),
        hipMemcpyHostToDevice));
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "kmeans_invert_kernel");
    LSB_Res();
    int blocks = (npoints + localWorkSize - 1) / localWorkSize;
    hipLaunchKernelGGL(
        invert_mapping_kernel,
        dim3(blocks),
        dim3(localWorkSize),
        0,
        0,
        feature_flipped_d,
        feature_d,
        npoints,
        nfeatures);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);
}

extern "C"
void deallocateMemory()
{
    free(membership_new);
    free(block_new_centers);

    HIP_CHECK(hipFree(feature_d));
    HIP_CHECK(hipFree(feature_flipped_d));
    HIP_CHECK(hipFree(membership_d));
    HIP_CHECK(hipFree(clusters_d));
}

extern "C"
int kmeansCuda(
    float** feature,
    int nfeatures,
    int npoints,
    int nclusters,
    int* membership,
    float** clusters,
    int* new_centers_len,
    float** new_centers)
{
    int delta = 0;

#ifndef PROFILE_OUTER_LOOP
    LSB_Set_Rparam_string("region", "device_side_h2d_copy");
    LSB_Res();
#endif

    HIP_CHECK(hipMemcpy(
        membership_d,
        membership_new,
        (size_t)npoints * sizeof(int),
        hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(
        clusters_d,
        clusters[0],
        (size_t)nclusters * nfeatures * sizeof(float),
        hipMemcpyHostToDevice));

#ifndef PROFILE_OUTER_LOOP
    LSB_Rec(0);
#endif

    if (cluster_invokations == 0) {
        printf("Working kernel memory: %fKiB\n", working_kernel_memory / 1024.0);
    }

#ifndef PROFILE_OUTER_LOOP
    LSB_Set_Rparam_string("region", "kmeans_kernel");
    LSB_Res();
#endif

    int blocks = (npoints + localWorkSize - 1) / localWorkSize;
    hipLaunchKernelGGL(
        kmeans_point_kernel,
        dim3(blocks),
        dim3(localWorkSize),
        0,
        0,
        feature_d,
        nfeatures,
        npoints,
        nclusters,
        membership_d,
        clusters_d);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

#ifndef PROFILE_OUTER_LOOP
    LSB_Rec(0);
#endif

#ifndef PROFILE_OUTER_LOOP
    LSB_Set_Rparam_string("region", "device_side_d2h_copy");
    LSB_Res();
#endif

    HIP_CHECK(hipMemcpy(
        membership_new,
        membership_d,
        (size_t)npoints * sizeof(int),
        hipMemcpyDeviceToHost));

#ifndef PROFILE_OUTER_LOOP
    LSB_Rec(0);
#endif

    for (int i = 0; i < npoints; i++) {
        int cluster_id = membership_new[i];

        new_centers_len[cluster_id]++;

        if (membership_new[i] != membership[i]) {
            delta++;
            membership[i] = membership_new[i];
        }

        for (int j = 0; j < nfeatures; j++) {
            new_centers[cluster_id][j] += feature[i][j];
        }
    }

    cluster_invokations++;
    return delta;
}

int main(int argc, char** argv)
{
    int rc = setup(argc, argv);
    HIP_CHECK(hipDeviceSynchronize());
    return rc;
}
