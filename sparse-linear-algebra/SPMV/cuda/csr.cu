#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

extern "C" {
#include "../../../include/lsb.h"
#include "../inc/common.h"
#include "../inc/sparse_formats.h"
}

#define MIN_TIME_SEC 2

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void csr_kernel(
    unsigned int num_rows,
    const unsigned int* Ap,
    const unsigned int* Aj,
    const float* Ax,
    const float* x,
    float* y)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float sum = y[row];

        const unsigned int row_start = Ap[row];
        const unsigned int row_end = Ap[row + 1];

        for (unsigned int jj = row_start; jj < row_end; jj++) {
            sum += Ax[jj] * x[Aj[jj]];
        }

        y[row] = sum;
    }
}

static const char* get_lsb_name(void)
{
    const char* lsb_name = getenv("ODW_LSB_NAME");
    if (lsb_name == NULL || lsb_name[0] == '\0') {
        lsb_name = "csr";
    }
    return lsb_name;
}

static void spmv_csr_cpu(const csr_matrix* csr, const float* x, const float* y, float* out)
{
    for (unsigned int row = 0; row < csr->num_rows; row++) {
        float sum = y[row];
        unsigned int row_start = csr->Ap[row];
        unsigned int row_end = csr->Ap[row + 1];

        for (unsigned int jj = row_start; jj < row_end; jj++) {
            sum += csr->Ax[jj] * x[csr->Aj[jj]];
        }

        out[row] = sum;
    }
}

static void float_array_comp(
    const float* control,
    const float* experimental,
    const unsigned int N,
    const unsigned int exec_num)
{
    for (unsigned int j = 0; j < N; j++) {
        float diff = experimental[j] - control[j];
        if (fabsf(diff) > 0.001f) {
            float perc = fabsf(diff / control[j]) * 100.0f;
            fprintf(stderr,
                    "Possible error on exec #%u, difference of %.3f (%.1f%% error) "
                    "[control=%.3f, experimental=%.3f] at row %u\n",
                    exec_num, diff, perc, control[j], experimental[j], j);
        }
    }
}

int main(int argc, char** argv)
{
    int verbosity = 0;
    int do_print = 0;
    int do_affirm = 0;
    unsigned int num_execs = 1;
    unsigned int num_matrices = 0;
    unsigned int i, j, k;
    size_t wg_size = 256;
    char* file_path = NULL;

    const char* usage =
        "Usage: %s -i <file_path> [-v] [-p] [-a] [-r <num_execs>] [-w <wg_size>]\n";

    const char* lsb_name = get_lsb_name();

    LSB_Init(lsb_name, 0);
    LSB_Set_Rparam_int("number_of_matrices", 0);
    LSB_Set_Rparam_int("workgroup_size", 0);
    LSB_Set_Rparam_int("execution_number", 0);
    LSB_Set_Rparam_int("repeats_to_two_seconds", 0);

    LSB_Set_Rparam_string("region", "runtime_initialization");
    LSB_Res();
    CUDA_CHECK(cudaFree(NULL));
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "host_side_setup");
    LSB_Res();

    for (int argi = 1; argi < argc; argi++) {
        if (strcmp(argv[argi], "-v") == 0 || strcmp(argv[argi], "--verbose") == 0) {
            verbosity++;
        } else if (strcmp(argv[argi], "-p") == 0 || strcmp(argv[argi], "--print") == 0) {
            do_print = 1;
        } else if (strcmp(argv[argi], "-a") == 0 || strcmp(argv[argi], "--affirm") == 0) {
            do_affirm = 1;
        } else if ((strcmp(argv[argi], "-i") == 0 || strcmp(argv[argi], "--input_file") == 0) && argi + 1 < argc) {
            file_path = argv[++argi];
            printf("Reading Input from '%s'\n", file_path);
        } else if ((strcmp(argv[argi], "-r") == 0 || strcmp(argv[argi], "--repeat") == 0) && argi + 1 < argc) {
            num_execs = (unsigned int)atoi(argv[++argi]);
            printf("Executing %u times\n", num_execs);
        } else if ((strcmp(argv[argi], "-w") == 0 || strcmp(argv[argi], "--wg_size") == 0) && argi + 1 < argc) {
            wg_size = (size_t)atoi(argv[++argi]);
        } else {
            fprintf(stderr, usage, argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (!file_path) {
        fprintf(stderr, "-i Option must be supplied\n\n");
        fprintf(stderr, usage, argv[0]);
        exit(EXIT_FAILURE);
    }

    csr_matrix* csr = read_csr(&num_matrices, file_path);

    if (do_print) {
        print_csr_arr_std(csr, num_matrices, stdout);
    } else if (verbosity) {
        printf("Number of input matrices: %u\nMatrix 0 Metadata:\n", num_matrices);
        print_csr_metadata(&csr[0], stdout);
    }

    float** device_out = (float**)calloc(num_matrices, sizeof(float*));
    float* x_host = NULL;
    float* y_host = NULL;
    float* host_out = NULL;
    unsigned int max_row_len = 0;
    unsigned int max_col_len = 0;

    for (i = 0; i < num_matrices; i++) {
        device_out[i] = (float*)float_new_array(
            csr[i].num_rows,
            "csr.cuda() - Heap Overflow! Cannot Allocate Space for device_out");

        if (max_row_len < csr[i].num_rows) {
            max_row_len = csr[i].num_rows;
            y_host = (float*)float_array_realloc(
                y_host,
                csr[i].num_rows,
                "csr.cuda() - Heap Overflow! Cannot Allocate Space for y_host");

            if (do_affirm) {
                host_out = (float*)realloc(host_out, sizeof(float) * max_row_len);
                check(host_out != NULL, "csr.cuda() - Heap Overflow! Cannot Allocate Space for host_out");
            }
        }

        if (max_col_len < csr[i].num_cols) {
            max_col_len = csr[i].num_cols;
            x_host = (float*)float_array_realloc(
                x_host,
                csr[i].num_cols,
                "csr.cuda() - Heap Overflow! Cannot Allocate Space for x_host");
        }
    }

    for (i = 0; i < max_col_len; i++) {
        x_host[i] = rand() / (RAND_MAX + 1.0f);
        if (do_print) printf("x[%u] = %6.2f\n", i, x_host[i]);
    }

    for (i = 0; i < max_row_len; i++) {
        y_host[i] = rand() / (RAND_MAX + 2.0f);
        if (do_print) printf("y[%u] = %6.2f\n", i, y_host[i]);
    }

    if (verbosity) printf("Input Generated.\n");

    LSB_Rec(0);

    LSB_Set_Rparam_int("number_of_matrices", (int)num_matrices);
    LSB_Set_Rparam_int("workgroup_size", (int)wg_size);

    unsigned int** csr_ap_d = (unsigned int**)calloc(num_matrices, sizeof(unsigned int*));
    unsigned int** csr_aj_d = (unsigned int**)calloc(num_matrices, sizeof(unsigned int*));
    float** csr_ax_d = (float**)calloc(num_matrices, sizeof(float*));
    float** x_d = (float**)calloc(num_matrices, sizeof(float*));
    float** y_d = (float**)calloc(num_matrices, sizeof(float*));

    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();

    for (k = 0; k < num_matrices; k++) {
        CUDA_CHECK(cudaMalloc((void**)&csr_ap_d[k], sizeof(unsigned int) * (csr[k].num_rows + 1)));
        CUDA_CHECK(cudaMalloc((void**)&csr_aj_d[k], sizeof(unsigned int) * csr[k].num_nonzeros));
        CUDA_CHECK(cudaMalloc((void**)&csr_ax_d[k], sizeof(float) * csr[k].num_nonzeros));
        CUDA_CHECK(cudaMalloc((void**)&x_d[k], sizeof(float) * csr[k].num_cols));
        CUDA_CHECK(cudaMalloc((void**)&y_d[k], sizeof(float) * csr[k].num_rows));
    }

    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "kernel_creation");
    LSB_Res();
    LSB_Rec(0);

    printf("Executing with WG Size #1 of 1: %zu...\n", wg_size);

    for (i = 0; i < num_execs; i++) {
        if (verbosity) printf("Beginning execution #%u of %u\n", i + 1, num_execs);
        LSB_Set_Rparam_int("execution_number", (int)i);

        for (k = 0; k < num_matrices; k++) {
            printf("Working kernel memory: %fKiB\n",
                   (sizeof(unsigned int) * (csr[k].num_rows + 1) +
                    sizeof(unsigned int) * csr[k].num_nonzeros +
                    sizeof(float) * csr[k].num_nonzeros +
                    sizeof(float) * csr[k].num_cols +
                    sizeof(float) * csr[k].num_rows) / 1024.0);

            int lsb_timing_repeats = 0;
            struct timeval startTime, currentTime, elapsedTime;
            gettimeofday(&startTime, NULL);

            do {
                LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

                LSB_Set_Rparam_string("region", "device_side_h2d_copy");
                LSB_Res();

                CUDA_CHECK(cudaMemcpy(csr_ap_d[k], csr[k].Ap,
                                      sizeof(unsigned int) * (csr[k].num_rows + 1),
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(csr_aj_d[k], csr[k].Aj,
                                      sizeof(unsigned int) * csr[k].num_nonzeros,
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(csr_ax_d[k], csr[k].Ax,
                                      sizeof(float) * csr[k].num_nonzeros,
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(x_d[k], x_host,
                                      sizeof(float) * csr[k].num_cols,
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(y_d[k], y_host,
                                      sizeof(float) * csr[k].num_rows,
                                      cudaMemcpyHostToDevice));

                LSB_Rec((int)i);

                LSB_Set_Rparam_string("region", "setting_kernel_arguments");
                LSB_Res();
                unsigned int global_size = csr[k].num_rows;
                unsigned int blocks = (global_size + (unsigned int)wg_size - 1U) / (unsigned int)wg_size;
                LSB_Rec((int)i);

                LSB_Set_Rparam_string("region", "spmv_kernel");
                LSB_Res();

                csr_kernel<<<blocks, (unsigned int)wg_size>>>(
                    global_size,
                    csr_ap_d[k],
                    csr_aj_d[k],
                    csr_ax_d[k],
                    x_d[k],
                    y_d[k]);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                LSB_Rec((int)i);

                LSB_Set_Rparam_string("region", "device_side_d2h_copy");
                LSB_Res();

                CUDA_CHECK(cudaMemcpy(device_out[k], y_d[k],
                                      sizeof(float) * csr[k].num_rows,
                                      cudaMemcpyDeviceToHost));

                LSB_Rec((int)i);

                lsb_timing_repeats++;
                gettimeofday(&currentTime, NULL);
                timersub(&currentTime, &startTime, &elapsedTime);
            } while (elapsedTime.tv_sec < MIN_TIME_SEC);
        }

        if (do_print) {
            for (k = 0; k < num_matrices; k++) {
                printf("\nMatrix #%u of %u:\n", k + 1, num_matrices);
                for (j = 0; j < csr[k].num_rows; j++) {
                    printf("\trow: %u\toutput: %6.2f \n", j, device_out[k][j]);
                }
            }
        }

        if (do_affirm) {
            if (verbosity) printf("Validating results with serial C code on CPU...\n");
            for (k = 0; k < num_matrices; k++) {
                spmv_csr_cpu(&csr[k], x_host, y_host, host_out);
                float_array_comp(host_out, device_out[k], csr[k].num_rows, i + 1);
            }
        }
    }

    LSB_Set_Rparam_string("region", "device_side_buffer_cleanup");
    LSB_Res();

    for (k = 0; k < num_matrices; k++) {
        CUDA_CHECK(cudaFree(csr_ap_d[k]));
        CUDA_CHECK(cudaFree(csr_aj_d[k]));
        CUDA_CHECK(cudaFree(csr_ax_d[k]));
        CUDA_CHECK(cudaFree(x_d[k]));
        CUDA_CHECK(cudaFree(y_d[k]));
    }

    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "runtime_finalization");
    LSB_Res();
    CUDA_CHECK(cudaDeviceSynchronize());
    LSB_Rec(0);

    LSB_Finalize();

    double csr_checksum = 0.0;
    double csr_abs_checksum = 0.0;
    unsigned long long csr_values = 0ULL;

    for (k = 0; k < num_matrices; k++) {
        for (j = 0; j < csr[k].num_rows; j++) {
            double v = (double)device_out[k][j];
            csr_checksum += v * (double)(csr_values + 1ULL);
            csr_abs_checksum += fabs(v);
            csr_values++;
        }
    }

    printf("CSR_CHECKSUM matrices=%u values=%llu value=%0.17e abs=%0.17e\n",
           num_matrices,
           csr_values,
           csr_checksum,
           csr_abs_checksum);

    for (k = 0; k < num_matrices; k++) {
        free(device_out[k]);
    }

    free(device_out);
    free(csr_ap_d);
    free(csr_aj_d);
    free(csr_ax_d);
    free(x_d);
    free(y_d);
    free(x_host);
    free(y_host);
    if (do_affirm) free(host_out);
    free_csr(csr, num_matrices);

    return 0;
}
