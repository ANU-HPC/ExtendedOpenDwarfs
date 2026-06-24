#include <hip/hip_runtime.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../../../include/lsb.h"

#define BLOCK_SIZE 16
#define MIN_TIME_SEC 2

#define HIP_CHECK(call) do { \
    hipError_t err__ = (call); \
    if (err__ != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err__)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static void usage(const char* argv0)
{
    fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lambda> <no. of iter>\n", argv0);
    exit(EXIT_FAILURE);
}

static const char* get_lsb_name(void)
{
    const char* lsb_name = getenv("ODW_LSB_NAME");
    if (lsb_name == NULL || lsb_name[0] == '\0') {
        lsb_name = "srad";
    }
    return lsb_name;
}

static void random_matrix(float* I, int rows, int cols)
{
    srand(7);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            I[i * cols + j] = rand() / (float)RAND_MAX;
        }
    }
}

__global__ void srad_cuda_1(
    float* E_C,
    float* W_C,
    float* N_C,
    float* S_C,
    float* J_cuda,
    float* C_cuda,
    int cols,
    int rows,
    float q0sqr)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
    int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_w = cols * BLOCK_SIZE * by + cols * ty - 1 + BLOCK_SIZE * bx;
    int index_e = cols * BLOCK_SIZE * by + cols * ty + BLOCK_SIZE + BLOCK_SIZE * bx;

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    if (by == 0) {
        index_n = BLOCK_SIZE * bx + tx;
    } else if (by == gridDim.y - 1) {
        index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx;
    }

    if (bx == 0) {
        index_w = cols * BLOCK_SIZE * by + cols * ty;
    } else if (bx == gridDim.x - 1) {
        index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE - 1;
    }

    int index_yp1_x = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * (ty + 1) + tx;
    int index_ym1_x = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * (ty - 1) + tx;
    int index_y_xp1 = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + (tx + 1);
    int index_y_xm1 = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + (tx - 1);

    jc = J_cuda[index];

    if (ty == 0 && tx == 0) {
        n = J_cuda[index_n] - jc;
        s = J_cuda[index_yp1_x] - jc;
        w = J_cuda[index_w] - jc;
        e = J_cuda[index_y_xp1] - jc;
    } else if (ty == 0 && tx == BLOCK_SIZE - 1) {
        n = J_cuda[index_n] - jc;
        s = J_cuda[index_yp1_x] - jc;
        w = J_cuda[index_y_xm1] - jc;
        e = J_cuda[index_e] - jc;
    } else if (ty == BLOCK_SIZE - 1 && tx == BLOCK_SIZE - 1) {
        n = J_cuda[index_ym1_x] - jc;
        s = J_cuda[index_s] - jc;
        w = J_cuda[index_y_xm1] - jc;
        e = J_cuda[index_e] - jc;
    } else if (ty == BLOCK_SIZE - 1 && tx == 0) {
        n = J_cuda[index_ym1_x] - jc;
        s = J_cuda[index_s] - jc;
        w = J_cuda[index_w] - jc;
        e = J_cuda[index_y_xp1] - jc;
    } else if (ty == 0) {
        n = J_cuda[index_n] - jc;
        s = J_cuda[index_yp1_x] - jc;
        w = J_cuda[index_y_xm1] - jc;
        e = J_cuda[index_y_xp1] - jc;
    } else if (tx == BLOCK_SIZE - 1) {
        n = J_cuda[index_ym1_x] - jc;
        s = J_cuda[index_yp1_x] - jc;
        w = J_cuda[index_y_xm1] - jc;
        e = J_cuda[index_e] - jc;
    } else if (ty == BLOCK_SIZE - 1) {
        n = J_cuda[index_ym1_x] - jc;
        s = J_cuda[index_s] - jc;
        w = J_cuda[index_y_xm1] - jc;
        e = J_cuda[index_y_xp1] - jc;
    } else if (tx == 0) {
        n = J_cuda[index_ym1_x] - jc;
        s = J_cuda[index_yp1_x] - jc;
        w = J_cuda[index_w] - jc;
        e = J_cuda[index_y_xp1] - jc;
    } else {
        n = J_cuda[index_ym1_x] - jc;
        s = J_cuda[index_yp1_x] - jc;
        w = J_cuda[index_y_xm1] - jc;
        e = J_cuda[index_y_xp1] - jc;
    }

    g2 = (n * n + s * s + w * w + e * e) / (jc * jc);
    l = (n + s + w + e) / jc;

    num = (0.5f * g2) - ((1.0f / 16.0f) * (l * l));
    den = 1.0f + (0.25f * l);
    qsqr = num / (den * den);

    den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
    c = 1.0f / (1.0f + den);

    if (c < 0.0f) {
        C_cuda[index] = 0.0f;
    } else if (c > 1.0f) {
        C_cuda[index] = 1.0f;
    } else {
        C_cuda[index] = c;
    }

    E_C[index] = e;
    W_C[index] = w;
    S_C[index] = s;
    N_C[index] = n;
}

__global__ void srad_cuda_2(
    float* E_C,
    float* W_C,
    float* N_C,
    float* S_C,
    float* J_cuda,
    float* C_cuda,
    int cols,
    int rows,
    float lambda,
    float q0sqr)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

    int index_yp1_x = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * (ty + 1) + tx;
    int index_y_xp1 = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + (tx + 1);

    float cc = C_cuda[index];
    float cn = cc;
    float cw = cc;
    float cs, ce;

    if (ty == BLOCK_SIZE - 1 && tx == BLOCK_SIZE - 1) {
        if (by == gridDim.y - 1)
            cs = C_cuda[index];
        else
            cs = C_cuda[index_s];

        if (bx == gridDim.x - 1)
            ce = C_cuda[index];
        else
            ce = C_cuda[index_e];
    } else if (tx == BLOCK_SIZE - 1) {
        cs = C_cuda[index_yp1_x];

        if (bx == gridDim.x - 1)
            ce = C_cuda[index];
        else
            ce = C_cuda[index_e];
    } else if (ty == BLOCK_SIZE - 1) {
        if (by == gridDim.y - 1)
            cs = C_cuda[index];
        else
            cs = C_cuda[index_s];

        ce = C_cuda[index_y_xp1];
    } else {
        cs = C_cuda[index_yp1_x];
        ce = C_cuda[index_y_xp1];
    }

    float d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

    J_cuda[index] = J_cuda[index] + 0.25f * lambda * d_sum;
}

int main(int argc, char** argv)
{
    if (argc != 9) {
        usage(argv[0]);
    }

    int rows = atoi(argv[1]);
    int cols = atoi(argv[2]);
    unsigned int r1 = (unsigned int)atoi(argv[3]);
    unsigned int r2 = (unsigned int)atoi(argv[4]);
    unsigned int c1 = (unsigned int)atoi(argv[5]);
    unsigned int c2 = (unsigned int)atoi(argv[6]);
    float lambda = (float)atof(argv[7]);
    int niter = atoi(argv[8]);

    if ((rows % BLOCK_SIZE) != 0 || (cols % BLOCK_SIZE) != 0) {
        fprintf(stderr, "rows and cols must be multiples of %d\n", BLOCK_SIZE);
        return EXIT_FAILURE;
    }

    if (r1 > r2 || c1 > c2 || r2 >= (unsigned int)rows || c2 >= (unsigned int)cols) {
        fprintf(stderr,
                "Invalid SRAD ROI: rows=%d cols=%d r1=%u r2=%u c1=%u c2=%u\n",
                rows, cols, r1, r2, c1, c2);
        return EXIT_FAILURE;
    }

    int size_I = rows * cols;
    int size_R = (int)((r2 - r1 + 1) * (c2 - c1 + 1));

    LSB_Init(get_lsb_name(), 0);
    LSB_Set_Rparam_int("repeats_to_two_seconds", 0);
    LSB_Set_Rparam_int("rows", rows);
    LSB_Set_Rparam_int("cols", cols);
    LSB_Set_Rparam_int("niter", niter);

    LSB_Set_Rparam_string("region", "runtime_initialization");
    LSB_Res();
    HIP_CHECK(hipFree(NULL));
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "host_side_setup");
    LSB_Res();

    float* I = (float*)malloc(sizeof(float) * size_I);
    float* J = (float*)malloc(sizeof(float) * size_I);
    float* O = (float*)malloc(sizeof(float) * size_I);

    if (!I || !J || !O) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    printf("Randomizing the input matrix\n");
    random_matrix(I, rows, cols);

    for (int k = 0; k < size_I; k++) {
        J[k] = expf(I[k]);
        O[k] = 0.0f;
    }

    LSB_Rec(0);

    float* J_cuda = NULL;
    float* C_cuda = NULL;
    float* E_C = NULL;
    float* W_C = NULL;
    float* N_C = NULL;
    float* S_C = NULL;

    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();

    HIP_CHECK(hipMalloc((void**)&J_cuda, sizeof(float) * (size_I + 1)));
    HIP_CHECK(hipMalloc((void**)&C_cuda, sizeof(float) * size_I));
    HIP_CHECK(hipMalloc((void**)&E_C, sizeof(float) * size_I));
    HIP_CHECK(hipMalloc((void**)&W_C, sizeof(float) * size_I));
    HIP_CHECK(hipMalloc((void**)&N_C, sizeof(float) * size_I));
    HIP_CHECK(hipMalloc((void**)&S_C, sizeof(float) * size_I));

    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "kernel_creation");
    LSB_Res();
    LSB_Rec(0);

    printf("Start the SRAD main loop\n");

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(cols / BLOCK_SIZE, rows / BLOCK_SIZE);

    for (int iter = 0; iter < niter; iter++) {
        float sum = 0.0f;
        float sum2 = 0.0f;

        for (unsigned int i = r1; i <= r2; i++) {
            for (unsigned int j = c1; j <= c2; j++) {
                float tmp = J[i * cols + j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }

        float meanROI = sum / (float)size_R;
        float varROI = (sum2 / (float)size_R) - meanROI * meanROI;
        float q0sqr = varROI / (meanROI * meanROI);

        printf("Working kernel memory: %fKiB\n",
               (sizeof(float) * size_I * 6) / 1024.0);

        int lsb_timing_repeats = 0;
        struct timeval startTime, currentTime, elapsedTime;
        gettimeofday(&startTime, NULL);

        do {
            LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

            LSB_Set_Rparam_string("region", "device_side_h2d_copy");
            LSB_Res();

            HIP_CHECK(hipMemcpy(J_cuda, J, sizeof(float) * size_I, hipMemcpyHostToDevice));

            LSB_Rec(0);

            LSB_Set_Rparam_string("region", "setting_srad1_kernel_arguments");
            LSB_Res();
            LSB_Rec(0);

            LSB_Set_Rparam_string("region", "srad1_kernel");
            LSB_Res();

            srad_cuda_1<<<grid, block>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            LSB_Rec(0);

            LSB_Set_Rparam_string("region", "setting_srad2_kernel_arguments");
            LSB_Res();
            LSB_Rec(0);

            LSB_Set_Rparam_string("region", "srad2_kernel");
            LSB_Res();

            srad_cuda_2<<<grid, block>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            LSB_Rec(0);

            LSB_Set_Rparam_string("region", "device_side_d2h_copy");
            LSB_Res();

            HIP_CHECK(hipMemcpy(O, J_cuda, sizeof(float) * size_I, hipMemcpyDeviceToHost));

            LSB_Rec(0);

            lsb_timing_repeats++;
            gettimeofday(&currentTime, NULL);
            timersub(&currentTime, &startTime, &elapsedTime);
        } while (elapsedTime.tv_sec < MIN_TIME_SEC);
    }

    printf("Computation Done\n");

    LSB_Set_Rparam_string("region", "device_side_buffer_cleanup");
    LSB_Res();

    HIP_CHECK(hipFree(C_cuda));
    HIP_CHECK(hipFree(J_cuda));
    HIP_CHECK(hipFree(E_C));
    HIP_CHECK(hipFree(W_C));
    HIP_CHECK(hipFree(N_C));
    HIP_CHECK(hipFree(S_C));

    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "runtime_finalization");
    LSB_Res();
    HIP_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);

    LSB_Finalize();

    double srad_checksum = 0.0;
    double srad_abs_checksum = 0.0;
    int srad_finite_values = 0;
    int srad_nan_values = 0;
    int srad_inf_values = 0;

    for (int i = 0; i < size_I; i++) {
        double v = (double)O[i];

        if (isnan(v)) {
            srad_nan_values++;
            continue;
        }

        if (isinf(v)) {
            srad_inf_values++;
            continue;
        }

        srad_checksum += v * (double)(i + 1);
        srad_abs_checksum += fabs(v);
        srad_finite_values++;
    }

    printf("SRAD_CHECKSUM rows=%d cols=%d values=%d finite=%d nan=%d inf=%d value=%0.17e abs=%0.17e\n",
           rows,
           cols,
           size_I,
           srad_finite_values,
           srad_nan_values,
           srad_inf_values,
           srad_checksum,
           srad_abs_checksum);

    free(I);
    free(J);
    free(O);

    return EXIT_SUCCESS;
}
