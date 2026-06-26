#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include "lsb.h"

#define MIN_TIME_SEC 2
#define THREADS 256

#define FFT_PI       3.14159265359f
#define FFT_SQRT_1_2 0.707106781187f
#define FFT_C8       0.923879532511f
#define FFT_S8       0.382683432365f

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d while running %s: %s\n", \
                __FILE__, __LINE__, #call, cudaGetErrorString(err__)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__device__ __forceinline__ float2 cadd(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 csub(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float2 mul(float2 a, float2 b)
{
    return make_float2(fmaf(a.x, b.x, -a.y * b.y),
                       fmaf(a.x, b.y,  a.y * b.x));
}

__device__ __forceinline__ float2 twiddle_1_2(float2 a)
{
    return make_float2(a.y, -a.x);
}

__device__ __forceinline__ float2 twiddle_1_4(float2 a)
{
    return make_float2(FFT_SQRT_1_2 * (a.x + a.y),
                       FFT_SQRT_1_2 * (-a.x + a.y));
}

__device__ __forceinline__ float2 twiddle_3_4(float2 a)
{
    return make_float2(FFT_SQRT_1_2 * (-a.x + a.y),
                       FFT_SQRT_1_2 * (-a.x - a.y));
}

__device__ __forceinline__ float2 twiddle_1_8(float2 a)
{
    return mul(a, make_float2(FFT_C8, -FFT_S8));
}

__device__ __forceinline__ float2 twiddle_3_8(float2 a)
{
    return mul(a, make_float2(FFT_S8, -FFT_C8));
}

__device__ __forceinline__ float2 twiddle_5_8(float2 a)
{
    return mul(a, make_float2(-FFT_S8, -FFT_C8));
}

__device__ __forceinline__ float2 twiddle_7_8(float2 a)
{
    return mul(a, make_float2(-FFT_C8, -FFT_S8));
}

__device__ __forceinline__ float2 twiddle(float2 a, int k, float alpha)
{
    float sn, cs;
    sincosf((float)k * alpha, &sn, &cs);
    return mul(a, make_float2(cs, sn));
}

#define DFT2(a,b) do { float2 tmp__ = csub((a), (b)); (a) = cadd((a), (b)); (b) = tmp__; } while (0)

#define DFT4(a,b,c,d) do { \
    DFT2((a), (c)); \
    DFT2((b), (d)); \
    (d) = twiddle_1_2((d)); \
    DFT2((a), (b)); \
    DFT2((c), (d)); \
} while (0)

#define DFT8(a,b,c,d,aa,bb,cc,dd) do { \
    DFT2((a), (aa)); DFT2((b), (bb)); DFT2((c), (cc)); DFT2((d), (dd)); \
    (bb) = twiddle_1_4((bb)); (cc) = twiddle_1_2((cc)); (dd) = twiddle_3_4((dd)); \
    DFT4((a), (b), (c), (d)); DFT4((aa), (bb), (cc), (dd)); \
} while (0)

#define DFT16(a0,a1,a2,a3,a4,a5,a6,a7,b0,b1,b2,b3,b4,b5,b6,b7) do { \
    DFT2((a0), (b0)); DFT2((a1), (b1)); DFT2((a2), (b2)); DFT2((a3), (b3)); \
    DFT2((a4), (b4)); DFT2((a5), (b5)); DFT2((a6), (b6)); DFT2((a7), (b7)); \
    (b1) = twiddle_1_8((b1)); (b2) = twiddle_1_4((b2)); (b3) = twiddle_3_8((b3)); \
    (b4) = twiddle_1_2((b4)); (b5) = twiddle_5_8((b5)); (b6) = twiddle_3_4((b6)); (b7) = twiddle_7_8((b7)); \
    DFT8((a0), (a1), (a2), (a3), (a4), (a5), (a6), (a7)); \
    DFT8((b0), (b1), (b2), (b3), (b4), (b5), (b6), (b7)); \
} while (0)

__global__ void fft_radix2_kernel(const float2* x, float2* y, int t, int p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t) return;

    int k = i & (p - 1);
    int j = ((i - k) << 1) + k;
    float alpha = -FFT_PI * (float)k / (float)p;

    float2 u0 = x[i];
    float2 u1 = twiddle(x[i + t], 1, alpha);

    DFT2(u0, u1);

    y[j] = u0;
    y[j + p] = u1;
}

__global__ void fft_radix4_kernel(const float2* x, float2* y, int t, int p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t) return;

    int k = i & (p - 1);
    int j = ((i - k) << 2) + k;
    float alpha = -FFT_PI * (float)k / (float)(2 * p);

    float2 u0 = x[i];
    float2 u1 = twiddle(x[i + t], 1, alpha);
    float2 u2 = twiddle(x[i + 2 * t], 2, alpha);
    float2 u3 = twiddle(x[i + 3 * t], 3, alpha);

    DFT4(u0, u1, u2, u3);

    y[j] = u0;
    y[j + p] = u2;
    y[j + 2 * p] = u1;
    y[j + 3 * p] = u3;
}

__global__ void fft_radix8_kernel(const float2* x, float2* y, int t, int p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t) return;

    int k = i & (p - 1);
    int j = ((i - k) << 3) + k;
    float alpha = -FFT_PI * (float)k / (float)(4 * p);

    float2 u0 = x[i];
    float2 u1 = twiddle(x[i + t], 1, alpha);
    float2 u2 = twiddle(x[i + 2 * t], 2, alpha);
    float2 u3 = twiddle(x[i + 3 * t], 3, alpha);
    float2 u4 = twiddle(x[i + 4 * t], 4, alpha);
    float2 u5 = twiddle(x[i + 5 * t], 5, alpha);
    float2 u6 = twiddle(x[i + 6 * t], 6, alpha);
    float2 u7 = twiddle(x[i + 7 * t], 7, alpha);

    DFT8(u0, u1, u2, u3, u4, u5, u6, u7);

    y[j] = u0;
    y[j + p] = u4;
    y[j + 2 * p] = u2;
    y[j + 3 * p] = u6;
    y[j + 4 * p] = u1;
    y[j + 5 * p] = u5;
    y[j + 6 * p] = u3;
    y[j + 7 * p] = u7;
}

__global__ void fft_radix16_kernel(const float2* x, float2* y, int t, int p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t) return;

    int k = i & (p - 1);
    int j = ((i - k) << 4) + k;
    float alpha = -FFT_PI * (float)k / (float)(8 * p);

    float2 u0 = x[i];
    float2 u1 = twiddle(x[i + t], 1, alpha);
    float2 u2 = twiddle(x[i + 2 * t], 2, alpha);
    float2 u3 = twiddle(x[i + 3 * t], 3, alpha);
    float2 u4 = twiddle(x[i + 4 * t], 4, alpha);
    float2 u5 = twiddle(x[i + 5 * t], 5, alpha);
    float2 u6 = twiddle(x[i + 6 * t], 6, alpha);
    float2 u7 = twiddle(x[i + 7 * t], 7, alpha);
    float2 u8 = twiddle(x[i + 8 * t], 8, alpha);
    float2 u9 = twiddle(x[i + 9 * t], 9, alpha);
    float2 u10 = twiddle(x[i + 10 * t], 10, alpha);
    float2 u11 = twiddle(x[i + 11 * t], 11, alpha);
    float2 u12 = twiddle(x[i + 12 * t], 12, alpha);
    float2 u13 = twiddle(x[i + 13 * t], 13, alpha);
    float2 u14 = twiddle(x[i + 14 * t], 14, alpha);
    float2 u15 = twiddle(x[i + 15 * t], 15, alpha);

    DFT16(u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15);

    y[j] = u0;
    y[j + p] = u8;
    y[j + 2 * p] = u4;
    y[j + 3 * p] = u12;
    y[j + 4 * p] = u2;
    y[j + 5 * p] = u10;
    y[j + 6 * p] = u6;
    y[j + 7 * p] = u14;
    y[j + 8 * p] = u1;
    y[j + 9 * p] = u9;
    y[j + 10 * p] = u5;
    y[j + 11 * p] = u13;
    y[j + 12 * p] = u3;
    y[j + 13 * p] = u11;
    y[j + 14 * p] = u7;
    y[j + 15 * p] = u15;
}

static const char* get_lsb_name()
{
    const char* name = getenv("ODW_LSB_NAME");
    return (name && name[0]) ? name : "fft";
}

static bool is_power_of_two(int x)
{
    return x > 1 && ((x & (x - 1)) == 0);
}

static void fill_sine(float2* x, int n)
{
    for (int i = 0; i < n; i++) {
        float z = -10.0f + 20.0f * (float)i / (float)n;
        x[i] = make_float2(sinf(z), 0.0f);
    }
}

static void print_fft_checksum(const char* label, const float2* y, int n)
{
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;
    double checksum = 0.0;
    double abs_checksum = 0.0;

    for (int i = 0; i < n; i++) {
        const double vals[2] = {(double)y[i].x, (double)y[i].y};
        for (int c = 0; c < 2; c++) {
            const double v = vals[c];
            const int idx = 2 * i + c;

            if (std::isnan(v)) {
                nan_count++;
                continue;
            }
            if (std::isinf(v)) {
                inf_count++;
                continue;
            }

            finite_count++;
            checksum += v * (double)(idx + 1);
            abs_checksum += std::fabs(v);
        }
    }

    printf("FFT_CHECKSUM label=%s signal_length=%d values=%d finite=%d nan=%d inf=%d value=%0.17e abs=%0.17e\n",
           label, n, 2 * n, finite_count, nan_count, inf_count, checksum, abs_checksum);
}

static void launch_fft_stage(const float2* in, float2* out, int n, int p, int radix)
{
    int t = n / radix;
    int blocks = (t + THREADS - 1) / THREADS;

    if (radix == 16) {
        fft_radix16_kernel<<<blocks, THREADS>>>(in, out, t, p);
    } else if (radix == 8) {
        fft_radix8_kernel<<<blocks, THREADS>>>(in, out, t, p);
    } else if (radix == 4) {
        fft_radix4_kernel<<<blocks, THREADS>>>(in, out, t, p);
    } else {
        fft_radix2_kernel<<<blocks, THREADS>>>(in, out, t, p);
    }

    CUDA_CHECK(cudaGetLastError());
}

static int run_fft(int n)
{
    const size_t bytes = (size_t)n * sizeof(float2);

    float2* h_x = (float2*)malloc(bytes);
    float2* h_y = (float2*)malloc(bytes);

    if (!h_x || !h_y) {
        fprintf(stderr, "Unable to allocate host FFT buffers\n");
        free(h_x);
        free(h_y);
        return EXIT_FAILURE;
    }

    LSB_Set_Rparam_string("region", "host_side_setup");
    LSB_Res();
    fill_sine(h_x, n);
    memset(h_y, 0, bytes);
    printf("Working kernel memory: %fKiB\n", (3.0 * (double)bytes) / 1024.0);
    LSB_Rec(0);

    float2* d_src = nullptr;
    float2* d_a = nullptr;
    float2* d_b = nullptr;

    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();
    CUDA_CHECK(cudaMalloc((void**)&d_src, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "device_side_h2d_copy");
    LSB_Res();
    CUDA_CHECK(cudaMemcpy(d_src, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    LSB_Rec(0);

    int repeats = 0;
    float2* final_out = d_a;

    struct timeval startTime, currentTime, elapsedTime;
    gettimeofday(&startTime, nullptr);

    do {
        LSB_Set_Rparam_int("repeats_to_two_seconds", repeats);

        CUDA_CHECK(cudaMemcpy(d_a, d_src, bytes, cudaMemcpyDeviceToDevice));

        float2* in = d_a;
        float2* out = d_b;

        LSB_Set_Rparam_string("region", "fft_kernel");
        LSB_Res();

        for (int p = 1; p < n;) {
            int radix = 2;
            if ((p << 4) <= n) {
                radix = 16;
            } else if ((p << 3) <= n) {
                radix = 8;
            } else if ((p << 2) <= n) {
                radix = 4;
            }

            launch_fft_stage(in, out, n, p, radix);

            float2* tmp = in;
            in = out;
            out = tmp;
            p *= radix;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        LSB_Rec(0);

        final_out = in;

        repeats++;
        gettimeofday(&currentTime, nullptr);
        timersub(&currentTime, &startTime, &elapsedTime);
    } while (elapsedTime.tv_sec < MIN_TIME_SEC);

    LSB_Set_Rparam_string("region", "device_side_d2h_copy");
    LSB_Res();
    CUDA_CHECK(cudaMemcpy(h_y, final_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    LSB_Rec(0);

    print_fft_checksum("fft", h_y, n);

    LSB_Set_Rparam_string("region", "runtime_finalization");
    LSB_Res();
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    LSB_Rec(0);

    free(h_x);
    free(h_y);

    return EXIT_SUCCESS;
}

static void print_help()
{
    printf("fft performs a one dimensional fast Fourier transform over signal length N.\n");
    printf("Usage:\n");
    printf("  ./fft-cuda-* [N]\n");
}

int main(int argc, char** argv)
{
    int n = 128;

    if (argc >= 2) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_help();
            return EXIT_SUCCESS;
        }

        n = atoi(argv[1]);
    }

    if (!is_power_of_two(n)) {
        fprintf(stderr, "N must be a power of 2 greater than 1, got %d\n", n);
        return EXIT_FAILURE;
    }

    LSB_Init(get_lsb_name(), 0);
    LSB_Set_Rparam_int("repeats_to_two_seconds", 0);
    LSB_Set_Rparam_int("signal_length", n);

    LSB_Set_Rparam_string("region", "runtime_initialization");
    LSB_Res();
    CUDA_CHECK(cudaFree(nullptr));
    LSB_Rec(0);

    int rc = run_fft(n);

    LSB_Finalize();
    return rc;
}
