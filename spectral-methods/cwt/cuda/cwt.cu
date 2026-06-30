#include <cuda_runtime.h>
#include <liblsb.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using real_t = double;

static void check_cuda(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        std::ostringstream os;
        os << what << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(os.str());
    }
}

static std::vector<real_t> linspace(real_t lo, real_t hi, std::int64_t n)
{
    std::vector<real_t> v(n);
    if (n <= 1) {
        if (n == 1) v[0] = lo;
        return v;
    }

    for (std::int64_t i = 0; i < n; ++i) {
        const real_t t = real_t(i) / real_t(n - 1);
        v[i] = lo + (hi - lo) * t;
    }
    return v;
}

static std::vector<real_t> build_test_signal(std::int64_t N)
{
    const real_t pi = real_t(M_PI);
    std::vector<real_t> fx(N);

    for (std::int64_t i = 0; i < N; ++i) {
        const real_t x = real_t(i) / real_t(N - 1);
        fx[i] =
            std::sin(real_t(40) * pi * x) *
                std::exp(real_t(-100) * pi * (x - real_t(2)) * (x - real_t(2))) +
            (std::sin(real_t(40) * pi * x) +
             real_t(2) * std::cos(real_t(160) * pi * x)) *
                std::exp(real_t(-50) * pi * (x - real_t(0.5)) * (x - real_t(0.5))) +
            real_t(2) * std::sin(real_t(160) * pi * x) *
                std::exp(real_t(-100) * pi * (x - real_t(0.8)) * (x - real_t(0.8)));
    }

    return fx;
}

__host__ __device__
static real_t mexican_hat(real_t x)
{
    const real_t x2 = x * x;
    return (real_t(1) - x2) * exp(real_t(-0.5) * x2);
}

__global__ void cwt_forward_kernel(const real_t *fx,
                                   const real_t *scales,
                                   const real_t *trans,
                                   real_t *cwt,
                                   long B,
                                   long A,
                                   long N,
                                   real_t dt)
{
    const long gid = blockIdx.x * blockDim.x + threadIdx.x;
    const long total = B * A * N;

    if (gid >= total) return;

    const long n = gid % N;
    const long a_idx = (gid / N) % A;
    const long b = gid / (A * N);

    const real_t aval = scales[a_idx];
    const real_t inv_sqrt_a = real_t(1) / sqrt(aval);
    const real_t tn = trans[n];

    real_t sum = 0;

    for (long k = 0; k < N; ++k) {
        const real_t tk = trans[k];
        const real_t x = (tk - tn) / aval;
        const real_t w = inv_sqrt_a * mexican_hat(x);
        sum += fx[b * N + k] * w;
    }

    cwt[gid] = sum * dt;
}

static real_t cpu_cwt_at(const std::vector<real_t> &fx,
                         const std::vector<real_t> &scales,
                         const std::vector<real_t> &trans,
                         std::int64_t B,
                         std::int64_t A,
                         std::int64_t N,
                         std::int64_t b,
                         std::int64_t a_idx,
                         std::int64_t n)
{
    (void)A;
    const real_t dt = trans[1] - trans[0];
    const real_t aval = scales[a_idx];
    const real_t inv_sqrt_a = real_t(1) / std::sqrt(aval);
    const real_t tn = trans[n];

    real_t sum = 0;
    for (std::int64_t k = 0; k < N; ++k) {
        const real_t tk = trans[k];
        const real_t x = (tk - tn) / aval;
        const real_t w = inv_sqrt_a * mexican_hat(x);
        sum += fx[b * N + k] * w;
    }

    return sum * dt;
}

static bool all_finite(const std::vector<real_t> &v)
{
    for (real_t x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

static double checksum_abs(const std::vector<real_t> &v)
{
    long double acc = 0.0L;
    for (real_t x : v) acc += std::fabs((long double)x);
    return (double)acc;
}

static double approx_gflops_forward(std::int64_t B, std::int64_t A, std::int64_t N, double seconds)
{
    const long double flops =
        2.0L * (long double)B * (long double)A * (long double)N * (long double)N;

    return seconds > 0.0 ? (double)(flops / seconds / 1.0e9L) : 0.0;
}

static void write_cwt_ppm(const std::string &path,
                          const std::vector<real_t> &cwt,
                          std::int64_t A,
                          std::int64_t N)
{
    real_t lo = std::numeric_limits<real_t>::infinity();
    real_t hi = -std::numeric_limits<real_t>::infinity();

    for (real_t v : cwt) {
        if (!std::isfinite(v)) continue;
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }

    if (!std::isfinite(lo) || !std::isfinite(hi) || lo == hi) {
        lo = real_t(0);
        hi = real_t(1);
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open output PPM: " + path);

    f << "P6\n" << N << " " << A << "\n255\n";

    for (std::int64_t a = 0; a < A; ++a) {
        for (std::int64_t n = 0; n < N; ++n) {
            const real_t v = cwt[a * N + n];
            real_t t = std::isfinite(v) ? (v - lo) / (hi - lo) : real_t(0);
            t = std::max(real_t(0), std::min(real_t(1), t));
            const unsigned char g = (unsigned char)std::lround(t * real_t(255));
            const unsigned char rgb[3] = {g, g, g};
            f.write((const char *)rgb, 3);
        }
    }
}

struct Args {
    std::int64_t N = 512;
    std::int64_t B = 1;
};

static Args parse_args(int argc, char **argv)
{
    Args args;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        auto need = [&](const char *name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
            return argv[++i];
        };

        if (a == "--N") args.N = std::stoll(need("--N"));
        else if (a == "--B") args.B = std::stoll(need("--B"));
        else positional.push_back(a);
    }

    if (!positional.empty()) args.N = std::stoll(positional[0]);
    if (positional.size() >= 2) args.B = std::stoll(positional[1]);

    if (args.N < 2) throw std::runtime_error("N must be >= 2");
    if (args.B < 1) throw std::runtime_error("B must be >= 1");

    return args;
}

int main(int argc, char **argv)
{
    try {
        const Args args = parse_args(argc, argv);

        const std::int64_t N = args.N;
        const std::int64_t A = N;
        const std::int64_t B = args.B;

        std::vector<real_t> fx1 = build_test_signal(N);
        std::vector<real_t> fx(B * N);
        for (std::int64_t b = 0; b < B; ++b) {
            std::copy(fx1.begin(), fx1.end(), fx.begin() + b * N);
        }

        std::vector<real_t> scales = linspace(real_t(0.01), real_t(0.10), A);
        std::vector<real_t> trans = linspace(real_t(-1.0), real_t(1.0), N);
        std::vector<real_t> cwt(B * A * N, real_t(0));

        const real_t dt = trans[1] - trans[0];
        const double working_kib =
            double(sizeof(real_t) * (fx.size() + scales.size() + trans.size() + cwt.size())) / 1024.0;

        std::string label = "cwt_cuda";
        const char *env_lsb = std::getenv("ODW_LSB_NAME");
        if (env_lsb && env_lsb[0]) label = env_lsb;

        LSB_Init(label.c_str(), 0);
        LSB_Set_Rparam_int("signal_length", (int)N);
        LSB_Set_Rparam_int("batch_size", (int)B);
        LSB_Set_Rparam_int("num_scales", (int)A);

        LSB_Set_Rparam_string("region", "runtime_initialization");
        LSB_Res();
        check_cuda(cudaFree(0), "cuda runtime initialization");
        cudaDeviceProp prop{};
        int dev = 0;
        check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
        check_cuda(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "host_side_setup");
        LSB_Res();
        std::cout << "Working kernel memory: " << working_kib << "KiB\n";
        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "device_side_buffer_setup");
        LSB_Res();
        real_t *d_fx = nullptr;
        real_t *d_scales = nullptr;
        real_t *d_trans = nullptr;
        real_t *d_cwt = nullptr;

        check_cuda(cudaMalloc((void **)&d_fx, sizeof(real_t) * fx.size()), "cudaMalloc fx");
        check_cuda(cudaMalloc((void **)&d_scales, sizeof(real_t) * scales.size()), "cudaMalloc scales");
        check_cuda(cudaMalloc((void **)&d_trans, sizeof(real_t) * trans.size()), "cudaMalloc trans");
        check_cuda(cudaMalloc((void **)&d_cwt, sizeof(real_t) * cwt.size()), "cudaMalloc cwt");
        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "device_side_h2d_copy");
        LSB_Res();
        check_cuda(cudaMemcpy(d_fx, fx.data(), sizeof(real_t) * fx.size(), cudaMemcpyHostToDevice), "copy fx");
        check_cuda(cudaMemcpy(d_scales, scales.data(), sizeof(real_t) * scales.size(), cudaMemcpyHostToDevice), "copy scales");
        check_cuda(cudaMemcpy(d_trans, trans.data(), sizeof(real_t) * trans.size(), cudaMemcpyHostToDevice), "copy trans");
        check_cuda(cudaDeviceSynchronize(), "sync h2d");
        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "setting_kernel_arguments");
        LSB_Res();
        const long kB = (long)B;
        const long kA = (long)A;
        const long kN = (long)N;
        const size_t total = (size_t)(B * A * N);
        const int block = 128;
        const int grid = (int)((total + block - 1) / block);
        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "kernel");
        LSB_Res();
        cwt_forward_kernel<<<grid, block>>>(d_fx, d_scales, d_trans, d_cwt, kB, kA, kN, dt);
        check_cuda(cudaGetLastError(), "launch cwt_forward_kernel");
        check_cuda(cudaDeviceSynchronize(), "sync kernel");
        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "device_side_d2h_copy");
        LSB_Res();
        check_cuda(cudaMemcpy(cwt.data(), d_cwt, sizeof(real_t) * cwt.size(), cudaMemcpyDeviceToHost), "copy cwt");
        check_cuda(cudaDeviceSynchronize(), "sync d2h");
        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "runtime_finalization");
        LSB_Res();
        check_cuda(cudaDeviceSynchronize(), "runtime final sync");
        LSB_Rec(0);

        LSB_Finalize();

        const bool finite = all_finite(cwt);
        const double checksum = checksum_abs(cwt);

        const std::vector<std::int64_t> sample_a = {0, A / 4, A / 2, A - 1};
        const std::vector<std::int64_t> sample_n = {0, N / 4, N / 2, N - 1};

        long double max_abs_err = 0.0L;
        long double max_rel_err = 0.0L;
        int checked = 0;

        for (std::int64_t b = 0; b < B; ++b) {
            for (std::int64_t a_idx : sample_a) {
                for (std::int64_t n : sample_n) {
                    const std::int64_t idx = (b * A + a_idx) * N + n;
                    const real_t ref = cpu_cwt_at(fx, scales, trans, B, A, N, b, a_idx, n);
                    const long double abs_err = std::fabs((long double)cwt[idx] - (long double)ref);
                    max_abs_err = std::max(max_abs_err, abs_err);

                    if (std::fabs((long double)ref) > 1.0e-8L) {
                        const long double rel_err = abs_err / std::fabs((long double)ref);
                        max_rel_err = std::max(max_rel_err, rel_err);
                    }
                    ++checked;
                }
            }
        }

        const std::string ppm_path =
            "cwt_scalogram_N" + std::to_string(N) +
            "_B" + std::to_string(B) +
            "_A" + std::to_string(A) + ".ppm";
        write_cwt_ppm(ppm_path, cwt, A, N);

        std::cout << "CWT_CUDA device=\"" << prop.name << "\""
                  << " N=" << N
                  << " B=" << B
                  << " A=" << A
                  << " values=" << cwt.size()
                  << " working_memory_kib=" << working_kib
                  << " finite=" << (finite ? 1 : 0)
                  << " checksum_abs=" << std::setprecision(17) << checksum
                  << " sampled_checks=" << checked
                  << " max_abs_err=" << (double)max_abs_err
                  << " max_rel_err=" << (double)max_rel_err
                  << " approx_gflops=" << approx_gflops_forward(B, A, N, 1.0)
                  << "\n";

        std::cout << "Output image written to: ./results/" << ppm_path << "\n";

        cudaFree(d_fx);
        cudaFree(d_scales);
        cudaFree(d_trans);
        cudaFree(d_cwt);

        if (!finite || max_abs_err > 1.0e-8) return 1;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
