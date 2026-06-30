#include <CL/cl.h>
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

static void check_cl(cl_int err, const char *what)
{
    if (err != CL_SUCCESS) {
        std::ostringstream os;
        os << what << " failed with OpenCL error " << err;
        throw std::runtime_error(os.str());
    }
}

static std::string read_file(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open " + path);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<real_t> linspace(real_t lo, real_t hi, std::int64_t n)
{
    std::vector<real_t> v(n);

    if (n <= 1) {
        if (n == 1) {
            v[0] = lo;
        }
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

static real_t mexican_hat(real_t x)
{
    const real_t x2 = x * x;
    return (real_t(1) - x2) * std::exp(real_t(-0.5) * x2);
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
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

static double checksum_abs(const std::vector<real_t> &v)
{
    long double acc = 0.0L;
    for (real_t x : v) {
        acc += std::fabs((long double)x);
    }
    return (double)acc;
}

static double gflops_forward(std::int64_t B, std::int64_t A, std::int64_t N, double seconds)
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
        if (!std::isfinite(v)) {
            continue;
        }
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }

    if (!std::isfinite(lo) || !std::isfinite(hi) || lo == hi) {
        lo = real_t(0);
        hi = real_t(1);
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Failed to open output PPM: " + path);
    }

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
    int platform = 0;
    int device = 0;
    int threads = 1;
    std::int64_t N = 512;
    std::int64_t B = 1;
    std::string kernel = "cwt_kernel.cl";
};

static Args parse_args(int argc, char **argv)
{
    Args args;
    bool after_opencl_sep = false;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        auto need = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };

        if (a == "--") {
            after_opencl_sep = true;
        } else if (!after_opencl_sep && a == "-p") {
            args.platform = std::stoi(need("-p"));
        } else if (!after_opencl_sep && a == "-d") {
            args.device = std::stoi(need("-d"));
        } else if (!after_opencl_sep && a == "-t") {
            args.threads = std::stoi(need("-t"));
        } else if (a == "--N") {
            args.N = std::stoll(need("--N"));
        } else if (a == "--B") {
            args.B = std::stoll(need("--B"));
        } else if (a == "--kernel") {
            args.kernel = need("--kernel");
        } else {
            positional.push_back(a);
        }
    }

    if (!positional.empty()) {
        args.N = std::stoll(positional[0]);
    }

    if (positional.size() >= 2) {
        args.B = std::stoll(positional[1]);
    }

    if (args.N < 2) {
        throw std::runtime_error("N must be >= 2");
    }

    if (args.B < 1) {
        throw std::runtime_error("B must be >= 1");
    }

    return args;
}

static cl_device_id select_device(int platform_idx, int device_idx)
{
    cl_uint platform_count = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs count");

    if (platform_count == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    std::vector<cl_platform_id> platforms(platform_count);
    check_cl(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");

    if (platform_idx < 0 || platform_idx >= (int)platforms.size()) {
        throw std::runtime_error("Requested OpenCL platform does not exist");
    }

    cl_uint device_count = 0;
    cl_int err = clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);

    if (err != CL_SUCCESS || device_count == 0) {
        check_cl(clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count),
                 "clGetDeviceIDs fallback count");
    }

    std::vector<cl_device_id> devices(device_count);
    err = clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr);

    if (err != CL_SUCCESS) {
        check_cl(clGetDeviceIDs(platforms[platform_idx], CL_DEVICE_TYPE_ALL, device_count, devices.data(), nullptr),
                 "clGetDeviceIDs fallback");
    }

    if (device_idx < 0 || device_idx >= (int)devices.size()) {
        throw std::runtime_error("Requested OpenCL device does not exist");
    }

    return devices[device_idx];
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

        std::string label = "cwt_opencl";
        const char *env_lsb = std::getenv("ODW_LSB_NAME");
        if (env_lsb && env_lsb[0]) {
            label = env_lsb;
        }

        LSB_Init(label.c_str(), 0);
        LSB_Set_Rparam_int("signal_length", (int)N);
        LSB_Set_Rparam_int("batch_size", (int)B);
        LSB_Set_Rparam_int("num_scales", (int)A);

        LSB_Set_Rparam_string("region", "runtime_initialization");
        LSB_Res();

        cl_int err = CL_SUCCESS;
        cl_device_id dev = select_device(args.platform, args.device);

        char device_name[512] = {0};
        clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);

        cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
        check_cl(err, "clCreateContext");

        cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
        check_cl(err, "clCreateCommandQueue");

        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "host_side_setup");
        LSB_Res();

        const std::string src = read_file(args.kernel);
        const char *src_ptr = src.c_str();
        const size_t src_len = src.size();

        cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
        check_cl(err, "clCreateProgramWithSource");

        err = clBuildProgram(prog, 1, &dev, "", nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::vector<char> log(log_size + 1);
            clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            std::cerr << log.data() << "\n";
            check_cl(err, "clBuildProgram");
        }

        cl_kernel kernel = clCreateKernel(prog, "cwt_forward_kernel", &err);
        check_cl(err, "clCreateKernel");

        std::cout << "Working kernel memory: " << working_kib << "KiB\n";

        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "device_side_buffer_setup");
        LSB_Res();

        cl_mem d_fx = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(real_t) * fx.size(), nullptr, &err);
        check_cl(err, "clCreateBuffer fx");

        cl_mem d_scales = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(real_t) * scales.size(), nullptr, &err);
        check_cl(err, "clCreateBuffer scales");

        cl_mem d_trans = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(real_t) * trans.size(), nullptr, &err);
        check_cl(err, "clCreateBuffer trans");

        cl_mem d_cwt = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(real_t) * cwt.size(), nullptr, &err);
        check_cl(err, "clCreateBuffer cwt");

        LSB_Set_Rparam_string("region", "device_side_h2d_copy");
        LSB_Res();

        check_cl(clEnqueueWriteBuffer(q, d_fx, CL_FALSE, 0, sizeof(real_t) * fx.size(), fx.data(), 0, nullptr, nullptr),
                 "write fx");
        check_cl(clEnqueueWriteBuffer(q, d_scales, CL_FALSE, 0, sizeof(real_t) * scales.size(), scales.data(), 0, nullptr, nullptr),
                 "write scales");
        check_cl(clEnqueueWriteBuffer(q, d_trans, CL_FALSE, 0, sizeof(real_t) * trans.size(), trans.data(), 0, nullptr, nullptr),
                 "write trans");
        check_cl(clFinish(q), "finish h2d");
        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "setting_kernel_arguments");
        LSB_Res();

        int argi = 0;
        check_cl(clSetKernelArg(kernel, argi++, sizeof(cl_mem), &d_fx), "arg fx");
        check_cl(clSetKernelArg(kernel, argi++, sizeof(cl_mem), &d_scales), "arg scales");
        check_cl(clSetKernelArg(kernel, argi++, sizeof(cl_mem), &d_trans), "arg trans");
        check_cl(clSetKernelArg(kernel, argi++, sizeof(cl_mem), &d_cwt), "arg cwt");
        check_cl(clSetKernelArg(kernel, argi++, sizeof(cl_long), &B), "arg B");
        check_cl(clSetKernelArg(kernel, argi++, sizeof(cl_long), &A), "arg A");
        check_cl(clSetKernelArg(kernel, argi++, sizeof(cl_long), &N), "arg N");
        check_cl(clSetKernelArg(kernel, argi++, sizeof(real_t), &dt), "arg dt");

        LSB_Rec(0);

        const size_t total = (size_t)(B * A * N);
        const size_t local = 128;
        const size_t global = ((total + local - 1) / local) * local;

        LSB_Set_Rparam_string("region", "kernel");
        LSB_Res();
        check_cl(clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                 "enqueue cwt_forward_kernel");
        check_cl(clFinish(q), "finish kernel");
        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "device_side_d2h_copy");
        LSB_Res();
        check_cl(clEnqueueReadBuffer(q, d_cwt, CL_TRUE, 0, sizeof(real_t) * cwt.size(), cwt.data(), 0, nullptr, nullptr),
                 "read cwt");
        LSB_Rec(0);

        LSB_Set_Rparam_string("region", "runtime_finalization");
        LSB_Res();
        clFinish(q);
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
                        const long double rel_err =
                            abs_err / std::fabs((long double)ref);
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

        std::cout << "CWT_OPENCL device=\"" << device_name << "\""
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
                  << " approx_gflops=" << gflops_forward(B, A, N, 1.0)
                  << "\n";

        std::cout << "Output image written to: ./results/" << ppm_path << "\n";

        clReleaseMemObject(d_fx);
        clReleaseMemObject(d_scales);
        clReleaseMemObject(d_trans);
        clReleaseMemObject(d_cwt);
        clReleaseKernel(kernel);
        clReleaseProgram(prog);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);

        if (!finite || max_abs_err > 1.0e-8) {
            return 1;
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
