#pragma OPENCL EXTENSION cl_khr_fp64 : enable

static double mexican_hat(double x)
{
    double x2 = x * x;
    return (1.0 - x2) * exp(-0.5 * x2);
}

__kernel void cwt_forward_kernel(
    __global const double *fx,
    __global const double *scales,
    __global const double *trans,
    __global double *cwt,
    const long B,
    const long A,
    const long N,
    const double dt)
{
    const long gid = get_global_id(0);
    const long total = B * A * N;

    if (gid >= total) {
        return;
    }

    const long n = gid % N;
    const long a_idx = (gid / N) % A;
    const long b = gid / (A * N);

    const double aval = scales[a_idx];
    const double inv_sqrt_a = 1.0 / sqrt(aval);
    const double tn = trans[n];

    double sum = 0.0;

    for (long k = 0; k < N; ++k) {
        const double tk = trans[k];
        const double x = (tk - tn) / aval;
        const double w = inv_sqrt_a * mexican_hat(x);
        sum += fx[b * N + k] * w;
    }

    cwt[gid] = sum * dt;
}
