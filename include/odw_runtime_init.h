#pragma once

/*
 * Shared device start-up for the CUDA and HIP variants of every benchmark. The OpenCL
 * counterpart is ocd_runtime_init() in common_args.c; the two do the same work.
 *
 * The LSB region "runtime_initialization" must contain exactly this call (plus, for an
 * application that creates its own streams, those stream creations) and nothing else, in
 * every backend, so that the region means the same thing everywhere:
 *
 *   LSB_Set_Rparam_string("region", "runtime_initialization");
 *   LSB_Res();
 *   odw_runtime_init(device);
 *   LSB_Rec(0);
 *
 * Why it does more than cudaFree(0): runtimes defer different amounts of work. HIP
 * initialises on the first runtime call; SCALE defers most of it to the first allocation or
 * stream creation (cudaFree(0) alone returns in microseconds). Timing only cudaFree(0)
 * charges one runtime and not the other, and the deferred cost then lands in whichever later
 * region happens to allocate first. This helper therefore touches, once and in order:
 *   1. device selection and runtime/context initialisation,
 *   2. stream creation,
 *   3. the device allocator (allocate a small buffer),
 *   4. the host-to-device copy path (one small asynchronous copy on the stream),
 *   5. synchronisation, then releases everything.
 *
 * It deliberately does NOT launch a kernel: loading an application's own device code
 * belongs to that application's first kernel launch, timed in its kernel region.
 *
 * Argument parsing and printing must happen BEFORE the region, and no device call
 * (property queries, stream creation, allocation) may precede it. On failure it prints the
 * error and exits.
 */

#include <stdio.h>
#include <stdlib.h>

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define ODW_RT(call) hip##call
typedef hipError_t odw_rt_error_t;
typedef hipStream_t odw_rt_stream_t;
#define ODW_RT_SUCCESS hipSuccess
#define odw_rt_error_string hipGetErrorString
#else
#include <cuda_runtime.h>
#define ODW_RT(call) cuda##call
typedef cudaError_t odw_rt_error_t;
typedef cudaStream_t odw_rt_stream_t;
#define ODW_RT_SUCCESS cudaSuccess
#define odw_rt_error_string cudaGetErrorString
#endif

#define ODW_RT_INIT_CHECK(expr)                                                              \
    do {                                                                                     \
        odw_rt_error_t odw_err_ = (expr);                                                    \
        if (odw_err_ != ODW_RT_SUCCESS) {                                                    \
            fprintf(stderr, "odw_runtime_init: %s failed: %s\n", #expr,                      \
                    odw_rt_error_string(odw_err_));                                          \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

static inline void odw_runtime_init(int device)
{
    char host_bytes[4096] = {0};
    void* dev_bytes = NULL;
    odw_rt_stream_t stream;

    ODW_RT_INIT_CHECK(ODW_RT(SetDevice)(device));
    ODW_RT_INIT_CHECK(ODW_RT(Free)(NULL));
    ODW_RT_INIT_CHECK(ODW_RT(StreamCreate)(&stream));
    ODW_RT_INIT_CHECK(ODW_RT(Malloc)(&dev_bytes, sizeof host_bytes));
    ODW_RT_INIT_CHECK(ODW_RT(MemcpyAsync)(dev_bytes, host_bytes, sizeof host_bytes,
                                          ODW_RT(MemcpyHostToDevice), stream));
    ODW_RT_INIT_CHECK(ODW_RT(StreamSynchronize)(stream));
    ODW_RT_INIT_CHECK(ODW_RT(DeviceSynchronize)());
    ODW_RT_INIT_CHECK(ODW_RT(Free)(dev_bytes));
    ODW_RT_INIT_CHECK(ODW_RT(StreamDestroy)(stream));
}
