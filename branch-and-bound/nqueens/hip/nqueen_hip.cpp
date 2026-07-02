// HIP host-side N-Queen solver

#include "nqueen_hip.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>
#include <sys/time.h>

#if __has_include(<liblsb.h>)
#include <liblsb.h>
#define HAVE_LSB 1
#else
#define HAVE_LSB 0
#endif

#define MIN_TIME_SEC 2

#define HIP_CHECK(expr)                                                       \
    do {                                                                      \
        hipError_t _err = (expr);                                             \
        if (_err != hipSuccess) {                                             \
            std::ostringstream _oss;                                          \
            _oss << "HIP error at line " << __LINE__ << ": "                 \
                 << hipGetErrorString(_err);                                 \
            throw HipError(_oss.str(), __LINE__);                             \
        }                                                                     \
    } while (0)

extern "C" __global__ void nqueen_hip(int board_size,
                                  int level,
                                  int threads,
                                  int pitch,
                                  const unsigned int* params,
                                  unsigned int* results,
                                  const unsigned int* forbidden);

extern "C" __global__ void nqueen1_hip(int board_size,
                                   int level,
                                   int threads,
                                   int pitch,
                                   const unsigned int* params,
                                   unsigned int* results,
                                   const unsigned int* forbidden);

HipError::HipError(const std::string& msg, int)
    : std::runtime_error(msg)
{
}

static inline int bit_scan(unsigned int x)
{
    int res = 0;
    res |= (x & 0xaaaaaaaau) ? 1 : 0;
    res |= (x & 0xccccccccu) ? 2 : 0;
    res |= (x & 0xf0f0f0f0u) ? 4 : 0;
    res |= (x & 0xff00ff00u) ? 8 : 0;
    res |= (x & 0xffff0000u) ? 16 : 0;
    return res;
}

NQueenSolverHIP::NQueenSolverHIP(int threads,
                                 int block_size,
                                 bool force_local,
                                 bool force_no_atomics,
                                 bool force_no_vec,
                                 bool force_vec4)
    : m_nThreads(threads),
      m_nBlockSize(block_size),
      m_nMaxWorkItems(0),
      m_nLastTotalSize(0),
      m_bForceLocal(force_local),
      m_bForceNoAtomics(force_no_atomics),
      m_bForceNoVectorization(force_no_vec),
      m_bForceVec4(force_vec4),
      d_ParamBuffer(nullptr),
      d_ResultBuffer(nullptr),
      d_ForbiddenBuffer(nullptr)
{
    InitDevice(threads, block_size);
}

NQueenSolverHIP::~NQueenSolverHIP()
{
    FreeBuffers();
}

void NQueenSolverHIP::InitDevice(int requested_threads, int requested_block_size)
{
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    int block = requested_block_size > 0 ? requested_block_size : 256;
    if (block > prop.maxThreadsPerBlock) {
        block = prop.maxThreadsPerBlock;
    }

    m_nMaxWorkItems = block;
    m_nBlockSize = block;

    if (requested_threads > 0) {
        m_nThreads = requested_threads;
    }
    else {
        m_nThreads = m_nMaxWorkItems * prop.multiProcessorCount * 4;
    }

    if (m_nThreads < m_nBlockSize) {
        m_nThreads = m_nBlockSize;
    }

    if (m_nThreads % m_nBlockSize != 0) {
        m_nThreads += m_nBlockSize - (m_nThreads % m_nBlockSize);
    }

    if (m_bForceLocal) {
        std::fprintf(stderr, "Note: -local is ignored in the scalar HIP port\n");
    }
    if (!m_bForceNoAtomics) {
        std::fprintf(stderr, "Note: atomics are not used in the scalar HIP port\n");
    }
    if (!m_bForceNoVectorization || m_bForceVec4) {
        std::fprintf(stderr, "Note: vectorized OpenCL paths are not used in this HIP port\n");
    }
}

void NQueenSolverHIP::AllocateBuffers(int max_pitch)
{
    FreeBuffers();

    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_ParamBuffer),
                        max_pitch * sizeof(unsigned int) * (32 + 32)));

    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_ResultBuffer),
                        max_pitch * sizeof(unsigned int) * 64));

    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_ForbiddenBuffer),
                        32 * sizeof(unsigned int)));
}

void NQueenSolverHIP::FreeBuffers()
{
    if (d_ParamBuffer != nullptr) {
        hipFree(d_ParamBuffer);
        d_ParamBuffer = nullptr;
    }

    if (d_ResultBuffer != nullptr) {
        hipFree(d_ResultBuffer);
        d_ResultBuffer = nullptr;
    }

    if (d_ForbiddenBuffer != nullptr) {
        hipFree(d_ForbiddenBuffer);
        d_ForbiddenBuffer = nullptr;
    }
}

long long NQueenSolverHIP::Compute(int board_size, long long* unique)
{
    long long total = 10000000000LL;
    total /= 10;

    int level = 0;
    int i_tmp = board_size;
    while (total > 0 && i_tmp > 0) {
        total /= ((i_tmp + 1) / 2);
        i_tmp--;
        level++;
    }

    if (level > board_size - 2) {
        level = board_size - 2;
    }

    if (level > 11) {
        level = 11;
    }

    int threads = m_nThreads;
    int max_threads = threads;
    int max_pitch = (max_threads + 15) & ~0xf;

#if HAVE_LSB
    LSB_Set_Rparam_string("region", "runtime_initialization");
    LSB_Res();
#endif

    HIP_CHECK(hipFree(0));

#if HAVE_LSB
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();
#endif

    AllocateBuffers(max_pitch);

#if HAVE_LSB
    LSB_Rec(0);
#endif

    std::fprintf(stderr, "Working kernel memory: %fKiB\n",
                 ((max_pitch * sizeof(unsigned int) * (32 + 32)) +
                  (max_pitch * sizeof(unsigned int) * 64) +
                  (32 * sizeof(unsigned int))) / 1024.0);

    std::vector<unsigned int> mask_vector(max_pitch * (4 + 32), 0);
    std::vector<unsigned int> results(max_pitch * 4, 0);
    std::vector<unsigned int> forbidden(32, 0);

    long long solutions = 0;
    long long unique_solutions = 0;

    unsigned int board_mask = (board_size == 32)
        ? 0xffffffffu
        : ((1u << board_size) - 1u);

    int lsb_timing_repeats = 0;
    struct timeval startTime, currentTime, elapsedTime;

    gettimeofday(&startTime, NULL);

    do {
#if HAVE_LSB
        LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);
#endif

        solutions = 0;
        unique_solutions = 0;

        int total_size = 0;
        bool forbidden_written = false;

        for (int j = 0; j < board_size / 2; j++) {
        unsigned int masks[32] = {0};
        unsigned int left_masks[32] = {0};
        unsigned int right_masks[32] = {0};
        unsigned int ms[32] = {0};
        unsigned int ns[32] = {0};

        unsigned int border_mask = 0;
        int idx = 0;
        int i = 0;

        masks[0] = (1u << j);
        left_masks[0] = 1u << (j + 1);
        right_masks[0] = (1u << j) >> 1;
        ms[0] = masks[0] | left_masks[0] | right_masks[0];
        ns[0] = (1u << j);

        for (int k = 0; k < j; k++) {
            border_mask |= (1u << k);
            border_mask |= (1u << (board_size - k - 1));
        }

        for (int k = 0; k < board_size; k++) {
            if (k == board_size - 2) {
                forbidden[k] = border_mask;
            }
            else if ((k + 1) < j || (k + 1) > board_size - j - 1) {
                forbidden[k] = 1u | (1u << (board_size - 1));
            }
            else {
                forbidden[k] = 0;
            }
        }

        forbidden[board_size - 1] = 0xffffffffu;
        forbidden_written = false;

        while (i >= 0) {
#if HAVE_LSB
            LSB_Set_Rparam_int("j", j);
#endif

            if (j == 0) {
                if (i >= 1) {
                    unsigned int m = ms[i] | (i + 1 < idx ? 2u : 0u);
                    ns[i + 1] = (m + 1u) & ~m;
                }
                else {
                    ns[i + 1] = ((ms[i] + 1u) & ~ms[i]);
                    if (i == 0) {
                        idx = bit_scan(ns[i + 1]);
                    }
                }
            }
            else {
                unsigned int m = ms[i] | forbidden[i];
                ns[i + 1] = (m + 1u) & ~m;
            }

            if (i == board_size - level - 1) {
                mask_vector[total_size] = masks[i];
                mask_vector[total_size + max_pitch] = left_masks[i];
                mask_vector[total_size + max_pitch * 2] = right_masks[i];

                if (j == 0) {
                    mask_vector[total_size + max_pitch * 3] =
                        idx - i < 0 ? 0u : static_cast<unsigned int>(idx - i);
                }
                else {
                    mask_vector[total_size + max_pitch * 3] =
                        static_cast<unsigned int>(j);
                }

                for (int k = 0; k <= i; k++) {
                    mask_vector[total_size + max_pitch * (k + 4)] = ns[k];
                }

                total_size++;

                if (total_size == max_threads) {
#if HAVE_LSB
                    LSB_Set_Rparam_string("region", "device_side_h2d_copy");
                    LSB_Res();
#endif

                    if (!forbidden_written) {
                        HIP_CHECK(hipMemcpy(d_ForbiddenBuffer,
                                            forbidden.data() + board_size - level - 1,
                                            (level + 1) * sizeof(unsigned int),
                                            hipMemcpyHostToDevice));
                        forbidden_written = true;
                    }

                    HIP_CHECK(hipMemcpy(d_ParamBuffer,
                                        mask_vector.data(),
                                        max_pitch * sizeof(unsigned int) * (4 + 32),
                                        hipMemcpyHostToDevice));

#if HAVE_LSB
                    LSB_Rec(i);
                    LSB_Set_Rparam_string("region", "setting_queen_arguments");
                    LSB_Res();
#endif

                    int arg_threads = threads;
                    int arg_pitch = max_pitch;
                    dim3 block(m_nBlockSize);
                    dim3 grid((arg_threads + block.x - 1) / block.x);

#if HAVE_LSB
                    LSB_Rec(i);
                    LSB_Set_Rparam_string("region", "queen_kernel");
                    LSB_Res();
#endif

                    if (j == 0) {
                        hipLaunchKernelGGL(nqueen1_hip,
                                           grid,
                                           block,
                                           0,
                                           0,
                                           board_size,
                                           level,
                                           arg_threads,
                                           arg_pitch,
                                           d_ParamBuffer,
                                           d_ResultBuffer,
                                           d_ForbiddenBuffer);
                    }
                    else {
                        hipLaunchKernelGGL(nqueen_hip,
                                           grid,
                                           block,
                                           0,
                                           0,
                                           board_size,
                                           level,
                                           arg_threads,
                                           arg_pitch,
                                           d_ParamBuffer,
                                           d_ResultBuffer,
                                           d_ForbiddenBuffer);
                    }

                    HIP_CHECK(hipGetLastError());
                    HIP_CHECK(hipDeviceSynchronize());

#if HAVE_LSB
                    LSB_Rec(i);
                    LSB_Set_Rparam_string("region", "device_side_d2h_copy");
                    LSB_Res();
#endif

                    HIP_CHECK(hipMemcpy(results.data(),
                                        d_ResultBuffer,
                                        max_pitch * sizeof(unsigned int) * 4,
                                        hipMemcpyDeviceToHost));

#if HAVE_LSB
                    LSB_Rec(i);
#endif

                    m_nLastTotalSize = threads;

                    for (int k = 0; k < m_nLastTotalSize; k++) {
                        if (results[k + max_pitch * 2] !=
                            results[k + max_pitch * 3]) {
                            throw HipError("HIP kernel execution failed: completion check mismatch", __LINE__);
                        }

                        solutions += results[k];
                        unique_solutions += results[k + max_pitch];
                    }

                    total_size = 0;
                }

                i--;
            }
            else if ((ns[i + 1] & board_mask) != 0) {
                ms[i] |= ns[i + 1];
                masks[i + 1] = masks[i] | ns[i + 1];
                left_masks[i + 1] = (left_masks[i] | ns[i + 1]) << 1;
                right_masks[i + 1] = (right_masks[i] | ns[i + 1]) >> 1;
                ms[i + 1] = masks[i + 1] | left_masks[i + 1] | right_masks[i + 1];
                i++;
            }
            else {
                i--;
            }
        }

        while (total_size > 0) {
            for (int k = total_size; k < max_threads; k++) {
                mask_vector[k] = 0xffffffffu;
                mask_vector[k + max_pitch] = 0xffffffffu;
                mask_vector[k + max_pitch * 2] = 0xffffffffu;
                mask_vector[k + max_pitch * 3] = 0u;
                mask_vector[k + max_pitch * 4] = 0u;
            }

            int t_size = std::min(total_size, threads);

#if HAVE_LSB
            LSB_Set_Rparam_string("region", "device_side_h2d_copy");
            LSB_Res();
#endif

            if (!forbidden_written) {
                HIP_CHECK(hipMemcpy(d_ForbiddenBuffer,
                                    forbidden.data() + board_size - level - 1,
                                    (level + 1) * sizeof(unsigned int),
                                    hipMemcpyHostToDevice));
                forbidden_written = true;
            }

            HIP_CHECK(hipMemcpy(d_ParamBuffer,
                                mask_vector.data(),
                                max_pitch * sizeof(unsigned int) * (4 + 32),
                                hipMemcpyHostToDevice));

#if HAVE_LSB
            LSB_Rec(total_size);
            LSB_Set_Rparam_string("region", "setting_queen_arguments");
            LSB_Res();
#endif

            int arg_threads = t_size;
            int arg_pitch = max_pitch;

            int launch_threads = t_size;
            if (launch_threads % m_nBlockSize != 0) {
                launch_threads += m_nBlockSize - (launch_threads % m_nBlockSize);
            }

            dim3 block(m_nBlockSize);
            dim3 grid((launch_threads + block.x - 1) / block.x);

#if HAVE_LSB
            LSB_Rec(total_size);
            LSB_Set_Rparam_string("region", "queen_kernel");
            LSB_Res();
#endif

            if (j == 0) {
                hipLaunchKernelGGL(nqueen1_hip,
                                   grid,
                                   block,
                                   0,
                                   0,
                                   board_size,
                                   level,
                                   arg_threads,
                                   arg_pitch,
                                   d_ParamBuffer,
                                   d_ResultBuffer,
                                   d_ForbiddenBuffer);
            }
            else {
                hipLaunchKernelGGL(nqueen_hip,
                                   grid,
                                   block,
                                   0,
                                   0,
                                   board_size,
                                   level,
                                   arg_threads,
                                   arg_pitch,
                                   d_ParamBuffer,
                                   d_ResultBuffer,
                                   d_ForbiddenBuffer);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

#if HAVE_LSB
            LSB_Rec(total_size);
            LSB_Set_Rparam_string("region", "device_side_d2h_copy");
            LSB_Res();
#endif

            HIP_CHECK(hipMemcpy(results.data(),
                                d_ResultBuffer,
                                max_pitch * sizeof(unsigned int) * 4,
                                hipMemcpyDeviceToHost));

#if HAVE_LSB
            LSB_Rec(total_size);
#endif

            m_nLastTotalSize = t_size;

            for (int k = 0; k < m_nLastTotalSize; k++) {
                if (results[k + max_pitch * 2] !=
                    results[k + max_pitch * 3]) {
                    throw HipError("HIP kernel execution failed: completion check mismatch", __LINE__);
                }

                solutions += results[k];
                unique_solutions += results[k + max_pitch];
            }

            if (total_size > t_size) {
                for (int k = 0; k < 4 + board_size - level; k++) {
                    std::memcpy(&mask_vector[max_pitch * k],
                                &mask_vector[t_size + max_pitch * k],
                                (total_size - t_size) * sizeof(unsigned int));
                }
            }

            total_size -= t_size;
        }
    }

        lsb_timing_repeats++;
        gettimeofday(&currentTime, NULL);
        timersub(&currentTime, &startTime, &elapsedTime);
    } while (elapsedTime.tv_sec < MIN_TIME_SEC);

    FreeBuffers();

    if (unique != nullptr) {
        *unique = unique_solutions;
    }

    return solutions;
}
