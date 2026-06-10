#include <hip/hip_runtime.h>
#include <cstdint>

__device__ __forceinline__ int bit_scan_u32(uint32_t x)
{
    int res = 0;
    res |= (x & 0xaaaaaaaau) ? 1 : 0;
    res |= (x & 0xccccccccu) ? 2 : 0;
    res |= (x & 0xf0f0f0f0u) ? 4 : 0;
    res |= (x & 0xff00ff00u) ? 8 : 0;
    res |= (x & 0xffff0000u) ? 16 : 0;
    return res;
}

#define BOARD_VALUE(x) \
    (((x) < board_size - level) ? params[idx + pitch * (4 + (x))] : ns[(x) - board_size + level])

extern "C" __global__ void nqueen_hip(
    int board_size,
    int level,
    int threads,
    int pitch,
    const uint32_t* __restrict__ params,
    uint32_t* __restrict__ results,
    const uint32_t* __restrict__ forbidden)
{
    int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= threads) {
        return;
    }

    uint32_t mask = params[idx];
    uint32_t left_mask = params[idx + pitch];
    uint32_t right_mask = params[idx + pitch * 2];
    uint32_t index = params[idx + pitch * 3];
    uint32_t board_mask = (board_size == 32)
        ? 0xffffffffu
        : ((1u << board_size) - 1u);

    uint32_t left_masks[32];
    uint32_t right_masks[32];
    uint32_t ms[32];
    uint32_t ns[33];

    uint32_t solutions = 0;
    uint32_t unique_solutions = 0;

    int board_array[32];
    int t_array[32];

    int i = 0;
    int j;

    ms[0] = mask | left_mask | right_mask | forbidden[0];
    ns[0] = ((ms[0] + 1u) & ~ms[0]);
    left_masks[0] = left_mask;
    right_masks[0] = right_mask;

    results[idx + pitch * 2] = params[idx];

    while (i >= 0) {
        if ((ns[i] & board_mask) != 0) {
            mask |= ns[i];
            left_masks[i + 1] = (left_masks[i] | ns[i]) << 1;
            right_masks[i + 1] = (right_masks[i] | ns[i]) >> 1;
            ms[i + 1] = mask | left_masks[i + 1] | right_masks[i + 1] | forbidden[i + 1];
            ns[i + 1] = ((ms[i + 1] + 1u) & ~ms[i + 1]);
            i++;
        }
        else {
            if (i == level) {
                int repeat_times = 8;
                bool repeat = false;
                bool equal = true;

                bool rotate1 = (BOARD_VALUE(index) == (1u << (board_size - 1)));
                bool rotate2 = (BOARD_VALUE(board_size - index - 1) == 1u);
                bool rotate3 = (ns[level - 1] == (1u << (board_size - index - 1)));

                if (rotate1 || rotate2 || rotate3) {
                    for (j = 0; j < board_size - level; j++) {
                        board_array[j] = bit_scan_u32(params[idx + pitch * (4 + j)]);
                    }
                    for (j = 0; j < level; j++) {
                        board_array[j + board_size - level] = bit_scan_u32(ns[j]);
                    }

                    if (rotate3) {
                        equal = true;
                        for (j = 0; j < board_size; j++) {
                            t_array[board_size - j - 1] = board_size - board_array[j] - 1;
                        }

                        for (j = 0; j < board_size; j++) {
                            if (t_array[j] < board_array[j]) {
                                repeat = true;
                                equal = false;
                                break;
                            }
                            else if (t_array[j] > board_array[j]) {
                                equal = false;
                                break;
                            }
                        }

                        if (equal) {
                            repeat_times = 4;
                        }
                    }

                    if (!repeat && rotate1) {
                        equal = true;
                        for (j = 0; j < board_size; j++) {
                            t_array[board_size - board_array[j] - 1] = j;
                        }

                        for (j = 0; j < board_size; j++) {
                            if (t_array[j] < board_array[j]) {
                                repeat = true;
                                equal = false;
                                break;
                            }
                            else if (t_array[j] > board_array[j]) {
                                equal = false;
                                break;
                            }
                        }

                        if (equal) {
                            repeat_times = 2;
                        }
                    }

                    if (!repeat && rotate2) {
                        equal = true;
                        for (j = 0; j < board_size; j++) {
                            t_array[board_array[j]] = board_size - j - 1;
                        }

                        for (j = 0; j < board_size; j++) {
                            if (t_array[j] < board_array[j]) {
                                repeat = true;
                                equal = false;
                                break;
                            }
                            else if (t_array[j] > board_array[j]) {
                                equal = false;
                                break;
                            }
                        }

                        if (equal) {
                            repeat_times = 2;
                        }
                    }

                    if (!repeat) {
                        solutions += repeat_times;
                        unique_solutions++;
                    }
                }
                else {
                    solutions += 8;
                    unique_solutions++;
                }
            }

            i--;

            if (i >= 0) {
                mask &= ~ns[i];
                ms[i] |= ns[i];
                ns[i] = ((ms[i] + 1u) & ~ms[i]);
            }
        }
    }

    results[idx] = solutions;
    results[idx + pitch] = unique_solutions;
    results[idx + pitch * 3] = params[idx];
}

extern "C" __global__ void nqueen1_hip(
    int board_size,
    int level,
    int threads,
    int pitch,
    const uint32_t* __restrict__ params,
    uint32_t* __restrict__ results,
    const uint32_t* __restrict__ forbidden)
{
    int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= threads) {
        return;
    }

    uint32_t mask = params[idx];
    uint32_t left_mask = params[idx + pitch];
    uint32_t right_mask = params[idx + pitch * 2];
    int second_row = static_cast<int>(params[idx + pitch * 3]);

    uint32_t board_mask = (board_size == 32)
        ? 0xffffffffu
        : ((1u << board_size) - 1u);

    uint32_t left_masks[32];
    uint32_t right_masks[32];
    uint32_t ms[32];
    uint32_t ns[33];

    uint32_t solutions = 0;
    int i = 0;

    ms[0] = mask | left_mask | right_mask | (i < second_row ? 2u : 0u);
    ns[0] = ((ms[0] + 1u) & ~ms[0]);
    left_masks[0] = left_mask;
    right_masks[0] = right_mask;

    results[idx + pitch * 2] = params[idx];

    while (i >= 0) {
        if ((ns[i] & board_mask) != 0) {
            mask |= ns[i];
            left_masks[i + 1] = (left_masks[i] | ns[i]) << 1;
            right_masks[i + 1] = (right_masks[i] | ns[i]) >> 1;
            ms[i + 1] = mask | left_masks[i + 1] | right_masks[i + 1] |
                        (i + 1 < second_row ? 2u : 0u);
            ns[i + 1] = ((ms[i + 1] + 1u) & ~ms[i + 1]);
            i++;
        }
        else {
            if (i == level) {
                solutions++;
            }

            i--;

            if (i >= 0) {
                mask &= ~ns[i];
                ms[i] |= ns[i];
                ns[i] = ((ms[i] + 1u) & ~ms[i]);
            }
        }
    }

    results[idx] = solutions * 8u;
    results[idx + pitch] = solutions;
    results[idx + pitch * 3] = params[idx];
}

#undef BOARD_VALUE
