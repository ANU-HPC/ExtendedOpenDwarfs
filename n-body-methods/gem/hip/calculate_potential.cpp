#include "calculations.h"
#include "defines.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <sys/time.h>

#include "lsb.h"

#define ALPHA_OF 0.571412f
#define REACTION_POTENTIAL 2
#define TOTAL_POTENTIAL 3
#define MIN_TIME_SEC 2

#define HIP_CHECK(call)                                                        \
    do {                                                                        \
        hipError_t err__ = (call);                                              \
        if (err__ != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d: %s\n",                           \
                    __FILE__, __LINE__, hipGetErrorString(err__));             \
            exit(EXIT_FAILURE);                                                  \
        }                                                                       \
    } while (0)

typedef struct
{
    int phiType;
    int region;

    float one_plus_alpha;
    float beta;
    float alpha_beta;
    float one_plus_alpha_beta;
    float alpha_by_one_minus_beta;
    float inverse_one_plus_ab_by_diel_ext;
    float kappa;
    float Asq;
    float Asq_minus_rnautsq;
    float Asq_minus_rsq;
    float Asq_by_dsq;
} analytical_definitions_struct_cuda;

static int BLOCK_DIM_X = 16;
static int BLOCK_DIM_Y = 16;
static int BLOCKS = 120;

static struct timeval start_time;

static void init_time()
{
    gettimeofday(&start_time, NULL);
}

static long long get_time()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (long long)(t.tv_sec - start_time.tv_sec) * 1000000LL +
           (long long)(t.tv_usec - start_time.tv_usec);
}

static const char* get_lsb_name()
{
    const char* lsb_name = getenv("ODW_LSB_NAME");
    if (lsb_name == NULL || lsb_name[0] == '\0') {
        lsb_name = "gem";
    }
    return lsb_name;
}

static void record_region_start(const char* region)
{
    LSB_Set_Rparam_string("region", region);
    LSB_Res();
}

static void record_region_end(int id)
{
    LSB_Rec(id);
}

__device__ inline float dist2D_cuda(
    float x, float y, float z,
    float x2, float y2, float z2)
{
    float xd = x - x2;
    float yd = y - y2;
    float zd = z - z2;
    return xd * xd + yd * yd + zd * zd;
}

__device__ inline float distD_cuda(
    float x, float y, float z,
    float x2, float y2, float z2)
{
    return sqrtf(dist2D_cuda(x, y, z, x2, y2, z2));
}

__global__
void calc_potential_single_step_dev(
    int nres,
    int nvert,
    float A,
    float proj_len,
    float diel_int,
    float diel_ext,
    float sal,
    float ion_exc_rad,
    int phiType,
    int eye,
    int step_size,
    const unsigned int* atom_addrs,
    const unsigned int* atom_lengths,
    float r,
    float r0,
    float rprime,
    const float* res_c_s,
    const float* res_x_s,
    const float* res_y_s,
    const float* res_z_s,
    const float* at_c_s,
    const float* at_x_s,
    const float* at_y_s,
    const float* at_z_s,
    float* vert_c_s,
    const float* vert_x_s,
    const float* vert_y_s,
    const float* vert_z_s,
    const float* vert_x_p_s,
    const float* vert_y_p_s,
    const float* vert_z_p_s)
{
    int j;
    int k;
    int natoms;

    float d;
    float dprime;
    float charge;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float salt = 0.0f;
    float coulomb = 0.0f;
    float to_return = 0.0f;
    float one_over_one_plus_kappa_rprime;

    analytical_definitions_struct_cuda defs;

    int local_eye = eye +
        (gridDim.x * gridDim.y * blockIdx.z) +
        (gridDim.x * blockIdx.y) +
        blockIdx.x;

    local_eye = eye + blockIdx.x * blockDim.x + threadIdx.x;

    if (local_eye >= nvert) {
        return;
    }

    defs.Asq = A * A;
    defs.Asq_minus_rsq = defs.Asq - (r * r);

    defs.kappa = 0.316f * sqrtf(sal);
    defs.beta = diel_int / diel_ext;
    defs.alpha_by_one_minus_beta = ALPHA_OF * (1.0f - defs.beta);
    defs.alpha_beta = ALPHA_OF * defs.beta;
    defs.one_plus_alpha_beta = 1.0f + defs.alpha_beta;
    defs.one_plus_alpha = 1.0f + ALPHA_OF;
    defs.inverse_one_plus_ab_by_diel_ext = 1.0f / (defs.one_plus_alpha_beta * diel_ext);
    defs.phiType = phiType;

    if (r > rprime) {
        defs.region = 3;
    } else if (r > A) {
        defs.region = 2;
    } else {
        defs.region = 1;
    }

    r0 = A / 2.0f;
    vert_c_s[local_eye] = 0.0f;
    defs.Asq_minus_rnautsq = defs.Asq - r0 * r0;

    for (k = 0; k < nres; k++) {
        natoms = atom_lengths[k];

        for (j = 0; j < natoms; j++) {
            d = distD_cuda(
                vert_x_s[local_eye], vert_y_s[local_eye], vert_z_s[local_eye],
                at_x_s[atom_addrs[k] + j],
                at_y_s[atom_addrs[k] + j],
                at_z_s[atom_addrs[k] + j]);

            dprime = distD_cuda(
                vert_x_p_s[local_eye], vert_y_p_s[local_eye], vert_z_p_s[local_eye],
                at_x_s[atom_addrs[k] + j],
                at_y_s[atom_addrs[k] + j],
                at_z_s[atom_addrs[k] + j]);

            defs.Asq_by_dsq = A * A * d * d;
            charge = at_c_s[atom_addrs[k] + j];

            if (defs.phiType != TOTAL_POTENTIAL) {
                coulomb = charge / (d * diel_int);
            }

            if (defs.phiType & REACTION_POTENTIAL) {
                if (defs.region == 3) {
                    sum1 = (charge * expf(-defs.kappa * (d - dprime)) / d) *
                           (defs.one_plus_alpha / (1.0f + defs.kappa * dprime));

                    sum2 = charge *
                           (defs.alpha_by_one_minus_beta *
                            expf(-defs.kappa * (r - rprime)) /
                            (r * (1.0f + defs.kappa * rprime)));

                    to_return = defs.inverse_one_plus_ab_by_diel_ext * (sum1 - sum2);
                } else if (defs.region == 2) {
                    one_over_one_plus_kappa_rprime = 1.0f / (1.0f + (defs.kappa * rprime));

                    sum1 = defs.one_plus_alpha * charge / d;
                    sum2 = defs.alpha_by_one_minus_beta * charge / r;

                    sum3 = charge *
                           (defs.alpha_by_one_minus_beta / rprime) *
                           (1.0f - one_over_one_plus_kappa_rprime);

                    sum4 = defs.one_plus_alpha * charge *
                           (1.0f / (dprime + defs.kappa * dprime * dprime) -
                            (1.0f / dprime));

                    to_return = defs.inverse_one_plus_ab_by_diel_ext *
                                (sum1 - sum2 + sum3 + sum4);
                } else {
                    one_over_one_plus_kappa_rprime = 1.0f / (1.0f + (defs.kappa * rprime));

                    sum1 = 1.0f / (d * diel_int);

                    sum2 = (1.0f / diel_int - 1.0f / diel_ext) /
                           (defs.one_plus_alpha_beta * A);

                    sum3 = defs.Asq /
                           sqrtf(defs.Asq_minus_rnautsq *
                                 defs.Asq_minus_rsq +
                                 defs.Asq_by_dsq);

                    salt = defs.inverse_one_plus_ab_by_diel_ext *
                           (defs.one_plus_alpha / dprime *
                                (1.0f / (1.0f + defs.kappa * dprime) - 1.0f) +
                            defs.alpha_by_one_minus_beta / rprime *
                                (1.0f - one_over_one_plus_kappa_rprime));

                    to_return = (sum1 - sum2 * sum3 + salt) * charge;
                }

                if (defs.phiType == REACTION_POTENTIAL) {
                    to_return -= coulomb;
                }
            } else {
                to_return = coulomb;
            }

            vert_c_s[local_eye] += to_return;
        }
    }
}

void calc_potential_single_step(
    residue* residues,
    int nres,
    vertx* vert,
    int nvert,
    float A,
    float proj_len,
    float diel_int,
    float diel_ext,
    float sal,
    float ion_exc_rad,
    int phiType,
    int* i,
    int step_size)
{
    int it;
    int eye;
    int bound;
    int natoms;

    float* res_c;
    float* res_x;
    float* res_y;
    float* res_z;
    float* at_c;
    float* at_x;
    float* at_y;
    float* at_z;
    float* vert_c;
    float* vert_x;
    float* vert_y;
    float* vert_z;
    float* vert_x_p;
    float* vert_y_p;
    float* vert_z_p;

    unsigned int* atom_addrs;
    unsigned int* atom_lengths;

    float *res_c_d, *res_x_d, *res_y_d, *res_z_d;
    float *at_c_d, *at_x_d, *at_y_d, *at_z_d;
    float *vert_c_d, *vert_x_d, *vert_y_d, *vert_z_d;
    float *vert_x_p_d, *vert_y_p_d, *vert_z_p_d;
    unsigned int *atom_addrs_d, *atom_lengths_d;

    LSB_Init(get_lsb_name(), 0);
    LSB_Set_Rparam_int("number_of_residues", nres);
    LSB_Set_Rparam_int("number_of_vertices", nvert);
    LSB_Set_Rparam_int("repeats_to_two_seconds", 0);

    record_region_start("runtime_initialization");
    HIP_CHECK(hipFree(nullptr));
    record_region_end(0);

    record_region_start("kernel_creation");
    record_region_end(0);

    record_region_start("host_side_setup");
    init_time();

    eye = *i;
    bound = eye + step_size;
    if (nvert < bound) {
        bound = nvert;
    }

    natoms = 0;
    for (it = 0; it < nres; it++) {
        natoms += residues[it].natoms;
    }

    res_c = (float*)malloc(sizeof(float) * nres);
    res_x = (float*)malloc(sizeof(float) * nres);
    res_y = (float*)malloc(sizeof(float) * nres);
    res_z = (float*)malloc(sizeof(float) * nres);
    at_c = (float*)malloc(sizeof(float) * natoms);
    at_x = (float*)malloc(sizeof(float) * natoms);
    at_y = (float*)malloc(sizeof(float) * natoms);
    at_z = (float*)malloc(sizeof(float) * natoms);
    vert_c = (float*)malloc(sizeof(float) * nvert);
    vert_x = (float*)malloc(sizeof(float) * nvert);
    vert_y = (float*)malloc(sizeof(float) * nvert);
    vert_z = (float*)malloc(sizeof(float) * nvert);
    vert_x_p = (float*)malloc(sizeof(float) * nvert);
    vert_y_p = (float*)malloc(sizeof(float) * nvert);
    vert_z_p = (float*)malloc(sizeof(float) * nvert);
    atom_addrs = (unsigned int*)malloc(sizeof(unsigned int) * nres);
    atom_lengths = (unsigned int*)malloc(sizeof(unsigned int) * nres);

    natoms = 0;
    for (it = 0; it < nres; it++) {
        atom_addrs[it] = natoms;

        res_c[it] = residues[it].total_charge;
        res_x[it] = residues[it].x;
        res_y[it] = residues[it].y;
        res_z[it] = residues[it].z;

        for (int j = 0; j < residues[it].natoms; j++) {
            at_c[natoms + j] = residues[it].atoms[j].charge;
            at_x[natoms + j] = residues[it].atoms[j].x;
            at_y[natoms + j] = residues[it].atoms[j].y;
            at_z[natoms + j] = residues[it].atoms[j].z;
        }

        natoms += residues[it].natoms;
        atom_lengths[it] = residues[it].natoms;
    }

    for (it = 0; it < nvert; it++) {
        vert_c[it] = 0.0f;

        vert_x[it] = vert[it].x + proj_len * vert[it].xNorm;
        vert_y[it] = vert[it].y + proj_len * vert[it].yNorm;
        vert_z[it] = vert[it].z + proj_len * vert[it].zNorm;

        vert_x_p[it] = vert[it].x + ion_exc_rad * vert[it].xNorm;
        vert_y_p[it] = vert[it].y + ion_exc_rad * vert[it].yNorm;
        vert_z_p[it] = vert[it].z + ion_exc_rad * vert[it].zNorm;
    }

    float r = A + proj_len;
    float r0 = A / 2.0f;
    float rprime = A + ion_exc_rad;

    record_region_end(0);

    record_region_start("device_side_buffer_setup");

    HIP_CHECK(hipMalloc((void**)&res_c_d, sizeof(float) * nres));
    HIP_CHECK(hipMalloc((void**)&res_x_d, sizeof(float) * nres));
    HIP_CHECK(hipMalloc((void**)&res_y_d, sizeof(float) * nres));
    HIP_CHECK(hipMalloc((void**)&res_z_d, sizeof(float) * nres));

    HIP_CHECK(hipMalloc((void**)&at_c_d, sizeof(float) * natoms));
    HIP_CHECK(hipMalloc((void**)&at_x_d, sizeof(float) * natoms));
    HIP_CHECK(hipMalloc((void**)&at_y_d, sizeof(float) * natoms));
    HIP_CHECK(hipMalloc((void**)&at_z_d, sizeof(float) * natoms));

    HIP_CHECK(hipMalloc((void**)&vert_c_d, sizeof(float) * nvert));
    HIP_CHECK(hipMalloc((void**)&vert_x_d, sizeof(float) * nvert));
    HIP_CHECK(hipMalloc((void**)&vert_y_d, sizeof(float) * nvert));
    HIP_CHECK(hipMalloc((void**)&vert_z_d, sizeof(float) * nvert));
    HIP_CHECK(hipMalloc((void**)&vert_x_p_d, sizeof(float) * nvert));
    HIP_CHECK(hipMalloc((void**)&vert_y_p_d, sizeof(float) * nvert));
    HIP_CHECK(hipMalloc((void**)&vert_z_p_d, sizeof(float) * nvert));

    HIP_CHECK(hipMalloc((void**)&atom_addrs_d, sizeof(unsigned int) * nres));
    HIP_CHECK(hipMalloc((void**)&atom_lengths_d, sizeof(unsigned int) * nres));

    printf("Working kernel memory: %fKiB\n",
        ((sizeof(float) * nres) * 4 +
         (sizeof(float) * natoms) * 4 +
         (sizeof(float) * nvert) * 7 +
         (sizeof(unsigned int) * nres) * 2) / 1024.0);

    record_region_end(0);

    int lsb_timing_repeats = 0;
    struct timeval startTime;
    struct timeval currentTime;
    struct timeval elapsedTime;

    gettimeofday(&startTime, NULL);

    do {
        eye = *i;
        LSB_Set_Rparam_int("repeats_to_two_seconds", lsb_timing_repeats);

        record_region_start("device_side_h2d_copy");

        HIP_CHECK(hipMemcpy(res_c_d, res_c, sizeof(float) * nres, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(res_x_d, res_x, sizeof(float) * nres, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(res_y_d, res_y, sizeof(float) * nres, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(res_z_d, res_z, sizeof(float) * nres, hipMemcpyHostToDevice));

        HIP_CHECK(hipMemcpy(at_c_d, at_c, sizeof(float) * natoms, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(at_x_d, at_x, sizeof(float) * natoms, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(at_y_d, at_y, sizeof(float) * natoms, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(at_z_d, at_z, sizeof(float) * natoms, hipMemcpyHostToDevice));

        HIP_CHECK(hipMemcpy(vert_c_d, vert_c, sizeof(float) * nvert, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(vert_x_d, vert_x, sizeof(float) * nvert, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(vert_y_d, vert_y, sizeof(float) * nvert, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(vert_z_d, vert_z, sizeof(float) * nvert, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(vert_x_p_d, vert_x_p, sizeof(float) * nvert, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(vert_y_p_d, vert_y_p, sizeof(float) * nvert, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(vert_z_p_d, vert_z_p, sizeof(float) * nvert, hipMemcpyHostToDevice));

        HIP_CHECK(hipMemcpy(atom_addrs_d, atom_addrs, sizeof(unsigned int) * nres, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(atom_lengths_d, atom_lengths, sizeof(unsigned int) * nres, hipMemcpyHostToDevice));

        HIP_CHECK(hipDeviceSynchronize());

        record_region_end(0);

        record_region_start("setting_kernel_arguments");
        record_region_end(0);

        int threads = BLOCK_DIM_X * BLOCK_DIM_Y;
        int blocks_per_launch = BLOCKS;
        dim3 block(threads);
        dim3 grid(blocks_per_launch);

        record_region_start("gem_kernel");

        for (; eye < bound; eye += threads * blocks_per_launch) {
            calc_potential_single_step_dev<<<grid, block>>>(
                nres,
                nvert,
                A,
                proj_len,
                diel_int,
                diel_ext,
                sal,
                ion_exc_rad,
                phiType,
                eye,
                step_size,
                atom_addrs_d,
                atom_lengths_d,
                r,
                r0,
                rprime,
                res_c_d,
                res_x_d,
                res_y_d,
                res_z_d,
                at_c_d,
                at_x_d,
                at_y_d,
                at_z_d,
                vert_c_d,
                vert_x_d,
                vert_y_d,
                vert_z_d,
                vert_x_p_d,
                vert_y_p_d,
                vert_z_p_d);

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            if (eye + step_size > bound) {
                step_size = bound - eye;
            }

            record_region_end(eye);
            record_region_start("gem_kernel");
        }

        record_region_end(0);

        lsb_timing_repeats++;
        gettimeofday(&currentTime, NULL);
        timersub(&currentTime, &startTime, &elapsedTime);
    } while (elapsedTime.tv_sec < MIN_TIME_SEC);

    record_region_start("device_side_d2h_copy");

    if (eye > bound) {
        eye = bound;
    }

    HIP_CHECK(hipMemcpy(vert_c, vert_c_d, sizeof(float) * nvert, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    record_region_end(0);

    printf("runtime:%lld\n", get_time());

    record_region_start("device_side_buffer_cleanup");

    HIP_CHECK(hipFree(res_c_d));
    HIP_CHECK(hipFree(res_x_d));
    HIP_CHECK(hipFree(res_y_d));
    HIP_CHECK(hipFree(res_z_d));
    HIP_CHECK(hipFree(at_c_d));
    HIP_CHECK(hipFree(at_x_d));
    HIP_CHECK(hipFree(at_y_d));
    HIP_CHECK(hipFree(at_z_d));
    HIP_CHECK(hipFree(vert_c_d));
    HIP_CHECK(hipFree(vert_x_d));
    HIP_CHECK(hipFree(vert_y_d));
    HIP_CHECK(hipFree(vert_z_d));
    HIP_CHECK(hipFree(vert_x_p_d));
    HIP_CHECK(hipFree(vert_y_p_d));
    HIP_CHECK(hipFree(vert_z_p_d));
    HIP_CHECK(hipFree(atom_addrs_d));
    HIP_CHECK(hipFree(atom_lengths_d));

    record_region_end(0);

    record_region_start("runtime_finalization");
    HIP_CHECK(hipDeviceSynchronize());
    record_region_end(0);

    LSB_Finalize();

    double gem_checksum = 0.0;
    for (it = 0; it < nvert; it++) {
        gem_checksum += (double)vert_c[it] * (double)(it + 1);
    }
    printf("GEM_CHECKSUM nvert=%d value=%0.17e\n", nvert, gem_checksum);

    for (it = 0; it < nvert; it++) {
        vert[it].potential = vert_c[it];
    }

    *i = eye;

    free(res_c);
    free(res_x);
    free(res_y);
    free(res_z);
    free(at_c);
    free(at_x);
    free(at_y);
    free(at_z);
    free(vert_c);
    free(vert_x);
    free(vert_y);
    free(vert_z);
    free(vert_x_p);
    free(vert_y_p);
    free(vert_z_p);
    free(atom_addrs);
    free(atom_lengths);
}
