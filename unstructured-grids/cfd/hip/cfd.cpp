#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cstring>

#include "lsb.h"
#include "portable_memory.h"

#define AOCL_ALIGNMENT 64

#define GAMMA 1.4f
#define iterations 1

#define NDIM 3
#define NNB 4
#define RK 3

#define ff_mach 1.2f
#define deg_angle_of_attack 0.0f

#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR (VAR_DENSITY_ENERGY + 1)

#ifndef block_length
#define block_length 128
#endif

#define HIP_CHECK(call) do { \
    hipError_t err__ = (call); \
    if (err__ != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err__)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static size_t working_kernel_memory = 0;

__host__ __device__
static inline void compute_velocity(float density, float3 momentum, float3* velocity)
{
    velocity->x = momentum.x / density;
    velocity->y = momentum.y / density;
    velocity->z = momentum.z / density;
}

__host__ __device__
static inline float compute_speed_sqd(float3 velocity)
{
    return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}

__host__ __device__
static inline float compute_pressure(float density, float density_energy, float speed_sqd)
{
    return (GAMMA - 1.0f) * (density_energy - 0.5f * density * speed_sqd);
}

__host__ __device__
static inline float compute_speed_of_sound(float density, float pressure)
{
    return sqrtf(GAMMA * pressure / density);
}

__host__ __device__
static inline void compute_flux_contribution(
    float density,
    float3 momentum,
    float density_energy,
    float pressure,
    float3 velocity,
    float3* fc_momentum_x,
    float3* fc_momentum_y,
    float3* fc_momentum_z,
    float3* fc_density_energy)
{
    fc_momentum_x->x = velocity.x * momentum.x + pressure;
    fc_momentum_x->y = velocity.x * momentum.y;
    fc_momentum_x->z = velocity.x * momentum.z;

    fc_momentum_y->x = fc_momentum_x->y;
    fc_momentum_y->y = velocity.y * momentum.y + pressure;
    fc_momentum_y->z = velocity.y * momentum.z;

    fc_momentum_z->x = fc_momentum_x->z;
    fc_momentum_z->y = fc_momentum_y->z;
    fc_momentum_z->z = velocity.z * momentum.z + pressure;

    float de_p = density_energy + pressure;
    fc_density_energy->x = velocity.x * de_p;
    fc_density_energy->y = velocity.y * de_p;
    fc_density_energy->z = velocity.z * de_p;
}

__global__ void initialize_variables(int nelr, float* variables, const float* ff_variable)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelr) return;

    for (int j = 0; j < NVAR; j++) {
        variables[i + j * nelr] = ff_variable[j];
    }
}

__global__ void compute_step_factor(int nelr, const float* variables, const float* areas, float* step_factors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelr) return;

    float density = variables[i + VAR_DENSITY * nelr];

    float3 momentum;
    momentum.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
    momentum.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
    momentum.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

    float density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];

    float3 velocity;
    compute_velocity(density, momentum, &velocity);

    float speed_sqd = compute_speed_sqd(velocity);
    float pressure = compute_pressure(density, density_energy, speed_sqd);
    float speed_of_sound = compute_speed_of_sound(density, pressure);

    step_factors[i] = 0.5f / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
}

__global__ void compute_flux_contributions(
    int nelr,
    const float* variables,
    float* fc_momentum_x,
    float* fc_momentum_y,
    float* fc_momentum_z,
    float* fc_density_energy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelr) return;

    float density_i = variables[i + VAR_DENSITY * nelr];

    float3 momentum_i;
    momentum_i.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
    momentum_i.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
    momentum_i.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

    float density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];

    float3 velocity_i;
    compute_velocity(density_i, momentum_i, &velocity_i);

    float speed_sqd_i = compute_speed_sqd(velocity_i);
    float pressure_i = compute_pressure(density_i, density_energy_i, speed_sqd_i);

    float3 fc_i_momentum_x, fc_i_momentum_y, fc_i_momentum_z, fc_i_density_energy;
    compute_flux_contribution(
        density_i,
        momentum_i,
        density_energy_i,
        pressure_i,
        velocity_i,
        &fc_i_momentum_x,
        &fc_i_momentum_y,
        &fc_i_momentum_z,
        &fc_i_density_energy);

    fc_momentum_x[i + 0 * nelr] = fc_i_momentum_x.x;
    fc_momentum_x[i + 1 * nelr] = fc_i_momentum_x.y;
    fc_momentum_x[i + 2 * nelr] = fc_i_momentum_x.z;

    fc_momentum_y[i + 0 * nelr] = fc_i_momentum_y.x;
    fc_momentum_y[i + 1 * nelr] = fc_i_momentum_y.y;
    fc_momentum_y[i + 2 * nelr] = fc_i_momentum_y.z;

    fc_momentum_z[i + 0 * nelr] = fc_i_momentum_z.x;
    fc_momentum_z[i + 1 * nelr] = fc_i_momentum_z.y;
    fc_momentum_z[i + 2 * nelr] = fc_i_momentum_z.z;

    fc_density_energy[i + 0 * nelr] = fc_i_density_energy.x;
    fc_density_energy[i + 1 * nelr] = fc_i_density_energy.y;
    fc_density_energy[i + 2 * nelr] = fc_i_density_energy.z;
}

__global__ void compute_flux(
    int nelr,
    const int* elements_surrounding_elements,
    const float* normals,
    const float* variables,
    const float* fc_momentum_x,
    const float* fc_momentum_y,
    const float* fc_momentum_z,
    const float* fc_density_energy,
    float* fluxes,
    const float* ff_variable,
    const float3* ff_fc_momentum_x,
    const float3* ff_fc_momentum_y,
    const float3* ff_fc_momentum_z,
    const float3* ff_fc_density_energy)
{
    const float smoothing_coefficient = 0.2f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelr) return;

    float density_i = variables[i + VAR_DENSITY * nelr];

    float3 momentum_i;
    momentum_i.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
    momentum_i.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
    momentum_i.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

    float density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];

    float3 velocity_i;
    compute_velocity(density_i, momentum_i, &velocity_i);

    float speed_sqd_i = compute_speed_sqd(velocity_i);
    float speed_i = sqrtf(speed_sqd_i);
    float pressure_i = compute_pressure(density_i, density_energy_i, speed_sqd_i);
    float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);

    float3 fc_i_momentum_x;
    fc_i_momentum_x.x = fc_momentum_x[i + 0 * nelr];
    fc_i_momentum_x.y = fc_momentum_x[i + 1 * nelr];
    fc_i_momentum_x.z = fc_momentum_x[i + 2 * nelr];

    float3 fc_i_momentum_y;
    fc_i_momentum_y.x = fc_momentum_y[i + 0 * nelr];
    fc_i_momentum_y.y = fc_momentum_y[i + 1 * nelr];
    fc_i_momentum_y.z = fc_momentum_y[i + 2 * nelr];

    float3 fc_i_momentum_z;
    fc_i_momentum_z.x = fc_momentum_z[i + 0 * nelr];
    fc_i_momentum_z.y = fc_momentum_z[i + 1 * nelr];
    fc_i_momentum_z.z = fc_momentum_z[i + 2 * nelr];

    float3 fc_i_density_energy;
    fc_i_density_energy.x = fc_density_energy[i + 0 * nelr];
    fc_i_density_energy.y = fc_density_energy[i + 1 * nelr];
    fc_i_density_energy.z = fc_density_energy[i + 2 * nelr];

    float flux_i_density = 0.0f;
    float3 flux_i_momentum;
    flux_i_momentum.x = 0.0f;
    flux_i_momentum.y = 0.0f;
    flux_i_momentum.z = 0.0f;
    float flux_i_density_energy = 0.0f;

    for (int j = 0; j < NNB; j++) {
        int nb = elements_surrounding_elements[i + j * nelr];

        float3 normal;
        normal.x = normals[i + (j + 0 * NNB) * nelr];
        normal.y = normals[i + (j + 1 * NNB) * nelr];
        normal.z = normals[i + (j + 2 * NNB) * nelr];

        float normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        float factor;

        if (nb >= 0) {
            float density_nb = variables[nb + VAR_DENSITY * nelr];

            float3 momentum_nb;
            momentum_nb.x = variables[nb + (VAR_MOMENTUM + 0) * nelr];
            momentum_nb.y = variables[nb + (VAR_MOMENTUM + 1) * nelr];
            momentum_nb.z = variables[nb + (VAR_MOMENTUM + 2) * nelr];

            float density_energy_nb = variables[nb + VAR_DENSITY_ENERGY * nelr];

            float3 velocity_nb;
            compute_velocity(density_nb, momentum_nb, &velocity_nb);

            float speed_sqd_nb = compute_speed_sqd(velocity_nb);
            float pressure_nb = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
            float speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);

            float3 fc_nb_momentum_x;
            fc_nb_momentum_x.x = fc_momentum_x[nb + 0 * nelr];
            fc_nb_momentum_x.y = fc_momentum_x[nb + 1 * nelr];
            fc_nb_momentum_x.z = fc_momentum_x[nb + 2 * nelr];

            float3 fc_nb_momentum_y;
            fc_nb_momentum_y.x = fc_momentum_y[nb + 0 * nelr];
            fc_nb_momentum_y.y = fc_momentum_y[nb + 1 * nelr];
            fc_nb_momentum_y.z = fc_momentum_y[nb + 2 * nelr];

            float3 fc_nb_momentum_z;
            fc_nb_momentum_z.x = fc_momentum_z[nb + 0 * nelr];
            fc_nb_momentum_z.y = fc_momentum_z[nb + 1 * nelr];
            fc_nb_momentum_z.z = fc_momentum_z[nb + 2 * nelr];

            float3 fc_nb_density_energy;
            fc_nb_density_energy.x = fc_density_energy[nb + 0 * nelr];
            fc_nb_density_energy.y = fc_density_energy[nb + 1 * nelr];
            fc_nb_density_energy.z = fc_density_energy[nb + 2 * nelr];

            factor = -normal_len * smoothing_coefficient * 0.5f *
                     (speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);

            flux_i_density += factor * (density_i - density_nb);
            flux_i_density_energy += factor * (density_energy_i - density_energy_nb);
            flux_i_momentum.x += factor * (momentum_i.x - momentum_nb.x);
            flux_i_momentum.y += factor * (momentum_i.y - momentum_nb.y);
            flux_i_momentum.z += factor * (momentum_i.z - momentum_nb.z);

            factor = 0.5f * normal.x;
            flux_i_density += factor * (momentum_nb.x + momentum_i.x);
            flux_i_density_energy += factor * (fc_nb_density_energy.x + fc_i_density_energy.x);
            flux_i_momentum.x += factor * (fc_nb_momentum_x.x + fc_i_momentum_x.x);
            flux_i_momentum.y += factor * (fc_nb_momentum_y.x + fc_i_momentum_y.x);
            flux_i_momentum.z += factor * (fc_nb_momentum_z.x + fc_i_momentum_z.x);

            factor = 0.5f * normal.y;
            flux_i_density += factor * (momentum_nb.y + momentum_i.y);
            flux_i_density_energy += factor * (fc_nb_density_energy.y + fc_i_density_energy.y);
            flux_i_momentum.x += factor * (fc_nb_momentum_x.y + fc_i_momentum_x.y);
            flux_i_momentum.y += factor * (fc_nb_momentum_y.y + fc_i_momentum_y.y);
            flux_i_momentum.z += factor * (fc_nb_momentum_z.y + fc_i_momentum_z.y);

            factor = 0.5f * normal.z;
            flux_i_density += factor * (momentum_nb.z + momentum_i.z);
            flux_i_density_energy += factor * (fc_nb_density_energy.z + fc_i_density_energy.z);
            flux_i_momentum.x += factor * (fc_nb_momentum_x.z + fc_i_momentum_x.z);
            flux_i_momentum.y += factor * (fc_nb_momentum_y.z + fc_i_momentum_z.y);
            flux_i_momentum.z += factor * (fc_nb_momentum_z.z + fc_i_momentum_z.z);
        } else if (nb == -1) {
            flux_i_momentum.x += normal.x * pressure_i;
            flux_i_momentum.y += normal.y * pressure_i;
            flux_i_momentum.z += normal.z * pressure_i;
        } else if (nb == -2) {
            factor = 0.5f * normal.x;
            flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 0] + momentum_i.x);
            flux_i_density_energy += factor * (ff_fc_density_energy[0].x + fc_i_density_energy.x);
            flux_i_momentum.x += factor * (ff_fc_momentum_x[0].x + fc_i_momentum_x.x);
            flux_i_momentum.y += factor * (ff_fc_momentum_y[0].x + fc_i_momentum_x.y);
            flux_i_momentum.z += factor * (ff_fc_momentum_z[0].x + fc_i_momentum_x.z);

            factor = 0.5f * normal.y;
            flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y);
            flux_i_density_energy += factor * (ff_fc_density_energy[0].y + fc_i_density_energy.y);
            flux_i_momentum.x += factor * (ff_fc_momentum_x[0].y + fc_i_momentum_y.x);
            flux_i_momentum.y += factor * (ff_fc_momentum_y[0].y + fc_i_momentum_y.y);
            flux_i_momentum.z += factor * (ff_fc_momentum_z[0].y + fc_i_momentum_y.z);

            factor = 0.5f * normal.z;
            flux_i_density += factor * (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z);
            flux_i_density_energy += factor * (ff_fc_density_energy[0].z + fc_i_density_energy.z);
            flux_i_momentum.x += factor * (ff_fc_momentum_x[0].z + fc_i_momentum_z.x);
            flux_i_momentum.y += factor * (ff_fc_momentum_y[0].z + fc_i_momentum_z.y);
            flux_i_momentum.z += factor * (ff_fc_momentum_z[0].z + fc_i_momentum_z.z);
        }
    }

    fluxes[i + VAR_DENSITY * nelr] = flux_i_density;
    fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x;
    fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y;
    fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z;
    fluxes[i + VAR_DENSITY_ENERGY * nelr] = flux_i_density_energy;
}

__global__ void time_step(
    int j,
    int nelr,
    const float* old_variables,
    float* variables,
    const float* step_factors,
    const float* fluxes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelr) return;

    float factor = step_factors[i] / (float)(RK + 1 - j);

    variables[i + VAR_DENSITY * nelr] =
        old_variables[i + VAR_DENSITY * nelr] + factor * fluxes[i + VAR_DENSITY * nelr];

    variables[i + VAR_DENSITY_ENERGY * nelr] =
        old_variables[i + VAR_DENSITY_ENERGY * nelr] + factor * fluxes[i + VAR_DENSITY_ENERGY * nelr];

    variables[i + (VAR_MOMENTUM + 0) * nelr] =
        old_variables[i + (VAR_MOMENTUM + 0) * nelr] + factor * fluxes[i + (VAR_MOMENTUM + 0) * nelr];

    variables[i + (VAR_MOMENTUM + 1) * nelr] =
        old_variables[i + (VAR_MOMENTUM + 1) * nelr] + factor * fluxes[i + (VAR_MOMENTUM + 1) * nelr];

    variables[i + (VAR_MOMENTUM + 2) * nelr] =
        old_variables[i + (VAR_MOMENTUM + 2) * nelr] + factor * fluxes[i + (VAR_MOMENTUM + 2) * nelr];
}

template <typename T>
static T* device_alloc(size_t n)
{
    T* ptr = nullptr;
    HIP_CHECK(hipMalloc((void**)&ptr, sizeof(T) * n));
    working_kernel_memory += sizeof(T) * n;
    return ptr;
}

template <typename T>
static void upload(T* dst, const T* src, size_t n)
{
    LSB_Set_Rparam_string("region", "device_side_h2d_copy");
    LSB_Res();
    HIP_CHECK(hipMemcpy(dst, src, sizeof(T) * n, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);
}

template <typename T>
static void download(T* dst, const T* src, size_t n)
{
    LSB_Set_Rparam_string("region", "device_side_d2h_copy");
    LSB_Res();
    HIP_CHECK(hipMemcpy(dst, src, sizeof(T) * n, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);
}

static const char* get_lsb_name()
{
    const char* lsb_name = getenv("ODW_LSB_NAME");
    return (lsb_name && lsb_name[0]) ? lsb_name : "cfd";
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Usage ./cfd <data input file>\n");
        return 0;
    }

    const char* data_file_name = argv[1];

    LSB_Init(get_lsb_name(), 0);
    LSB_Set_Rparam_int("repeats_to_two_seconds", 0);
    LSB_Set_Rparam_string("input_file", data_file_name);

    LSB_Set_Rparam_string("region", "runtime_initialization");
    LSB_Res();
    HIP_CHECK(hipFree(NULL));
    LSB_Rec(0);

    int nel = 0;
    int nelr = 0;

    float* h_areas = nullptr;
    int* h_elements_surrounding_elements = nullptr;
    float* h_normals = nullptr;

    LSB_Set_Rparam_string("region", "host_side_setup");
    LSB_Res();

    std::ifstream file(data_file_name);
    if (!file.is_open()) {
        std::cerr << "Unable to open file " << data_file_name << std::endl;
        LSB_Finalize();
        return 2;
    }

    file >> nel;
    nelr = block_length * ((nel / block_length) + std::min(1, nel % block_length));

    LSB_Set_Rparam_int("nel", nel);
    LSB_Set_Rparam_int("nelr", nelr);

    h_areas = (float*)memalign(AOCL_ALIGNMENT, nelr * sizeof(float));
    h_elements_surrounding_elements = (int*)memalign(AOCL_ALIGNMENT, nelr * NNB * sizeof(int));
    h_normals = (float*)memalign(AOCL_ALIGNMENT, nelr * NDIM * NNB * sizeof(float));

    if (!h_areas || !h_elements_surrounding_elements || !h_normals) {
        std::cerr << "Unable to allocate host mesh buffers" << std::endl;
        return 3;
    }

    for (int i = 0; i < nel; i++) {
        file >> h_areas[i];
        for (int j = 0; j < NNB; j++) {
            file >> h_elements_surrounding_elements[i + j * nelr];
            if (h_elements_surrounding_elements[i + j * nelr] < 0) {
                h_elements_surrounding_elements[i + j * nelr] = -1;
            }
            h_elements_surrounding_elements[i + j * nelr]--;

            for (int k = 0; k < NDIM; k++) {
                file >> h_normals[i + (j + k * NNB) * nelr];
                h_normals[i + (j + k * NNB) * nelr] = -h_normals[i + (j + k * NNB) * nelr];
            }
        }
    }

    int last = nel - 1;
    for (int i = nel; i < nelr; i++) {
        h_areas[i] = h_areas[last];
        for (int j = 0; j < NNB; j++) {
            h_elements_surrounding_elements[i + j * nelr] =
                h_elements_surrounding_elements[last + j * nelr];
            for (int k = 0; k < NDIM; k++) {
                h_normals[i + (j + k * NNB) * nelr] =
                    h_normals[last + (j + k * NNB) * nelr];
            }
        }
    }

    float h_ff_variable[NVAR];
    const float angle_of_attack = 3.1415926535897931f / 180.0f * deg_angle_of_attack;

    h_ff_variable[VAR_DENSITY] = 1.4f;

    float ff_pressure = 1.0f;
    float ff_speed_of_sound = sqrtf(GAMMA * ff_pressure / h_ff_variable[VAR_DENSITY]);
    float ff_speed = ff_mach * ff_speed_of_sound;

    float3 ff_velocity;
    ff_velocity.x = ff_speed * cosf(angle_of_attack);
    ff_velocity.y = ff_speed * sinf(angle_of_attack);
    ff_velocity.z = 0.0f;

    h_ff_variable[VAR_MOMENTUM + 0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
    h_ff_variable[VAR_MOMENTUM + 1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
    h_ff_variable[VAR_MOMENTUM + 2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;

    h_ff_variable[VAR_DENSITY_ENERGY] =
        h_ff_variable[VAR_DENSITY] * (0.5f * ff_speed * ff_speed) +
        ff_pressure / (GAMMA - 1.0f);

    float3 h_ff_momentum;
    h_ff_momentum.x = h_ff_variable[VAR_MOMENTUM + 0];
    h_ff_momentum.y = h_ff_variable[VAR_MOMENTUM + 1];
    h_ff_momentum.z = h_ff_variable[VAR_MOMENTUM + 2];

    float3 h_ff_fc_momentum_x;
    float3 h_ff_fc_momentum_y;
    float3 h_ff_fc_momentum_z;
    float3 h_ff_fc_density_energy;

    compute_flux_contribution(
        h_ff_variable[VAR_DENSITY],
        h_ff_momentum,
        h_ff_variable[VAR_DENSITY_ENERGY],
        ff_pressure,
        ff_velocity,
        &h_ff_fc_momentum_x,
        &h_ff_fc_momentum_y,
        &h_ff_fc_momentum_z,
        &h_ff_fc_density_energy);

    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "device_side_buffer_setup");
    LSB_Res();

    float* areas = device_alloc<float>(nelr);
    int* elements_surrounding_elements = device_alloc<int>(nelr * NNB);
    float* normals = device_alloc<float>(nelr * NDIM * NNB);

    float* ff_variable = device_alloc<float>(NVAR);
    float3* ff_fc_momentum_x = device_alloc<float3>(1);
    float3* ff_fc_momentum_y = device_alloc<float3>(1);
    float3* ff_fc_momentum_z = device_alloc<float3>(1);
    float3* ff_fc_density_energy = device_alloc<float3>(1);

    float* variables = device_alloc<float>(nelr * NVAR);
    float* old_variables = device_alloc<float>(nelr * NVAR);
    float* fluxes = device_alloc<float>(nelr * NVAR);
    float* step_factors = device_alloc<float>(nelr);
    float* fc_momentum_x = device_alloc<float>(nelr * NDIM);
    float* fc_momentum_y = device_alloc<float>(nelr * NDIM);
    float* fc_momentum_z = device_alloc<float>(nelr * NDIM);
    float* fc_density_energy = device_alloc<float>(nelr * NDIM);

    HIP_CHECK(hipMemset(step_factors, 0, sizeof(float) * nelr));
    HIP_CHECK(hipDeviceSynchronize());

    LSB_Rec(0);

    upload<float>(areas, h_areas, nelr);
    upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr * NNB);
    upload<float>(normals, h_normals, nelr * NDIM * NNB);

    upload<float>(ff_variable, h_ff_variable, NVAR);
    upload<float3>(ff_fc_momentum_x, &h_ff_fc_momentum_x, 1);
    upload<float3>(ff_fc_momentum_y, &h_ff_fc_momentum_y, 1);
    upload<float3>(ff_fc_momentum_z, &h_ff_fc_momentum_z, 1);
    upload<float3>(ff_fc_density_energy, &h_ff_fc_density_energy, 1);

    LSB_Set_Rparam_string("region", "kernel_creation");
    LSB_Res();
    dim3 block(block_length);
    dim3 grid((nelr + block_length - 1) / block_length);
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "setting_kernel_initialize_variables_arguments");
    LSB_Res();
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "kernel_initialize_variables_kernel");
    LSB_Res();
    initialize_variables<<<grid, block>>>(nelr, variables, ff_variable);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "setting_kernel_initialize_variables_arguments");
    LSB_Res();
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "kernel_initialize_variables_kernel");
    LSB_Res();
    initialize_variables<<<grid, block>>>(nelr, old_variables, ff_variable);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "setting_kernel_initialize_variables_arguments");
    LSB_Res();
    LSB_Rec(0);

    LSB_Set_Rparam_string("region", "kernel_initialize_variables_kernel");
    LSB_Res();
    initialize_variables<<<grid, block>>>(nelr, fluxes, ff_variable);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);

    printf("Working kernel memory: %fKiB\n", working_kernel_memory / 1024.0);
    std::cout << "Starting..." << std::endl;

    for (int i = 0; i < iterations; i++) {
        LSB_Set_Rparam_string("region", "device_side_d2d_copy");
        LSB_Res();
        HIP_CHECK(hipMemcpy(old_variables, variables, sizeof(float) * nelr * NVAR, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipDeviceSynchronize());
        LSB_Rec(i);

        LSB_Set_Rparam_string("region", "setting_kernel_compute_step_factor_arguments");
        LSB_Res();
        LSB_Rec(i);

        LSB_Set_Rparam_string("region", "kernel_compute_step_factor_kernel");
        LSB_Res();
        compute_step_factor<<<grid, block>>>(nelr, variables, areas, step_factors);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        LSB_Rec(i);

        for (int j = 0; j < RK; j++) {
            LSB_Set_Rparam_string("region", "setting_kernel_compute_flux_contributions_arguments");
            LSB_Res();
            LSB_Rec(j);

            LSB_Set_Rparam_string("region", "kernel_compute_flux_contributions_kernel");
            LSB_Res();
            compute_flux_contributions<<<grid, block>>>(
                nelr,
                variables,
                fc_momentum_x,
                fc_momentum_y,
                fc_momentum_z,
                fc_density_energy);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            LSB_Rec(j);

            LSB_Set_Rparam_string("region", "setting_kernel_compute_flux_arguments");
            LSB_Res();
            LSB_Rec(j);

            LSB_Set_Rparam_string("region", "kernel_compute_flux_kernel");
            LSB_Res();
            compute_flux<<<grid, block>>>(
                nelr,
                elements_surrounding_elements,
                normals,
                variables,
                fc_momentum_x,
                fc_momentum_y,
                fc_momentum_z,
                fc_density_energy,
                fluxes,
                ff_variable,
                ff_fc_momentum_x,
                ff_fc_momentum_y,
                ff_fc_momentum_z,
                ff_fc_density_energy);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            LSB_Rec(j);

            LSB_Set_Rparam_string("region", "setting_kernel_time_step_arguments");
            LSB_Res();
            LSB_Rec(j);

            LSB_Set_Rparam_string("region", "kernel_time_step_kernel");
            LSB_Res();
            time_step<<<grid, block>>>(j, nelr, old_variables, variables, step_factors, fluxes);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            LSB_Rec(j);
        }
    }

    LSB_Set_Rparam_string("region", "runtime_finalization");
    LSB_Res();
    HIP_CHECK(hipDeviceSynchronize());
    LSB_Rec(0);

    LSB_Finalize();

    float* checksum_variables = (float*)memalign(AOCL_ALIGNMENT, nelr * NVAR * sizeof(float));
    if (checksum_variables == NULL) {
        std::cerr << "Unable to allocate checksum buffer" << std::endl;
        return 3;
    }

    HIP_CHECK(hipMemcpy(
        checksum_variables,
        variables,
        sizeof(float) * nelr * NVAR,
        hipMemcpyDeviceToHost));

    double cfd_checksum = 0.0;
    double cfd_abs_checksum = 0.0;
    int cfd_finite_values = 0;
    int cfd_nan_values = 0;
    int cfd_inf_values = 0;

    for (int idx = 0; idx < nel * NVAR; idx++) {
        double v = (double)checksum_variables[idx];

        if (std::isnan(v)) {
            cfd_nan_values++;
            continue;
        }

        if (std::isinf(v)) {
            cfd_inf_values++;
            continue;
        }

        cfd_checksum += v * (double)(idx + 1);
        cfd_abs_checksum += std::fabs(v);
        cfd_finite_values++;
    }

    printf("CFD_CHECKSUM nel=%d nelr=%d values=%d finite=%d nan=%d inf=%d value=%0.17e abs=%0.17e\n",
           nel,
           nelr,
           nel * NVAR,
           cfd_finite_values,
           cfd_nan_values,
           cfd_inf_values,
           cfd_checksum,
           cfd_abs_checksum);

    free(checksum_variables);
    free(h_areas);
    free(h_elements_surrounding_elements);
    free(h_normals);

    HIP_CHECK(hipFree(areas));
    HIP_CHECK(hipFree(elements_surrounding_elements));
    HIP_CHECK(hipFree(normals));
    HIP_CHECK(hipFree(ff_variable));
    HIP_CHECK(hipFree(ff_fc_momentum_x));
    HIP_CHECK(hipFree(ff_fc_momentum_y));
    HIP_CHECK(hipFree(ff_fc_momentum_z));
    HIP_CHECK(hipFree(ff_fc_density_energy));
    HIP_CHECK(hipFree(variables));
    HIP_CHECK(hipFree(old_variables));
    HIP_CHECK(hipFree(fluxes));
    HIP_CHECK(hipFree(step_factors));
    HIP_CHECK(hipFree(fc_momentum_x));
    HIP_CHECK(hipFree(fc_momentum_y));
    HIP_CHECK(hipFree(fc_momentum_z));
    HIP_CHECK(hipFree(fc_density_energy));

    std::cout << "Done..." << std::endl;
    return 0;
}
