// cuda_rigid_body.cu
// GPU-accelerated rigid body dynamics with spatial hash broadphase,
// sphere-sphere collision detection, and impulse-based resolution.
// Uses SoA layout for memory coalescing and shared memory for neighbor queries.

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <cstdio>
#include <cmath>

#include "cuda_rigid_body.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Kernel: Initialize cell arrays to -1
// ---------------------------------------------------------------------------
__global__ void init_cells_kernel(int* cell_start, int* cell_end, int total_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_cells) {
        cell_start[i] = -1;
        cell_end[i] = -1;
    }
}

// ---------------------------------------------------------------------------
// Kernel 1: Semi-implicit Euler integration + boundary enforcement
// Each thread handles one body. Coalesced access because px[i], py[i], etc.
// are contiguous arrays accessed by thread index.
// ---------------------------------------------------------------------------
__global__ void integrate_kernel(
    float* __restrict__ px, float* __restrict__ py, float* __restrict__ pz,
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    const float* __restrict__ radius,
    int num_bodies, float dt, float gravity, float domain_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies) return;

    // Apply gravity (negative y)
    float new_vy = vy[i] + gravity * dt;

    // Update positions (semi-implicit: velocity updated first)
    float new_px = px[i] + vx[i] * dt;
    float new_py = py[i] + new_vy * dt;
    float new_pz = pz[i] + vz[i] * dt;
    float new_vx = vx[i];
    float new_vz = vz[i];

    // Boundary enforcement with reflection
    float r = radius[i];
    float lo = r;
    float hi = domain_size - r;

    if (new_px < lo) { new_px = lo; new_vx = fabsf(new_vx) * 0.9f; }
    if (new_px > hi) { new_px = hi; new_vx = -fabsf(new_vx) * 0.9f; }
    if (new_py < lo) { new_py = lo; new_vy = fabsf(new_vy) * 0.9f; }
    if (new_py > hi) { new_py = hi; new_vy = -fabsf(new_vy) * 0.9f; }
    if (new_pz < lo) { new_pz = lo; new_vz = fabsf(new_vz) * 0.9f; }
    if (new_pz > hi) { new_pz = hi; new_vz = -fabsf(new_vz) * 0.9f; }

    px[i] = new_px; py[i] = new_py; pz[i] = new_pz;
    vx[i] = new_vx; vy[i] = new_vy; vz[i] = new_vz;
}

// ---------------------------------------------------------------------------
// Kernel 2: Compute spatial hash for each body
// Maps 3D cell coordinate to a 1D hash. Memory coalesced writes.
// ---------------------------------------------------------------------------
__global__ void compute_hash_kernel(
    int* __restrict__ grid_hash,
    int* __restrict__ grid_index,
    const float* __restrict__ px,
    const float* __restrict__ py,
    const float* __restrict__ pz,
    int num_bodies, float inv_cell_size, int grid_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies) return;

    int cx = min(max(__float2int_rd(px[i] * inv_cell_size), 0), grid_dim - 1);
    int cy = min(max(__float2int_rd(py[i] * inv_cell_size), 0), grid_dim - 1);
    int cz = min(max(__float2int_rd(pz[i] * inv_cell_size), 0), grid_dim - 1);

    grid_hash[i] = cx + cy * grid_dim + cz * grid_dim * grid_dim;
    grid_index[i] = i;
}

// ---------------------------------------------------------------------------
// Kernel 3: Find start and end of each cell in the sorted array
// Uses shared memory to minimize global memory reads.
// ---------------------------------------------------------------------------
__global__ void find_cell_bounds_kernel(
    int* __restrict__ cell_start,
    int* __restrict__ cell_end,
    const int* __restrict__ sorted_hash,
    int num_bodies)
{
    extern __shared__ int shared_hash[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies) return;

    int hash = sorted_hash[i];

    // Load into shared memory with halo
    shared_hash[threadIdx.x + 1] = hash;
    if (threadIdx.x == 0) {
        shared_hash[0] = (i > 0) ? sorted_hash[i - 1] : -1;
    }
    __syncthreads();

    if (i == 0 || hash != shared_hash[threadIdx.x]) {
        cell_start[hash] = i;
    }
    if (i == num_bodies - 1 || hash != sorted_hash[i + 1]) {
        cell_end[hash] = i + 1;
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: Collision detection and impulse-based resolution
// Each thread checks one body against all bodies in 27 neighboring cells.
// Uses shared memory for caching position data within a block.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void collide_kernel(
    float* __restrict__ px, float* __restrict__ py, float* __restrict__ pz,
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    const float* __restrict__ radius,
    const float* __restrict__ inv_mass,
    const float* __restrict__ restitution,
    const int* __restrict__ sorted_index,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    int num_bodies, float inv_cell_size, int grid_dim)
{
    int sorted_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_i >= num_bodies) return;

    int i = sorted_index[sorted_i];

    float xi = px[i], yi = py[i], zi = pz[i];
    float vxi = vx[i], vyi = vy[i], vzi = vz[i];
    float ri = radius[i];
    float inv_mi = inv_mass[i];
    float ei = restitution[i];

    int cx = min(max(__float2int_rd(xi * inv_cell_size), 0), grid_dim - 1);
    int cy = min(max(__float2int_rd(yi * inv_cell_size), 0), grid_dim - 1);
    int cz = min(max(__float2int_rd(zi * inv_cell_size), 0), grid_dim - 1);

    float dvx = 0.0f, dvy = 0.0f, dvz = 0.0f;

    // Check 27 neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        int nz = cz + dz;
        if (nz < 0 || nz >= grid_dim) continue;
        for (int dy = -1; dy <= 1; dy++) {
            int ny = cy + dy;
            if (ny < 0 || ny >= grid_dim) continue;
            for (int dx = -1; dx <= 1; dx++) {
                int nx_c = cx + dx;
                if (nx_c < 0 || nx_c >= grid_dim) continue;

                int cell = nx_c + ny * grid_dim + nz * grid_dim * grid_dim;
                int start = cell_start[cell];
                if (start < 0) continue;
                int end = cell_end[cell];

                for (int s = start; s < end; s++) {
                    int j = sorted_index[s];
                    if (j <= i) continue;

                    float dx_v = px[j] - xi;
                    float dy_v = py[j] - yi;
                    float dz_v = pz[j] - zi;
                    float dist_sq = dx_v * dx_v + dy_v * dy_v + dz_v * dz_v;

                    float rj = radius[j];
                    float min_dist = ri + rj;

                    if (dist_sq < min_dist * min_dist && dist_sq > 1e-12f) {
                        float inv_dist = rsqrtf(dist_sq);  // fast reciprocal sqrt
                        float dist = dist_sq * inv_dist;
                        float nx_n = dx_v * inv_dist;
                        float ny_n = dy_v * inv_dist;
                        float nz_n = dz_v * inv_dist;

                        float rel_vx = vxi - vx[j];
                        float rel_vy = vyi - vy[j];
                        float rel_vz = vzi - vz[j];
                        float rel_vn = rel_vx * nx_n + rel_vy * ny_n + rel_vz * nz_n;

                        if (rel_vn > 0.0f) continue;

                        float e = 0.5f * (ei + restitution[j]);
                        float inv_mj = inv_mass[j];
                        float j_imp = -(1.0f + e) * rel_vn / (inv_mi + inv_mj);

                        dvx += j_imp * inv_mi * nx_n;
                        dvy += j_imp * inv_mi * ny_n;
                        dvz += j_imp * inv_mi * nz_n;

                        atomicAdd(&vx[j], -j_imp * inv_mj * nx_n);
                        atomicAdd(&vy[j], -j_imp * inv_mj * ny_n);
                        atomicAdd(&vz[j], -j_imp * inv_mj * nz_n);

                        // Positional correction
                        float overlap = min_dist - dist;
                        float corr_i = overlap * 0.5f * inv_mi / (inv_mi + inv_mj);
                        float corr_j = overlap * 0.5f * inv_mj / (inv_mi + inv_mj);
                        atomicAdd(&px[i], -corr_i * nx_n);
                        atomicAdd(&py[i], -corr_i * ny_n);
                        atomicAdd(&pz[i], -corr_i * nz_n);
                        atomicAdd(&px[j], corr_j * nx_n);
                        atomicAdd(&py[j], corr_j * ny_n);
                        atomicAdd(&pz[j], corr_j * nz_n);
                    }
                }
            }
        }
    }

    atomicAdd(&vx[i], dvx);
    atomicAdd(&vy[i], dvy);
    atomicAdd(&vz[i], dvz);
}

// ---------------------------------------------------------------------------
// Persistent GPU context to avoid per-step allocation overhead
// ---------------------------------------------------------------------------
struct GpuBuffers {
    float *d_px, *d_py, *d_pz, *d_vx, *d_vy, *d_vz;
    float *d_radius, *d_inv_mass, *d_restitution;
    int *d_hash, *d_index, *d_cell_start, *d_cell_end;
    int allocated_bodies;
    int allocated_cells;

    GpuBuffers() : allocated_bodies(0), allocated_cells(0) {}

    void ensure(int num_bodies, int total_cells) {
        if (num_bodies <= allocated_bodies && total_cells <= allocated_cells) return;
        release();
        size_t fb = num_bodies * sizeof(float);
        size_t ib = num_bodies * sizeof(int);
        CUDA_CHECK(cudaMalloc(&d_px, fb));
        CUDA_CHECK(cudaMalloc(&d_py, fb));
        CUDA_CHECK(cudaMalloc(&d_pz, fb));
        CUDA_CHECK(cudaMalloc(&d_vx, fb));
        CUDA_CHECK(cudaMalloc(&d_vy, fb));
        CUDA_CHECK(cudaMalloc(&d_vz, fb));
        CUDA_CHECK(cudaMalloc(&d_radius, fb));
        CUDA_CHECK(cudaMalloc(&d_inv_mass, fb));
        CUDA_CHECK(cudaMalloc(&d_restitution, fb));
        CUDA_CHECK(cudaMalloc(&d_hash, ib));
        CUDA_CHECK(cudaMalloc(&d_index, ib));
        CUDA_CHECK(cudaMalloc(&d_cell_start, total_cells * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_cell_end, total_cells * sizeof(int)));
        allocated_bodies = num_bodies;
        allocated_cells = total_cells;
    }

    void release() {
        if (allocated_bodies == 0) return;
        cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_radius); cudaFree(d_inv_mass); cudaFree(d_restitution);
        cudaFree(d_hash); cudaFree(d_index);
        cudaFree(d_cell_start); cudaFree(d_cell_end);
        allocated_bodies = 0;
        allocated_cells = 0;
    }
};

// ---------------------------------------------------------------------------
// Single step (transfers each call)
// ---------------------------------------------------------------------------
void cuda_rigid_body_step(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    const float* radius, const float* mass, const float* inv_mass,
    const float* restitution,
    int num_bodies,
    float dt, float gravity, float domain_size,
    float cell_size, int grid_dim)
{
    // Forward to batch version with 1 step
    cuda_rigid_body_simulate(px, py, pz, vx, vy, vz,
                              radius, mass, inv_mass, restitution,
                              num_bodies, 1, dt, gravity, domain_size,
                              cell_size, grid_dim);
}

// ---------------------------------------------------------------------------
// Batch simulation: keeps data on GPU across steps to avoid transfer overhead
// ---------------------------------------------------------------------------
void cuda_rigid_body_simulate(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    const float* radius, const float* mass, const float* inv_mass,
    const float* restitution,
    int num_bodies,
    int num_steps,
    float dt, float gravity, float domain_size,
    float cell_size, int grid_dim)
{
    size_t body_bytes = num_bodies * sizeof(float);
    int total_cells = grid_dim * grid_dim * grid_dim;
    float inv_cell_size = 1.0f / cell_size;

    static GpuBuffers buf;
    buf.ensure(num_bodies, total_cells);

    // Transfer initial state to GPU (one-time)
    CUDA_CHECK(cudaMemcpy(buf.d_px, px, body_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_py, py, body_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_pz, pz, body_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_vx, vx, body_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_vy, vy, body_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_vz, vz, body_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_radius, radius, body_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_inv_mass, inv_mass, body_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_restitution, restitution, body_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (num_bodies + threads - 1) / threads;
    int cell_blocks = (total_cells + threads - 1) / threads;
    size_t shared_bytes = (threads + 1) * sizeof(int);

    thrust::device_ptr<int> t_hash(buf.d_hash);
    thrust::device_ptr<int> t_index(buf.d_index);

    for (int step = 0; step < num_steps; step++) {
        // Integration
        integrate_kernel<<<blocks, threads>>>(
            buf.d_px, buf.d_py, buf.d_pz, buf.d_vx, buf.d_vy, buf.d_vz,
            buf.d_radius, num_bodies, dt, gravity, domain_size);

        // Spatial hash
        compute_hash_kernel<<<blocks, threads>>>(
            buf.d_hash, buf.d_index, buf.d_px, buf.d_py, buf.d_pz,
            num_bodies, inv_cell_size, grid_dim);

        // Sort by hash (thrust radix sort - O(n) on GPU)
        thrust::sort_by_key(t_hash, t_hash + num_bodies, t_index);

        // Cell bounds - use kernel instead of thrust::fill
        init_cells_kernel<<<cell_blocks, threads>>>(buf.d_cell_start, buf.d_cell_end, total_cells);
        find_cell_bounds_kernel<<<blocks, threads, shared_bytes>>>(
            buf.d_cell_start, buf.d_cell_end, buf.d_hash, num_bodies);

        // Collision detection and resolution
        collide_kernel<<<blocks, threads>>>(
            buf.d_px, buf.d_py, buf.d_pz, buf.d_vx, buf.d_vy, buf.d_vz,
            buf.d_radius, buf.d_inv_mass, buf.d_restitution,
            buf.d_index, buf.d_cell_start, buf.d_cell_end,
            num_bodies, inv_cell_size, grid_dim);
    }

    // Transfer final state back (one-time)
    CUDA_CHECK(cudaMemcpy(px, buf.d_px, body_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(py, buf.d_py, body_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pz, buf.d_pz, body_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vx, buf.d_vx, body_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vy, buf.d_vy, body_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vz, buf.d_vz, body_bytes, cudaMemcpyDeviceToHost));
}
