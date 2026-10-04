// cpu_rigid_body.cpp
// Single-threaded CPU reference implementation of rigid body simulation.
// Uses the same spatial hash broadphase and impulse-based collision as the GPU version.

#include "cpu_rigid_body.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

struct CpuHashEntry {
    int hash;
    int index;
};

static inline int compute_cell_hash(float x, float y, float z,
                                     float cell_size, int grid_dim) {
    int cx = std::min(std::max((int)(x / cell_size), 0), grid_dim - 1);
    int cy = std::min(std::max((int)(y / cell_size), 0), grid_dim - 1);
    int cz = std::min(std::max((int)(z / cell_size), 0), grid_dim - 1);
    return cx + cy * grid_dim + cz * grid_dim * grid_dim;
}

void cpu_rigid_body_simulate(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    const float* radius, const float* mass, const float* inv_mass,
    const float* restitution,
    int num_bodies,
    int num_steps,
    float dt, float gravity, float domain_size,
    float cell_size, int grid_dim)
{
    int total_cells = grid_dim * grid_dim * grid_dim;
    std::vector<CpuHashEntry> entries(num_bodies);
    std::vector<int> cell_start(total_cells);
    std::vector<int> cell_end(total_cells);

    for (int step = 0; step < num_steps; step++) {
        // 1. Integration + boundary
        for (int i = 0; i < num_bodies; i++) {
            vy[i] += gravity * dt;
            px[i] += vx[i] * dt;
            py[i] += vy[i] * dt;
            pz[i] += vz[i] * dt;

            float r = radius[i];
            float lo = r, hi = domain_size - r;
            if (px[i] < lo) { px[i] = lo; vx[i] = fabsf(vx[i]) * 0.9f; }
            if (px[i] > hi) { px[i] = hi; vx[i] = -fabsf(vx[i]) * 0.9f; }
            if (py[i] < lo) { py[i] = lo; vy[i] = fabsf(vy[i]) * 0.9f; }
            if (py[i] > hi) { py[i] = hi; vy[i] = -fabsf(vy[i]) * 0.9f; }
            if (pz[i] < lo) { pz[i] = lo; vz[i] = fabsf(vz[i]) * 0.9f; }
            if (pz[i] > hi) { pz[i] = hi; vz[i] = -fabsf(vz[i]) * 0.9f; }
        }

        // 2. Compute spatial hashes
        for (int i = 0; i < num_bodies; i++) {
            entries[i].hash = compute_cell_hash(px[i], py[i], pz[i],
                                                 cell_size, grid_dim);
            entries[i].index = i;
        }

        // 3. Sort by hash
        std::sort(entries.begin(), entries.end(),
                  [](const CpuHashEntry& a, const CpuHashEntry& b) {
                      return a.hash < b.hash;
                  });

        // 4. Find cell boundaries
        std::fill(cell_start.begin(), cell_start.end(), -1);
        std::fill(cell_end.begin(), cell_end.end(), -1);
        for (int i = 0; i < num_bodies; i++) {
            int h = entries[i].hash;
            if (i == 0 || h != entries[i - 1].hash) cell_start[h] = i;
            if (i == num_bodies - 1 || h != entries[i + 1].hash) cell_end[h] = i + 1;
        }

        // 5. Collision detection and resolution
        for (int si = 0; si < num_bodies; si++) {
            int i = entries[si].index;
            float xi = px[i], yi = py[i], zi = pz[i];
            float ri = radius[i];
            float inv_mi = inv_mass[i];
            float ei = restitution[i];

            int cx = std::min(std::max((int)(xi / cell_size), 0), grid_dim - 1);
            int cy = std::min(std::max((int)(yi / cell_size), 0), grid_dim - 1);
            int cz = std::min(std::max((int)(zi / cell_size), 0), grid_dim - 1);

            for (int dz = -1; dz <= 1; dz++) {
                int nz = cz + dz;
                if (nz < 0 || nz >= grid_dim) continue;
                for (int dy = -1; dy <= 1; dy++) {
                    int ny = cy + dy;
                    if (ny < 0 || ny >= grid_dim) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = cx + dx;
                        if (nx < 0 || nx >= grid_dim) continue;

                        int cell = nx + ny * grid_dim + nz * grid_dim * grid_dim;
                        int s_start = cell_start[cell];
                        if (s_start < 0) continue;
                        int s_end = cell_end[cell];

                        for (int sj = s_start; sj < s_end; sj++) {
                            int j = entries[sj].index;
                            if (j <= i) continue;

                            float dx_v = px[j] - xi;
                            float dy_v = py[j] - yi;
                            float dz_v = pz[j] - zi;
                            float dist_sq = dx_v * dx_v + dy_v * dy_v + dz_v * dz_v;
                            float rj = radius[j];
                            float min_dist = ri + rj;

                            if (dist_sq < min_dist * min_dist && dist_sq > 1e-12f) {
                                float dist = sqrtf(dist_sq);
                                float nx_n = dx_v / dist;
                                float ny_n = dy_v / dist;
                                float nz_n = dz_v / dist;

                                float rel_vx = vx[j] - vx[i];
                                float rel_vy = vy[j] - vy[i];
                                float rel_vz = vz[j] - vz[i];
                                float rel_vn = rel_vx * nx_n + rel_vy * ny_n + rel_vz * nz_n;

                                if (rel_vn >= 0.0f) continue;  // separating

                                float e = 0.5f * (ei + restitution[j]);
                                float inv_mj = inv_mass[j];
                                float j_imp = -(1.0f + e) * rel_vn / (inv_mi + inv_mj);

                                vx[i] -= j_imp * inv_mi * nx_n;
                                vy[i] -= j_imp * inv_mi * ny_n;
                                vz[i] -= j_imp * inv_mi * nz_n;
                                vx[j] += j_imp * inv_mj * nx_n;
                                vy[j] += j_imp * inv_mj * ny_n;
                                vz[j] += j_imp * inv_mj * nz_n;

                                float overlap = min_dist - dist;
                                float corr_i = overlap * 0.5f * inv_mi / (inv_mi + inv_mj);
                                float corr_j = overlap * 0.5f * inv_mj / (inv_mi + inv_mj);
                                px[i] -= corr_i * nx_n; py[i] -= corr_i * ny_n; pz[i] -= corr_i * nz_n;
                                px[j] += corr_j * nx_n; py[j] += corr_j * ny_n; pz[j] += corr_j * nz_n;
                            }
                        }
                    }
                }
            }
        }
    }
}
