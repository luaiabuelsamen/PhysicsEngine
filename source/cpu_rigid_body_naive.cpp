// cpu_rigid_body_naive.cpp
// Naive O(n^2) brute-force CPU rigid body simulation.
// No spatial acceleration - checks every pair. This is the standard
// approach in basic CPU physics implementations.

#include "cpu_rigid_body_naive.h"
#include <cmath>

void cpu_rigid_body_simulate_naive(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    const float* radius, const float* mass, const float* inv_mass,
    const float* restitution,
    int num_bodies,
    int num_steps,
    float dt, float gravity, float domain_size)
{
    for (int step = 0; step < num_steps; step++) {
        // Integration + boundary
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

        // Brute-force O(n^2) collision detection
        for (int i = 0; i < num_bodies; i++) {
            for (int j = i + 1; j < num_bodies; j++) {
                float dx = px[j] - px[i];
                float dy = py[j] - py[i];
                float dz = pz[j] - pz[i];
                float dist_sq = dx * dx + dy * dy + dz * dz;
                float min_dist = radius[i] + radius[j];

                if (dist_sq < min_dist * min_dist && dist_sq > 1e-12f) {
                    float dist = sqrtf(dist_sq);
                    float nx = dx / dist;
                    float ny = dy / dist;
                    float nz = dz / dist;

                    float rel_vx = vx[i] - vx[j];
                    float rel_vy = vy[i] - vy[j];
                    float rel_vz = vz[i] - vz[j];
                    float rel_vn = rel_vx * nx + rel_vy * ny + rel_vz * nz;

                    if (rel_vn > 0.0f) continue;

                    float e = 0.5f * (restitution[i] + restitution[j]);
                    float inv_mi = inv_mass[i];
                    float inv_mj = inv_mass[j];
                    float j_imp = -(1.0f + e) * rel_vn / (inv_mi + inv_mj);

                    vx[i] += j_imp * inv_mi * nx;
                    vy[i] += j_imp * inv_mi * ny;
                    vz[i] += j_imp * inv_mi * nz;
                    vx[j] -= j_imp * inv_mj * nx;
                    vy[j] -= j_imp * inv_mj * ny;
                    vz[j] -= j_imp * inv_mj * nz;

                    float overlap = min_dist - dist;
                    float corr_i = overlap * 0.5f * inv_mi / (inv_mi + inv_mj);
                    float corr_j = overlap * 0.5f * inv_mj / (inv_mi + inv_mj);
                    px[i] -= corr_i * nx; py[i] -= corr_i * ny; pz[i] -= corr_i * nz;
                    px[j] += corr_j * nx; py[j] += corr_j * ny; pz[j] += corr_j * nz;
                }
            }
        }
    }
}
