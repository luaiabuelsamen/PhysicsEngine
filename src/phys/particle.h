#pragma once

// Particle solver: per-body routines shared by the CPU and CUDA backends.
// Spheres without rotation or friction, bouncing off axis-aligned walls;
// suited to very large single scenes.
//
// A step runs in phases: integrate, (grid only) hash + sort + cell bounds,
// collide, apply. Within a phase every routine reads the state left by the
// previous phase and writes only the body it was called for, so the result
// does not depend on how bodies are scheduled onto threads. Contacts are
// accumulated into separate delta buffers (Jacobi style) instead of being
// scattered with atomics, and neighbours are always visited in the same
// order, which makes both backends deterministic.

#include "phys/common.h"

namespace phys {
namespace detail {

// Semi-implicit Euler step, then reflection off the walls.
PHYS_HD void integrate_body(const Params& p, const Buffers& b, int g) {
    if (!b.enabled[g]) return;
    float r = b.radius[g % p.nbody];
    for (int a = 0; a < 3; a++) {
        float v = b.vel[a][g] + p.gravity[a] * p.dt;
        float x = b.pos[a][g] + v * p.dt;
        float lo = p.lo[a] + r;
        float hi = p.hi[a] - r;
        if (x < lo) { x = lo; v = fabsf(v) * p.wall_restitution; }
        if (x > hi) { x = hi; v = -fabsf(v) * p.wall_restitution; }
        b.pos[a][g] = x;
        b.vel[a][g] = v;
    }
}

PHYS_HD int grid_coord(const Params& p, int axis, float x) {
    int c = (int)floorf((x - p.lo[axis]) * p.inv_cell);
    return c < 0 ? 0 : (c >= p.dim[axis] ? p.dim[axis] - 1 : c);
}

PHYS_HD int grid_cell(const Params& p, int cx, int cy, int cz) {
    return cx + p.dim[0] * (cy + p.dim[1] * cz);
}

PHYS_HD void compute_key(const Params& p, const Buffers& b, int g) {
    if (!b.enabled[g]) {
        b.key[g] = p.sentinel_key;
        return;
    }
    int cell = grid_cell(p, grid_coord(p, 0, b.pos[0][g]),
                            grid_coord(p, 1, b.pos[1][g]),
                            grid_coord(p, 2, b.pos[2][g]));
    b.key[g] = (g / p.nbody) * p.ncell + cell;
}

// Record where each cell's run of sorted slots starts and ends.
PHYS_HD void find_cell_bounds(const Params& p, const Buffers& b, int s, int n) {
    int k = b.sorted_key[s];
    if (k == p.sentinel_key) return;
    if (s == 0 || b.sorted_key[s - 1] != k) b.cell_start[k] = s;
    if (s == n - 1 || b.sorted_key[s + 1] != k) b.cell_end[k] = s + 1;
}

// Add to dv/dp the response of body g to its contact with body h.
// The pair is always evaluated with the lower index first, so both bodies
// compute bitwise-identical impulses of opposite sign: the pair conserves
// momentum no matter which thread handles which body.
PHYS_HD void accumulate_contact(const Params& p, const Buffers& b, int g, int h,
                                float dv[3], float dp[3]) {
    int a = g < h ? g : h;
    int c = g < h ? h : g;
    int ia = a % p.nbody;
    int ic = c % p.nbody;

    float d[3];
    float dist_sq = 0.0f;
    for (int k = 0; k < 3; k++) {
        d[k] = b.pos[k][c] - b.pos[k][a];
        dist_sq += d[k] * d[k];
    }
    float min_dist = b.radius[ia] + b.radius[ic];
    if (dist_sq >= min_dist * min_dist || dist_sq <= 1e-12f) return;

    float w_a = b.inv_mass[ia];
    float w_c = b.inv_mass[ic];
    float w = w_a + w_c;
    if (w <= 0.0f) return;  // two immovable bodies

    float dist = sqrtf(dist_sq);
    float n[3] = {d[0] / dist, d[1] / dist, d[2] / dist};  // from a to c

    // Velocity of c relative to a along n; negative means they approach.
    float vn = 0.0f;
    for (int k = 0; k < 3; k++) vn += (b.vel[k][c] - b.vel[k][a]) * n[k];

    float impulse = 0.0f;
    if (vn < 0.0f) {
        float e = 0.5f * (b.restitution[ia] + b.restitution[ic]);
        impulse = -(1.0f + e) * vn / w;
    }
    float correction = p.contact_correction * (min_dist - dist) / w;

    // a is pushed along -n, c along +n, each scaled by its inverse mass.
    float s = (g == a) ? -w_a : w_c;
    for (int k = 0; k < 3; k++) {
        dv[k] += s * impulse * n[k];
        dp[k] += s * correction * n[k];
    }
}

PHYS_HD void store_deltas(const Buffers& b, int g, const float dv[3], const float dp[3]) {
    for (int k = 0; k < 3; k++) {
        b.dvel[k][g] = dv[k];
        b.dpos[k][g] = dp[k];
    }
}

PHYS_HD void collide_all_pairs(const Params& p, const Buffers& b, int g) {
    float dv[3] = {0.0f, 0.0f, 0.0f};
    float dp[3] = {0.0f, 0.0f, 0.0f};
    if (b.enabled[g]) {
        int base = (g / p.nbody) * p.nbody;
        for (int h = base; h < base + p.nbody; h++) {
            if (h != g && b.enabled[h]) accumulate_contact(p, b, g, h, dv, dp);
        }
    }
    store_deltas(b, g, dv, dp);
}

// Visit the 27 cells around g in a fixed order. Disabled bodies carry the
// sentinel key, so they never appear in a cell.
PHYS_HD void collide_grid(const Params& p, const Buffers& b, int g) {
    float dv[3] = {0.0f, 0.0f, 0.0f};
    float dp[3] = {0.0f, 0.0f, 0.0f};
    if (b.enabled[g]) {
        int env_base = (g / p.nbody) * p.ncell;
        int cx = grid_coord(p, 0, b.pos[0][g]);
        int cy = grid_coord(p, 1, b.pos[1][g]);
        int cz = grid_coord(p, 2, b.pos[2][g]);
        for (int z = cz - 1; z <= cz + 1; z++) {
            if (z < 0 || z >= p.dim[2]) continue;
            for (int y = cy - 1; y <= cy + 1; y++) {
                if (y < 0 || y >= p.dim[1]) continue;
                for (int x = cx - 1; x <= cx + 1; x++) {
                    if (x < 0 || x >= p.dim[0]) continue;
                    int k = env_base + grid_cell(p, x, y, z);
                    int start = b.cell_start[k];
                    if (start < 0) continue;
                    int end = b.cell_end[k];
                    for (int s = start; s < end; s++) {
                        int h = b.sorted_index[s];
                        if (h != g) accumulate_contact(p, b, g, h, dv, dp);
                    }
                }
            }
        }
    }
    store_deltas(b, g, dv, dp);
}

PHYS_HD void apply_deltas(const Buffers& b, int g) {
    for (int k = 0; k < 3; k++) {
        b.pos[k][g] += b.dpos[k][g];
        b.vel[k][g] += b.dvel[k][g];
    }
}

}  // namespace detail
}  // namespace phys
