#pragma once

// CPU single-threaded rigid body simulation (reference implementation for benchmarking).
// Identical physics to the GPU version: semi-implicit Euler + spatial hash broadphase +
// sphere-sphere collision with impulse-based resolution.

void cpu_rigid_body_simulate(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    const float* radius, const float* mass, const float* inv_mass,
    const float* restitution,
    int num_bodies,
    int num_steps,
    float dt,
    float gravity,
    float domain_size,
    float cell_size,
    int grid_dim
);
