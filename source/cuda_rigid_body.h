#pragma once

// GPU-accelerated rigid body simulation interface.
// All functions transfer data to/from GPU internally.

// Run one simulation step on the GPU:
//   1. Semi-implicit Euler integration
//   2. Spatial hash broadphase
//   3. Sphere-sphere collision detection + impulse resolution
//   4. Boundary enforcement
void cuda_rigid_body_step(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    const float* radius, const float* mass, const float* inv_mass,
    const float* restitution,
    int num_bodies,
    float dt,
    float gravity,
    float domain_size,
    float cell_size,
    int grid_dim
);

// Batch version: run multiple steps without host-device round-trips
void cuda_rigid_body_simulate(
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
