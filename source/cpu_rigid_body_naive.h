#pragma once

// Naive O(n^2) CPU rigid body simulation - brute force collision detection.
// This represents the baseline CPU approach without spatial acceleration.
void cpu_rigid_body_simulate_naive(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    const float* radius, const float* mass, const float* inv_mass,
    const float* restitution,
    int num_bodies,
    int num_steps,
    float dt,
    float gravity,
    float domain_size
);
