#pragma once

// Hand-written single-threaded CPU baseline for the benchmark.
// Same integrator, spatial hash broadphase and contact model as libphys, but it
// visits each pair once and updates both bodies in place (Gauss-Seidel style),
// which is the fastest way to do it on one core. Its trajectories therefore
// differ slightly from libphys; libphys's own CPU backend is the exact reference.

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
