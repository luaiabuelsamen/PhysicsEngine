#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>

// Structure-of-Arrays layout for GPU memory coalescing.
// Each field is a contiguous array so that threads in a warp access
// adjacent memory locations, maximizing memory bandwidth on the GPU.
struct RigidBodySystem {
    // Positions
    float* px;
    float* py;
    float* pz;

    // Velocities
    float* vx;
    float* vy;
    float* vz;

    // Per-body properties
    float* radius;
    float* mass;
    float* inv_mass;
    float* restitution;

    // Spatial hash grid
    int* grid_hash;
    int* grid_index;  // original index before sorting

    int num_bodies;

    void allocate(int n) {
        num_bodies = n;
        px = new float[n]; py = new float[n]; pz = new float[n];
        vx = new float[n]; vy = new float[n]; vz = new float[n];
        radius = new float[n];
        mass = new float[n];
        inv_mass = new float[n];
        restitution = new float[n];
        grid_hash = new int[n];
        grid_index = new int[n];
    }

    void free() {
        delete[] px; delete[] py; delete[] pz;
        delete[] vx; delete[] vy; delete[] vz;
        delete[] radius; delete[] mass; delete[] inv_mass; delete[] restitution;
        delete[] grid_hash; delete[] grid_index;
    }

    void initRandom(int n, float domain_size, float min_radius, float max_radius) {
        allocate(n);
        for (int i = 0; i < n; i++) {
            radius[i] = min_radius + ((float)rand() / RAND_MAX) * (max_radius - min_radius);
            mass[i] = (4.0f / 3.0f) * M_PI * radius[i] * radius[i] * radius[i]; // density=1
            inv_mass[i] = 1.0f / mass[i];
            restitution[i] = 0.8f;

            px[i] = ((float)rand() / RAND_MAX) * domain_size;
            py[i] = ((float)rand() / RAND_MAX) * domain_size;
            pz[i] = ((float)rand() / RAND_MAX) * domain_size;

            float speed = 1.0f;
            vx[i] = (((float)rand() / RAND_MAX) - 0.5f) * speed;
            vy[i] = (((float)rand() / RAND_MAX) - 0.5f) * speed;
            vz[i] = (((float)rand() / RAND_MAX) - 0.5f) * speed;
        }
    }

    void copyFrom(const RigidBodySystem& other) {
        allocate(other.num_bodies);
        int n = other.num_bodies;
        memcpy(px, other.px, n * sizeof(float));
        memcpy(py, other.py, n * sizeof(float));
        memcpy(pz, other.pz, n * sizeof(float));
        memcpy(vx, other.vx, n * sizeof(float));
        memcpy(vy, other.vy, n * sizeof(float));
        memcpy(vz, other.vz, n * sizeof(float));
        memcpy(radius, other.radius, n * sizeof(float));
        memcpy(mass, other.mass, n * sizeof(float));
        memcpy(inv_mass, other.inv_mass, n * sizeof(float));
        memcpy(restitution, other.restitution, n * sizeof(float));
        memcpy(grid_hash, other.grid_hash, n * sizeof(int));
        memcpy(grid_index, other.grid_index, n * sizeof(int));
    }
};

// Spatial grid parameters
struct GridParams {
    float cell_size;
    int grid_dim;     // number of cells per axis
    float origin_x, origin_y, origin_z;

    int totalCells() const { return grid_dim * grid_dim * grid_dim; }

    void init(float domain_size, float max_radius) {
        cell_size = max_radius * 2.0f;  // cell must be >= max object diameter
        grid_dim = (int)ceilf(domain_size / cell_size);
        origin_x = origin_y = origin_z = 0.0f;
    }
};
