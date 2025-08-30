// cuda_mass_spring.cu
#include <cuda_runtime.h>
#include <iostream>


extern "C" __global__ void simulate_coupled_mass_spring(
    float* positions, float* velocities,
    const float* initialPositions, const float* initialVelocities,
    const float* masses, const float* dampings, const float* springConstants,
    const int* couplingA, const int* couplingB, const float* couplingConstants, int numCouplings,
    int numMasses, int steps, float h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numMasses) return;

    float x = initialPositions[idx];
    float v = initialVelocities[idx];
    float m = masses[idx];
    float c = dampings[idx];
    float k = springConstants[idx];

    for (int i = 0; i < steps; ++i) {
        // Compute spring force
        float force = -k * x - c * v;
        // Add coupling forces
        for (int j = 0; j < numCouplings; ++j) {
            if (couplingA[j] == idx) {
                force += -couplingConstants[j] * (x - initialPositions[couplingB[j]]);
            } else if (couplingB[j] == idx) {
                force += -couplingConstants[j] * (x - initialPositions[couplingA[j]]);
            }
        }
        float a = force / m;
        v += a * h;
        x += v * h;
    }
    positions[idx] = x;
    velocities[idx] = v;
}

#include "cuda_mass_spring.h"
#include <cuda_runtime.h>
#include <vector>

void cuda_coupled_mass_spring(
    float* positions, float* velocities,
    const float* initialPositions, const float* initialVelocities,
    const float* masses, const float* dampings, const float* springConstants,
    const int* couplingA, const int* couplingB, const float* couplingConstants, int numCouplings,
    int numMasses, int steps, float h)
{
    float *d_pos, *d_vel, *d_init_pos, *d_init_vel, *d_masses, *d_dampings, *d_springConstants, *d_couplingConstants;
    int *d_couplingA, *d_couplingB;

    cudaMalloc(&d_pos, numMasses * sizeof(float));
    cudaMalloc(&d_vel, numMasses * sizeof(float));
    cudaMalloc(&d_init_pos, numMasses * sizeof(float));
    cudaMalloc(&d_init_vel, numMasses * sizeof(float));
    cudaMalloc(&d_masses, numMasses * sizeof(float));
    cudaMalloc(&d_dampings, numMasses * sizeof(float));
    cudaMalloc(&d_springConstants, numMasses * sizeof(float));
    cudaMalloc(&d_couplingConstants, numCouplings * sizeof(float));
    cudaMalloc(&d_couplingA, numCouplings * sizeof(int));
    cudaMalloc(&d_couplingB, numCouplings * sizeof(int));

    cudaMemcpy(d_init_pos, initialPositions, numMasses * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_init_vel, initialVelocities, numMasses * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, masses, numMasses * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dampings, dampings, numMasses * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_springConstants, springConstants, numMasses * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_couplingConstants, couplingConstants, numCouplings * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_couplingA, couplingA, numCouplings * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_couplingB, couplingB, numCouplings * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (numMasses + threads - 1) / threads;
    simulate_coupled_mass_spring<<<blocks, threads>>>(d_pos, d_vel, d_init_pos, d_init_vel, d_masses, d_dampings, d_springConstants,
        d_couplingA, d_couplingB, d_couplingConstants, numCouplings, numMasses, steps, h);

    cudaMemcpy(positions, d_pos, numMasses * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, d_vel, numMasses * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_pos); cudaFree(d_vel); cudaFree(d_init_pos); cudaFree(d_init_vel);
    cudaFree(d_masses); cudaFree(d_dampings); cudaFree(d_springConstants);
    cudaFree(d_couplingConstants); cudaFree(d_couplingA); cudaFree(d_couplingB);
}


