#pragma once

#include <vector>

// Launches the CUDA kernel for a coupled mass-spring system
// positions, velocities: input/output arrays (size numMasses)
// initialPositions, initialVelocities: input arrays (size numMasses)
// masses, dampings, springConstants: input arrays (size numMasses)
// couplings: pairs of indices (size numCouplings)
// couplingConstants: input array (size numCouplings)
// steps: number of integration steps
// h: timestep
extern void cuda_coupled_mass_spring(
    float* positions, float* velocities,
    const float* initialPositions, const float* initialVelocities,
    const float* masses, const float* dampings, const float* springConstants,
    const int* couplingA, const int* couplingB, const float* couplingConstants, int numCouplings,
    int numMasses, int steps, float h);
