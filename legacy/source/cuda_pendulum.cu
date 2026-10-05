// cuda_pendulum.cu
// CUDA kernel for massively parallel double pendulum integration.
// Each thread integrates one independent double pendulum using RK4.

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Double pendulum derivatives computed on device
__device__ void pendulum_derivs(
    float theta1, float theta2, float omega1, float omega2,
    float g, float L1, float L2, float m1, float m2,
    float& dtheta1, float& dtheta2, float& domega1, float& domega2)
{
    float dt = theta1 - theta2;
    float sin_dt = sinf(dt);
    float cos_dt = cosf(dt);
    float denom = 2.0f * m1 + m2 - m2 * cosf(2.0f * dt);

    dtheta1 = omega1;
    dtheta2 = omega2;

    domega1 = (-g * (2*m1+m2) * sinf(theta1)
               - m2 * g * sinf(theta1 - 2*theta2)
               - 2 * sin_dt * m2 * (omega2*omega2*L2 + omega1*omega1*L1*cos_dt))
              / (L1 * denom);

    domega2 = (2 * sin_dt * (omega1*omega1*L1*(m1+m2)
               + g * (m1+m2) * cosf(theta1)
               + omega2*omega2*L2*m2*cos_dt))
              / (L2 * denom);
}

// Each thread integrates one double pendulum for `substeps` RK4 steps
__global__ void integrate_pendulums_kernel(
    float* __restrict__ theta1, float* __restrict__ theta2,
    float* __restrict__ omega1, float* __restrict__ omega2,
    float* __restrict__ tip_x,  float* __restrict__ tip_y,
    int num_pendulums, int substeps, float dt,
    float g, float L1, float L2, float m1, float m2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pendulums) return;

    float t1 = theta1[i], t2 = theta2[i];
    float w1 = omega1[i], w2 = omega2[i];

    for (int s = 0; s < substeps; s++) {
        // RK4
        float k1t1, k1t2, k1w1, k1w2;
        float k2t1, k2t2, k2w1, k2w2;
        float k3t1, k3t2, k3w1, k3w2;
        float k4t1, k4t2, k4w1, k4w2;

        pendulum_derivs(t1, t2, w1, w2, g, L1, L2, m1, m2,
                         k1t1, k1t2, k1w1, k1w2);

        pendulum_derivs(t1 + 0.5f*dt*k1t1, t2 + 0.5f*dt*k1t2,
                         w1 + 0.5f*dt*k1w1, w2 + 0.5f*dt*k1w2,
                         g, L1, L2, m1, m2,
                         k2t1, k2t2, k2w1, k2w2);

        pendulum_derivs(t1 + 0.5f*dt*k2t1, t2 + 0.5f*dt*k2t2,
                         w1 + 0.5f*dt*k2w1, w2 + 0.5f*dt*k2w2,
                         g, L1, L2, m1, m2,
                         k3t1, k3t2, k3w1, k3w2);

        pendulum_derivs(t1 + dt*k3t1, t2 + dt*k3t2,
                         w1 + dt*k3w1, w2 + dt*k3w2,
                         g, L1, L2, m1, m2,
                         k4t1, k4t2, k4w1, k4w2);

        t1 += dt/6.0f * (k1t1 + 2*k2t1 + 2*k3t1 + k4t1);
        t2 += dt/6.0f * (k1t2 + 2*k2t2 + 2*k3t2 + k4t2);
        w1 += dt/6.0f * (k1w1 + 2*k2w1 + 2*k3w1 + k4w1);
        w2 += dt/6.0f * (k1w2 + 2*k2w2 + 2*k3w2 + k4w2);
    }

    theta1[i] = t1; theta2[i] = t2;
    omega1[i] = w1; omega2[i] = w2;

    // Compute tip position
    float x1 = L1 * sinf(t1);
    float y1 = L1 * cosf(t1);
    tip_x[i] = x1 + L2 * sinf(t2);
    tip_y[i] = y1 + L2 * cosf(t2);
}

// Host wrapper
extern "C" void cuda_integrate_pendulums(
    float* theta1, float* theta2,
    float* omega1, float* omega2,
    float* tip_x, float* tip_y,
    int num_pendulums, int substeps, float dt,
    float g, float L1, float L2, float m1, float m2)
{
    size_t bytes = num_pendulums * sizeof(float);
    float *d_t1, *d_t2, *d_w1, *d_w2, *d_tx, *d_ty;

    CUDA_CHECK(cudaMalloc(&d_t1, bytes));
    CUDA_CHECK(cudaMalloc(&d_t2, bytes));
    CUDA_CHECK(cudaMalloc(&d_w1, bytes));
    CUDA_CHECK(cudaMalloc(&d_w2, bytes));
    CUDA_CHECK(cudaMalloc(&d_tx, bytes));
    CUDA_CHECK(cudaMalloc(&d_ty, bytes));

    CUDA_CHECK(cudaMemcpy(d_t1, theta1, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t2, theta2, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, omega1, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, omega2, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (num_pendulums + threads - 1) / threads;

    integrate_pendulums_kernel<<<blocks, threads>>>(
        d_t1, d_t2, d_w1, d_w2, d_tx, d_ty,
        num_pendulums, substeps, dt, g, L1, L2, m1, m2);

    CUDA_CHECK(cudaMemcpy(theta1, d_t1, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(theta2, d_t2, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(omega1, d_w1, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(omega2, d_w2, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tip_x, d_tx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tip_y, d_ty, bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_t1); cudaFree(d_t2);
    cudaFree(d_w1); cudaFree(d_w2);
    cudaFree(d_tx); cudaFree(d_ty);
}
