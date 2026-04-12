// benchmark.cpp
// Benchmark harness comparing CPU (naive + optimized) vs GPU rigid body simulation.

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "RigidBody.h"
#include "cpu_rigid_body.h"
#include "cpu_rigid_body_naive.h"
#include "cuda_rigid_body.h"

using Clock = std::chrono::high_resolution_clock;

struct BenchResult {
    int num_bodies;
    int num_steps;
    double cpu_naive_ms;
    double cpu_opt_ms;
    double gpu_ms;
    double speedup_vs_naive;
    double speedup_vs_opt;
};

static BenchResult run_benchmark(int num_bodies, int num_steps,
                                  bool run_cpu_naive, bool run_cpu_opt) {
    // Domain sized for ~20% packing fraction - realistic dense scenario
    float min_radius = 0.3f;
    float max_radius = 0.5f;
    float avg_radius = 0.5f * (min_radius + max_radius);
    float avg_vol = (4.0f / 3.0f) * M_PI * avg_radius * avg_radius * avg_radius;
    float total_vol = avg_vol * num_bodies;
    float domain_size = cbrtf(total_vol / 0.20f);
    float dt = 0.001f;
    float gravity = -9.81f;

    float cell_size = max_radius * 2.0f;
    int grid_dim = (int)ceilf(domain_size / cell_size);
    if (grid_dim > 512) grid_dim = 512;

    RigidBodySystem sys;
    sys.initRandom(num_bodies, domain_size, min_radius, max_radius);

    BenchResult result;
    result.num_bodies = num_bodies;
    result.num_steps = num_steps;
    result.cpu_naive_ms = 0;
    result.cpu_opt_ms = 0;
    result.gpu_ms = 0;

    // --- Naive CPU benchmark (O(n^2)) ---
    if (run_cpu_naive) {
        RigidBodySystem cpu_sys;
        cpu_sys.copyFrom(sys);
        auto t0 = Clock::now();
        cpu_rigid_body_simulate_naive(
            cpu_sys.px, cpu_sys.py, cpu_sys.pz,
            cpu_sys.vx, cpu_sys.vy, cpu_sys.vz,
            cpu_sys.radius, cpu_sys.mass, cpu_sys.inv_mass, cpu_sys.restitution,
            num_bodies, num_steps, dt, gravity, domain_size);
        auto t1 = Clock::now();
        result.cpu_naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cpu_sys.free();
    }

    // --- Optimized CPU benchmark (spatial hash) ---
    if (run_cpu_opt) {
        RigidBodySystem cpu_sys;
        cpu_sys.copyFrom(sys);
        auto t0 = Clock::now();
        cpu_rigid_body_simulate(
            cpu_sys.px, cpu_sys.py, cpu_sys.pz,
            cpu_sys.vx, cpu_sys.vy, cpu_sys.vz,
            cpu_sys.radius, cpu_sys.mass, cpu_sys.inv_mass, cpu_sys.restitution,
            num_bodies, num_steps, dt, gravity, domain_size, cell_size, grid_dim);
        auto t1 = Clock::now();
        result.cpu_opt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cpu_sys.free();
    }

    // --- GPU benchmark ---
    {
        RigidBodySystem gpu_sys;
        gpu_sys.copyFrom(sys);

        // Warm-up run
        cuda_rigid_body_simulate(
            gpu_sys.px, gpu_sys.py, gpu_sys.pz,
            gpu_sys.vx, gpu_sys.vy, gpu_sys.vz,
            gpu_sys.radius, gpu_sys.mass, gpu_sys.inv_mass, gpu_sys.restitution,
            num_bodies, 2, dt, gravity, domain_size, cell_size, grid_dim);
        gpu_sys.free();

        // Fresh state for actual benchmark
        gpu_sys.copyFrom(sys);
        auto t0 = Clock::now();
        cuda_rigid_body_simulate(
            gpu_sys.px, gpu_sys.py, gpu_sys.pz,
            gpu_sys.vx, gpu_sys.vy, gpu_sys.vz,
            gpu_sys.radius, gpu_sys.mass, gpu_sys.inv_mass, gpu_sys.restitution,
            num_bodies, num_steps, dt, gravity, domain_size, cell_size, grid_dim);
        auto t1 = Clock::now();
        result.gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        gpu_sys.free();
    }

    result.speedup_vs_naive = (result.cpu_naive_ms > 0) ? result.cpu_naive_ms / result.gpu_ms : 0;
    result.speedup_vs_opt = (result.cpu_opt_ms > 0) ? result.cpu_opt_ms / result.gpu_ms : 0;
    sys.free();
    return result;
}

int main(int argc, char** argv) {
    srand(42);

    std::cout << "================================================================\n";
    std::cout << "  GPU-Accelerated Rigid Body Physics Engine - Benchmark\n";
    std::cout << "================================================================\n";
    std::cout << "  Sphere-sphere collisions | Impulse-based resolution\n";
    std::cout << "  GPU: Spatial hash broadphase + CUDA parallel narrowphase\n";
    std::cout << "  SoA memory layout for GPU memory coalescing\n";
    std::cout << "  CPU Naive: O(n^2) brute-force | CPU Opt: spatial hash\n";
    std::cout << "================================================================\n\n";

    struct Config {
        int num_bodies;
        int num_steps;
        bool run_naive;
        bool run_opt;
    };

    std::vector<Config> configs = {
        {1000,    50,  true,  true},
        {5000,    50,  true,  true},
        {10000,   50,  true,  true},
        {25000,   50,  true,  true},
        {50000,   50,  true,  true},
        {100000,  50,  false, true},
        {200000,  20,  false, false},
        {500000,  10,  false, false},
    };

    if (argc > 1) {
        int n = atoi(argv[1]);
        int steps = (argc > 2) ? atoi(argv[2]) : 50;
        bool naive = (argc > 3) ? (atoi(argv[3]) != 0) : (n <= 50000);
        bool opt = (argc > 4) ? (atoi(argv[4]) != 0) : (n <= 100000);
        configs = {{n, steps, naive, opt}};
    }

    std::cout << std::left
              << std::setw(10) << "Bodies"
              << std::setw(8)  << "Steps"
              << std::setw(14) << "CPU Naive"
              << std::setw(14) << "CPU Opt"
              << std::setw(14) << "GPU (ms)"
              << std::setw(14) << "vs Naive"
              << std::setw(14) << "vs Opt"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    for (auto& cfg : configs) {
        std::cout << std::flush;
        auto r = run_benchmark(cfg.num_bodies, cfg.num_steps, cfg.run_naive, cfg.run_opt);

        std::cout << std::left << std::fixed << std::setprecision(1)
                  << std::setw(10) << r.num_bodies
                  << std::setw(8)  << r.num_steps;

        if (cfg.run_naive)
            std::cout << std::setw(14) << r.cpu_naive_ms;
        else
            std::cout << std::setw(14) << "---";

        if (cfg.run_opt)
            std::cout << std::setw(14) << r.cpu_opt_ms;
        else
            std::cout << std::setw(14) << "---";

        std::cout << std::setw(14) << r.gpu_ms;

        if (cfg.run_naive)
            std::cout << std::setw(14) << (std::to_string((int)r.speedup_vs_naive) + "x");
        else
            std::cout << std::setw(14) << "---";

        if (cfg.run_opt)
            std::cout << std::setw(14) << (std::to_string((int)r.speedup_vs_opt) + "x");
        else
            std::cout << std::setw(14) << "---";

        std::cout << "\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << "  Benchmark complete.\n";
    std::cout << "================================================================\n";

    return 0;
}
