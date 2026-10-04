// benchmark.cpp
// Benchmark harness: libphys on the GPU against single-threaded CPU code.
//
//   CPU Naive  hand-written O(n^2) brute force (cpu_rigid_body_naive.cpp)
//   CPU Opt    hand-written spatial hash, one pass per pair (cpu_rigid_body.cpp)
//   CPU Ref    libphys CPU backend: same algorithm as the GPU, bit-exact
//   GPU        libphys CUDA backend
//
// Every run starts from the same random scene. The GPU and CPU Ref columns
// time only the steps (state already resident), which is how an RL loop or a
// long simulation uses them.

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cpu_rigid_body.h"
#include "cpu_rigid_body_naive.h"
#include "phys/phys.h"

using Clock = std::chrono::high_resolution_clock;

namespace {

const float kDt = 0.001f;
const float kGravity = -9.81f;
const float kMinRadius = 0.3f;
const float kMaxRadius = 0.5f;

struct Scene {
    phys::ModelDesc desc;
    phys::HostState state;
    std::vector<float> radius, mass, inv_mass, restitution;  // for the baselines
    float domain_size;
};

// Random spheres in a cube sized for ~20% packing fraction.
Scene make_scene(int num_bodies) {
    Scene sc;
    float avg_radius = 0.5f * (kMinRadius + kMaxRadius);
    float avg_vol = (4.0f / 3.0f) * M_PI * avg_radius * avg_radius * avg_radius;
    sc.domain_size = cbrtf(avg_vol * num_bodies / 0.20f);

    sc.desc.solver = phys::Solver::Particle;
    sc.desc.gravity = {0.0f, kGravity, 0.0f};
    sc.desc.bounds_lo = {0.0f, 0.0f, 0.0f};
    sc.desc.bounds_hi = {sc.domain_size, sc.domain_size, sc.domain_size};
    sc.desc.wall_restitution = 0.9f;
    sc.desc.broadphase = phys::Broadphase::Grid;

    sc.state.resize(num_bodies);
    for (int i = 0; i < num_bodies; i++) {
        float r = kMinRadius + ((float)rand() / RAND_MAX) * (kMaxRadius - kMinRadius);
        float m = (4.0f / 3.0f) * M_PI * r * r * r;  // density 1
        phys::BodyDesc body = phys::BodyDesc::sphere(r, m);
        body.restitution = 0.8f;
        sc.desc.bodies.push_back(body);
        sc.radius.push_back(r);
        sc.mass.push_back(m);
        sc.inv_mass.push_back(1.0f / m);
        sc.restitution.push_back(body.restitution);

        sc.state.px[i] = ((float)rand() / RAND_MAX) * sc.domain_size;
        sc.state.py[i] = ((float)rand() / RAND_MAX) * sc.domain_size;
        sc.state.pz[i] = ((float)rand() / RAND_MAX) * sc.domain_size;
        sc.state.vx[i] = ((float)rand() / RAND_MAX) - 0.5f;
        sc.state.vy[i] = ((float)rand() / RAND_MAX) - 0.5f;
        sc.state.vz[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }
    return sc;
}

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

template <typename Fn>
double time_baseline(const Scene& sc, Fn simulate) {
    phys::HostState s = sc.state;
    auto t0 = Clock::now();
    simulate(s);
    return ms_since(t0);
}

// Times `num_steps` steps on an already-loaded world; fills `out` with the result.
double time_phys(const Scene& sc, phys::Device device, int num_steps, phys::HostState& out) {
    phys::World world(sc.desc, 1, device);
    if (device == phys::Device::CUDA) {  // warm up kernels and caches
        world.set_state(sc.state);
        world.step(kDt, 2);
        world.synchronize();
    }
    world.set_state(sc.state);
    world.synchronize();
    auto t0 = Clock::now();
    world.step(kDt, num_steps);
    world.synchronize();
    double ms = ms_since(t0);
    world.get_state(out);
    return ms;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::string fmt_ms(double ms, bool ran) {
    if (!ran) return "---";
    std::ostringstream os;
    os << std::fixed << std::setprecision(1) << ms;
    return os.str();
}

std::string fmt_speedup(double base_ms, double gpu_ms, bool ran) {
    if (!ran) return "---";
    std::ostringstream os;
    os << std::fixed << std::setprecision(1) << base_ms / gpu_ms << "x";
    return os.str();
}

}  // namespace

int main(int argc, char** argv) {
    srand(42);

    std::cout << "================================================================\n";
    std::cout << "  GPU-Accelerated Rigid Body Physics Engine - Benchmark\n";
    std::cout << "================================================================\n";
    std::cout << "  CPU Naive: O(n^2) brute force | CPU Opt: hand-written spatial hash\n";
    std::cout << "  CPU Ref:   libphys CPU backend (bit-exact reference for GPU)\n";
    std::cout << "  GPU:       libphys CUDA backend, state resident on the GPU\n";
    std::cout << "================================================================\n\n";

    struct Config {
        int num_bodies;
        int num_steps;
        bool run_naive;
        bool run_cpu;  // CPU Opt and CPU Ref
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
        bool cpu = (argc > 4) ? (atoi(argv[4]) != 0) : (n <= 100000);
        configs = {{n, steps, naive, cpu}};
    }

    std::cout << std::left
              << std::setw(10) << "Bodies"
              << std::setw(7)  << "Steps"
              << std::setw(13) << "CPU Naive"
              << std::setw(12) << "CPU Opt"
              << std::setw(12) << "CPU Ref"
              << std::setw(10) << "GPU"
              << std::setw(10) << "vs Naive"
              << std::setw(9)  << "vs Opt"
              << std::setw(9)  << "vs Ref"
              << "GPU==Ref"
              << "\n";
    std::cout << std::string(100, '-') << "\n";

    for (const Config& cfg : configs) {
        Scene sc = make_scene(cfg.num_bodies);
        int n = cfg.num_bodies;
        float cell_size = 2.0f * kMaxRadius;
        int grid_dim = (int)ceilf(sc.domain_size / cell_size);

        double naive_ms = 0, opt_ms = 0, ref_ms = 0;
        if (cfg.run_naive) {
            naive_ms = time_baseline(sc, [&](phys::HostState& s) {
                cpu_rigid_body_simulate_naive(
                    s.px.data(), s.py.data(), s.pz.data(), s.vx.data(), s.vy.data(), s.vz.data(),
                    sc.radius.data(), sc.mass.data(), sc.inv_mass.data(),
                    sc.restitution.data(), n, cfg.num_steps, kDt, kGravity, sc.domain_size);
            });
        }
        phys::HostState ref_out, gpu_out;
        if (cfg.run_cpu) {
            opt_ms = time_baseline(sc, [&](phys::HostState& s) {
                cpu_rigid_body_simulate(
                    s.px.data(), s.py.data(), s.pz.data(), s.vx.data(), s.vy.data(), s.vz.data(),
                    sc.radius.data(), sc.mass.data(), sc.inv_mass.data(),
                    sc.restitution.data(), n, cfg.num_steps, kDt, kGravity, sc.domain_size,
                    cell_size, grid_dim);
            });
            ref_ms = time_phys(sc, phys::Device::CPU, cfg.num_steps, ref_out);
        }
        double gpu_ms = time_phys(sc, phys::Device::CUDA, cfg.num_steps, gpu_out);

        std::string match = "---";
        if (cfg.run_cpu) {
            bool same = same_bits(ref_out.px, gpu_out.px) && same_bits(ref_out.py, gpu_out.py) &&
                        same_bits(ref_out.pz, gpu_out.pz) && same_bits(ref_out.vx, gpu_out.vx) &&
                        same_bits(ref_out.vy, gpu_out.vy) && same_bits(ref_out.vz, gpu_out.vz);
            match = same ? "yes" : "no";
        }

        std::cout << std::left
                  << std::setw(10) << n
                  << std::setw(7)  << cfg.num_steps
                  << std::setw(13) << fmt_ms(naive_ms, cfg.run_naive)
                  << std::setw(12) << fmt_ms(opt_ms, cfg.run_cpu)
                  << std::setw(12) << fmt_ms(ref_ms, cfg.run_cpu)
                  << std::setw(10) << fmt_ms(gpu_ms, true)
                  << std::setw(10) << fmt_speedup(naive_ms, gpu_ms, cfg.run_naive)
                  << std::setw(9)  << fmt_speedup(opt_ms, gpu_ms, cfg.run_cpu)
                  << std::setw(9)  << fmt_speedup(ref_ms, gpu_ms, cfg.run_cpu)
                  << match
                  << std::endl;
    }

    std::cout << "\n================================================================\n";
    std::cout << "  Benchmark complete. Times in ms.\n";
    std::cout << "================================================================\n";
    return 0;
}
