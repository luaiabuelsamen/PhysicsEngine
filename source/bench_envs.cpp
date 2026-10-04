// bench_envs.cpp
// Throughput of the rigid solver on batched environments, the RL use case.
// Every env holds a ground plane and a pile of mixed shapes (spheres,
// capsules, boxes) that fall and settle, so contacts are active throughout.
// Reports env-steps per second (one env advanced by one 1/60 s step).
//
//   ./bench_envs [bodies_per_env] [substeps]

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

#include "phys/phys.h"

using namespace phys;
using Clock = std::chrono::high_resolution_clock;

namespace {

const float kDt = 1.0f / 60.0f;

ModelDesc make_model(int nbody, int substeps) {
    ModelDesc desc;
    desc.substeps = substeps;
    desc.bodies.push_back(BodyDesc::plane());
    for (int i = 1; i < nbody; i++) {
        switch (i % 3) {
            case 0: desc.bodies.push_back(BodyDesc::sphere(0.2f, 1.0f)); break;
            case 1: desc.bodies.push_back(BodyDesc::capsule(0.1f, 0.25f, 1.0f)); break;
            default: desc.bodies.push_back(BodyDesc::box({0.2f, 0.15f, 0.25f}, 1.0f)); break;
        }
    }
    return desc;
}

HostState make_state(int nbody, int nenv) {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    HostState s(nbody * nenv);
    for (int e = 0; e < nenv; e++) {
        for (int i = 1; i < nbody; i++) {
            int g = e * nbody + i;
            s.px[g] = unit(rng) - 0.5f;
            s.py[g] = 0.3f + 0.45f * i;
            s.pz[g] = unit(rng) - 0.5f;
            float ax = unit(rng) - 0.5f, ay = unit(rng) - 0.5f, az = unit(rng) - 0.5f;
            float n = std::sqrt(ax * ax + ay * ay + az * az) + 1e-6f;
            float angle = 3.0f * unit(rng), sn = std::sin(angle / 2) / n;
            s.qw[g] = std::cos(angle / 2);
            s.qx[g] = ax * sn; s.qy[g] = ay * sn; s.qz[g] = az * sn;
        }
    }
    return s;
}

// Seconds for `nsteps` steps, after a warm-up step.
double time_world(World& world, const HostState& s, int nsteps) {
    world.set_state(s);
    world.step(kDt, 1);
    world.synchronize();
    world.set_state(s);
    world.synchronize();
    auto t0 = Clock::now();
    world.step(kDt, nsteps);
    world.synchronize();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    int nbody = argc > 1 ? std::atoi(argv[1]) : 11;
    int substeps = argc > 2 ? std::atoi(argv[2]) : 10;
    const int nsteps = 60;  // one simulated second

    std::printf("Rigid solver throughput: %d bodies per env (ground + %d shapes), %d substeps,\n"
                "%d steps of 1/60 s per run. Higher is better.\n\n",
                nbody, nbody - 1, substeps, nsteps);
    std::printf("%-8s %16s %16s %10s\n", "Envs", "CPU env-steps/s", "GPU env-steps/s", "GPU/CPU");
    std::printf("----------------------------------------------------------\n");

    ModelDesc desc = make_model(nbody, substeps);
    for (int nenv : {1, 16, 64, 256, 1024, 4096, 16384}) {
        HostState s = make_state(nbody, nenv);
        double cpu_rate = 0.0;
        if (nenv <= 256) {
            World cpu(desc, nenv, Device::CPU);
            cpu_rate = nenv * nsteps / time_world(cpu, s, nsteps);
        }
        World gpu(desc, nenv, Device::CUDA);
        double gpu_rate = nenv * nsteps / time_world(gpu, s, nsteps);

        if (cpu_rate > 0)
            std::printf("%-8d %16.0f %16.0f %9.1fx\n", nenv, cpu_rate, gpu_rate, gpu_rate / cpu_rate);
        else
            std::printf("%-8d %16s %16.0f %10s\n", nenv, "---", gpu_rate, "---");
        std::fflush(stdout);
    }
    return 0;
}
