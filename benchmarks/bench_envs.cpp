// bench_envs.cpp
// Throughput of the rigid solver on batched environments, the RL use case.
// Reports env-steps per second (one env advanced by one 1/60 s step).
//
//   pile  a ground plane and a pile of mixed shapes (spheres, capsules,
//         boxes) that fall and settle, so contacts are active throughout
//   hand  a crude dexterous hand: a palm, four fingers of three capsule
//         links on position-driven hinges, and a cube held among them,
//         with every finger curling and opening on its own schedule
//
//   ./bench_envs [pile|hand|all] [substeps]

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "phys/phys.h"

using namespace phys;
using Clock = std::chrono::high_resolution_clock;

namespace {

const float kDt = 1.0f / 60.0f;

ModelDesc make_pile(int nbody, int substeps) {
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

HostState make_pile_state(int nbody, int nenv) {
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

const int kFingers = 4, kLinks = 3;
const float kLinkHalf = 0.02f, kLinkRadius = 0.01f;

// Body 0: palm (static). Bodies 1..12: finger links, finger-major. Body 13:
// a 5 cm cube resting on the palm between the fingers.
ModelDesc make_hand(int substeps) {
    ModelDesc desc;
    desc.substeps = substeps;
    desc.bodies.push_back(BodyDesc::box({0.06f, 0.01f, 0.05f}, 0.0f));
    for (int f = 0; f < kFingers; f++) {
        for (int l = 0; l < kLinks; l++) {
            BodyDesc link = BodyDesc::capsule(kLinkRadius, kLinkHalf, 0.02f);
            link.friction = 1.0f;
            desc.bodies.push_back(link);
            int child = 1 + f * kLinks + l;
            // Finger bases sit on the palm's edges, pointing up (+y); each
            // joint bends about the axis across the palm's width.
            JointDesc j = l == 0
                ? JointDesc::hinge(0, child, {-0.045f + 0.03f * f, 0.01f, f % 2 ? 0.045f : -0.045f},
                                   {0, -kLinkHalf - kLinkRadius, 0}, {1, 0, 0})
                : JointDesc::hinge(child - 1, child, {0, kLinkHalf + kLinkRadius, 0},
                                   {0, -kLinkHalf - kLinkRadius, 0}, {1, 0, 0});
            j.limited = true;
            j.lower = -1.6f;
            j.upper = 1.6f;
            j.actuator = Actuator::Position;
            j.kp = 0.5f;
            j.kd = 0.01f;
            j.max_force = 0.5f;
            desc.joints.push_back(j);
        }
    }
    BodyDesc cube = BodyDesc::box({0.025f, 0.025f, 0.025f}, 0.05f);
    cube.friction = 1.0f;
    desc.bodies.push_back(cube);
    return desc;
}

HostState make_hand_state(const ModelDesc& desc, int nenv) {
    int nbody = (int)desc.bodies.size();
    HostState s(nbody * nenv);
    for (int e = 0; e < nenv; e++) {
        for (int f = 0; f < kFingers; f++) {
            const JointDesc& base = desc.joints[f * kLinks];
            for (int l = 0; l < kLinks; l++) {
                int g = e * nbody + 1 + f * kLinks + l;
                s.px[g] = base.parent_anchor.x;
                s.py[g] = base.parent_anchor.y + (2 * l + 1) * (kLinkHalf + kLinkRadius);
                s.pz[g] = base.parent_anchor.z;
            }
        }
        s.py[e * nbody + nbody - 1] = 0.01f + 0.025f;  // cube on the palm
    }
    return s;
}

// Finger curl targets for one step: every joint follows its own sine wave.
void hand_controls(int nenv, int njoint, int step, std::vector<float>& ctrl) {
    float t = step * kDt;
    for (int e = 0; e < nenv; e++) {
        for (int j = 0; j < njoint; j++) {
            float phase = 0.37f * e + 0.9f * j;
            float side = (j / kLinks) % 2 ? 1.0f : -1.0f;  // fingers on both sides curl inwards
            ctrl[e * njoint + j] = side * (0.6f + 0.5f * std::sin(2.0f * t + phase));
        }
    }
}

// Seconds for `nsteps` steps, after a warm-up step. With `hand`, controls
// are uploaded every step (part of the timed loop, as in RL).
double time_world(World& world, const HostState& s, int nsteps, bool hand) {
    std::vector<float> ctrl((size_t)world.nenv() * world.njoint());
    world.set_state(s);
    world.step(kDt, 1);
    world.synchronize();
    world.set_state(s);
    world.synchronize();
    auto t0 = Clock::now();
    if (hand) {
        for (int i = 0; i < nsteps; i++) {
            hand_controls(world.nenv(), world.njoint(), i, ctrl);
            world.set_controls(ctrl);
            world.step(kDt, 1);
        }
    } else {
        world.step(kDt, nsteps);
    }
    world.synchronize();
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

void run(const char* name, const ModelDesc& desc, HostState (*make_state)(const ModelDesc&, int),
         bool hand) {
    const int nsteps = 60;  // one simulated second
    std::printf("\n%s: %d bodies, %d joints per env, %d substeps, %d steps of 1/60 s per run.\n",
                name, (int)desc.bodies.size(), (int)desc.joints.size(), desc.substeps, nsteps);
    std::printf("%-8s %16s %16s %10s\n", "Envs", "CPU env-steps/s", "GPU env-steps/s", "GPU/CPU");
    std::printf("----------------------------------------------------------\n");
    for (int nenv : {1, 16, 64, 256, 1024, 4096, 16384}) {
        HostState s = make_state(desc, nenv);
        double cpu_rate = 0.0;
        if (nenv <= 256) {
            World cpu(desc, nenv, Device::CPU);
            cpu_rate = nenv * nsteps / time_world(cpu, s, nsteps, hand);
        }
        World gpu(desc, nenv, Device::CUDA);
        double gpu_rate = nenv * nsteps / time_world(gpu, s, nsteps, hand);

        if (cpu_rate > 0)
            std::printf("%-8d %16.0f %16.0f %9.1fx\n", nenv, cpu_rate, gpu_rate, gpu_rate / cpu_rate);
        else
            std::printf("%-8d %16s %16.0f %10s\n", nenv, "---", gpu_rate, "---");
        std::fflush(stdout);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string which = argc > 1 ? argv[1] : "all";
    int substeps = argc > 2 ? std::atoi(argv[2]) : 10;
    std::printf("Rigid solver throughput (higher is better).\n");
    if (which == "pile" || which == "all") {
        run("pile", make_pile(11, substeps),
            [](const ModelDesc& d, int n) { return make_pile_state((int)d.bodies.size(), n); }, false);
    }
    if (which == "hand" || which == "all") run("hand", make_hand(substeps), make_hand_state, true);
    return 0;
}
