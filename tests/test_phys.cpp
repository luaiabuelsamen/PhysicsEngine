// Tests for libphys. Run through CTest (`ctest`) or directly (`./test_phys`).
// `./test_phys <substring>` runs only the tests whose name contains it.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "phys/phys.h"

using namespace phys;

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "  %s:%d: CHECK failed: %s\n", __FILE__,      \
                         __LINE__, #cond);                                     \
            g_failures++;                                                      \
        }                                                                      \
    } while (0)

static const Device kDevices[] = {Device::CPU, Device::CUDA};
static const char* device_name(Device d) { return d == Device::CPU ? "cpu" : "cuda"; }

// --- helpers -----------------------------------------------------------------

template <typename T>
static bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

static bool same_bits(const HostState& a, const HostState& b) {
    return same_bits(a.px, b.px) && same_bits(a.py, b.py) && same_bits(a.pz, b.pz) &&
           same_bits(a.vx, b.vx) && same_bits(a.vy, b.vy) && same_bits(a.vz, b.vz) &&
           same_bits(a.qw, b.qw) && same_bits(a.qx, b.qx) && same_bits(a.qy, b.qy) &&
           same_bits(a.qz, b.qz) && same_bits(a.wx, b.wx) && same_bits(a.wy, b.wy) &&
           same_bits(a.wz, b.wz) && same_bits(a.enabled, b.enabled);
}

// Copy env `src_env` of `src` into env `dst_env` of `dst`.
static void copy_env(const HostState& src, int src_env, HostState& dst, int dst_env, int nbody) {
    for (int i = 0; i < nbody; i++) {
        int s = src_env * nbody + i, d = dst_env * nbody + i;
        dst.px[d] = src.px[s]; dst.py[d] = src.py[s]; dst.pz[d] = src.pz[s];
        dst.vx[d] = src.vx[s]; dst.vy[d] = src.vy[s]; dst.vz[d] = src.vz[s];
        dst.qw[d] = src.qw[s]; dst.qx[d] = src.qx[s]; dst.qy[d] = src.qy[s]; dst.qz[d] = src.qz[s];
        dst.wx[d] = src.wx[s]; dst.wy[d] = src.wy[s]; dst.wz[d] = src.wz[s];
        dst.enabled[d] = src.enabled[s];
    }
}

static HostState simulate(const ModelDesc& desc, int nenv, Device device,
                          const HostState& initial, float dt, int nsteps) {
    World world(desc, nenv, device);
    world.set_state(initial);
    world.step(dt, nsteps);
    HostState out;
    world.get_state(out);
    return out;
}

static Quat axis_angle(Vec3 axis, float angle) {
    float n = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    float s = std::sin(angle / 2) / n;
    return {std::cos(angle / 2), axis.x * s, axis.y * s, axis.z * s};
}

static void set_quat(HostState& s, int g, Quat q) {
    s.qw[g] = q.w; s.qx[g] = q.x; s.qy[g] = q.y; s.qz[g] = q.z;
}

// Body g's local axis `axis` (0, 1, 2) expressed in the world frame.
static Vec3 body_axis(const HostState& s, int g, int axis) {
    float w = s.qw[g], x = s.qx[g], y = s.qy[g], z = s.qz[g];
    switch (axis) {
        case 0: return {1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)};
        case 1: return {2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)};
        default: return {2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)};
    }
}

static float speed(const HostState& s, int g) {
    return std::sqrt(s.vx[g] * s.vx[g] + s.vy[g] * s.vy[g] + s.vz[g] * s.vz[g]);
}

static float spin(const HostState& s, int g) {
    return std::sqrt(s.wx[g] * s.wx[g] + s.wy[g] * s.wy[g] + s.wz[g] * s.wz[g]);
}

// --- particle solver ---------------------------------------------------------

static ModelDesc particle_desc() {
    ModelDesc desc;
    desc.solver = Solver::Particle;
    return desc;
}

// A dense random box of spheres, like the benchmark scene.
static void random_particles(int nbody, int nenv, unsigned seed, float packing,
                             ModelDesc& desc, HostState& state) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    desc = particle_desc();
    float volume = 0.0f;
    for (int i = 0; i < nbody; i++) {
        float r = 0.3f + 0.2f * unit(rng);
        BodyDesc b = BodyDesc::sphere(r, 4.18879f * r * r * r);
        b.restitution = 0.8f;
        desc.bodies.push_back(b);
        volume += 4.18879f * r * r * r;
    }
    float side = std::cbrt(volume / packing);
    desc.bounds_hi = {side, side, side};

    state.resize(nbody * nenv);
    for (int g = 0; g < nbody * nenv; g++) {
        state.px[g] = side * unit(rng);
        state.py[g] = side * unit(rng);
        state.pz[g] = side * unit(rng);
        state.vx[g] = unit(rng) - 0.5f;
        state.vy[g] = unit(rng) - 0.5f;
        state.vz[g] = unit(rng) - 0.5f;
    }
}

// Two spheres meeting head on, with exact elastic-collision results.
static void test_particle_head_on() {
    for (Device dev : kDevices) {
        for (float m2 : {1.0f, 100.0f}) {
            ModelDesc desc = particle_desc();
            for (float m : {1.0f, m2}) {
                BodyDesc b = BodyDesc::sphere(0.5f, m);
                b.restitution = 1.0f;
                desc.bodies.push_back(b);
            }
            desc.gravity = {0.0f, 0.0f, 0.0f};
            desc.bounds_lo = {-10.0f, -10.0f, -10.0f};
            desc.bounds_hi = {10.0f, 10.0f, 10.0f};

            HostState s(2);
            s.px = {-1.0f, 1.0f};
            s.vx = {1.0f, m2 == 1.0f ? -1.0f : 0.0f};
            HostState out = simulate(desc, 1, dev, s, 0.01f, 150);
            if (m2 == 1.0f) {  // equal masses swap velocities
                CHECK(std::fabs(out.vx[0] + 1.0f) < 1e-5f);
                CHECK(std::fabs(out.vx[1] - 1.0f) < 1e-5f);
                CHECK(out.px[0] < -1.0f && out.px[1] > 1.0f);
                CHECK(out.px[1] - out.px[0] >= 1.0f - 1e-4f);
            } else {  // v1' = (m1-m2)/(m1+m2) v1, v2' = 2 m1/(m1+m2) v1
                CHECK(std::fabs(out.vx[0] - (-99.0f / 101.0f)) < 1e-4f);
                CHECK(std::fabs(out.vx[1] - (2.0f / 101.0f)) < 1e-4f);
            }
        }
    }
}

// Without walls or gravity, contacts must conserve total momentum.
static void test_particle_momentum() {
    for (Device dev : kDevices) {
        ModelDesc desc;
        HostState s;
        random_particles(2000, 1, 7, 0.3f, desc, s);
        desc.gravity = {0.0f, 0.0f, 0.0f};
        desc.bounds_lo = {-1000.0f, -1000.0f, -1000.0f};
        desc.bounds_hi = {1000.0f, 1000.0f, 1000.0f};

        auto momentum = [&](const HostState& st, double p[3]) {
            p[0] = p[1] = p[2] = 0.0;
            for (int i = 0; i < st.size(); i++) {
                p[0] += (double)desc.bodies[i].mass * st.vx[i];
                p[1] += (double)desc.bodies[i].mass * st.vy[i];
                p[2] += (double)desc.bodies[i].mass * st.vz[i];
            }
        };
        double scale = 0.0;
        for (const BodyDesc& b : desc.bodies) scale += b.mass * 0.5;

        double before[3], after[3];
        momentum(s, before);
        momentum(simulate(desc, 1, dev, s, 0.002f, 100), after);
        double err = 0.0;
        for (int k = 0; k < 3; k++) err = std::fmax(err, std::fabs(after[k] - before[k]));
        std::printf("  [%s] max momentum drift = %.3g (scale %.3g)\n", device_name(dev), err, scale);
        CHECK(err < 1e-4 * scale);
    }
}

// Same input, same output, every time - and the CPU backend matches the GPU.
static void test_particle_determinism_and_parity() {
    struct Case { const char* name; int nbody, nenv; Broadphase bp; };
    for (Case c : {Case{"grid, 1 env", 5000, 1, Broadphase::Grid},
                   Case{"grid, 16 envs", 300, 16, Broadphase::Grid},
                   Case{"all-pairs, 256 envs", 24, 256, Broadphase::AllPairs}}) {
        ModelDesc desc;
        HostState s;
        random_particles(c.nbody, c.nenv, 3, 0.25f, desc, s);
        desc.broadphase = c.bp;
        HostState cpu = simulate(desc, c.nenv, Device::CPU, s, 0.001f, 50);
        HostState gpu = simulate(desc, c.nenv, Device::CUDA, s, 0.001f, 50);
        HostState gpu2 = simulate(desc, c.nenv, Device::CUDA, s, 0.001f, 50);
        CHECK(same_bits(gpu, gpu2));
        float max_diff = 0.0f;
        for (int g = 0; g < cpu.size(); g++)
            max_diff = std::fmax(max_diff, std::fabs(cpu.px[g] - gpu.px[g]));
        std::printf("  [%s] max |cpu - gpu| px = %.3g\n", c.name, max_diff);
#ifdef PHYS_STRICT_FP
        CHECK(same_bits(cpu, gpu));
#else
        CHECK(max_diff < 1e-2f);
#endif
    }
}

// Every env in a batch evolves exactly as it would on its own.
static void test_particle_env_independence() {
    for (Broadphase bp : {Broadphase::AllPairs, Broadphase::Grid}) {
        for (Device dev : kDevices) {
            const int nbody = 24, nenv = 64;
            ModelDesc desc;
            HostState batch;
            random_particles(nbody, nenv, 11, 0.3f, desc, batch);
            desc.broadphase = bp;
            HostState batch_out = simulate(desc, nenv, dev, batch, 0.002f, 100);
            for (int env : {0, 17, 63}) {
                HostState single(nbody), slice(nbody);
                copy_env(batch, env, single, 0, nbody);
                copy_env(batch_out, env, slice, 0, nbody);
                CHECK(same_bits(simulate(desc, 1, dev, single, 0.002f, 100), slice));
            }
        }
    }
}

// Disabled bodies stay put and are invisible to everything else.
static void test_particle_disabled_bodies() {
    for (Device dev : kDevices) {
        ModelDesc desc = particle_desc();
        for (int i = 0; i < 3; i++) desc.bodies.push_back(BodyDesc::sphere(0.5f, 1.0f));
        desc.bounds_hi = {10.0f, 10.0f, 10.0f};

        HostState with(3), without(3);
        for (HostState* s : {&with, &without}) {
            s->px = {5.0f, 5.2f, 2.0f};
            s->py = {5.0f, 5.0f, 2.0f};
            s->pz = {5.0f, 5.0f, 2.0f};
            s->enabled = {1, 0, 1};  // body 1 overlaps body 0 but is disabled
        }
        without.px[1] = 9.0f;  // ... and here it is far away

        HostState a = simulate(desc, 1, dev, with, 0.01f, 50);
        HostState b = simulate(desc, 1, dev, without, 0.01f, 50);
        CHECK(a.px[0] == b.px[0] && a.py[0] == b.py[0] && a.vy[0] == b.vy[0]);
        CHECK(a.px[1] == 5.2f && a.py[1] == 5.0f && a.vy[1] == 0.0f);
    }
}

// Bodies never leave the box, and a dropped ball comes to rest on the floor.
static void test_particle_walls() {
    for (Device dev : kDevices) {
        ModelDesc desc = particle_desc();
        desc.bodies.push_back(BodyDesc::sphere(0.25f, 1.0f));
        desc.bounds_hi = {2.0f, 4.0f, 3.0f};
        desc.wall_restitution = 0.5f;

        HostState s(1);
        s.px = {1.0f}; s.py = {3.0f}; s.pz = {1.5f};
        s.vx = {5.0f}; s.vz = {-7.0f};

        World world(desc, 1, dev);
        world.set_state(s);
        bool inside = true;
        HostState out;
        for (int i = 0; i < 400; i++) {
            world.step(0.005f, 5);
            world.get_state(out);
            inside = inside && out.px[0] >= 0.25f && out.px[0] <= 1.75f &&
                     out.py[0] >= 0.25f && out.py[0] <= 3.75f &&
                     out.pz[0] >= 0.25f && out.pz[0] <= 2.75f;
        }
        CHECK(inside);
        CHECK(std::fabs(out.py[0] - 0.25f) < 1e-3f);
    }
}

// --- rigid solver ------------------------------------------------------------

const float kDt = 1.0f / 60.0f;

// A ground plane (body 0) plus `bodies`.
static ModelDesc ground_scene(const std::vector<BodyDesc>& bodies) {
    ModelDesc desc;
    desc.bodies.push_back(BodyDesc::plane());
    desc.bodies.insert(desc.bodies.end(), bodies.begin(), bodies.end());
    return desc;
}

static BodyDesc with(BodyDesc b, float friction, float restitution) {
    b.friction = friction;
    b.restitution = restitution;
    return b;
}

static void test_rigid_sphere_rests() {
    for (Device dev : kDevices) {
        ModelDesc desc = ground_scene({with(BodyDesc::sphere(0.5f, 1.0f), 0.5f, 0.0f)});
        HostState s(2);
        s.py[1] = 2.0f;
        HostState out = simulate(desc, 1, dev, s, kDt, 180);
        std::printf("  [%s] y = %.5f, |v| = %.2g\n", device_name(dev), out.py[1], speed(out, 1));
        CHECK(std::fabs(out.py[1] - 0.5f) < 2e-3f);
        CHECK(speed(out, 1) < 1e-2f);
        CHECK(out.px[1] == 0.0f && out.pz[1] == 0.0f);
    }
}

// An elastic ball keeps (nearly) all its height when it bounces.
static void test_rigid_bounce() {
    for (Device dev : kDevices) {
        ModelDesc desc = ground_scene({with(BodyDesc::sphere(0.5f, 1.0f), 0.0f, 1.0f)});
        desc.bodies[0].restitution = 1.0f;
        World world(desc, 1, dev);
        HostState s(2);
        s.py[1] = 2.0f;
        world.set_state(s);
        float apex = 0.0f;
        for (int i = 0; i < 100; i++) {  // fall 0.55 s, bounce back up by ~1.1 s
            world.step(kDt);
            world.get_state(s);
            if (i > 45) apex = std::fmax(apex, s.py[1]);
        }
        std::printf("  [%s] apex after bounce = %.4f (dropped from 2.0)\n", device_name(dev), apex);
        CHECK(apex > 1.9f && apex < 2.01f);
    }
}

// A box dropped flat lands without drifting, and a box placed at rest stays
// exactly where it is. (Restitution is 0: an elastic box bouncing flat is
// ill-conditioned - the slightest tilt makes one edge land first.)
static void test_rigid_box_rests() {
    for (Device dev : kDevices) {
        ModelDesc desc = ground_scene({with(BodyDesc::box({0.5f, 0.25f, 0.5f}, 1.0f), 0.5f, 0.0f)});
        desc.bodies[0].restitution = 0.0f;
        for (float start_y : {0.75f, 0.25f}) {
            World world(desc, 1, dev);
            HostState s(2);
            s.py[1] = start_y;
            world.set_state(s);
            world.step(kDt, 180);
            world.get_state(s);
            std::vector<int> contacts;
            world.get_contact_counts(contacts);
            float drift = std::fabs(s.px[1]) + std::fabs(s.pz[1]);
            float rot = std::fabs(s.qx[1]) + std::fabs(s.qy[1]) + std::fabs(s.qz[1]);
            std::printf("  [%s] from y=%.2f: y = %.5f, drift = %.2g, rotation = %.2g, contacts = %d\n",
                        device_name(dev), start_y, s.py[1], drift, rot, contacts[0]);
            CHECK(std::fabs(s.py[1] - 0.25f) < 2e-3f);
            CHECK(drift < 2e-4f && rot < 2e-4f);
            CHECK(speed(s, 1) < 1e-3f && spin(s, 1) < 1e-3f);
            CHECK(contacts[0] == 4);
        }
    }
}

// A box dropped on its edge tips over and settles on a face.
static void test_rigid_box_tips_over() {
    for (Device dev : kDevices) {
        ModelDesc desc = ground_scene({BodyDesc::box({0.5f, 0.5f, 0.5f}, 1.0f)});
        HostState s(2);
        s.py[1] = 1.5f;
        set_quat(s, 1, axis_angle({0.0f, 0.0f, 1.0f}, 0.5f));
        HostState out = simulate(desc, 1, dev, s, kDt, 300);
        float up = 0.0f;
        for (int k = 0; k < 3; k++) up = std::fmax(up, std::fabs(body_axis(out, 1, k).y));
        std::printf("  [%s] y = %.5f, face alignment = %.6f\n", device_name(dev), out.py[1], up);
        CHECK(std::fabs(out.py[1] - 0.5f) < 5e-3f);
        CHECK(up > 0.9999f);
        CHECK(speed(out, 1) < 1e-2f && spin(out, 1) < 1e-2f);
    }
}

// Five unit cubes stacked on the ground stay stacked.
static void test_rigid_box_stack() {
    for (Device dev : kDevices) {
        const int n = 5;
        std::vector<BodyDesc> boxes(n, BodyDesc::box({0.5f, 0.5f, 0.5f}, 1.0f));
        ModelDesc desc = ground_scene(boxes);
        desc.substeps = 20;
        HostState s(n + 1);
        for (int i = 0; i < n; i++) s.py[i + 1] = 0.5f + 1.0f * i;
        HostState out = simulate(desc, 1, dev, s, kDt, 300);
        float drift = 0.0f, max_speed = 0.0f;
        for (int i = 1; i <= n; i++) {
            drift = std::fmax(drift, std::fabs(out.px[i]) + std::fabs(out.pz[i]));
            max_speed = std::fmax(max_speed, speed(out, i));
        }
        std::printf("  [%s] top y = %.4f, drift = %.2g, max |v| = %.2g\n", device_name(dev),
                    out.py[n], drift, max_speed);
        CHECK(std::fabs(out.py[n] - (0.5f + n - 1)) < 0.05f);
        CHECK(drift < 1e-2f);
        CHECK(max_speed < 0.05f);
    }
}

// A box sliding with friction stops after v^2 / (2 mu g); without friction it
// keeps going.
static void test_rigid_friction() {
    for (Device dev : kDevices) {
        for (float mu : {0.4f, 0.0f}) {
            ModelDesc desc =
                ground_scene({with(BodyDesc::box({0.5f, 0.5f, 0.5f}, 1.0f), mu, 0.0f)});
            desc.bodies[0].friction = mu;
            HostState s(2);
            s.py[1] = 0.5f;
            s.vx[1] = 5.0f;
            HostState out = simulate(desc, 1, dev, s, kDt, 120);
            if (mu > 0.0f) {
                float expected = 25.0f / (2.0f * mu * 9.81f);
                std::printf("  [%s] stopped after %.3f (expected %.3f)\n", device_name(dev),
                            out.px[1], expected);
                CHECK(std::fabs(out.px[1] - expected) < 0.05f * expected);
                CHECK(speed(out, 1) < 1e-2f);
            } else {
                CHECK(std::fabs(out.vx[1] - 5.0f) < 1e-3f);
                CHECK(std::fabs(out.px[1] - 10.0f) < 0.02f);
            }
            CHECK(std::fabs(body_axis(out, 1, 1).y - 1.0f) < 1e-4f);  // did not tip
        }
    }
}

static void test_rigid_capsule_rests() {
    for (Device dev : kDevices) {
        ModelDesc desc = ground_scene({BodyDesc::capsule(0.2f, 0.5f, 1.0f)});
        HostState s(2);
        s.py[1] = 1.0f;
        set_quat(s, 1, axis_angle({0.0f, 0.0f, 1.0f}, 1.4f));  // nearly horizontal
        HostState out = simulate(desc, 1, dev, s, kDt, 240);
        float axis_up = body_axis(out, 1, 1).y;
        std::printf("  [%s] y = %.5f, axis up = %.2g\n", device_name(dev), out.py[1], axis_up);
        CHECK(std::fabs(out.py[1] - 0.2f) < 2e-3f);
        CHECK(std::fabs(axis_up) < 1e-2f);
        CHECK(speed(out, 1) < 1e-2f && spin(out, 1) < 1e-2f);
    }
}

// A sphere lands on a static box.
static void test_rigid_static_obstacle() {
    for (Device dev : kDevices) {
        ModelDesc desc = ground_scene({BodyDesc::box({1.0f, 0.5f, 1.0f}, 0.0f),
                                       with(BodyDesc::sphere(0.3f, 1.0f), 0.5f, 0.0f)});
        HostState s(3);
        s.py[1] = 0.5f;  // static box resting on the ground
        s.py[2] = 3.0f;
        HostState out = simulate(desc, 1, dev, s, kDt, 180);
        CHECK(out.py[1] == 0.5f);
        CHECK(std::fabs(out.py[2] - 1.3f) < 2e-3f);
        CHECK(speed(out, 2) < 1e-2f);
    }
}

// Torque-free tumbling conserves angular momentum.
static void test_rigid_free_rotation() {
    for (Device dev : kDevices) {
        ModelDesc desc;
        desc.bodies.push_back(BodyDesc::box({0.1f, 0.3f, 0.5f}, 1.0f));
        desc.gravity = {0.0f, 0.0f, 0.0f};
        HostState s(1);
        s.wx = {0.1f}; s.wy = {3.0f}; s.wz = {0.1f};  // near the unstable middle axis

        // Principal moments for half extents (0.1, 0.3, 0.5), mass 1.
        const float I[3] = {(0.09f + 0.25f) / 3, (0.01f + 0.25f) / 3, (0.01f + 0.09f) / 3};
        auto momentum = [&](const HostState& st) {
            Vec3 L = {0, 0, 0};
            for (int k = 0; k < 3; k++) {
                Vec3 a = body_axis(st, 0, k);
                float wk = a.x * st.wx[0] + a.y * st.wy[0] + a.z * st.wz[0];
                L.x += I[k] * wk * a.x; L.y += I[k] * wk * a.y; L.z += I[k] * wk * a.z;
            }
            return L;
        };
        Vec3 L0 = momentum(s);
        HostState out = simulate(desc, 1, dev, s, kDt, 180);
        Vec3 L1 = momentum(out);
        float d = std::sqrt((L1.x - L0.x) * (L1.x - L0.x) + (L1.y - L0.y) * (L1.y - L0.y) +
                            (L1.z - L0.z) * (L1.z - L0.z));
        float n0 = std::sqrt(L0.x * L0.x + L0.y * L0.y + L0.z * L0.z);
        std::printf("  [%s] |dL| / |L| = %.3g, body y axis now = %.3f\n", device_name(dev), d / n0,
                    body_axis(out, 0, 1).y);
        CHECK(d / n0 < 0.02f);
    }
}

// A random pile of mixed shapes in many envs.
static void random_rigid_scene(int nenv, unsigned seed, ModelDesc& desc, HostState& s) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    desc = ground_scene({BodyDesc::sphere(0.3f, 1.0f), BodyDesc::sphere(0.4f, 2.0f),
                         BodyDesc::box({0.3f, 0.2f, 0.4f}, 1.5f),
                         BodyDesc::box({0.25f, 0.25f, 0.25f}, 1.0f),
                         BodyDesc::capsule(0.15f, 0.4f, 1.0f),
                         BodyDesc::capsule(0.2f, 0.3f, 1.2f)});
    int nbody = (int)desc.bodies.size();
    s.resize(nenv * nbody);
    for (int e = 0; e < nenv; e++) {
        for (int i = 1; i < nbody; i++) {
            int g = e * nbody + i;
            s.px[g] = unit(rng) * 1.5f - 0.75f;
            s.py[g] = 0.5f + 0.8f * i + unit(rng) * 0.3f;
            s.pz[g] = unit(rng) * 1.5f - 0.75f;
            s.vx[g] = unit(rng) - 0.5f;
            s.wy[g] = 2.0f * unit(rng) - 1.0f;
            set_quat(s, g, axis_angle({unit(rng) - 0.5f, unit(rng) - 0.5f, unit(rng) - 0.5f},
                                      3.0f * unit(rng)));
        }
    }
}

static void test_rigid_determinism_and_parity() {
    const int nenv = 64;
    ModelDesc desc;
    HostState s;
    random_rigid_scene(nenv, 5, desc, s);
    HostState cpu = simulate(desc, nenv, Device::CPU, s, kDt, 120);
    HostState gpu = simulate(desc, nenv, Device::CUDA, s, kDt, 120);
    HostState gpu2 = simulate(desc, nenv, Device::CUDA, s, kDt, 120);
    CHECK(same_bits(gpu, gpu2));
    float max_diff = 0.0f, min_y = 1e9f;
    for (int g = 0; g < cpu.size(); g++) {
        max_diff = std::fmax(max_diff, std::fabs(cpu.py[g] - gpu.py[g]));
        if (g % (int)desc.bodies.size() != 0) min_y = std::fmin(min_y, gpu.py[g]);
    }
    std::printf("  max |cpu - gpu| py = %.3g, lowest body y = %.3f\n", max_diff, min_y);
#ifdef PHYS_STRICT_FP
    CHECK(same_bits(cpu, gpu));
#else
    CHECK(max_diff < 1e-2f);
#endif
    CHECK(min_y > 0.1f);  // nothing fell through the floor
}

static void test_rigid_env_independence() {
    for (Device dev : kDevices) {
        const int nenv = 32;
        ModelDesc desc;
        HostState batch;
        random_rigid_scene(nenv, 9, desc, batch);
        int nbody = (int)desc.bodies.size();
        HostState batch_out = simulate(desc, nenv, dev, batch, kDt, 60);
        for (int env : {0, 13, 31}) {
            HostState single(nbody), slice(nbody);
            copy_env(batch, env, single, 0, nbody);
            copy_env(batch_out, env, slice, 0, nbody);
            CHECK(same_bits(simulate(desc, 1, dev, single, kDt, 60), slice));
        }
    }
}

static void test_rigid_disabled_bodies() {
    for (Device dev : kDevices) {
        ModelDesc desc = ground_scene({BodyDesc::box({0.5f, 0.5f, 0.5f}, 1.0f),
                                       BodyDesc::sphere(0.5f, 1.0f)});
        HostState s(3);
        s.py[1] = 0.5f;
        s.py[2] = 1.2f;  // overlaps the box, but disabled
        s.enabled[2] = 0;
        HostState out = simulate(desc, 1, dev, s, kDt, 60);
        CHECK(std::fabs(out.py[1] - 0.5f) < 2e-3f && speed(out, 1) < 1e-2f);
        CHECK(out.py[2] == 1.2f && out.vy[2] == 0.0f);
    }
}


// --- joints ------------------------------------------------------------------

static Quat about_z(float angle) { return axis_angle({0.0f, 0.0f, 1.0f}, angle); }

// Point `local` of body g, in world coordinates.
static Vec3 body_point(const HostState& s, int g, Vec3 local) {
    Vec3 ax = body_axis(s, g, 0), ay = body_axis(s, g, 1), az = body_axis(s, g, 2);
    return {s.px[g] + ax.x * local.x + ay.x * local.y + az.x * local.z,
            s.py[g] + ax.y * local.x + ay.y * local.y + az.y * local.z,
            s.pz[g] + ax.z * local.x + ay.z * local.y + az.z * local.z};
}

static float dist(Vec3 a, Vec3 b) {
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
                     (a.z - b.z) * (a.z - b.z));
}

// Place body g so that its local point `anchor` sits at world point `pivot`,
// rotated by `angle` about z.
static void place(HostState& s, int g, Vec3 pivot, Vec3 anchor, float angle) {
    float c = std::cos(angle), sn = std::sin(angle);
    s.px[g] = pivot.x - (c * anchor.x - sn * anchor.y);
    s.py[g] = pivot.y - (sn * anchor.x + c * anchor.y);
    s.pz[g] = pivot.z - anchor.z;
    set_quat(s, g, about_z(angle));
}

// A rod of length 2 * half hanging from a hinge about z at its top end.
static BodyDesc rod(float half, float mass) { return BodyDesc::box({0.05f, half, 0.05f}, mass); }

static float rod_inertia_z(float half, float mass) { return mass / 3.0f * (0.0025f + half * half); }

// Classical RK4 for a system x' = f(x).
template <typename F>
static void rk4(std::vector<double>& x, double h, F f) {
    size_t n = x.size();
    std::vector<double> k1 = f(x), t(n);
    for (size_t i = 0; i < n; i++) t[i] = x[i] + 0.5 * h * k1[i];
    std::vector<double> k2 = f(t);
    for (size_t i = 0; i < n; i++) t[i] = x[i] + 0.5 * h * k2[i];
    std::vector<double> k3 = f(t);
    for (size_t i = 0; i < n; i++) t[i] = x[i] + h * k3[i];
    std::vector<double> k4 = f(t);
    for (size_t i = 0; i < n; i++) x[i] += h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
}

// Compound pendulum: small-angle period, energy at large amplitude, and how
// well the hinge holds its anchor.
static void test_joint_pendulum() {
    const float half = 0.5f, m = 1.0f, g = 9.81f;
    const double I_pivot = rod_inertia_z(half, m) + m * half * half;
    for (Device dev : kDevices) {
        ModelDesc desc;
        desc.bodies.push_back(rod(half, m));
        desc.joints.push_back(JointDesc::hinge(-1, 0, {0, 0, 0}, {0, half, 0}, {0, 0, 1}));

        // Period at 0.1 rad: 2 pi sqrt(I / (m g d)) (1 + theta^2 / 16).
        {
            World world(desc, 1, dev);
            HostState s(1);
            place(s, 0, {0, 0, 0}, {0, half, 0}, 0.1f);
            world.set_state(s);
            std::vector<float> q, qd;
            float prev = 0.1f, t = 0.0f, first = -1.0f, last = -1.0f;
            int crossings = 0;
            for (int i = 0; i < 600; i++) {
                world.step(kDt);
                world.get_joint_state(q, qd);
                t += kDt;
                if (prev > 0.0f && q[0] <= 0.0f) {  // downward zero crossing
                    float tc = t - kDt * q[0] / (q[0] - prev);
                    if (first < 0.0f) first = tc;
                    last = tc;
                    crossings++;
                }
                prev = q[0];
            }
            double expected = 2.0 * M_PI * std::sqrt(I_pivot / (m * g * half)) * (1.0 + 0.01 / 16.0);
            double period = (last - first) / (crossings - 1);
            std::printf("  [%s] period %.5f s (expected %.5f)\n", device_name(dev), period, expected);
            CHECK(crossings >= 5);
            CHECK(std::fabs(period - expected) < 2e-3 * expected);
        }
        // Energy and anchor drift at 1.5 rad over 10 s. Position-based joints
        // lose energy at first order in the substep length (each substep
        // moves bodies along a tangent and projects them back), so the loss
        // must halve as the substeps double.
        double loss_at[2] = {0.0, 0.0};
        for (int run = 0; run < 2; run++) {
            desc.substeps = run == 0 ? 20 : 40;
            World world(desc, 1, dev);
            HostState s(1);
            place(s, 0, {0, 0, 0}, {0, half, 0}, 1.5f);
            world.set_state(s);
            auto energy = [&](float q, float qd) {
                return 0.5 * I_pivot * qd * qd + m * g * half * (1.0 - std::cos(q));
            };
            double e0 = energy(1.5f, 0.0f), worst = 0.0;
            float anchor_err = 0.0f;
            std::vector<float> q, qd;
            for (int i = 0; i < 600; i++) {
                world.step(kDt);
                world.get_joint_state(q, qd);
                world.get_state(s);
                worst = std::fmax(worst, std::fabs(energy(q[0], qd[0]) - e0) / e0);
                anchor_err = std::fmax(anchor_err, dist(body_point(s, 0, {0, half, 0}), {0, 0, 0}));
            }
            loss_at[run] = worst;
            std::printf("  [%s] %d substeps: energy loss over 10 s %.3f, anchor error %.2g\n",
                        device_name(dev), desc.substeps, worst, anchor_err);
            CHECK(anchor_err < 1e-4f);
        }
        CHECK(loss_at[1] < 0.6 * loss_at[0]);  // first-order convergence
        CHECK(loss_at[1] < 0.12);
    }
}

// Double pendulum of two rods against its exact equations of motion.
static void test_joint_double_pendulum() {
    const float half = 0.5f, m = 1.0f;
    const double g = 9.81, l1 = 2 * half, lc = half, I = rod_inertia_z(half, m);
    const float q1_0 = 1.0f, q2_0 = 0.5f;

    // Relative angles from hanging straight down.
    std::vector<double> x = {q1_0, q2_0, 0.0, 0.0};
    auto f = [&](const std::vector<double>& s) {
        double q2 = s[1], w1 = s[2], w2 = s[3];
        double M11 = 2 * I + m * (lc * lc + l1 * l1 + lc * lc + 2 * l1 * lc * std::cos(q2));
        double M12 = I + m * (lc * lc + l1 * lc * std::cos(q2));
        double M22 = I + m * lc * lc;
        double h = m * l1 * lc * std::sin(q2);
        double C1 = -h * (2 * w1 * w2 + w2 * w2), C2 = h * w1 * w1;
        double G1 = (m * lc + m * l1) * g * std::sin(s[0]) + m * g * lc * std::sin(s[0] + q2);
        double G2 = m * g * lc * std::sin(s[0] + q2);
        double r1 = -C1 - G1, r2 = -C2 - G2, det = M11 * M22 - M12 * M12;
        return std::vector<double>{w1, w2, (M22 * r1 - M12 * r2) / det, (M11 * r2 - M12 * r1) / det};
    };

    for (Device dev : kDevices) {
      float err_at[2];
      for (int run = 0; run < 2; run++) {
        ModelDesc desc;
        desc.substeps = run == 0 ? 10 : 40;
        desc.bodies = {rod(half, m), rod(half, m)};
        desc.joints = {JointDesc::hinge(-1, 0, {0, 0, 0}, {0, half, 0}, {0, 0, 1}),
                       JointDesc::hinge(0, 1, {0, -half, 0}, {0, half, 0}, {0, 0, 1})};
        HostState s(2);
        place(s, 0, {0, 0, 0}, {0, half, 0}, q1_0);
        Vec3 elbow = body_point(s, 0, {0, -half, 0});
        place(s, 1, elbow, {0, half, 0}, q1_0 + q2_0);

        World world(desc, 1, dev);
        world.set_state(s);
        std::vector<double> ref = x;
        std::vector<float> q, qd;
        float worst = 0.0f;
        for (int i = 1; i <= 90; i++) {  // 1.5 s, before the motion turns chaotic
            world.step(kDt);
            for (int k = 0; k < 100; k++) rk4(ref, kDt / 100.0, f);
            world.get_joint_state(q, qd);
            worst = std::fmax(worst, std::fmax(std::fabs(q[0] - (float)ref[0]),
                                               std::fabs(q[1] - (float)ref[1])));
        }
        std::printf("  [%s] %d substeps: max angle error vs exact dynamics over 1.5 s: %.4f rad\n",
                    device_name(dev), desc.substeps, worst);
        err_at[run] = worst;
      }
      CHECK(err_at[1] < 0.5f * err_at[0]);
      CHECK(err_at[1] < 0.015f);
    }
}

// Cart-pole driven by a force on the cart, against its exact equations.
static void test_joint_cartpole() {
    const float M = 1.0f, m = 0.1f, half = 0.5f;
    const double g = 9.81, l = half, I = 0.1 / 3.0 * (0.0004 + half * half);
    const float theta0 = 0.2f;
    auto force = [](double t) { return 2.0 * std::sin(3.0 * t); };

    auto wrap = [](double a) { return std::remainder(a, 2.0 * M_PI); };
    for (Device dev : kDevices) {
      float err_at[2];
      for (int run = 0; run < 2; run++) {
        ModelDesc desc;
        desc.substeps = run == 0 ? 10 : 40;
        desc.bodies = {BodyDesc::none(M, {0.1f, 0.1f, 0.1f}),
                       BodyDesc::box({0.02f, half, 0.02f}, m)};
        JointDesc rail = JointDesc::slider(-1, 0, {0, 0, 0}, {0, 0, 0}, {1, 0, 0});
        rail.actuator = Actuator::Torque;
        desc.joints = {rail, JointDesc::hinge(0, 1, {0, 0, 0}, {0, -half, 0}, {0, 0, 1})};
        HostState s(2);
        place(s, 1, {0, 0, 0}, {0, -half, 0}, theta0);

        World world(desc, 1, dev);
        world.set_state(s);
        // State: x, theta, x', theta'. theta is the pole's rotation about z
        // from upright, so its centre is at (x - l sin theta, l cos theta).
        std::vector<double> ref = {0.0, theta0, 0.0, 0.0};
        std::vector<float> q, qd;
        float worst_x = 0.0f, worst_th = 0.0f;
        double t = 0.0;
        // 1 s: the pole falls from near upright, an unstable equilibrium that
        // amplifies any error exponentially, so longer horizons test chaos
        // rather than the integrator.
        for (int i = 0; i < 60; i++) {
            double F = force(t);
            world.set_controls({(float)F, 0.0f});
            world.step(kDt);
            auto f = [&](const std::vector<double>& st) {
                double th = st[1], w = st[3], c = std::cos(th), sn = std::sin(th);
                // (M+m) x'' - m l c th'' = F - m l sn w^2
                // -m l c x'' + (I + m l^2) th'' = m g l sn
                double a11 = M + m, a12 = -m * l * c, a22 = I + m * l * l;
                double b1 = F - m * l * sn * w * w, b2 = m * g * l * sn;
                double det = a11 * a22 - a12 * a12;
                return std::vector<double>{st[2], w, (a22 * b1 - a12 * b2) / det,
                                           (a11 * b2 - a12 * b1) / det};
            };
            for (int k = 0; k < 100; k++) rk4(ref, kDt / 100.0, f);
            t += kDt;
            world.get_joint_state(q, qd);
            worst_x = std::fmax(worst_x, std::fabs(q[0] - (float)ref[0]));
            worst_th = std::fmax(worst_th, (float)std::fabs(wrap(q[1] - ref[1])));
        }
        std::printf("  [%s] %d substeps, 1 s: x = %.4f (exact %.4f), theta = %.4f (exact %.4f); "
                    "max errors %.4f m, %.4f rad\n", device_name(dev), desc.substeps, q[0], ref[0],
                    q[1], wrap(ref[1]), worst_x, worst_th);
        err_at[run] = worst_th;
        if (run == 1) {
            CHECK(worst_x < 0.005f);
            CHECK(worst_th < 0.01f);
        }
      }
      CHECK(err_at[1] < 0.5f * err_at[0]);
    }
}

// Position drives reach the analytic steady state under gravity, and stay
// stable with stiff gains on a light, finger-sized link.
static void test_joint_position_drive() {
    for (Device dev : kDevices) {
        {
            const float half = 0.5f, m = 1.0f, kp = 50.0f, target = 0.8f;
            ModelDesc desc;
            desc.bodies.push_back(rod(half, m));
            JointDesc j = JointDesc::hinge(-1, 0, {0, 0, 0}, {0, half, 0}, {0, 0, 1});
            j.actuator = Actuator::Position;
            j.kp = kp;
            j.kd = 5.0f;
            desc.joints.push_back(j);
            HostState s(1);
            place(s, 0, {0, 0, 0}, {0, half, 0}, 0.0f);
            World world(desc, 1, dev);
            world.set_state(s);
            world.set_controls({target});
            world.step(kDt, 240);
            std::vector<float> q, qd;
            world.get_joint_state(q, qd);
            // kp (target - q) = m g d sin q, solved by bisection.
            double lo = 0.0, hi = target;
            for (int i = 0; i < 60; i++) {
                double mid = 0.5 * (lo + hi);
                (kp * (target - mid) > m * 9.81 * half * std::sin(mid) ? lo : hi) = mid;
            }
            std::printf("  [%s] q = %.5f (steady state %.5f), qd = %.2g\n", device_name(dev), q[0],
                        lo, qd[0]);
            CHECK(std::fabs(q[0] - lo) < 2e-3);
            CHECK(std::fabs(qd[0]) < 1e-3f);
        }
        {
            // 10 g, 4 cm link with kp = 50 N m / rad: natural period ~0.2 ms,
            // far below the 1.7 ms substep. Explicit PD would explode.
            ModelDesc desc;
            desc.bodies.push_back(BodyDesc::capsule(0.008f, 0.012f, 0.01f));
            JointDesc j = JointDesc::hinge(-1, 0, {0, 0, 0}, {0, 0.02f, 0}, {0, 0, 1});
            j.actuator = Actuator::Position;
            j.kp = 50.0f;
            j.kd = 0.05f;
            desc.joints.push_back(j);
            HostState s(1);
            place(s, 0, {0, 0, 0}, {0, 0.02f, 0}, 0.0f);
            World world(desc, 1, dev);
            world.set_state(s);
            world.set_controls({1.2f});
            world.step(kDt, 60);
            std::vector<float> q, qd;
            world.get_joint_state(q, qd);
            world.get_state(s);
            std::printf("  [%s] stiff light finger: q = %.5f (target 1.2), qd = %.2g\n",
                        device_name(dev), q[0], qd[0]);
            CHECK(std::isfinite(q[0]) && std::fabs(q[0] - 1.2f) < 1e-3f);
            CHECK(std::fabs(qd[0]) < 1e-2f);
            CHECK(dist(body_point(s, 0, {0, 0.02f, 0}), {0, 0, 0}) < 1e-5f);
        }
    }
}

// Velocity drives reach their target speed, and a force limit caps the
// acceleration at max_force / I.
static void test_joint_velocity_drive() {
    for (Device dev : kDevices) {
        for (float max_force : {0.0f, 0.2f}) {
            ModelDesc desc;
            desc.gravity = {0, 0, 0};
            desc.bodies.push_back(BodyDesc::box({0.5f, 0.1f, 0.1f}, 1.0f));  // spins about its centre
            JointDesc j = JointDesc::hinge(-1, 0, {0, 0, 0}, {0, 0, 0}, {0, 0, 1});
            j.actuator = Actuator::Velocity;
            j.max_force = max_force;
            desc.joints.push_back(j);
            World world(desc, 1, dev);
            world.set_state(HostState(1));
            world.set_controls({2.0f});
            world.step(kDt, 30);
            std::vector<float> q, qd;
            world.get_joint_state(q, qd);
            if (max_force == 0.0f) {
                CHECK(std::fabs(qd[0] - 2.0f) < 1e-4f);
            } else {
                float expected = max_force / (1.0f / 3.0f * (0.25f + 0.01f)) * 0.5f;  // tau t / I
                std::printf("  [%s] force-limited: qd = %.4f after 0.5 s (expected %.4f)\n",
                            device_name(dev), qd[0], expected);
                CHECK(std::fabs(qd[0] - expected) < 0.01f * expected);
            }
        }
    }
}

// Hinge and slider limits hold against gravity.
static void test_joint_limits() {
    for (Device dev : kDevices) {
        ModelDesc desc;
        desc.gravity = {9.81f, -9.81f, 0.0f};  // pushes the pendulum towards +q, the cart to +x
        desc.bodies = {rod(0.5f, 1.0f), BodyDesc::none(1.0f, {0.1f, 0.1f, 0.1f})};
        JointDesc hinge = JointDesc::hinge(-1, 0, {0, 0, 0}, {0, 0.5f, 0}, {0, 0, 1});
        hinge.limited = true;
        hinge.lower = -0.3f;
        hinge.upper = 0.4f;
        hinge.damping = 0.5f;
        JointDesc rail = JointDesc::slider(-1, 1, {0, 2, 0}, {0, 0, 0}, {1, 0, 0});
        rail.limited = true;
        rail.lower = -1.0f;
        rail.upper = 0.25f;
        desc.joints = {hinge, rail};
        HostState s(2);
        place(s, 0, {0, 0, 0}, {0, 0.5f, 0}, 0.0f);
        s.py[1] = 2.0f;
        World world(desc, 1, dev);
        world.set_state(s);
        std::vector<float> q, qd;
        float max_q = -1e9f, max_x = -1e9f;
        for (int i = 0; i < 180; i++) {
            world.step(kDt);
            world.get_joint_state(q, qd);
            max_q = std::fmax(max_q, q[0]);
            max_x = std::fmax(max_x, q[1]);
        }
        std::printf("  [%s] hinge: max %.4f final %.4f (limit 0.4); slider: max %.4f final %.4f (limit 0.25)\n",
                    device_name(dev), max_q, q[0], max_x, q[1]);
        CHECK(max_q < 0.4f + 2e-3f && std::fabs(q[0] - 0.4f) < 1e-3f);
        CHECK(max_x < 0.25f + 1e-3f && std::fabs(q[1] - 0.25f) < 1e-3f);
    }
}

// A ball-joint pendulum swings conically with its anchor held; two bodies
// welded by a fixed joint tumble as one.
static void test_joint_ball_and_fixed() {
    for (Device dev : kDevices) {
        {
            ModelDesc desc;
            desc.bodies.push_back(rod(0.5f, 1.0f));
            desc.joints.push_back(JointDesc::ball(-1, 0, {0, 0, 0}, {0, 0.5f, 0}));
            HostState s(1);
            place(s, 0, {0, 0, 0}, {0, 0.5f, 0}, 0.6f);
            s.vz[0] = 1.5f;  // sideways push: conical swing
            World world(desc, 1, dev);
            world.set_state(s);
            float err = 0.0f, max_z = 0.0f;
            for (int i = 0; i < 180; i++) {
                world.step(kDt);
                world.get_state(s);
                err = std::fmax(err, dist(body_point(s, 0, {0, 0.5f, 0}), {0, 0, 0}));
                max_z = std::fmax(max_z, std::fabs(s.pz[0]));
            }
            std::printf("  [%s] ball anchor error %.2g, max |z| = %.3f\n", device_name(dev), err, max_z);
            CHECK(err < 1e-4f);
            CHECK(max_z > 0.1f);  // it really swung out of the plane
        }
        {
            ModelDesc desc;
            desc.gravity = {0, 0, 0};
            desc.bodies = {BodyDesc::box({0.3f, 0.2f, 0.1f}, 1.0f), BodyDesc::sphere(0.2f, 0.5f)};
            desc.joints.push_back(JointDesc::fixed(0, 1, {0.6f, 0, 0}, {0, 0, 0}));
            HostState s(2);
            s.px[1] = 0.6f;
            s.wx[0] = 1.0f; s.wy[0] = 2.0f; s.wz[0] = 0.5f;  // tumbles; the sphere must follow
            s.vy[1] = 0.3f;
            World world(desc, 1, dev);
            world.set_state(s);
            world.step(kDt, 120);
            world.get_state(s);
            float err = dist(body_point(s, 0, {0.6f, 0, 0}), {s.px[1], s.py[1], s.pz[1]});
            float rot = 0.0f;  // relative rotation: the two frames stay equal
            for (int k = 0; k < 3; k++) {
                Vec3 a = body_axis(s, 0, k), b = body_axis(s, 1, k);
                rot = std::fmax(rot, dist(a, b));
            }
            std::printf("  [%s] fixed: anchor error %.2g, frame error %.2g\n", device_name(dev), err, rot);
            CHECK(err < 1e-4f && rot < 1e-4f);
            CHECK(spin(s, 0) > 0.5f);
        }
    }
}

// Jointed bodies don't collide with each other by default; group / mask
// bits switch other pairs off.
static void test_joint_collision_filter() {
    for (Device dev : kDevices) {
        ModelDesc desc = ground_scene({BodyDesc::box({0.3f, 0.3f, 0.3f}, 1.0f),
                                       BodyDesc::box({0.3f, 0.3f, 0.3f}, 1.0f),
                                       BodyDesc::sphere(0.2f, 1.0f)});
        // Two overlapping boxes hinged together, hanging in the air.
        JointDesc top = JointDesc::fixed(-1, 1, {0, 2, 0}, {0, 0, 0});
        desc.joints = {top, JointDesc::hinge(1, 2, {0.2f, 0, 0}, {-0.2f, 0, 0}, {0, 0, 1})};
        desc.bodies[3].collision_mask = 0;  // the sphere ignores everything
        HostState s(4);
        s.py[1] = 2.0f;
        s.px[2] = 0.4f; s.py[2] = 2.0f;
        s.py[3] = 1.0f;
        World world(desc, 1, dev);
        world.set_state(s);
        world.step(kDt, 60);
        world.get_state(s);
        std::vector<float> q, qd;
        world.get_joint_state(q, qd);
        std::printf("  [%s] hinged overlap: |v| = %.2g; sphere y = %.2f\n", device_name(dev),
                    speed(s, 2), s.py[3]);
        CHECK(speed(s, 1) < 1e-3f);      // nothing pushed the boxes apart
        CHECK(s.py[3] < -1.0f);          // fell through the floor
    }
}

// An actuated chain lying on the ground: deterministic, CPU == GPU, and each
// env independent.
static void test_joint_determinism_and_parity() {
    const int nlink = 5, nenv = 48;
    ModelDesc desc;
    desc.bodies.push_back(BodyDesc::plane());
    for (int i = 0; i < nlink; i++) desc.bodies.push_back(BodyDesc::capsule(0.08f, 0.15f, 0.5f));
    for (int i = 1; i < nlink; i++) {
        JointDesc j = JointDesc::hinge(i, i + 1, {0, 0.23f, 0}, {0, -0.23f, 0}, {0, 0, 1});
        j.limited = true;
        j.lower = -1.2f;
        j.upper = 1.2f;
        j.actuator = i % 2 ? Actuator::Position : Actuator::Torque;
        j.kp = 20.0f;
        j.kd = 0.5f;
        j.max_force = 3.0f;
        desc.joints.push_back(j);
    }
    int nbody = (int)desc.bodies.size(), njoint = (int)desc.joints.size();
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    HostState s(nbody * nenv);
    for (int e = 0; e < nenv; e++) {
        for (int i = 1; i < nbody; i++) {  // a straight chain along +x, lying down
            int g = e * nbody + i;
            s.px[g] = 0.46f * (i - 1);
            s.py[g] = 0.1f;
            set_quat(s, g, about_z(-1.5707963f));
            s.vy[g] = 0.2f * unit(rng);
        }
    }
    auto run = [&](Device dev, int n, const HostState& init) {
        World world(desc, n, dev);
        world.set_state(init);
        std::mt19937 ctrl_rng(7);
        std::vector<float> ctrl(n * njoint);
        for (int step = 0; step < 90; step++) {
            for (float& c : ctrl) c = unit(ctrl_rng);
            world.set_controls(ctrl);
            world.step(kDt);
        }
        HostState out;
        world.get_state(out);
        return out;
    };
    HostState cpu = run(Device::CPU, nenv, s), gpu = run(Device::CUDA, nenv, s);
    CHECK(same_bits(gpu, run(Device::CUDA, nenv, s)));
#ifdef PHYS_STRICT_FP
    CHECK(same_bits(cpu, gpu));
#endif
    float min_y = 1e9f, spread = 0.0f;
    for (int g = 0; g < gpu.size(); g++) {
        if (g % nbody == 0) continue;
        min_y = std::fmin(min_y, gpu.py[g]);
        spread = std::fmax(spread, std::fabs(gpu.px[g] - s.px[g]));
    }
    std::printf("  lowest link y = %.3f, max x displacement = %.3f\n", min_y, spread);
    CHECK(min_y > 0.05f);    // links stay on top of the ground
    CHECK(spread > 0.02f);   // and the actuators actually moved them
}

static void test_invalid_models_rejected() {
    auto throws = [](const std::function<void()>& f) {
        try { f(); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    ModelDesc ok;
    ok.bodies.push_back(BodyDesc::sphere(0.5f, 1.0f));

    ModelDesc dynamic_plane = ok;
    dynamic_plane.bodies[0] = BodyDesc::plane();
    dynamic_plane.bodies[0].mass = 1.0f;
    CHECK(throws([&] { World w(dynamic_plane, 1, Device::CPU); }));

    ModelDesc particle_box = ok;
    particle_box.solver = Solver::Particle;
    particle_box.bodies[0] = BodyDesc::box({1.0f, 1.0f, 1.0f}, 1.0f);
    CHECK(throws([&] { World w(particle_box, 1, Device::CPU); }));

    ModelDesc bad_bounds = ok;
    bad_bounds.solver = Solver::Particle;
    bad_bounds.bounds_hi = {1.0f, 0.0f, 1.0f};
    CHECK(throws([&] { World w(bad_bounds, 1, Device::CPU); }));

    ModelDesc bad_box = ok;
    bad_box.bodies[0] = BodyDesc::box({1.0f, 0.0f, 1.0f}, 1.0f);
    CHECK(throws([&] { World w(bad_box, 1, Device::CPU); }));

    CHECK(throws([&] { World w(ok, 0, Device::CPU); }));

    ModelDesc self_joint = ok;
    self_joint.joints.push_back(JointDesc::hinge(0, 0, {0, 0, 0}, {0, 0, 0}, {0, 0, 1}));
    CHECK(throws([&] { World w(self_joint, 1, Device::CPU); }));

    ModelDesc limited_ball = ok;
    limited_ball.joints.push_back(JointDesc::ball(-1, 0, {0, 0, 0}, {0, 0, 0}));
    limited_ball.joints[0].limited = true;
    CHECK(throws([&] { World w(limited_ball, 1, Device::CPU); }));

    ModelDesc particle_joint = ok;
    particle_joint.solver = Solver::Particle;
    particle_joint.joints.push_back(JointDesc::hinge(-1, 0, {0, 0, 0}, {0, 0, 0}, {0, 0, 1}));
    CHECK(throws([&] { World w(particle_joint, 1, Device::CPU); }));

    ModelDesc shapeless = ok;
    shapeless.bodies[0] = BodyDesc::none(1.0f, {0, 0, 0});
    CHECK(throws([&] { World w(shapeless, 1, Device::CPU); }));

    World world(ok, 2, Device::CPU);
    CHECK(throws([&] { world.set_state(HostState(1)); }));
}

int main(int argc, char** argv) {
    struct Test { const char* name; void (*fn)(); };
    const Test tests[] = {
        {"particle_head_on", test_particle_head_on},
        {"particle_momentum", test_particle_momentum},
        {"particle_determinism_and_parity", test_particle_determinism_and_parity},
        {"particle_env_independence", test_particle_env_independence},
        {"particle_disabled_bodies", test_particle_disabled_bodies},
        {"particle_walls", test_particle_walls},
        {"rigid_sphere_rests", test_rigid_sphere_rests},
        {"rigid_bounce", test_rigid_bounce},
        {"rigid_box_rests", test_rigid_box_rests},
        {"rigid_box_tips_over", test_rigid_box_tips_over},
        {"rigid_box_stack", test_rigid_box_stack},
        {"rigid_friction", test_rigid_friction},
        {"rigid_capsule_rests", test_rigid_capsule_rests},
        {"rigid_static_obstacle", test_rigid_static_obstacle},
        {"rigid_free_rotation", test_rigid_free_rotation},
        {"rigid_determinism_and_parity", test_rigid_determinism_and_parity},
        {"rigid_env_independence", test_rigid_env_independence},
        {"rigid_disabled_bodies", test_rigid_disabled_bodies},
        {"joint_pendulum", test_joint_pendulum},
        {"joint_double_pendulum", test_joint_double_pendulum},
        {"joint_cartpole", test_joint_cartpole},
        {"joint_position_drive", test_joint_position_drive},
        {"joint_velocity_drive", test_joint_velocity_drive},
        {"joint_limits", test_joint_limits},
        {"joint_ball_and_fixed", test_joint_ball_and_fixed},
        {"joint_collision_filter", test_joint_collision_filter},
        {"joint_determinism_and_parity", test_joint_determinism_and_parity},
        {"invalid_models_rejected", test_invalid_models_rejected},
    };
    const char* only = argc > 1 ? argv[1] : nullptr;
    for (const Test& t : tests) {
        if (only && !std::strstr(t.name, only)) continue;
        int before = g_failures;
        std::printf("%s\n", t.name);
        t.fn();
        std::printf("  %s\n", g_failures == before ? "ok" : "FAILED");
    }
    std::printf("\n%s (%d failed checks)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
