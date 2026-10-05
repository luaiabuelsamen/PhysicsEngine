// Tests for URDF import (phys/urdf.h) and, when the motion planner submodule
// is available, its agreement with the planner's kinematics.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "phys/urdf.h"

#ifdef PHYS_HAS_PLANNER
#include "robot.hpp"
#endif

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

static std::string ur5e_path() { return std::string(PHYS_SOURCE_DIR) + "/extern/MotionPlanning/src/assets/ur5e/ur5e.urdf"; }

static double dist(const double a[3], const double b[3]) {
    return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
}

// Angle between two rotations.
static double angle_between(const double a[4], const double b[4]) {
    double d = std::fabs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
    return 2 * std::acos(std::fmin(d, 1.0));
}

static std::vector<double> random_q(std::mt19937& rng, int n) {
    std::uniform_real_distribution<double> u(-2.0, 2.0);
    std::vector<double> q(n);
    for (double& v : q) v = u(rng);
    return q;
}

// A single link with a rotated, non-diagonal inertia: the loader must recover
// its principal moments.
static void test_principal_axes() {
    // R = rotation by 0.7 rad about (1, 2, 3)/|.|, I = R diag(1, 2, 3) R^T.
    double ax[3] = {1, 2, 3}, n = std::sqrt(14.0), a = 0.7;
    for (double& v : ax) v /= n;
    double c = std::cos(a), s = std::sin(a), C = 1 - c, x = ax[0], y = ax[1], z = ax[2];
    double R[3][3] = {{c + x * x * C, x * y * C - z * s, x * z * C + y * s},
                      {y * x * C + z * s, c + y * y * C, y * z * C - x * s},
                      {z * x * C - y * s, z * y * C + x * s, c + z * z * C}};
    double D[3] = {1, 2, 3}, I[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            I[i][j] = 0;
            for (int k = 0; k < 3; k++) I[i][j] += R[i][k] * D[k] * R[j][k];
        }
    std::string path = "/tmp/phys_test_inertia.urdf";
    std::ofstream f(path);
    f << "<robot name='t'><link name='base'/><link name='arm'><inertial><mass value='2'/>"
      << "<inertia ixx='" << I[0][0] << "' iyy='" << I[1][1] << "' izz='" << I[2][2] << "' ixy='" << I[0][1]
      << "' ixz='" << I[0][2] << "' iyz='" << I[1][2] << "'/></inertial></link>"
      << "<joint name='j' type='revolute'><parent link='base'/><child link='arm'/><axis xyz='0 0 1'/>"
      << "<limit lower='-1' upper='1' effort='10' velocity='1'/></joint></robot>";
    f.close();
    UrdfModel m = load_urdf(path);
    Vec3 I0 = m.model.bodies.at(0).inertia;
    std::vector<float> got = {I0.x, I0.y, I0.z};
    std::sort(got.begin(), got.end());
    std::printf("  principal moments %.6f %.6f %.6f (expected 1 2 3)\n", got[0], got[1], got[2]);
    CHECK(std::fabs(got[0] - 1) < 1e-5 && std::fabs(got[1] - 2) < 1e-5 && std::fabs(got[2] - 3) < 1e-5);
    CHECK(m.num_dofs() == 1 && m.model.joints.at(0).limited);
}

static void test_ur5e_structure() {
    UrdfModel m = load_urdf(ur5e_path());
    std::printf("  %d bodies, %d joints, %d dofs:", (int)m.model.bodies.size(), (int)m.model.joints.size(),
                m.num_dofs());
    for (const std::string& j : m.joint_names) std::printf(" %s", j.c_str());
    std::printf("\n");
    CHECK(m.num_dofs() == 6);
    CHECK(m.model.bodies.size() == 6);  // base_link_inertia is welded to the world
    auto it = std::find(m.body_links.begin(), m.body_links.end(), "upper_arm_link");
    CHECK(it != m.body_links.end() &&
          std::fabs(m.model.bodies[it - m.body_links.begin()].mass - 8.393f) < 1e-4f);
}

// The loader's tree FK against the planner's FK on random configurations.
static void test_fk_matches_planner() {
#ifdef PHYS_HAS_PLANNER
    UrdfModel m = load_urdf(ur5e_path());
    Robot planner(ur5e_path());
    std::mt19937 rng(3);
    double worst_pos = 0, worst_rot = 0;
    for (int i = 0; i < 200; i++) {
        std::vector<double> q = random_q(rng, 6);
        Pose ours = m.link_pose("tool0", q);
        auto theirs = planner.getEndEffectorPose(q);
        double p[3] = {theirs.first.x(), theirs.first.y(), theirs.first.z()};
        double r[4] = {theirs.second.w(), theirs.second.x(), theirs.second.y(), theirs.second.z()};
        worst_pos = std::fmax(worst_pos, dist(ours.p, p));
        worst_rot = std::fmax(worst_rot, angle_between(ours.q, r));
    }
    std::printf("  tool0 vs planner FK over 200 random configurations: max position difference %.2g m, "
                "max orientation difference %.3f rad\n", worst_pos, worst_rot);
    CHECK(worst_pos < 1e-9);
    CHECK(worst_rot < 1e-6);
#else
    std::printf("  (motion planner submodule not available: skipped)\n");
#endif
}

// Placing the arm at a configuration and holding it (no gravity) must not
// move it: the joint frames built from the URDF are consistent.
static void test_configuration_is_consistent() {
    for (Device dev : {Device::CPU, Device::CUDA}) {
        UrdfModel m = load_urdf(ur5e_path());
        m.model.gravity = {0, 0, 0};
        const int nenv = 16;
        World w(m.model, nenv, dev);
        HostState s(nenv * (int)m.model.bodies.size());
        std::mt19937 rng(5);
        std::vector<float> ctrl(nenv * m.num_dofs());
        std::vector<std::vector<double>> qs;
        for (int e = 0; e < nenv; e++) {
            qs.push_back(random_q(rng, 6));
            m.set_configuration(s, e, qs.back());
            for (int j = 0; j < 6; j++) ctrl[e * 6 + j] = (float)qs.back()[j];
        }
        w.set_state(s);
        w.set_controls(ctrl);
        w.step(1.0f / 240, 120);
        std::vector<float> q, qd;
        w.get_joint_state(q, qd);
        HostState out;
        w.get_state(out);
        double worst_q = 0, worst_x = 0;
        for (int e = 0; e < nenv; e++)
            for (int j = 0; j < 6; j++) worst_q = std::fmax(worst_q, std::fabs(q[e * 6 + j] - qs[e][j]));
        for (int g = 0; g < s.size(); g++) {
            double a[3] = {s.px[g], s.py[g], s.pz[g]}, b[3] = {out.px[g], out.py[g], out.pz[g]};
            worst_x = std::fmax(worst_x, dist(a, b));
        }
        std::printf("  [%s] held for 0.5 s without gravity: max joint error %.2g rad, max body drift %.2g m\n",
                    dev == Device::CPU ? "cpu" : "cuda", worst_q, worst_x);
        CHECK(worst_q < 1e-4);
        CHECK(worst_x < 1e-4);
    }
}

// Under gravity, the URDF masses and the actuators' effort limits hold the
// arm at its target.
static void test_holds_under_gravity() {
    UrdfModel m = load_urdf(ur5e_path());
    World w(m.model, 1, Device::CUDA);
    HostState s((int)m.model.bodies.size());
    std::vector<double> target = {0.3, -1.2, 1.4, -1.6, -1.57, 0.2};
    m.set_configuration(s, 0, target);
    w.set_state(s);
    std::vector<float> ctrl(target.begin(), target.end());
    w.set_controls(ctrl);
    w.step(1.0f / 240, 480);
    std::vector<float> q, qd;
    w.get_joint_state(q, qd);
    double worst = 0;
    for (int j = 0; j < 6; j++) worst = std::fmax(worst, std::fabs(q[j] - target[j]));
    std::vector<double> qm(q.begin(), q.end());
    double tip_err = dist(m.link_pose("tool0", qm).p, m.link_pose("tool0", target).p);
    std::printf("  under gravity after 2 s: max joint error %.2g rad, tool0 error %.2g mm\n", worst,
                tip_err * 1e3);
    CHECK(worst < 2e-3);
    CHECK(tip_err < 2e-3);
}

int main() {
    struct Test { const char* name; void (*fn)(); };
    const Test tests[] = {
        {"principal_axes", test_principal_axes},
        {"ur5e_structure", test_ur5e_structure},
        {"fk_matches_planner", test_fk_matches_planner},
        {"configuration_is_consistent", test_configuration_is_consistent},
        {"holds_under_gravity", test_holds_under_gravity},
    };
    for (const Test& t : tests) {
        int before = g_failures;
        std::printf("%s\n", t.name);
        try {
            t.fn();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  exception: %s\n", e.what());
            g_failures++;
        }
        std::printf("  %s\n", g_failures == before ? "ok" : "FAILED");
    }
    std::printf("\n%s (%d failed checks)\n", g_failures ? "FAILED" : "PASSED", g_failures);
    return g_failures ? 1 : 0;
}
