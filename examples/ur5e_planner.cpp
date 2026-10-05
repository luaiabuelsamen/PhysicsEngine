// UR5e: plan with the C++ motion planner (extern/MotionPlanning), execute in
// libphys.
//
// For every env a random reachable goal is drawn; the planner produces a
// MoveJ (minimum-jerk joint-space) and a MoveL (straight Cartesian line)
// trajectory; libphys simulates the UR5e - loaded from the same URDF, with
// its link masses and inertias, under gravity - tracking each plan with
// joint position actuators, all envs in parallel on the GPU. Reports how
// well the dynamic arm follows the kinematic plans. Plans are first checked
// for joint jumps and for reaching their goal; invalid plans (e.g. IK that
// did not converge) are counted separately - a physical arm cannot follow
// them. The planner flags its own failures (Trajectory::success); the
// example also checks every plan independently.
//
//   ./ur5e_planner [num_envs]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "phys/urdf.h"
#include "robot.hpp"

using namespace phys;
using Clock = std::chrono::steady_clock;

namespace {

const double kSimDt = 1.0 / 240.0;
const int kWaypoints = 60;
const double kPlanDt = 0.04;  // 60 waypoints -> 2.4 s per motion
const double kHold = 0.5;     // settle time after each plan (s)

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

double norm3(const double a[3], const double b[3]) {
    return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) +
                     (a[2] - b[2]) * (a[2] - b[2]));
}

// Distance from point p to the segment a-b.
double to_segment(const double p[3], const double a[3], const double b[3]) {
    double ab[3], ap[3], t = 0, len2 = 0;
    for (int i = 0; i < 3; i++) {
        ab[i] = b[i] - a[i];
        ap[i] = p[i] - a[i];
        t += ap[i] * ab[i];
        len2 += ab[i] * ab[i];
    }
    t = len2 > 0 ? std::fmin(std::fmax(t / len2, 0.0), 1.0) : 0.0;
    double c[3] = {a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]};
    return norm3(p, c);
}

// Joint target at time t of a planned trajectory (linear between waypoints).
std::vector<double> sample(const Trajectory& traj, double t) {
    double s = t / traj.dt;
    int i = (int)std::floor(s);
    if (i >= (int)traj.size() - 1) return traj.points.back().position;
    double a = s - i;
    std::vector<double> q(traj.dof);
    for (size_t j = 0; j < traj.dof; j++)
        q[j] = (1 - a) * traj.points[i].position[j] + a * traj.points[i + 1].position[j];
    return q;
}

struct Stats {
    std::vector<double> rms, max_err, final_tip, path_dev;
    int invalid = 0;
};

// Largest change of any joint between consecutive waypoints.
double largest_step(const Trajectory& t) {
    double big = 0;
    for (size_t i = 1; i < t.size(); i++)
        for (size_t j = 0; j < t.dof; j++)
            big = std::fmax(big, std::fabs(t.points[i].position[j] - t.points[i - 1].position[j]));
    return big;
}

void report(const char* name, const Stats& s, double plan_ms) {
    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    auto worst = [](const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); };
    std::printf("%-6s planning %.2f ms/trajectory, %d invalid plan(s)\n", name, plan_ms, s.invalid);
    std::printf("       tracking (valid plans): joint RMS %.4f rad (worst %.4f), max %.4f rad | "
                "final tool error %.2f mm (worst %.2f)",
                med(s.rms), worst(s.rms), med(s.max_err), 1e3 * med(s.final_tip), 1e3 * worst(s.final_tip));
    if (!s.path_dev.empty())
        std::printf(" | executed path off the straight line: %.2f mm (worst %.2f)", 1e3 * med(s.path_dev),
                    1e3 * worst(s.path_dev));
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    const int nenv = argc > 1 ? std::atoi(argv[1]) : 256;
    const std::string urdf = std::string(PHYS_SOURCE_DIR) + "/extern/MotionPlanning/src/assets/ur5e/ur5e.urdf";

    // Planner, with the waypoint spacing set.
    Robot planner(urdf);
    OptimizerConfig cfg = planner.getOptimizerConfig();
    cfg.dt = kPlanDt;
    planner.setOptimizerConfig(cfg);

    // Simulator, from the same URDF.
    UrdfOptions opt;
    opt.kp = 2e5f;
    opt.kd = 2e3f;
    UrdfModel arm = load_urdf(urdf, opt);
    arm.model.substeps = 8;
    const int nbody = (int)arm.model.bodies.size(), dof = arm.num_dofs();
    std::printf("UR5e: %d bodies, %d actuated joints; %d envs on the GPU, %.0f Hz simulation\n", nbody, dof, nenv,
                1 / kSimDt);

    // Random reachable goals around a home pose.
    const std::vector<double> home = {0.0, -1.57, 1.57, -1.57, -1.57, 0.0};
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-0.6, 0.6);
    std::vector<std::vector<double>> goals(nenv, home);
    for (auto& g : goals)
        for (double& v : g) v += u(rng);

    for (int mode = 0; mode < 2; mode++) {
        const char* name = mode == 0 ? "MoveJ" : "MoveL";
        // Plan every env's motion on the CPU.
        std::vector<Trajectory> plans;
        auto t0 = Clock::now();
        for (int e = 0; e < nenv; e++) {
            if (mode == 0) {
                plans.push_back(planner.moveJ(home, goals[e], kWaypoints));
            } else {
                auto pose = planner.getEndEffectorPose(goals[e]);
                plans.push_back(planner.moveL(home, pose.first, pose.second, kWaypoints));
            }
        }
        double plan_ms = 1e3 * seconds_since(t0) / nenv;

        // Validate: no joint jumps, and the final waypoint reaches the goal.
        std::vector<bool> valid(nenv);
        Stats st;
        for (int e = 0; e < nenv; e++) {
            auto want = planner.getEndEffectorPose(goals[e]);
            auto got = planner.getEndEffectorPose(plans[e].points.back().position);
            bool sound = largest_step(plans[e]) < 0.3 && (got.first - want.first).norm() < 1e-3;
            valid[e] = sound && plans[e].success;
            if (!valid[e]) {
                st.invalid++;
                std::printf("       env %d: invalid %s plan (%s; largest joint step %.2f rad, ends %.1f mm from "
                            "the goal)\n", e, name, plans[e].success ? "NOT flagged by the planner" : "flagged",
                            largest_step(plans[e]), 1e3 * (got.first - want.first).norm());
            }
        }

        // Execute all plans in parallel.
        World world(arm.model, nenv, Device::CUDA);
        HostState s(nenv * nbody);
        for (int e = 0; e < nenv; e++) arm.set_configuration(s, e, home);
        world.set_state(s);
        std::vector<float> ctrl(nenv * dof), q, qd;
        std::vector<double> sq(nenv, 0.0), mx(nenv, 0.0), dev(nenv, 0.0);
        const double duration = (kWaypoints - 1) * kPlanDt;
        const int steps = (int)std::ceil((duration + kHold) / kSimDt);
        Pose start_tip = arm.link_pose("tool0", home);
        auto t1 = Clock::now();
        for (int k = 1; k <= steps; k++) {
            double t = k * kSimDt;
            std::vector<std::vector<double>> targets(nenv);
            for (int e = 0; e < nenv; e++) {
                targets[e] = sample(plans[e], t);
                for (int j = 0; j < dof; j++) ctrl[e * dof + j] = (float)targets[e][j];
            }
            world.set_controls(ctrl);
            world.step((float)kSimDt);
            world.get_joint_state(q, qd);
            for (int e = 0; e < nenv; e++) {
                for (int j = 0; j < dof; j++) {
                    double err = q[e * dof + j] - targets[e][j];
                    sq[e] += err * err / dof;
                    mx[e] = std::fmax(mx[e], std::fabs(err));
                }
                if (mode == 1 && k % 8 == 0) {  // straightness of the executed tool path
                    std::vector<double> qm(q.begin() + e * dof, q.begin() + (e + 1) * dof);
                    Pose tip = arm.link_pose("tool0", qm);
                    Pose goal = arm.link_pose("tool0", plans[e].points.back().position);
                    dev[e] = std::fmax(dev[e], to_segment(tip.p, start_tip.p, goal.p));
                }
            }
        }
        double sim_s = seconds_since(t1);
        for (int e = 0; e < nenv; e++) {
            if (!valid[e]) continue;
            std::vector<double> qm(q.begin() + e * dof, q.begin() + (e + 1) * dof);
            Pose tip = arm.link_pose("tool0", qm);
            Pose want = arm.link_pose("tool0", plans[e].points.back().position);
            st.rms.push_back(std::sqrt(sq[e] / steps));
            st.max_err.push_back(mx[e]);
            st.final_tip.push_back(norm3(tip.p, want.p));
            if (mode == 1) st.path_dev.push_back(dev[e]);
        }
        report(name, st, plan_ms);
        std::printf("       simulated %d envs x %.1f s in %.2f s wall (%.0f env-steps/s, incl. host-side "
                    "control and metrics)\n",
                    nenv, steps * kSimDt, sim_s, nenv * steps / sim_s);
    }
    return 0;
}
