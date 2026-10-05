"""UR5e from Python: plan with the C++ motion planner, track the plans in
libphys, many envs at once on the GPU.

    PYTHONPATH=python:$PYTHONPATH python3 examples/ur5e_planner.py --envs 512
"""

import argparse
import os
import time

import numpy as np
import torch

import libphys as lp

URDF = os.path.join(os.path.dirname(__file__), "..", "extern", "MotionPlanning", "src", "assets", "ur5e",
                    "ur5e.urdf")
HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--waypoints", type=int, default=60)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    robot = lp.load_urdf(URDF, kp=2e5, kd=2e3)          # dynamics: masses, inertias, effort limits
    planner = lp.Planner(URDF, dt=0.04)                 # kinematics and trajectory optimisation

    # One random reachable goal per env; straight-line (MoveL) plans.
    rng = np.random.default_rng(0)
    goals = np.array(HOME) + rng.uniform(-0.6, 0.6, size=(args.envs, 6))
    t0 = time.time()
    plans, valid = [], []
    for g in goals:
        pos, quat = planner.fk(g.tolist())
        plan, ok = planner.move_l(HOME, pos, quat, args.waypoints)   # ok: IK converged along the line
        plans.append(plan)
        valid.append(ok)
    plan_time = time.time() - t0
    plans = torch.tensor(np.stack(plans), dtype=torch.float32, device=device)  # [envs, waypoints, dof]
    valid = torch.tensor(valid, device=device)

    world = lp.World(robot.model, num_envs=args.envs, device=device)
    world.set_configuration(robot, HOME)
    sim_dt = 1 / 240
    steps = int((args.waypoints - 1) * planner.dt / sim_dt) + 120
    track_err = torch.zeros(args.envs, device=device)
    t1 = time.time()
    for k in range(1, steps + 1):
        s = min(k * sim_dt / planner.dt, args.waypoints - 1)   # linear between waypoints
        i = min(int(s), args.waypoints - 2)
        target = torch.lerp(plans[:, i], plans[:, i + 1], s - i)
        world.ctrl[:] = target
        world.step(sim_dt)
        track_err = torch.maximum(track_err, (world.joint_q - target).abs().max(dim=1).values)
    if device == "cuda":
        torch.cuda.synchronize()
    sim_time = time.time() - t1

    ok = track_err[valid]
    print(f"planned {args.envs} MoveL trajectories in {plan_time:.2f} s on the CPU; "
          f"{int((~valid).sum())} invalid (IK did not converge)")
    print(f"tracked them in {sim_time:.2f} s ({args.envs * steps / sim_time:,.0f} env-steps/s); worst joint "
          f"tracking error on valid plans: median {ok.median().item():.4f} rad, max {ok.max().item():.4f} rad")


if __name__ == "__main__":
    main()
