"""UR5e from Python: plan with the C++ motion planner, track the plans in
libphys, many envs at once on the GPU.

    PYTHONPATH=python:$PYTHONPATH python3 examples/ur5e_planner.py --envs 512
    PYTHONPATH=python:$PYTHONPATH python3 examples/ur5e_planner.py --gif ur5e.gif   # also draw 12 envs
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


LINKS = ["base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link",
         "wrist_3_link", "tool0"]


def draw_envs(robot, planner, q, goals, valid, trails, t):
    """A 4 x 3 grid of envs: the arm as a skeleton through its link frames,
    the tool's path so far, and the goal."""
    from PIL import Image, ImageDraw
    from libphys import viz
    tile_w, tile_h, ss = 210, 180, 2
    cam = viz.Camera(eye=(1.7, -1.35, 1.1), target=(0.0, 0.1, 0.38), fov=42, size=(tile_w * ss, tile_h * ss))
    frame = Image.new("RGB", (4 * tile_w, 3 * tile_h + 24), (250, 249, 246))
    ImageDraw.Draw(frame).text((6, 6), f"UR5e, 12 of the envs: MoveL plans from the C++ planner tracked under "
                                       f"gravity   t = {t:.2f} s", fill=(20, 20, 20))
    for e in range(len(q)):
        img = Image.new("RGB", (tile_w * ss, tile_h * ss), (246, 244, 239))
        d = ImageDraw.Draw(img)
        # Floor grid.
        for v in np.linspace(-0.6, 0.6, 7):
            for a, b in ((( v, -0.6, 0), (v, 0.6, 0)), ((-0.6, v, 0), (0.6, v, 0))):
                x, y, _ = cam.project(np.array([a, b]))
                d.line(list(zip(x, y)), fill=(215, 212, 204), width=ss)
        pts = np.array([robot.link_pose(link, q[e].tolist())[0] for link in LINKS])
        trails[e].append(pts[-1])
        goal = np.array(planner.fk(goals[e].tolist())[0])
        x, y, _ = cam.project(np.array(trails[e]))
        if len(trails[e]) > 1:
            d.line(list(zip(x, y)), fill=(42, 157, 143), width=2 * ss)
        gx, gy, _ = cam.project(goal)
        r = 5 * ss
        d.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(214, 40, 57) if bool(valid[e]) else (150, 150, 150),
                  width=2 * ss)
        x, y, _ = cam.project(pts)
        d.line(list(zip(x, y)), fill=(38, 70, 83), width=7 * ss)
        for xi, yi in zip(x, y):
            d.ellipse([xi - 5 * ss, yi - 5 * ss, xi + 5 * ss, yi + 5 * ss], fill=(233, 196, 106))
        img = img.resize((tile_w, tile_h), Image.LANCZOS)
        frame.paste(img, ((e % 4) * tile_w, 24 + (e // 4) * tile_h))
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--waypoints", type=int, default=60)
    ap.add_argument("--gif", help="also render the first 12 envs to this GIF")
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
    frames, trails = [], [[] for _ in range(12)]
    t1 = time.time()
    for k in range(1, steps + 1):
        s = min(k * sim_dt / planner.dt, args.waypoints - 1)   # linear between waypoints
        i = min(int(s), args.waypoints - 2)
        target = torch.lerp(plans[:, i], plans[:, i + 1], s - i)
        world.ctrl[:] = target
        world.step(sim_dt)
        track_err = torch.maximum(track_err, (world.joint_q - target).abs().max(dim=1).values)
        if args.gif and k % 6 == 0:
            frames.append(draw_envs(robot, planner, world.joint_q[:12].cpu().numpy(), goals[:12], valid[:12].cpu(),
                                    trails, k * sim_dt))
    if device == "cuda":
        torch.cuda.synchronize()
    sim_time = time.time() - t1

    if args.gif:
        from libphys import viz
        viz.save_gif(frames, args.gif, fps=20)
        print(f"wrote {args.gif}")
    ok = track_err[valid]
    print(f"planned {args.envs} MoveL trajectories in {plan_time:.2f} s on the CPU; "
          f"{int((~valid).sum())} invalid (IK did not converge)")
    print(f"tracked them in {sim_time:.2f} s ({args.envs * steps / sim_time:,.0f} env-steps/s); worst joint "
          f"tracking error on valid plans: median {ok.median().item():.4f} rad, max {ok.max().item():.4f} rad")


if __name__ == "__main__":
    main()
