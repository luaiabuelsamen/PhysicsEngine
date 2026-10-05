"""URDF import and the motion planner from Python.
Run with: PYTHONPATH=python:$PYTHONPATH pytest python/tests"""

import os

import numpy as np
import pytest
import torch

import libphys as lp

URDF = os.path.join(os.path.dirname(__file__), "..", "..", "extern", "MotionPlanning", "src", "assets", "ur5e",
                    "ur5e.urdf")
HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

needs_urdf = pytest.mark.skipif(lp.UrdfModel is None or not os.path.exists(URDF),
                                reason="built without URDF support or no planner submodule")
needs_planner = pytest.mark.skipif(lp.Planner is None or not os.path.exists(URDF),
                                   reason="built without the motion planner")


@needs_urdf
def test_load_ur5e():
    robot = lp.load_urdf(URDF, kp=2e5, kd=2e3)
    assert robot.num_dofs == 6
    assert robot.joint_names[0] == "shoulder_pan_joint"
    world = lp.World(robot.model, num_envs=8, device=DEVICE)
    assert world.num_bodies == 6 and world.num_joints == 6


@needs_urdf
@needs_planner
def test_plan_and_track_movej():
    robot = lp.load_urdf(URDF, kp=2e5, kd=2e3)
    planner = lp.Planner(URDF, dt=0.04)
    rng = np.random.default_rng(0)
    n = 32
    goals = np.array(HOME) + rng.uniform(-0.5, 0.5, size=(n, 6))
    results = [planner.move_j(HOME, g.tolist(), 50) for g in goals]
    assert all(ok for _, ok in results)
    plans = torch.tensor(np.stack([plan for plan, _ in results]),
                         dtype=torch.float32, device=DEVICE)          # [envs, waypoints, dof]
    world = lp.World(robot.model, num_envs=n, device=DEVICE)
    world.set_configuration(robot, HOME)
    sim_dt = 1 / 240
    steps = int((plans.shape[1] - 1) * planner.dt / sim_dt) + 120
    for k in range(1, steps + 1):
        i = min(int(k * sim_dt / planner.dt), plans.shape[1] - 1)
        world.ctrl[:] = plans[:, i]
        world.step(sim_dt)
    world.synchronize()
    err = (world.joint_q - plans[:, -1]).abs().max().item()
    assert err < 2e-3
    # The simulated tool reaches the planner's FK of the goal.
    q = world.joint_q[0].cpu().tolist()
    sim_tip = np.array(robot.link_pose("tool0", q)[0])
    plan_tip = np.array(planner.fk(goals[0].tolist())[0])
    assert np.linalg.norm(sim_tip - plan_tip) < 2e-3


@needs_planner
def test_planner_reports_failures():
    planner = lp.Planner(URDF)
    assert planner.joint_names[0] == "shoulder_pan_joint"
    pos, quat = planner.fk(HOME)
    q, ok = planner.ik(pos, quat, [h + 0.2 for h in HOME])
    assert ok and np.allclose(planner.fk(q)[0], pos, atol=1e-4)
    _, ok = planner.ik((5.0, 0.0, 0.0), quat, HOME)     # out of reach
    assert not ok
    _, ok = planner.move_j(HOME, [7.0] + HOME[1:], 20)  # beyond the joint limit
    assert not ok
