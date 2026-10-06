"""The fragile-grasp RL environment (examples/grasp_env.py): an oracle that
knows each ball's mass and friction lifts every one; fixed grips cannot."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def run(env, policy):
    from grasp_env import EPISODE
    obs = env.reset()
    for _ in range(EPISODE):
        obs, r, done, info = env.step(policy(env))
    return info


def test_oracle_and_fixed_grip():
    from grasp_env import GraspEnv, action_towards
    env = GraspEnv(num_envs=256, obs="tactile", seed=1)
    assert env.reset().shape == (256, env.obs_size)
    oracle = run(env, lambda e: action_towards(e.grip, 1.25 * e.min_grip()))
    assert oracle["success"].float().mean() > 0.95
    fixed = run(env, lambda e: action_towards(e.grip, torch.full((e.n,), 2.0, device=e.device)))
    assert fixed["success"].float().mean() < 0.5
