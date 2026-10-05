"""Tests for the libphys Python API. Run with: PYTHONPATH=python pytest python/tests"""

import math

import numpy as np
import pytest
import torch

import libphys as lp

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def ground_and(*bodies, **settings):
    model = lp.Model(**settings)
    model.add_body(lp.Body.plane())
    ids = [model.add_body(b) for b in bodies]
    return model, ids


@pytest.mark.parametrize("device", DEVICES)
def test_state_views_are_zero_copy(device):
    model, (ball,) = ground_and(lp.Body.sphere(0.5, 1.0))
    world = lp.World(model, num_envs=4, device=device)
    s = world.state
    assert s.py.shape == (4, 2) and s.py.device.type == device
    s.py[:, ball] = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)  # set state by writing
    world.step(1 / 60)
    world.synchronize()
    # The same tensor now shows the stepped state: every ball fell a bit.
    assert torch.all(s.py[:, ball] < torch.tensor([1.0, 2.0, 3.0, 4.0], device=device))
    assert torch.all(s.vy[:, ball] < 0)


@pytest.mark.parametrize("device", DEVICES)
def test_ball_comes_to_rest_on_ground(device):
    model, (ball,) = ground_and(lp.Body.sphere(0.25, 1.0))
    model.bodies[ball].restitution = 0.0
    world = lp.World(model, num_envs=8, device=device)
    world.state.py[:, ball] = torch.linspace(0.5, 2.0, 8, device=device)
    world.step(1 / 60, 180)
    world.synchronize()
    assert torch.allclose(world.state.py[:, ball], torch.full((8,), 0.25, device=device), atol=2e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_position_actuator_tracks_targets(device):
    model = lp.Model(gravity=(0, 0, 0))
    link = model.add_body(lp.Body.box((0.05, 0.25, 0.05), 1.0))
    model.add_joint(lp.Joint.hinge(-1, link, (0, 0, 0), (0, 0.25, 0), (0, 0, 1)),
                    actuator="position", kp=50.0, kd=2.0)
    world = lp.World(model, num_envs=16, device=device)
    world.state.py[:, link] = -0.25
    targets = torch.linspace(-1.0, 1.0, 16, device=device)
    world.ctrl[:, 0] = targets
    world.step(1 / 60, 240)
    world.synchronize()
    assert world.joint_q.shape == (16, 1)
    assert torch.allclose(world.joint_q[:, 0], targets, atol=2e-3)


def test_cpu_and_cuda_agree_bit_for_bit():
    if "cuda" not in DEVICES:
        pytest.skip("no CUDA")
    rng = np.random.default_rng(0)
    results = []
    for device in ("cpu", "cuda"):
        model, ids = ground_and(lp.Body.box((0.2, 0.1, 0.15), 1.0), lp.Body.sphere(0.15, 0.5),
                                lp.Body.capsule(0.08, 0.2, 0.7))
        world = lp.World(model, num_envs=32, device=device)
        init = np.random.default_rng(0)
        for k, b in enumerate(ids):
            world.state.px[:, b] = torch.tensor(init.uniform(-0.3, 0.3, 32), dtype=torch.float32)
            world.state.py[:, b] = 0.3 + 0.4 * k
        world.step(1 / 60, 90)
        world.synchronize()
        results.append(world.state.positions().cpu())
    assert torch.equal(results[0], results[1])


def test_elastic_patch_matches_hertz():
    d = lp.ElasticPatchDesc()
    d.width = d.height = 4e-3
    d.cell = 8e-5
    d.youngs_modulus = 3e5
    d.poisson = 0.5
    patch = lp.ElasticPatch(d)
    ind = lp.Indenter()
    ind.radius = 5e-3
    depth = 2e-4
    for k in range(1, 9):
        patch.step(ind, (0.0, 0.0, -depth * k / 8))
    fz = patch.force()[2]
    e_star = 3e5 / 0.75
    hertz = 4 / 3 * e_star * math.sqrt(5e-3) * depth ** 1.5
    assert abs(fz / hertz - 1) < 0.02
    tr = patch.tractions()
    assert tr.shape == (patch.cells, 3) and tr[:, 2].max() > 0
