"""Tactile sensor pads from Python: tensors, shapes and a Hertz check.
Run with: PYTHONPATH=python:$PYTHONPATH pytest python/tests"""

import math

import pytest
import torch

import libphys as lp

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def pressed_pad(device, loads, resolution=(24, 24)):
    """A pad on the bottom of a box, pressed onto a fixed sphere by a slider
    force; returns the world after it settles."""
    m = lp.Model(gravity=(0, 0, 0), substeps=40)
    m.add_body(lp.Body.none(0.5, (1e-3, 1e-3, 1e-3)))
    box = m.add_body(lp.Body.box((0.02, 0.02, 0.005), 0.1), friction=0.5, restitution=0.0)
    m.add_body(lp.Body.sphere(0.01, 0.0), friction=0.5, restitution=0.0)
    m.add_joint(lp.Joint.slider(-1, 0, (0, 0, 0.015), (0, 0, 0), (0, 0, 1)), actuator="torque", damping=20.0)
    m.add_joint(lp.Joint.slider(0, box, (0, 0, 0), (0, 0, 0), (1, 0, 0)), actuator="torque", damping=20.0)
    m.add_tactile_sensor(box, origin=(0, 0, -0.005), frame=(0, 1, 0, 0), width=0.012, height=0.012,
                         resolution=resolution, youngs_modulus=3e5)
    w = lp.World(m, num_envs=len(loads), device=device)
    w.state.pz[:, :2] = 0.015
    w.ctrl[:, 0] = -torch.tensor(loads, device=w.ctrl.device)
    w.step(1 / 240, 240)
    return w


@pytest.mark.parametrize("device", DEVICES)
def test_tactile_tensors_and_hertz(device):
    loads = [0.5, 1.0, 2.0]
    w = pressed_pad(device, loads)
    w.synchronize()
    assert len(w.tactile) == 1
    t = w.tactile[0]
    assert t.shape == (3, len(lp.TACTILE_CHANNELS), 24, 24)
    assert w.tactile_force.shape == (3, 1, 3)
    cell_area = (0.012 / 24) ** 2
    pressure = t[:, lp.TACTILE_CHANNELS.index("pressure")]
    total = pressure.sum(dim=(1, 2)) * cell_area
    e_star = 3e5 / 0.75
    for e, W in enumerate(loads):
        assert abs(w.tactile_force[e, 0, 2].item() - W) < 0.01 * W
        assert abs(total[e].item() - W) < 1e-3 * W
        a = (3 * W * 0.01 / (4 * e_star)) ** (1 / 3)
        peak = 3 * W / (2 * math.pi * a * a)
        assert abs(pressure[e].max().item() - peak) < 0.06 * peak


def test_cpu_and_cuda_agree():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    a = pressed_pad("cpu", [1.0, 1.5], resolution=(12, 12))
    b = pressed_pad("cuda", [1.0, 1.5], resolution=(12, 12))
    b.synchronize()
    assert torch.equal(a.tactile[0], b.tactile[0].cpu())
