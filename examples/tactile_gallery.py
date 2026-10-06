"""Simulated tactile images: one gel pad pressed onto five different objects
(one per env) with a load that ramps up and back down.

Top row: the gel as a vision-based sensor would see it (shaded deflection);
bottom row: contact pressure.

    PYTHONPATH=python:$PYTHONPATH python3 examples/tactile_gallery.py   # writes tactile_gallery.gif
"""

import argparse
import math

import torch
from PIL import Image, ImageDraw

import libphys as lp
from libphys import viz

PAD, CELLS, HALF_H = 0.016, 32, 0.004
PEAK = 3.0  # N


def build(device):
    m = lp.Model(gravity=(0, 0, 0), substeps=40)
    carriage = m.add_body(lp.Body.none(0.5, (1e-3, 1e-3, 1e-3)))
    finger = m.add_body(lp.Body.box((PAD / 2, PAD / 2, HALF_H), 0.05), restitution=0.0)
    objects = {
        "sphere": [m.add_body(lp.Body.sphere(0.008, 0.0), restitution=0.0)],
        "capsule": [m.add_body(lp.Body.capsule(0.003, 0.004, 0.0), restitution=0.0)],
        "box edge": [m.add_body(lp.Body.box((0.005, 0.005, 0.005), 0.0), restitution=0.0)],
        "box corner": [m.add_body(lp.Body.box((0.005, 0.005, 0.005), 0.0), restitution=0.0)],
        "two spheres": [m.add_body(lp.Body.sphere(0.004, 0.0), restitution=0.0) for _ in range(2)],
    }
    m.add_joint(lp.Joint.slider(-1, carriage, (0, 0, HALF_H), (0, 0, 0), (0, 0, 1)), actuator="torque",
                damping=10.0)
    m.add_joint(lp.Joint.fixed(carriage, finger, (0, 0, 0), (0, 0, 0)))
    m.add_tactile_sensor(finger, origin=(0, 0, -HALF_H), frame=(0, 1, 0, 0), width=PAD, height=PAD,
                         resolution=(CELLS, CELLS), youngs_modulus=3e5)
    w = lp.World(m, num_envs=len(objects), device=device)
    s = w.state
    s.pz[:, :2] = HALF_H
    s.enabled[:, 2:] = 0
    (sphere,), (capsule,), (edge,), (corner,), pair = objects.values()
    s.pz[:, sphere] = -0.008
    s.pz[:, capsule] = -0.003
    s.qw[:, capsule] = s.qz[:, capsule] = math.sqrt(0.5)  # capsule axis turned to run diagonally
    s.qw[:, capsule], s.qz[:, capsule] = math.cos(math.pi / 8), math.sin(math.pi / 8)
    s.pz[:, edge] = -0.005 * math.sqrt(2.0)
    s.qw[:, edge], s.qy[:, edge] = math.cos(math.pi / 8), math.sin(math.pi / 8)
    # Corner up: rotate (1, 1, 1) onto +z.
    angle = math.acos(1 / math.sqrt(3))
    axis = (1 / math.sqrt(2), -1 / math.sqrt(2), 0.0)
    s.pz[:, corner] = -0.005 * math.sqrt(3.0)
    s.qw[:, corner] = math.cos(angle / 2)
    s.qx[:, corner] = axis[0] * math.sin(angle / 2)
    s.qy[:, corner] = axis[1] * math.sin(angle / 2)
    for k, b in enumerate(pair):
        s.px[:, b] = (-1) ** k * 0.0045
        s.py[:, b] = (-1) ** k * 0.002
        s.pz[:, b] = -0.004
    for env, bodies in enumerate(objects.values()):
        for b in bodies:
            s.enabled[env, b] = 1
    return w, list(objects)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tactile_gallery.gif")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    w, names = build(args.device)
    tile, gap, head = 150, 6, 30
    frames = []
    steps = 240
    for k in range(steps):
        load = PEAK * math.sin(math.pi * min(k / (steps - 20), 1.0)) ** 2
        w.ctrl[:, 0] = -max(load, 0.02)
        w.step(1 / 240)
        if k % 4:
            continue
        frame = Image.new("RGB", (len(names) * (tile + gap) + gap, head + 2 * (tile + gap)), (250, 249, 246))
        d = ImageDraw.Draw(frame)
        d.text((gap, 8), f"pad load {load:.2f} N (32 x 32 cells, 16 mm, 0.3 MPa gel)", fill=(20, 20, 20))
        for e, name in enumerate(names):
            x = gap + e * (tile + gap)
            reading = w.tactile[0][e]
            frame.paste(viz.gelsight(reading, PAD / CELLS, scale=6, gain=2.0).resize((tile, tile), Image.LANCZOS),
                        (x, head))
            frame.paste(viz.tactile_map(reading, scale=5, pressure_max=1.2e5, arrows=False, stick=False)
                        .resize((tile, tile), Image.NEAREST), (x, head + tile + gap))
            d.text((x + 4, head + 4), name, fill=(30, 30, 30))
        frames.append(frame)
    viz.save_gif(frames, args.out, fps=15)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
