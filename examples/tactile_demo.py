"""Tactile demo: a gel fingertip pressed onto, then dragged across, a sphere,
a capsule and the edge of a box - one per env, simulated together.

Columns: the scene; the gel as a vision-based sensor (GelSight / DIGIT) would
see it, shaded from the simulated deflection; pressure with shear arrows and
the sticking zone outlined in white.

    PYTHONPATH=python:$PYTHONPATH python3 examples/tactile_demo.py   # writes tactile_demo.gif
"""

import argparse
import math

import torch
from PIL import Image, ImageDraw

import libphys as lp
from libphys import viz

PAD, CELLS, HALF_H = 0.016, 24, 0.004
LOAD, SPEED = 1.5, 0.01  # N, m/s


def build(device):
    m = lp.Model(gravity=(0, 0, 0), substeps=40)
    carriage = m.add_body(lp.Body.none(0.5, (1e-3, 1e-3, 1e-3)))
    finger = m.add_body(lp.Body.box((PAD / 2, PAD / 2, HALF_H), 0.05), friction=0.5, restitution=0.0)
    sphere = m.add_body(lp.Body.sphere(0.008, 0.0), friction=0.5, restitution=0.0)
    capsule = m.add_body(lp.Body.capsule(0.004, 0.004, 0.0), friction=0.5, restitution=0.0)  # along y
    edge = m.add_body(lp.Body.box((0.006, 0.006, 0.006), 0.0), friction=0.5, restitution=0.0)
    table = m.add_body(lp.Body.plane())
    m.add_joint(lp.Joint.slider(-1, carriage, (-0.004, 0, HALF_H), (0, 0, 0), (0, 0, 1)),
                actuator="torque", damping=10.0)
    m.add_joint(lp.Joint.slider(carriage, finger, (0, 0, 0), (0, 0, 0), (1, 0, 0)),
                actuator="velocity", max_force=5.0)
    m.add_tactile_sensor(finger, origin=(0, 0, -HALF_H), frame=(0, 1, 0, 0), width=PAD, height=PAD,
                         resolution=(CELLS, CELLS), youngs_modulus=3e5)
    w = lp.World(m, num_envs=3, device=device)
    s = w.state
    s.px[:, :2] = -0.004
    s.pz[:, :2] = HALF_H
    s.pz[:, sphere] = -0.008
    s.pz[:, capsule] = -0.004
    s.pz[:, edge] = -0.006 * math.sqrt(2.0)  # turned 45 degrees about y: an edge on top
    s.qw[:, edge] = math.cos(math.pi / 8)
    s.qy[:, edge] = math.sin(math.pi / 8)
    s.pz[:, table] = -0.0085                 # a table under everything, facing +z
    s.qw[:, table] = s.qx[:, table] = math.sqrt(0.5)
    s.enabled[:, sphere:edge + 1] = 0
    for env, body in enumerate((sphere, capsule, edge)):
        s.enabled[env, body] = 1
    colors = {finger: (60, 72, 88), sphere: (233, 160, 70), capsule: (42, 157, 143), edge: (200, 90, 80),
              table: (225, 222, 214)}
    return w, ("sphere", "capsule", "box edge"), colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tactile_demo.gif")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    w, names, colors = build(args.device)
    cam = viz.Camera(eye=(0.045, -0.075, 0.03), target=(0.002, 0, -0.002), fov=32, size=(260, 200))
    dt, cell = 1 / 240, PAD / CELLS
    tile, head, gap = 200, 34, 8
    width = 260 + 2 * tile + 4 * gap
    frames = []
    for k in range(330):
        w.ctrl[:, 0] = -LOAD
        w.ctrl[:, 1] = 0.0 if k < 50 else SPEED
        w.step(dt)
        if k % 5:
            continue
        frame = Image.new("RGB", (width, head + 3 * (tile + gap)), (250, 249, 246))
        d = ImageDraw.Draw(frame)
        phase = "pressing" if k < 50 else f"sliding {SPEED * (k - 50) * dt * 1e3:.1f} mm"
        d.text((gap, 6), f"t = {k * dt:.2f} s  {phase}", fill=(20, 20, 20))
        for x, title in ((gap, "scene"), (260 + 2 * gap, "simulated GelSight image"),
                         (260 + tile + 3 * gap, "pressure, shear, stick zone")):
            d.text((x, 20), title, fill=(90, 90, 90))
        forces = w.tactile_force[:, 0].cpu()
        for e in range(3):
            y = head + e * (tile + gap)
            scene = viz.render(w, e, cam, colors=colors, hidden=(0,), plane_extent=0.03)
            frame.paste(scene, (gap, y))
            reading = w.tactile[0][e]
            frame.paste(viz.gelsight(reading, cell).resize((tile, tile), Image.LANCZOS), (260 + 2 * gap, y))
            frame.paste(viz.tactile_map(reading, pressure_max=6e4).resize((tile, tile), Image.NEAREST),
                        (260 + tile + 3 * gap, y))
            Wn, Q = forces[e, 2].item(), math.hypot(forces[e, 0].item(), forces[e, 1].item())
            d.text((gap + 4, y + 4), names[e], fill=(30, 30, 30))
            d.text((gap + 4, y + tile - 16), f"W {Wn:.2f} N   Q {Q:.2f} N", fill=(60, 60, 60))
        frames.append(frame)
    viz.save_gif(frames, args.out, fps=15)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
