"""A tactile fingertip pressed onto, then slid across, three objects - a
sphere, a capsule and the edge of a box - one per env, all on the GPU.

The fingertip is a box whose whole underside is a tactile pad (24 x 24
cells of a 0.3 MPa gel). A vertical slider presses it down with a constant force; a
horizontal slider then drags it sideways at constant speed. Each frame shows
the pad's pressure, its shear traction (arrows) and which cells stick (white)
or slip (red): watch the contact go from fully stuck, through partial slip
from the edges inwards, to full sliding.

    PYTHONPATH=python:$PYTHONPATH python3 examples/tactile_probe.py   # writes tactile_probe.gif
"""

import argparse
import math

import numpy as np
import torch

import libphys as lp

PAD, CELLS, HALF_H = 0.016, 24, 0.004
LOAD, SPEED = 1.5, 0.01   # N, m/s


def build(device):
    m = lp.Model(gravity=(0, 0, 0), substeps=40)
    carriage = m.add_body(lp.Body.none(0.5, (1e-3, 1e-3, 1e-3)))
    finger = m.add_body(lp.Body.box((PAD / 2, PAD / 2, HALF_H), 0.05), friction=0.5, restitution=0.0)
    # The objects, each with its top at z = 0; every env keeps one of them.
    sphere = m.add_body(lp.Body.sphere(0.008, 0.0), friction=0.5, restitution=0.0)
    capsule = m.add_body(lp.Body.capsule(0.004, 0.004, 0.0), friction=0.5, restitution=0.0)  # along y
    edge = m.add_body(lp.Body.box((0.006, 0.006, 0.006), 0.0), friction=0.5, restitution=0.0)
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
    # Box turned 45 degrees about y, so that an edge along y is on top.
    s.pz[:, edge] = -0.006 * math.sqrt(2.0)
    s.qw[:, edge] = math.cos(math.pi / 8)
    s.qy[:, edge] = math.sin(math.pi / 8)
    s.enabled[:, sphere:edge + 1] = 0
    for env, body in enumerate((sphere, capsule, edge)):
        s.enabled[env, body] = 1
    return w, ("sphere", "capsule", "box edge")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tactile_probe.gif")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    w, names = build(args.device)
    dt = 1 / 240
    ch = {name: i for i, name in enumerate(lp.TACTILE_CHANNELS)}
    frames, log = [], []
    steps = 360  # 0.25 s press, then slide
    for k in range(steps):
        w.ctrl[:, 0] = -LOAD
        w.ctrl[:, 1] = 0.0 if k < 60 else SPEED
        w.step(dt)
        if k % 6:
            continue
        t = w.tactile[0].detach().cpu().numpy()
        f = w.tactile_force[:, 0].detach().cpu().numpy()
        log.append(f.copy())
        fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.4))
        extent = [-PAD / 2 * 1e3, PAD / 2 * 1e3, -PAD / 2 * 1e3, PAD / 2 * 1e3]
        xs = (np.arange(CELLS) + 0.5) * PAD / CELLS * 1e3 - PAD / 2 * 1e3
        X, Y = np.meshgrid(xs, xs)
        for e in range(3):
            p = t[e, ch["pressure"]] * 1e-3
            qx, qy = t[e, ch["shear_x"]], t[e, ch["shear_y"]]
            contact = p > 0
            stick = t[e, ch["stick"]] > 0.5
            ax = axes[e, 0]
            ax.imshow(p, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=60)
            ax.set_ylabel(names[e])
            ax = axes[e, 1]
            ax.imshow(np.hypot(qx, qy) * 1e-3, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=30)
            sel = contact & ((np.indices(p.shape).sum(0) % 2) == 0)
            if sel.any():
                ax.quiver(X[sel], Y[sel], qx[sel], qy[sel], color="w", scale=4e5, width=0.006)
            ax = axes[e, 2]
            img = np.zeros(p.shape + (3,))
            img[contact & stick] = (1, 1, 1)
            img[contact & ~stick] = (0.85, 0.2, 0.2)
            ax.imshow(img, origin="lower", extent=extent)
            ax.set_title(f"W {f[e, 2]:.2f} N  Q {math.hypot(f[e, 0], f[e, 1]):.2f} N", fontsize=8)
            for a in axes[e]:
                a.set_xticks([])
                a.set_yticks([])
        axes[0, 0].set_title("pressure (0-60 kPa)", fontsize=9)
        axes[0, 1].set_title("shear (0-30 kPa)", fontsize=9)
        phase = "pressing" if k < 60 else f"sliding {SPEED * (k - 60) * dt * 1e3:.1f} mm"
        fig.suptitle(f"t = {k * dt:.2f} s, {phase}  (stick: white, slip: red)", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=80, loop=0)
    last = log[-1]
    for e in range(3):
        print(f"{names[e]:9s}: normal {last[e, 2]:.3f} N, shear {math.hypot(last[e, 0], last[e, 1]):.3f} N "
              f"(mu W = {0.5 * last[e, 2]:.3f})")
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
