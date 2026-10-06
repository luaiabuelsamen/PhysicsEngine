"""Figure: the tactile sensor pads in World against closed-form contact
mechanics. A pad on a box is pressed onto a fixed sphere (R = 10 mm, gel
E = 0.3 MPa) by slider forces, then sheared.

    PYTHONPATH=python:$PYTHONPATH python3 tools/plot_tactile_validation.py --out tactile_validation.png
"""

import argparse
import math

import numpy as np
import torch

import libphys as lp

R, E, NU, MU = 0.01, 3e5, 0.5, 0.5
PAD, CELLS = 0.012, 40


def run(loads, shears, device, average_steps=0):
    """Readings after 1 s; with average_steps, also the mean and standard
    deviation of the pad force over that many further steps."""
    m = lp.Model(gravity=(0, 0, 0), substeps=40)
    m.add_body(lp.Body.none(0.5, (1e-3, 1e-3, 1e-3)))
    box = m.add_body(lp.Body.box((0.02, 0.02, 0.005), 0.1), friction=MU, restitution=0.0)
    m.add_body(lp.Body.sphere(R, 0.0), friction=MU, restitution=0.0)
    m.add_joint(lp.Joint.slider(-1, 0, (0, 0, 0.015), (0, 0, 0), (0, 0, 1)), actuator="torque", damping=20.0)
    m.add_joint(lp.Joint.slider(0, box, (0, 0, 0), (0, 0, 0), (1, 0, 0)), actuator="torque", damping=20.0)
    m.add_tactile_sensor(box, origin=(0, 0, -0.005), frame=(0, 1, 0, 0), width=PAD, height=PAD,
                         resolution=(CELLS, CELLS), youngs_modulus=E, poisson=NU)
    w = lp.World(m, num_envs=len(loads), device=device)
    w.state.pz[:, :2] = 0.015
    w.ctrl[:, 0] = -torch.tensor(loads, device=w.ctrl.device)
    w.ctrl[:, 1] = torch.tensor(shears, device=w.ctrl.device)
    w.step(1 / 240, 240)
    w.synchronize()
    reading = w.tactile[0].cpu().numpy(), w.tactile_force[:, 0].cpu().numpy()
    if not average_steps:
        return reading
    history = []
    for _ in range(average_steps):
        w.step(1 / 240)
        history.append(w.tactile_force[:, 0, 2].cpu().numpy())
    history = np.array(history)
    return reading + (history.mean(axis=0), history.std(axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tactile_validation.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e_star = E / (1 - NU ** 2)
    cell = PAD / CELLS
    xs = (np.arange(CELLS) + 0.5) * cell - PAD / 2
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

    # (a) Hertz pressure profiles.
    loads = [0.5, 1.0, 2.0]
    t, f = run(loads, [0, 0, 0], args.device)
    r = np.linspace(-4.5e-3, 4.5e-3, 400)
    for e, W in enumerate(loads):
        a = (3 * W * R / (4 * e_star)) ** (1 / 3)
        p0 = 3 * W / (2 * math.pi * a * a)
        exact = p0 * np.sqrt(np.clip(1 - (r / a) ** 2, 0, None))
        row = t[e, 0][CELLS // 2 - 1:CELLS // 2 + 1].mean(axis=0)  # the two rows straddling y = 0
        line, = ax[0].plot(r * 1e3, exact * 1e-3, lw=1.5)
        ax[0].plot(xs * 1e3, row * 1e-3, "o", ms=3.5, color=line.get_color(), label=f"W = {W} N")
    ax[0].set(xlabel="x (mm)", ylabel="pressure (kPa)", title="Pressure under a sphere vs Hertz")
    ax[0].legend(frameon=False, fontsize=8)

    # (b) Mindlin stick radius.
    fr = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9])
    t, f = run([1.0] * len(fr), list(fr * MU), args.device)
    W = f[:, 2]
    Q = np.hypot(f[:, 0], f[:, 1])
    stick = (t[:, 6] > 0.5).sum(axis=(1, 2))
    c_sim = np.sqrt(stick * cell * cell / math.pi)
    a = (3 * W * R / (4 * e_star)) ** (1 / 3)
    g = np.linspace(0, 1, 200)
    ax[1].plot(g, g * 0 + 1, alpha=0)
    ax[1].plot(g, (1 - g) ** (1 / 3), "k", lw=1.5, label="Mindlin  c / a = (1 - Q / mu W)^(1/3)")
    ax[1].plot(Q / (MU * W), c_sim / a, "o", ms=5, label="simulated pad")
    ax[1].set(xlabel="Q / mu W", ylabel="stick radius / contact radius", title="Partial slip vs Mindlin")
    ax[1].legend(frameon=False, fontsize=8)

    # (c) Load reported by the pad vs applied: mean over 0.5 s, and the
    # step-to-step scatter of the rigid contact (error bars).
    loads = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0])
    t, f, mean, std = run(list(loads), [0] * len(loads), args.device, average_steps=120)
    total = t[:, 0].sum(axis=(1, 2)) * cell * cell
    assert np.allclose(total, f[:, 2], rtol=1e-3), "sum of cell pressures x area != pad force"
    ax[2].plot([0, 4.2], [0, 4.2], "k", lw=1)
    ax[2].errorbar(loads, mean, yerr=std, fmt="o", ms=5, capsize=3, label="pad force: mean, step-to-step scatter")
    ax[2].set(xlabel="applied load (N)", ylabel="reported (N)", title="Load balance (sum p A = pad force)")
    ax[2].legend(frameon=False, fontsize=8)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
