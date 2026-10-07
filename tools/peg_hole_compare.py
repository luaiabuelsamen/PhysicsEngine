"""Experiment E1: rigid square peg-in-hole, libphys vs MuJoCo, matched scenes.

A box peg (20 x 20 x 60 mm, 0.1 kg) is pushed by a 30 m/s^2 body load into a
square hole formed by four static box walls (30 mm deep), from 50 random
initial poses per cell: lateral offset |dx| <= clearance, tilt about y fixed
per cell. Friction 0.3 everywhere. Both simulators step at 2 ms (libphys:
10 substeps; MuJoCo: its defaults, and a stiffer-contact variant).

Per cell: success (peg bottom reaches 90% of the depth within 1 s), jam (no
success and peg speed < 1 mm/s at the end), peak interpenetration of the
peg's corners into the walls / floor, and throughput (env-steps per second
per simulator step; MuJoCo single CPU core, libphys batched on the GPU).

MuJoCo variants: defaults; stiffer contacts (solref 0.004); and a
near-rigid reference (0.2 ms step, solref 0.0004, elliptic cones, no-slip
iterations), used to judge which behaviour is physical.

    PYTHONPATH=python:$PYTHONPATH python3 tools/peg_hole_compare.py --out results/peg_hole
"""

import argparse
import json
import math
import os
import time

import numpy as np

A, HL, MASS = 0.010, 0.030, 0.1          # peg half width, half length (m), kg
DEPTH, WALL = 0.030, 0.010               # hole depth, wall half thickness
MU, PUSH = 0.3, 30.0                     # friction, body load (m/s^2)
DT, T = 0.002, 1.0
CLEARANCES = (0.002, 0.001, 0.0005, 0.0002)
TILTS = (0.0, 1.0, 2.0, 4.0)             # degrees
N = 50


def walls(c):
    """(centre, half extents) of the four wall boxes for clearance c per side."""
    h = A + c
    zc = -DEPTH / 2
    return [((h + WALL, 0.0, zc), (WALL, h + 2 * WALL, DEPTH / 2)),
            ((-(h + WALL), 0.0, zc), (WALL, h + 2 * WALL, DEPTH / 2)),
            ((0.0, h + WALL, zc), (h, WALL, DEPTH / 2)),
            ((0.0, -(h + WALL), zc), (h, WALL, DEPTH / 2))]


def initial_poses(c, tilt_deg, rng):
    th = math.radians(tilt_deg)
    dx = rng.uniform(-c, c, N)
    sign = rng.choice([-1.0, 1.0], N)
    z = 0.001 + HL * math.cos(th) + A * math.sin(th)
    quat = np.stack([np.full(N, math.cos(th / 2)), np.zeros(N), sign * math.sin(th / 2), np.zeros(N)], 1)
    pos = np.stack([dx, np.zeros(N), np.full(N, z)], 1)
    return pos, quat


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


CORNERS = np.array([[sx * A, sy * A, sz * HL] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def penetration(pos, quat, wall_boxes):
    """Deepest penetration (m) of the peg's corners into the walls or floor."""
    pts = CORNERS @ quat_mat(quat).T + pos
    worst = max(0.0, float(np.max(-DEPTH - pts[:, 2])))
    for centre, half in wall_boxes:
        d = np.asarray(half) - np.abs(pts - np.asarray(centre))
        inside = np.all(d > 0, axis=1)
        if inside.any():
            worst = max(worst, float(np.max(np.min(d[inside], axis=1))))
    return worst


def outcome(pos_hist, end_speed):
    success = pos_hist[:, 2].min() < HL - 0.9 * DEPTH + 1e-9  # centre low enough means bottom at 90% depth
    return bool(success), bool((not success) and end_speed < 1e-3)


# --- libphys -------------------------------------------------------------------

def run_libphys(c, poses_by_tilt):
    import torch
    import libphys as lp
    m = lp.Model(gravity=(0, 0, -PUSH), substeps=10)
    floor = m.add_body(lp.Body.plane(), friction=MU, restitution=0.0)
    wb = walls(c)
    wall_ids = [m.add_body(lp.Body.box(half, 0.0), friction=MU, restitution=0.0) for _, half in wb]
    peg = m.add_body(lp.Body.box((A, A, HL), MASS), friction=MU, restitution=0.0)
    n = sum(len(p[0]) for p in poses_by_tilt)
    w = lp.World(m, num_envs=n, device="cuda")
    s = w.state
    q = math.sqrt(0.5)
    s.qw[:, floor] = q
    s.qx[:, floor] = q
    s.pz[:, floor] = -DEPTH
    for b, (centre, _) in zip(wall_ids, wb):
        s.px[:, b], s.py[:, b], s.pz[:, b] = centre
    pos = np.concatenate([p for p, _ in poses_by_tilt])
    quat = np.concatenate([qq for _, qq in poses_by_tilt])
    dev = s.px.device
    s.px[:, peg] = torch.tensor(pos[:, 0], device=dev, dtype=torch.float32)
    s.py[:, peg] = torch.tensor(pos[:, 1], device=dev, dtype=torch.float32)
    s.pz[:, peg] = torch.tensor(pos[:, 2], device=dev, dtype=torch.float32)
    for k, f in enumerate((s.qw, s.qx, s.qy, s.qz)):
        f[:, peg] = torch.tensor(quat[:, k], device=dev, dtype=torch.float32)
    steps = int(T / DT)
    zmin = np.full(n, np.inf)
    pen = np.zeros(n)
    w.synchronize()
    t0 = time.time()
    sim_time = 0.0
    for k in range(steps):
        t1 = time.time()
        w.step(DT)
        w.synchronize()
        sim_time += time.time() - t1
        if k % 5 == 0 or k == steps - 1:
            P = torch.stack([s.px[:, peg], s.py[:, peg], s.pz[:, peg]], 1).cpu().numpy()
            Q = torch.stack([s.qw[:, peg], s.qx[:, peg], s.qy[:, peg], s.qz[:, peg]], 1).cpu().numpy()
            zmin = np.minimum(zmin, P[:, 2])
            for e in range(n):
                pen[e] = max(pen[e], penetration(P[e], Q[e], wb))
    speed = torch.sqrt(s.vx[:, peg] ** 2 + s.vy[:, peg] ** 2 + s.vz[:, peg] ** 2).cpu().numpy()
    return zmin, speed, pen, n * steps / sim_time


# --- MuJoCo --------------------------------------------------------------------

MUJOCO_VARIANTS = {
    # name: (timestep, geom contact attributes, option attributes)
    "mujoco_default": (DT, "", ""),
    "mujoco_stiff": (DT, 'solref="0.004 1"', ""),
    # Towards rigid contact: small step, stiff contacts, elliptic cones, no-slip.
    "mujoco_hard": (0.0002, 'solref="0.0004 1" solimp="0.99 0.999 0.0001"',
                    'cone="elliptic" noslip_iterations="30"'),
}


def mujoco_xml(c, variant):
    dt, geom_attr, opt_attr = MUJOCO_VARIANTS[variant]
    wb = walls(c)
    geoms = "\n".join(f'<geom type="box" pos="{p[0]} {p[1]} {p[2]}" size="{h[0]} {h[1]} {h[2]}"/>' for p, h in wb)
    return f"""
<mujoco>
  <option timestep="{dt}" gravity="0 0 -{PUSH}" {opt_attr}/>
  <default><geom friction="{MU} 0.005 0.0001" {geom_attr}/></default>
  <worldbody>
    <geom type="plane" pos="0 0 {-DEPTH}" size="1 1 0.1"/>
    {geoms}
    <body name="peg" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="{A} {A} {HL}" mass="{MASS}"/>
    </body>
  </worldbody>
</mujoco>"""


def run_mujoco(c, poses_by_tilt, variant):
    import mujoco
    model = mujoco.MjModel.from_xml_string(mujoco_xml(c, variant))
    dt = MUJOCO_VARIANTS[variant][0]
    every = max(1, int(round(5 * DT / dt)))
    data = mujoco.MjData(model)
    wb = walls(c)
    pos = np.concatenate([p for p, _ in poses_by_tilt])
    quat = np.concatenate([qq for _, qq in poses_by_tilt])
    n = len(pos)
    steps = int(round(T / dt))
    zmin = np.full(n, np.inf)
    pen = np.zeros(n)
    speed = np.zeros(n)
    sim_time = 0.0
    for e in range(n):
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = pos[e]
        data.qpos[3:7] = quat[e]
        mujoco.mj_forward(model, data)
        for k in range(steps):
            t1 = time.time()
            mujoco.mj_step(model, data)
            sim_time += time.time() - t1
            if k % every == 0 or k == steps - 1:
                zmin[e] = min(zmin[e], data.qpos[2])
                pen[e] = max(pen[e], penetration(data.qpos[:3].copy(), data.qpos[3:7].copy(), wb))
        speed[e] = np.linalg.norm(data.qvel[:3])
    return zmin, speed, pen, n * steps / sim_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/peg_hole")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for c in CLEARANCES:
        rng = np.random.default_rng(args.seed + int(c * 1e6))
        poses = [initial_poses(c, t, rng) for t in TILTS]
        sims = {"libphys": run_libphys(c, poses)}
        for variant in MUJOCO_VARIANTS:
            sims[variant] = run_mujoco(c, poses, variant)
        for name, (zmin, speed, pen, rate) in sims.items():
            for i, tilt in enumerate(TILTS):
                sl = slice(i * N, (i + 1) * N)
                outs = [outcome(np.array([[0, 0, z]]), v) for z, v in zip(zmin[sl], speed[sl])]
                succ = float(np.mean([o[0] for o in outs]))
                jam = float(np.mean([o[1] for o in outs]))
                row = {"sim": name, "clearance_mm": c * 1e3, "tilt_deg": tilt, "n": N, "success": succ, "jam": jam,
                       "peak_penetration_mm": float(pen[sl].max() * 1e3),
                       "median_penetration_mm": float(np.median(pen[sl]) * 1e3), "env_steps_per_s": rate}
                rows.append(row)
                print(f"{name:15s} c {c*1e3:4.1f} mm tilt {tilt:3.0f}: success {succ:.2f} jam {jam:.2f} "
                      f"peak pen {row['peak_penetration_mm']:.3f} mm  ({rate:,.0f} env-steps/s)", flush=True)
    with open(os.path.join(args.out, "e1_results.json"), "w") as f:
        json.dump({"config": {"peg_half": [A, A, HL], "mass": MASS, "depth": DEPTH, "mu": MU, "push": PUSH,
                              "dt": DT, "T": T, "n_per_cell": N, "seed": args.seed}, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
