"""Validate tactile contact models against real sensor data (Meta's Sparsh
force-estimation datasets: a probe presses into a DIGIT / GelSight Mini gel,
then slides across it, with an ATI Nano17 measuring 3-axis force).

For every trajectory the recorded probe path is replayed through three
contact models, and their predicted forces are compared with the measured
ones:

  elastic   phys::ElasticPatch (Boussinesq / Cerruti half-space, Kalker
            stick-slip) - the libphys tactile model
  winkler   independent springs per cell with per-cell stick-slip (the
            hydroelastic / brush family)
  coulomb   rigid contact with Coulomb friction (what rigid-body engines
            produce), behind a compliant holder: shear = min(ks u, mu N)

Every model also gets a holder stiffness ks: the robot, mount and force
sensor are compliant, so the contact moves less than the recorded pose.

Fitting is split so that the headline metric is out of sample:
  * per trajectory, only the gel surface height (contact onset) is fitted,
    and only from the press phase;
  * global parameters are fitted on a training split, with the same freedom
    for both compliant models: probe radius, a shear-compliance knob
    (elastic: tangential scale; winkler: shear ratio) and the holder
    stiffness; friction is the median steady sliding ratio;
  * the reported metric is the error of the predicted shear during the slide
    phase on the held-out split.

    python3 tools/sparsh_validate.py --data data/sparsh --sensor gelsight --probe sphere

Data: https://huggingface.co/datasets/facebook/gelsight-force-estimation and
https://huggingface.co/datasets/facebook/digit-force-estimation (only the
small dataset_slip_forces.pkl files are needed). Licence: CC BY-NC 4.0.
"""

import argparse
import ctypes
import functools
import multiprocessing
import glob
import os
import pickle
import sys

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))


# --- data ---------------------------------------------------------------------

def load(data_dir, sensor, probe):
    trajs = []
    for path in sorted(glob.glob(os.path.join(data_dir, f"{sensor}_{probe}_batch_*.pkl"))):
        for key, t in pickle.load(open(path, "rb"))["trajectories"].items():
            if "poses" not in t:
                continue
            P = np.asarray(t["poses"], dtype=np.float64)[:, :3] * 1e-3  # mm -> m
            F = np.asarray(t["forces"], dtype=np.float64)
            n = min(len(P), len(F))
            P, F = P[:n], F[:n]
            lat = np.linalg.norm(P[:, :2] - P[0, :2], axis=1)
            moving = lat > 2e-5
            if n < 10 or not moving.any():
                continue
            s = max(1, int(np.argmax(moving)) - 1)  # last sample before sliding
            if s < 5 or n - s < 5:
                continue
            trajs.append({"id": f"{os.path.basename(path)}:{key}", "P": P, "F": F - F[0] * [1, 1, 0],
                          "slide": s})
    return trajs


# --- models -------------------------------------------------------------------

_lib = None


def elastic_lib():
    global _lib
    if _lib is None:
        for cand in [os.environ.get("PHYS_TACTILE_LIB", ""),
                     os.path.join(HERE, "..", "build", "libphys_tactile.so")]:
            if cand and os.path.exists(cand):
                _lib = ctypes.CDLL(cand)
                break
        if _lib is None:
            sys.exit("libphys_tactile.so not found: build it and/or set PHYS_TACTILE_LIB")
        _lib.phys_patch_replay.restype = ctypes.c_int
    return _lib


def tips_from(traj, z0):
    """Probe tip in the patch frame: patch centred under the start position,
    surface at z = 0 (robot z = z0)."""
    P = traj["P"]
    tips = np.empty_like(P)
    tips[:, :2] = P[:, :2] - P[0, :2]
    tips[:, 2] = P[:, 2] - z0
    return tips


def patch_size(traj, radius):
    reach = np.abs(traj["P"][:, :2] - traj["P"][0, :2]).max()
    return 2 * (reach + 0.7 * radius + 5e-4)


def patch_replay(traj, z0, R, mu, cell, model, E=1.0, tscale=1.0, k=1.0, ratio=1.0, ks=0.0):
    """Replay a trajectory through phys::ElasticPatch (model 0: half-space,
    1: winkler), with a holder of tangential stiffness ks (0: rigid)."""
    tips = tips_from(traj, z0).astype(np.float32)
    size = patch_size(traj, R)
    patch = np.array([size, size, cell, 0.0, E, 0.5, mu, tscale, model, k, ratio, ks], dtype=np.float32)
    ind = np.array([0, R, 0], dtype=np.float32)
    out = np.zeros_like(tips)
    f = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if elastic_lib().phys_patch_replay(f(patch), f(ind), f(tips), len(tips), f(out)) != 0:
        raise RuntimeError("phys_patch_replay failed")
    return out.astype(np.float64)


def coulomb(traj, mu, fn, ks):
    """Rigid contact with Coulomb friction behind a holder of tangential
    stiffness ks: shear = min(ks * u, mu * normal) along the motion - what a
    rigid-body engine with a compliant robot produces."""
    P = traj["P"]
    s = traj["slide"]
    out = np.zeros((len(P), 3))
    out[:, 2] = fn
    e = slide_dir(traj)
    u = np.maximum((P[:, :2] - P[s, :2]) @ e, 0.0)
    out[:, :2] = np.minimum(ks * u, mu * fn)[:, None] * e
    return out


# --- fitting ------------------------------------------------------------------

def fit_onset(traj, law):
    """Gel surface height z0 (robot frame) and scale from the press phase,
    using the model's analytic normal law (n = 1.5 elastic, 2 winkler)."""
    s = traj["slide"]
    z, fz = traj["P"][:s + 1, 2], traj["F"][:s + 1, 2]

    def res(p):
        return p[1] * np.maximum(p[0] - z, 0.0) ** law - fz

    best = None
    for z0 in z[0] + np.array([0, 2e-4, 5e-4, 1e-3]):
        r = least_squares(res, [z0, max(fz.max(), 1e-3) / max(z0 - z.min(), 1e-4) ** law],
                          bounds=([z.min(), 0], [z[0] + 3e-3, np.inf]))
        if best is None or r.cost < best.cost:
            best = r
    return best.x  # z0, scale


SHEAR_SIGN = 1.0  # measured shear relative to the probe's motion; set per sensor


def slide_dir(traj):
    s = traj["slide"]
    d = traj["P"][-1, :2] - traj["P"][s, :2]
    return d / max(np.linalg.norm(d), 1e-12)


def slide_ratio(traj, forces, sign=1.0):
    """Shear along the slide direction over normal force, during the slide.
    The ratio cancels slow drifts of the normal force (the gel is not exactly
    parallel to the slide), isolating the stick-to-slip transition."""
    s = traj["slide"]
    shear = sign * forces[s:, :2] @ slide_dir(traj)
    return shear / np.maximum(forces[s:, 2], 1e-3)


def slide_error(traj, pred):
    """RMS error of the shear/normal ratio during the slide phase."""
    meas = slide_ratio(traj, traj["F"], SHEAR_SIGN)
    model = slide_ratio(traj, pred)
    return np.sqrt(np.mean((model - meas) ** 2))


def force_error(traj, pred):
    """RMS error of the shear force along the slide, normalised by the
    measured peak shear (affected by normal-force drift)."""
    s = traj["slide"]
    meas = SHEAR_SIGN * traj["F"][s:, :2] @ slide_dir(traj)
    model = pred[s:, :2] @ slide_dir(traj)
    return np.sqrt(np.mean((model - meas) ** 2)) / max(np.abs(meas).max(), 1e-6)


_POOL = None


def pmap(fn, items):
    return _POOL.map(fn, items) if _POOL else list(map(fn, items))


class Model:
    """A contact model with named free parameters (fitted in log space)."""

    def __init__(self, name, names, grid, lo, hi):
        self.name, self.names, self.grid, self.lo, self.hi = name, names, grid, np.log(lo), np.log(hi)

    def clip(self, logp):
        return np.minimum(np.maximum(logp, self.lo), self.hi)


def predict(args):
    """(model name, trajectory, params, globals) -> predicted forces."""
    name, t, p, g = args
    if name == "elastic":
        R, tscale, ks = p
        e_star = 3 * g["c_el"] / (4 * np.sqrt(R))
        return patch_replay(t, t["el"][0], R, g["mu"], g["cell"], 0, E=0.75 * e_star, tscale=tscale, ks=ks)
    if name == "winkler":
        R, ratio, ks = p
        return patch_replay(t, t["wk"][0], R, g["mu"], g["cell"], 1, k=g["c_wk"] / (np.pi * R), ratio=ratio, ks=ks)
    (ks,) = p
    return coulomb(t, g["mu"], t["F"][:, 2], ks)


def _err(args):
    global SHEAR_SIGN
    SHEAR_SIGN = args[3]["sign"]
    pred = predict(args)
    return slide_error(args[1], pred), force_error(args[1], pred)


def fit(model, train, g):
    from scipy.optimize import minimize
    cache = {}

    def loss(logp):
        logp = model.clip(np.asarray(logp))
        key = tuple(np.round(logp, 4))
        if key not in cache:
            p = np.exp(logp)
            cache[key] = float(np.mean([e for e, _ in pmap(_err, [(model.name, t, p, g) for t in train])]))
            print(f"  {model.name:8s} " + ", ".join(f"{n} {v:.3g}" for n, v in zip(model.names, p)) +
                  f": train ratio error {cache[key]:.4f}", flush=True)
        return cache[key]

    import itertools
    best = min(itertools.product(*model.grid), key=lambda p: loss(np.log(p)))
    r = minimize(loss, np.log(best), method="Nelder-Mead",
                 options={"xatol": 0.05, "fatol": 1e-4, "maxiter": 30 * len(model.names)})
    return np.exp(model.clip(r.x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "sparsh"))
    ap.add_argument("--sensor", default="gelsight")
    ap.add_argument("--probe", default="sphere")
    ap.add_argument("--cell", type=float, default=1.5e-4)
    ap.add_argument("--fit-cell", type=float, default=2.5e-4, help="grid spacing while fitting")
    ap.add_argument("--train", type=int, default=30)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--stride", type=int, default=1, help="use every n-th sample")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    trajs = load(args.data, args.sensor, args.probe)
    if not trajs:
        sys.exit("no trajectories found")
    for t in trajs:
        if args.stride > 1:
            keep = np.arange(0, len(t["P"]), args.stride)
            t["slide"] = int(np.searchsorted(keep, t["slide"]))
            t["P"], t["F"] = t["P"][keep], t["F"][keep]
    order = np.random.default_rng(0).permutation(len(trajs))
    train = [trajs[i] for i in order[:args.train]]
    test = [trajs[i] for i in order[args.train:args.train + args.test]]
    print(f"{args.sensor}/{args.probe}: {len(trajs)} trajectories, {len(train)} train, {len(test)} test",
          flush=True)

    # Sign convention of the measured shear relative to the motion, and
    # friction as the steady sliding ratio, from the training split.
    sign = float(np.sign(np.median([np.median(slide_ratio(t, t["F"])[-5:]) for t in train])))
    ratios = []
    for t in train:
        s = t["slide"]
        tail = slice(s + max(3, (len(t["P"]) - s) // 2), None)
        ratios.append(np.median(np.linalg.norm(t["F"][tail, :2], axis=1) / np.maximum(t["F"][tail, 2], 1e-3)))
    mu = float(np.median(ratios))

    # Press phase: per-trajectory onset, and the normal-law scale.
    for t in train + test:
        t["el"] = fit_onset(t, 1.5)
        t["wk"] = fit_onset(t, 2.0)
    g = {"mu": mu, "sign": sign, "cell": args.fit_cell,
         "c_el": float(np.median([t["el"][1] for t in train])),   # 4/3 E* sqrt(R)
         "c_wk": float(np.median([t["wk"][1] for t in train]))}   # pi k R

    global _POOL
    _POOL = multiprocessing.Pool(args.jobs)
    models = [
        Model("elastic", ["R", "tscale", "ks"], [[3e-3, 1e-2, 2.5e-2], [1.0, 3.0], [100.0, 300.0, 1e3, 1e5]],
              [1e-3, 0.3, 10.0], [3e-2, 30.0, 1e6]),
        Model("winkler", ["R", "ratio", "ks"], [[3e-3, 1e-2, 2.5e-2], [0.1, 0.4, 1.0], [100.0, 300.0, 1e3, 1e5]],
              [1e-3, 0.01, 10.0], [3e-2, 10.0, 1e6]),
        Model("coulomb", ["ks"], [[30.0, 100.0, 300.0, 1e3, 1e4]], [1.0], [1e6]),
    ]
    fitted = {}
    for m in models:
        fitted[m.name] = fit(m, train, g)
    print(f"fitted (mu = {mu:.3f}, shear sign {sign:+.0f}):")
    for m in models:
        print(f"  {m.name:8s} " + ", ".join(f"{n} = {v:.4g}" for n, v in zip(m.names, fitted[m.name])))

    g["cell"] = args.cell
    print(f"held-out slide phase, {len(test)} trajectories:")
    print(f"  {'model':8s} {'ratio RMS error (median / mean / p90)':40s} shear-force error / peak (median)")
    errs = {}
    for m in models:
        res = pmap(_err, [(m.name, t, fitted[m.name], g) for t in test])
        e = np.array([r[0] for r in res]); f = np.array([r[1] for r in res])
        errs[m.name] = e
        print(f"  {m.name:8s} {np.median(e):.3f} / {np.mean(e):.3f} / {np.percentile(e,90):.3f}{'':18s} "
              f"{100*np.median(f):5.1f}%", flush=True)
    e, w, c = errs["elastic"], errs["winkler"], errs["coulomb"]
    print(f"elastic beats winkler on {100*np.mean(e < w):.0f}% and coulomb on {100*np.mean(e < c):.0f}%; "
          f"winkler beats coulomb on {100*np.mean(w < c):.0f}% of held-out trajectories (ratio error)")


if __name__ == "__main__":
    main()
