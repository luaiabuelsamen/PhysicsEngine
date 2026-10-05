"""Which contact law do real tactile gels follow? Model-free fits on Meta's
Sparsh force datasets (see tools/sparsh_validate.py for the data).

Press phase: F(d) = F0 + c * max(0, d - d_c)^n, fitted per trajectory for
  n = 2 (Winkler / hydroelastic / independent springs, sphere probe),
  n = 1.5 (Hertz, elastic half-space, sphere probe), and free n.
Slide phase: shear / normal along the slide, as a function of slide distance
  u, fitted with r_max * (1 - (1 - u / u*)^m) for
  m = 2 (brush / Winkler), m = 1.5 (Cattaneo-Mindlin), and free m;
  u* is the slide distance at which full sliding sets in.

    python3 tools/sparsh_contact_laws.py --data data/sparsh
"""

import argparse
import glob
import os
import pickle

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))


def trajectories(data, pattern):
    for path in sorted(glob.glob(os.path.join(data, pattern))):
        for t in pickle.load(open(path, "rb"))["trajectories"].values():
            if "poses" in t:
                P = np.asarray(t["poses"])[:, :3]  # mm
                F = np.asarray(t["forces"])
                n = min(len(P), len(F))
                yield P[:n], F[:n]


def press_phase(P, F):
    lateral = np.linalg.norm(P[:, :2] - P[0, :2], axis=1)
    end = int(np.argmax(lateral > 0.05)) if np.any(lateral > 0.05) else len(P)
    return P[0, 2] - P[:end, 2], F[:end, 2]  # depth (mm), normal force


def fit_press(depth, fz, n_fixed=None):
    f0 = fz[0]

    def model(p, d):
        n = n_fixed if n_fixed else p[2]
        return f0 + p[0] * np.maximum(0.0, d - p[1]) ** n

    p0 = [1.0, 0.0] + ([] if n_fixed else [1.5])
    lo = [0.0, -0.5] + ([] if n_fixed else [0.5])
    hi = [100.0, 0.5] + ([] if n_fixed else [4.0])
    r = least_squares(lambda p: model(p, depth) - fz, p0, bounds=(lo, hi))
    return np.sqrt(np.mean(r.fun ** 2)) / max(fz.max() - f0, 1e-6), r.x


def slide_phase(P, F):
    lat = P[:, :2] - P[0, :2]
    moving = np.linalg.norm(lat, axis=1) > 0.02
    if not moving.any():
        return None
    s = int(np.argmax(moving)) - 1
    d = lat[-1] - lat[s]
    if np.linalg.norm(d) < 0.5:
        return None
    e = d / np.linalg.norm(d)
    u = (lat[s:] - lat[s]) @ e
    ft = (F[s:, :2] - F[s, :2]) @ e
    if ft[-20:].mean() < 0:  # sign conventions differ between rigs
        ft = -ft
    return u, ft, F[s:, 2]


def fit_slide(u, r, m_fixed=None):
    def model(p, x):
        m = m_fixed if m_fixed else p[2]
        return p[0] * (1.0 - (1.0 - np.minimum(x / p[1], 1.0)) ** m)

    p0 = [float(np.clip(r[-10:].mean(), 0.01, 4.9)), 0.5] + ([] if m_fixed else [1.7])
    lo = [0.0, 0.01] + ([] if m_fixed else [1.0])
    hi = [5.0, 5.0] + ([] if m_fixed else [4.0])
    res = least_squares(lambda p: model(p, u) - r, p0, bounds=(lo, hi))
    return np.sqrt(np.mean(res.fun ** 2)) / max(r.max(), 1e-6), res.x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "sparsh"))
    args = ap.parse_args()

    print("Press phase: normalised RMS error of each law (median, p90)")
    for pattern in ["digit_sphere_batch_*.pkl", "gelsight_sphere_batch_*.pkl", "digit_sharp_batch_*.pkl",
                    "gelsight_flat_batch_*.pkl"]:
        errs = {"winkler n=2": [], "hertz n=1.5": [], "free n": []}
        exps = []
        for P, F in trajectories(args.data, pattern):
            depth, fz = press_phase(P, F)
            if len(depth) < 10 or fz.max() - fz[0] < 0.1:
                continue
            errs["winkler n=2"].append(fit_press(depth, fz, 2.0)[0])
            errs["hertz n=1.5"].append(fit_press(depth, fz, 1.5)[0])
            e, x = fit_press(depth, fz)
            errs["free n"].append(e)
            exps.append(x[2])
        if not exps:
            continue
        print(f"  {pattern}: {len(exps)} trajectories, free exponent median {np.median(exps):.2f} "
              f"(IQR {np.percentile(exps, 25):.2f}-{np.percentile(exps, 75):.2f})")
        for k, v in errs.items():
            print(f"     {k:12s} {100*np.median(v):5.2f}%  {100*np.percentile(v, 90):5.2f}%")

    print("Slide phase (DIGIT, 60 fps): stick-to-slip transition shape (median, p90 error)")
    for pattern in ["digit_sphere_batch_*.pkl", "digit_sharp_batch_*.pkl", "digit_flat_batch_*.pkl"]:
        errs = {2.0: [], 1.5: [], None: []}
        ms, ustar = [], []
        for P, F in trajectories(args.data, pattern):
            out = slide_phase(P, F)
            if out is None:
                continue
            u, ft, fn = out
            m = u < 2.0
            if m.sum() < 15 or ft.max() < 0.05:
                continue
            for k in errs:
                e, x = fit_slide(u[m], ft[m] / np.maximum(fn[m], 1e-3), k)
                errs[k].append(e)
                if k is None:
                    ms.append(x[2])
                    ustar.append(x[1])
        if not ms:
            continue
        print(f"  {pattern}: {len(ms)} slides, free m median {np.median(ms):.2f} "
              f"(IQR {np.percentile(ms, 25):.2f}-{np.percentile(ms, 75):.2f}), "
              f"full slip after u* = {np.median(ustar):.2f} mm")
        for k, v in errs.items():
            name = {2.0: "brush/winkler m=2", 1.5: "cattaneo-mindlin m=1.5", None: "free m"}[k]
            print(f"     {name:22s} {100*np.median(v):5.2f}%  {100*np.percentile(v, 90):5.2f}%")


if __name__ == "__main__":
    main()
