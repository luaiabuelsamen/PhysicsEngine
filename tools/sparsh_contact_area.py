"""Contact radius versus normal force, measured from GelSight / DIGIT images.

For a sphere pressed into a gel the contact radius grows as a ~ F^(1/3) on an
elastic half-space (Hertz) but as a ~ F^(1/4) for independent springs
(Winkler / hydroelastic). The exponent does not depend on the probe radius,
gel stiffness or rig compliance, so it separates the two contact models
without any fitted parameters.

Measurement: each frame of the press phase is differenced against the last
no-contact frame before the trajectory, smoothed, and azimuthally averaged
around the imprint's centre (the centroid of its brightest part). Gel images
respond to surface slope, so a pressed sphere shows up as a ring; for both
contact models the slope peaks exactly at the contact edge, so the radius of
the ring's peak is the contact radius. log(radius) is then fitted against
log(force) per trajectory. calibrate() runs the same measurement on
synthetic images rendered from each model's exact surface shape.

    python3 tools/sparsh_contact_area.py --images data/sparsh/images --batch 1

Needs the dataset_gelsight_0*.pkl image files of a batch (about 230 MB) in
--images, named gelsight_sphere_b<batch>_<nn>.pkl, and the matching
data/sparsh/gelsight_sphere_batch_<batch>.pkl force file.
"""

import argparse
import glob
import io
import os
import pickle

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))


def load_frames(images, sensor, batch):
    frames = []
    for path in sorted(glob.glob(os.path.join(images, f"{sensor}_sphere_b{batch}_*.pkl"))):
        frames += pickle.load(open(path, "rb"))
    return frames


def decode(frame):
    return np.asarray(Image.open(io.BytesIO(frame)), dtype=np.float32)


def radial_profile(diff, centre, r_max):
    h, w = diff.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - centre[0], xx - centre[1])
    which = r.astype(int).ravel()
    ok = which < r_max
    sums = np.bincount(which[ok], weights=diff.ravel()[ok], minlength=r_max)
    counts = np.bincount(which[ok], minlength=r_max)
    return sums / np.maximum(counts, 1)


def ring_radius(profile, background, min_height):
    """Radius (pixels, parabolic interpolation) of the radial profile's peak;
    None if the imprint is too faint or the peak is at the centre."""
    k = int(np.argmax(profile[1:])) + 1
    if profile[k] - background < min_height or k >= len(profile) - 1:
        return None
    a, b, c = profile[k - 1], profile[k], profile[k + 1]
    den = a - 2 * b + c
    return k + 0.5 + (0.5 * (a - c) / den if den != 0 else 0.0)


def half_max_radius(profile, background, min_height):
    """Radius (pixels, interpolated) where the profile falls to half its
    height above background; None if the imprint is too faint."""
    top = profile[:3].mean()
    height = top - background
    if height < min_height:
        return None
    level = background + 0.5 * height
    below = np.nonzero(profile[1:] < level)[0]
    if len(below) == 0:
        return None
    k = below[0] + 1
    a, b = profile[k - 1], profile[k]
    return (k - 1) + (a - level) / max(a - b, 1e-9) + 0.5


def _surface(model, r, R, depth):
    """Surface depression of a gel pressed by a sphere of radius R to `depth`."""
    if model == "hertz":
        a = np.sqrt(R * depth)
        outside = ((2 * a * a - r ** 2) * np.arcsin(np.minimum(a / r, 1.0)) +
                   r * a * np.sqrt(np.maximum(1 - a * a / r ** 2, 0.0))) / (np.pi * R)
        return np.where(r <= a, depth - r ** 2 / (2 * R), outside)
    return np.maximum(depth - r ** 2 / (2 * R), 0.0)  # winkler: only the overlap deforms


def synthetic_beta(model, R, d_max, span, blur, min_height, extent, rng, trials,
                   contrast=35.0, noise=4.0):
    """Measured exponent on synthetic slope images of `model`: a sphere of
    radius R pixels pressed to depths spanning a force ratio `span`, at the
    real images' contrast and noise. Returns (betas, start radius, end radius)."""
    n = 1.5 if model == "hertz" else 2.0
    size = 2 * extent + 30
    yy, xx = np.mgrid[0:size, 0:size]
    centre = (size / 2.0, size / 2.0)
    r = np.hypot(yy - centre[0], xx - centre[1]) + 1e-9
    depths = np.linspace(d_max * span ** (-1.0 / n), d_max, 12)
    imgs = []
    for d in depths:
        gy, gx = np.gradient(_surface(model, r, R, d))
        imgs.append(np.hypot(gx, gy))
    scale = contrast / max(gaussian_filter(imgs[-1], blur).max(), 1e-12)
    betas, starts, ends = [], [], []
    for _ in range(trials):
        radii, forces = [], []
        for d, img in zip(depths, imgs):
            diff = gaussian_filter(np.abs(img * scale + rng.normal(0, noise, img.shape)), blur)
            prof = radial_profile(diff, centre, extent)
            rk = ring_radius(prof, np.median(prof[-10:]), min_height)
            if rk is not None:
                radii.append(rk)
                forces.append(d ** n)
        if len(radii) >= 5:
            betas.append(np.polyfit(np.log(forces), np.log(radii), 1)[0])
            starts.append(radii[0])
            ends.append(radii[-1])
    return np.array(betas), np.median(starts) if starts else 0.0, np.median(ends) if ends else 0.0


def calibrate(end_radius, span, blur, min_height, extent, trials=40):
    """For each contact model, find the synthetic sphere size whose rings end
    at the measured radius, then return the exponent the measurement reports
    on that model's images (blur biases it upward for small rings)."""
    rng = np.random.default_rng(1)
    out = {}
    for model in ("hertz", "winkler"):
        best = None
        for R in np.linspace(80, 800, 37):
            d_max = (0.8 if model == "hertz" else 0.4) * R / 300.0
            _, s0, s1 = synthetic_beta(model, R, d_max, span, blur, min_height, extent, rng, 2)
            if best is None or abs(s1 - end_radius) < abs(best[1] - end_radius):
                best = (R, s1, d_max)
        betas, s0, s1 = synthetic_beta(model, best[0], best[2], span, blur, min_height, extent, rng, trials)
        out[model] = (np.median(betas), 1.2533 * betas.std() / np.sqrt(max(len(betas), 1)), s0, s1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.path.join(HERE, "..", "data", "sparsh", "images"))
    ap.add_argument("--forces", default=os.path.join(HERE, "..", "data", "sparsh"))
    ap.add_argument("--sensor", default="gelsight")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--blur", type=float, default=2.0, help="Gaussian smoothing (pixels)")
    ap.add_argument("--min-span", type=float, default=2.0, help="min force ratio over the press")
    ap.add_argument("--min-height", type=float, default=6.0, help="min imprint contrast (summed RGB)")
    ap.add_argument("--radius", type=int, default=45, help="radial profile extent (pixels)")
    ap.add_argument("--no-calibrate", action="store_true", help="skip the synthetic calibration")
    args = ap.parse_args()

    frames = load_frames(args.images, args.sensor, args.batch)
    data = pickle.load(open(os.path.join(args.forces, f"{args.sensor}_sphere_batch_{args.batch}.pkl"), "rb"))
    in_contact = np.asarray(data["in_contact"])
    exps, kept, skipped = [], 0, {"short": 0, "edge": 0, "noisy": 0}
    starts, ends, spans = [], [], []
    for t in data["trajectories"].values():
        idx = np.asarray(t["indexes"])
        F = np.asarray(t["forces"])[:, 2]
        P = np.asarray(t["poses"])[:, :3]
        n = min(len(idx), len(F), len(P))
        idx, F, P = idx[:n], F[:n], P[:n]
        moving = np.linalg.norm(P[:, :2] - P[0, :2], axis=1) > 0.02
        end = int(np.argmax(moving)) if moving.any() else n
        press = np.arange(end)
        if len(press) < 6 or F[press].max() / max(F[press].min(), 1e-3) < args.min_span:
            skipped["short"] += 1
            continue
        ref_k = idx[0] - 1
        while ref_k > 0 and in_contact[ref_k]:
            ref_k -= 1
        ref = decode(frames[ref_k])
        # Imprint centre: intensity-weighted centroid of the deepest frame.
        deep = gaussian_filter(np.abs(decode(frames[idx[press[-1]]]) - ref).sum(2), args.blur)
        noise = np.median(deep)
        weight = np.maximum(deep - (noise + 0.5 * (deep.max() - noise)), 0.0)
        if weight.sum() == 0:
            skipped["noisy"] += 1
            continue
        yy, xx = np.mgrid[0:deep.shape[0], 0:deep.shape[1]]
        centre = ((yy * weight).sum() / weight.sum(), (xx * weight).sum() / weight.sum())
        if min(centre[0], centre[1], deep.shape[0] - centre[0], deep.shape[1] - centre[1]) < args.radius:
            skipped["edge"] += 1
            continue
        radii, forces = [], []
        for k in press:
            diff = gaussian_filter(np.abs(decode(frames[idx[k]]) - ref).sum(2), args.blur)
            prof = radial_profile(diff, centre, args.radius)
            rk = ring_radius(prof, np.median(prof[-10:]), args.min_height)
            if rk is not None:
                radii.append(rk)
                forces.append(F[k])
        radii, forces = np.array(radii), np.array(forces)
        good = forces > 0.05
        if good.sum() < 5 or np.ptp(np.log(forces[good])) < np.log(args.min_span) * 0.8:
            skipped["noisy"] += 1
            continue
        slope = np.polyfit(np.log(forces[good]), np.log(radii[good]), 1)[0]
        exps.append(slope)
        starts.append(radii[good][0])
        ends.append(radii[good][-1])
        spans.append(forces[good][-1] / forces[good][0])
        kept += 1

    exps = np.array(exps)
    print(f"{args.sensor} sphere batch {args.batch}: {kept} trajectories measured "
          f"(skipped: {skipped})")
    if kept:
        se = 1.2533 * exps.std() / np.sqrt(kept)  # standard error of the median
        print(f"contact radius ~ F^beta: median beta = {np.median(exps):.3f} +- {se:.3f} "
              f"(IQR {np.percentile(exps, 25):.3f} - {np.percentile(exps, 75):.3f})")
        print(f"  ring radius {np.median(starts):.1f} -> {np.median(ends):.1f} px over a force span of "
              f"{np.median(spans):.2f}x")
        print("  ideal measurement: elastic half-space (Hertz) 0.333, Winkler / hydroelastic 0.250")
        if not args.no_calibrate:
            cal = calibrate(np.median(ends), np.median(spans), args.blur, args.min_height, args.radius)
            print("  the same measurement on synthetic images at this scale (exact surfaces, real "
                  "contrast and noise):")
            for model, (b, se, s0, s1) in cal.items():
                print(f"    {model:8s} reads beta = {b:.3f} +- {se:.3f} (rings {s0:.1f} -> {s1:.1f} px)")


if __name__ == "__main__":
    main()
