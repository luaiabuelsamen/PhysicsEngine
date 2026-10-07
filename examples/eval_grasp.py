"""Evaluate fragile-grasp policies (examples/train_grasp.py): success rates on
held-out episodes, learning curves against fixed-grip and oracle baselines,
and a GIF of a policy at work.

    PYTHONPATH=python:$PYTHONPATH python3 examples/eval_grasp.py --runs runs --media docs/media

Every sub-directory of --runs with a policy.pt is evaluated; its name is the
observation set, optionally with a seed suffix (tactile, tactile_s1, ...).
"""

import argparse
import json
import math
import os

import torch
from PIL import Image, ImageDraw

from grasp_env import BALL_FRICTION, EPISODE, FINGER_FRICTION, MASSES, GraspEnv, action_towards
from train_grasp import ActorCritic, RunningNorm
from libphys import viz

KINDS = ("proprio", "force", "depth", "shear", "markers", "tactile")
LABELS = {"proprio": "proprioception", "force": "+ pad forces", "markers": "+ pad forces + deflection & displacement images",
          "depth": "+ pad forces + deflection images", "shear": "+ pad forces + displacement images",
          "tactile": "+ pad forces + stick fraction"}
COLORS = {"proprio": "tab:blue", "force": "tab:orange", "markers": "tab:purple", "tactile": "tab:green",
          "depth": "tab:brown", "shear": "tab:pink"}


def find_runs(roots):
    """{kind: [run dirs]} for every run directory under the roots."""
    runs = {}
    for root in roots:
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.exists(os.path.join(path, "policy.pt")):
                runs.setdefault(name.split("_s")[0], []).append(path)
    return runs


def load(run_dir, env):
    ck = torch.load(os.path.join(run_dir, "policy.pt"), map_location=env.device)
    net = ActorCritic(env.obs_size).to(env.device)
    net.load_state_dict(ck["net"])
    norm = RunningNorm(env.obs_size, env.device)
    norm.load_state_dict(ck["norm"])
    return lambda obs: net.actor(norm(obs)).squeeze(-1)  # the mean action


def rollout(env, policy, on_step=None):
    obs = env.reset()
    for k in range(EPISODE):
        with torch.no_grad():
            a = policy(obs)
        obs, r, done, info = env.step(a)
        if on_step:
            on_step(k, env)
    return info


def evaluate(policy_fn, kind, n, seed):
    env = GraspEnv(num_envs=n, obs=kind, seed=seed)
    info = rollout(env, policy_fn(env))
    ok, broken = info["success"].float(), info["broken"].float()
    by_variant = {}
    for v in range(len(env.balls)):
        sel = env.variant == v
        if sel.any():
            by_variant[v] = ok[sel].mean().item()
    return {"success": ok.mean().item(), "broken": broken.mean().item(),
            "dropped": (1 - ok - broken).clamp(min=0).mean().item(), "by_variant": by_variant}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs"])
    ap.add_argument("--media", default=".")
    ap.add_argument("--envs", type=int, default=4096)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = find_runs(args.runs)
    results = {}
    for kind in KINDS:
        for run in runs.get(kind, []):
            results.setdefault(kind, []).append(evaluate(lambda env, run=run: load(run, env), kind, args.envs, seed=123))
    baselines = {}
    for f in (1.0, 2.0, 4.0):
        baselines[f"fixed {f:g} N"] = evaluate(
            lambda env, f=f: (lambda obs: action_towards(env.grip, torch.full((env.n,), f, device=env.device))),
            "proprio", args.envs, seed=123)
    baselines["oracle (knows m, mu)"] = evaluate(
        lambda env: (lambda obs: action_towards(env.grip, 1.25 * env.min_grip())), "proprio", args.envs, seed=123)

    def summary(rs, key):
        v = [r[key] for r in rs]
        return sum(v) / len(v), min(v), max(v)

    print(f"{'policy':44s} seeds  success (min-max)    broken  dropped")
    for kind, rs in results.items():
        m, lo, hi = summary(rs, "success")
        print(f"{LABELS[kind]:44s} {len(rs):5d}  {m:.3f} ({lo:.2f}-{hi:.2f})   {summary(rs, 'broken')[0]:.3f}   "
              f"{summary(rs, 'dropped')[0]:.3f}")
    for name, r in baselines.items():
        print(f"{name:44s}     -  {r['success']:.3f}                {r['broken']:.3f}   {r['dropped']:.3f}")
    with open(os.path.join(args.runs[0], "eval.json"), "w") as f:
        json.dump({"policies": results, "baselines": baselines}, f, indent=1)

    # Learning curves: mean over seeds, shaded min-max.
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for kind in KINDS:
        logs = [json.load(open(os.path.join(r, "log.json"))) for r in runs.get(kind, [])]
        if not logs:
            continue
        n = min(len(l) for l in logs)
        steps = [logs[0][i]["steps"] / 1e6 for i in range(n)]
        curves = torch.tensor([[l[i]["success"] for i in range(n)] for l in logs])
        ax.plot(steps, curves.mean(0), color=COLORS[kind], label=f"{LABELS[kind]} ({len(logs)} seeds)")
        if len(logs) > 1:
            ax.fill_between(steps, curves.min(0).values, curves.max(0).values, color=COLORS[kind], alpha=0.15)
    ax.axhline(baselines["oracle (knows m, mu)"]["success"], color="k", ls="--", lw=1, label="oracle (knows mass, friction)")
    best_fixed = max((v["success"], k) for k, v in baselines.items() if k.startswith("fixed"))
    ax.axhline(best_fixed[0], color="gray", ls=":", lw=1, label=f"best fixed grip ({best_fixed[1][6:]})")
    ax.set(xlabel="env steps (millions)", ylabel="success rate during training",
           title="Fragile grasp: which observations let PPO learn it", ylim=(0, 1.02))
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.media, "grasp_learning.png"), dpi=120)

    # GIF of the best policy that sees the gel images, on four hidden balls.
    kind = "markers" if "markers" in results else "tactile"
    if kind in results:
        best = max(range(len(results[kind])), key=lambda i: results[kind][i]["success"])
        make_gif(runs[kind][best], kind, os.path.join(args.media, "grasp_tactile.gif"))


def make_gif(run, kind, path):
    env = GraspEnv(num_envs=4, obs=kind, seed=7)
    policy = load(run, env)
    picks = [(0.05, 0.8), (0.4, 0.8), (0.1, 0.3), (0.4, 0.3)]  # (mass, mu)
    mus = [0.5 * (FINGER_FRICTION + fb) for fb in BALL_FRICTION]
    env.reset()
    variants = torch.tensor([MASSES.index(m) * len(mus) + mus.index(mu) for m, mu in picks], device=env.device)
    env.variant = variants
    env.fragility = torch.full((4,), 1.6, device=env.device)
    s = env.world.state
    rows = torch.arange(4, device=env.device)
    for b in env.balls:
        s.enabled[:, b] = 0
        s.px[:, b] = 1.0
    chosen = env.balls_t[variants]
    s.enabled[rows, chosen] = 1
    s.px[rows, chosen] = 0.0
    env.world.step(1 / 30, 1)
    env.z0 = env.ball_z().clone()
    obs = env.observe()
    cam = viz.Camera(eye=(0.16, -0.26, 0.17), target=(0.0, 0.0, 0.07), fov=34, size=(220, 190))
    colors = {env.fingers[0]: (60, 72, 88), env.fingers[1]: (60, 72, 88), env.table: (225, 222, 214)}
    for b in env.balls:
        colors[b] = (233, 160, 70)
    frames, grips = [], [[] for _ in range(4)]
    tile_w, tile_h = 220, 190 + 100 + 30
    for k in range(EPISODE):
        with torch.no_grad():
            a = policy(obs)
        obs, r, done, info = env.step(a)
        for e in range(4):
            grips[e].append((env.grip[e].item(), env.min_grip()[e].item()))
        frame = Image.new("RGB", (4 * tile_w, tile_h + 22), (250, 249, 246))
        d = ImageDraw.Draw(frame)
        d.text((6, 4), f"Policy seeing {LABELS[kind][2:]}, t = {k / 30:.2f} s: grip force vs the least that "
                       f"holds (dashed) and the break limit (red)", fill=(20, 20, 20))
        for e in range(4):
            x0 = e * tile_w
            img = viz.render(env.world, e, cam, colors=colors, hidden=(env.carriage,), plane_extent=0.12)
            frame.paste(img, (x0, 22))
            gel = viz.gelsight(env.world.tactile[0][e], 2 * 0.015 / 12, scale=8, gain=2.0).resize((96, 96))
            tm = viz.tactile_map(env.world.tactile[0][e], scale=8, pressure_max=4e4).resize((96, 96), Image.NEAREST)
            frame.paste(gel, (x0 + 6, 22 + 190))
            frame.paste(tm, (x0 + 112, 22 + 190))
            m, mu = picks[e]
            status = "BROKEN" if info["broken"][e] else f"lifted {info['height'][e].item() * 100:.1f} cm"
            d.text((x0 + 6, 26), f"m {m} kg, mu {mu}", fill=(30, 30, 30))
            d.text((x0 + 6, 40), status, fill=(170, 30, 30) if info["broken"][e] else (30, 110, 60))
            # grip trace
            gx0, gy0, gw, gh = x0 + 6, 22 + 190 + 100, tile_w - 12, 26
            fmax = 1.0 + max(1.6 * g[1] for g in grips[e])
            pts = [(gx0 + gw * i / EPISODE, gy0 + gh - gh * min(g[0] / fmax, 1.0)) for i, g in enumerate(grips[e])]
            ymin = gy0 + gh - gh * grips[e][0][1] / fmax
            ybrk = gy0 + gh - gh * 1.6 * grips[e][0][1] / fmax
            for xx in range(gx0, gx0 + gw, 6):
                d.line([xx, ymin, xx + 3, ymin], fill=(120, 120, 120))
            d.line([gx0, ybrk, gx0 + gw, ybrk], fill=(200, 60, 60))
            if len(pts) > 1:
                d.line(pts, fill=(42, 157, 143), width=2)
        frames.append(frame)
    viz.save_gif(frames, path, fps=15)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
