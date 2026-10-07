"""Collect PufferLib 5.0 robot_arm training runs into a results file and plot
the learning curves.

Each run's log (PufferLib's `logs/robot_arm/<run_id>.ini`, written at the
end of training with `--sweep.downsample=64`) holds bin-means of every
metric over training. Steps/s per run is the final agent_steps divided by
the final uptime.

    python3 tools/puffer_robot_arm_results.py SEED=LOG.ini [SEED=LOG.ini ...] \\
        --out results/pufferlib_robot_arm --fig docs/media/robot_arm_curves.png
"""

import argparse
import configparser
import json
import os

CURVES = [("env/success_rate", "success"), ("env/lift_rate", "lift"), ("env/grasp_rate", "grasp")]


def read_metrics(path):
    p = configparser.ConfigParser(interpolation=None)
    p.optionxform = str
    p.read(path)
    m = {k: [float(x) for x in v.split(",")] for k, v in p["metrics"].items()}
    cfg = {s: dict(p[s]) for s in ("base", "train", "vec") if s in p}
    return m, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="SEED=path/to/run.ini")
    ap.add_argument("--out", default="results/pufferlib_robot_arm")
    ap.add_argument("--fig", default="docs/media/robot_arm_curves.png")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = []
    for spec in args.runs:
        seed, path = spec.split("=", 1)
        m, cfg = read_metrics(path)
        steps, up = m["agent_steps"][-1], m["uptime"][-1]
        run = {"seed": int(seed), "log": os.path.basename(path), "agent_steps": steps, "uptime_s": up,
               "mean_sps": steps / up, "final": {k: v[-1] for k, v in m.items() if k.startswith("env/")},
               "curves": m, "config": cfg}
        runs.append(run)
        print(f"seed {seed}: {steps/1e6:.1f}M steps in {up/3600:.2f} h = {steps/up:,.0f} steps/s; "
              + ", ".join(f"{n} {m[k][0]:.3f}->{m[k][-1]:.3f}" for k, n in CURVES if k in m))
    with open(os.path.join(args.out, "runs.json"), "w") as f:
        json.dump({"runs": runs}, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [(k, n) for k, n in CURVES if all(k in r["curves"] for r in runs)]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.3), squeeze=False)
    for ax, (k, n) in zip(axes[0], keys):
        for r in runs:
            ax.plot([s / 1e6 for s in r["curves"]["agent_steps"]], r["curves"][k], lw=1.6, label=f"seed {r['seed']}")
        ax.set_title(f"{n} rate")
        ax.set_xlabel("env steps (M)")
        top = max(max(r["curves"][k]) for r in runs)
        ax.set_ylim(0, max(0.05, 1.15 * top))
        ax.grid(alpha=0.3)
    axes[0][0].legend(frameon=False)
    fig.suptitle("PufferLib 5.0 robot_arm (Franka pick-and-place) on a Jetson Orin NX", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    fig.savefig(args.fig, dpi=130)


if __name__ == "__main__":
    main()
