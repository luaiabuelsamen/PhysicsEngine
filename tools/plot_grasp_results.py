"""Figure: held-out success of the fragile-grasp policies by observation set
(mean over 3 seeds, bars show the seed range), from results/grasp/eval.json.

    python3 tools/plot_grasp_results.py --out docs/media/grasp_results.png
"""

import argparse
import json

ORDER = [("proprio", "finger state only"), ("force", "+ pad force readings"),
         ("depth", "+ gel deflection images\n(sensor without markers)"),
         ("shear", "+ marker displacement\nimages"), ("markers", "+ deflection and\nmarker displacement")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/grasp/eval.json")
    ap.add_argument("--out", default="docs/media/grasp_results.png")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(args.results))
    names, means, lo, hi = [], [], [], []
    for key, label in ORDER:
        s = [r["success"] * 100 for r in d["policies"][key]]
        names.append(label)
        means.append(sum(s) / len(s))
        lo.append(sum(s) / len(s) - min(s))
        hi.append(max(s) - sum(s) / len(s))
    fixed = max(v["success"] for k, v in d["baselines"].items() if k.startswith("fixed")) * 100
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    colors = ["#9aa5b1", "#9aa5b1", "#9aa5b1", "#2a9d8f", "#2a9d8f"]
    bars = ax.bar(range(len(names)), means, yerr=[lo, hi], capsize=4, color=colors, width=0.62)
    for b, m, h in zip(bars, means, hi):
        ax.text(b.get_x() + b.get_width() / 2, m + h + 2.5, f"{m:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(fixed, color="#555", ls=":", lw=1, label=f"best fixed grip force ({fixed:.0f}%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("success (lifted, not broken)")
    ax.set_ylim(0, 100)
    ax.set_title("Fragile grasp: held-out success by what the policy observes\n"
                 "(3 seeds x 4,096 episodes; bars: seed range)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
