"""What does a gel-image grasp policy look at? Two probes of a trained
"markers" policy (examples/train_grasp.py):

  ablation  success when one input group is replaced by its training mean
            for the whole episode (causal: what the policy needs)
  saliency  the size of the gradient of its grip action with respect to
            every input during the lift, by input group, image channel and
            distance from the contact's centre (what it is sensitive to;
            inflated for inputs whose spread is tiny, as normalisation
            magnifies them)

    PYTHONPATH=python:$PYTHONPATH python3 examples/probe_grasp.py runs/markers_s0
"""

import argparse
import math

import torch

from grasp_env import EPISODE, GRASP_TIME, CONTROL_DT, PAD_CELLS, GraspEnv
from train_grasp import ActorCritic, RunningNorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--envs", type=int, default=1024)
    args = ap.parse_args()
    env = GraspEnv(num_envs=args.envs, obs="markers", seed=11)
    ck = torch.load(f"{args.run}/policy.pt", map_location=env.device)
    net = ActorCritic(env.obs_size).to(env.device)
    net.load_state_dict(ck["net"])
    norm = RunningNorm(env.obs_size, env.device)
    norm.load_state_dict(ck["norm"])

    groups = {"time": [0], "finger position / velocity": [1, 2, 3, 4], "grip": [5],
              "pad normal forces": [6, 7], "pad shear forces": [8, 9]}
    cells = PAD_CELLS * PAD_CELLS
    sal = torch.zeros(env.obs_size, device=env.device)
    # Image saliency by channel and normalised radius from the contact centre.
    bins = torch.linspace(0, 2.5, 11, device=env.device)
    radial = torch.zeros(3, len(bins) - 1, device=env.device)
    radial_n = torch.zeros(len(bins) - 1, device=env.device)
    obs = env.reset()
    samples = 0
    for k in range(EPISODE):
        o = norm(obs).detach().requires_grad_(True)
        a = net.actor(o).sum()
        (g,) = torch.autograd.grad(a, o)
        # Saliency of the raw input: d action / d normalised input / std.
        g = (g / torch.sqrt(norm.var + 1e-8)).abs() * torch.sqrt(norm.var + 1e-8)  # = |d a / d o_norm|
        lifting = (k * CONTROL_DT > GRASP_TIME) & (k * CONTROL_DT < GRASP_TIME + 1.0)
        with torch.no_grad():
            if lifting:
                live = ~env.broken
                sal += g[live].sum(0)
                samples += int(live.sum())
                for pad, t in enumerate(env.world.tactile):
                    p = t[:, 0]                                  # pressure [n, 12, 12]
                    contact = p > 0
                    area = contact.sum(dim=(1, 2)).float()
                    ok = live & (area > 3)
                    if not ok.any():
                        continue
                    yy, xx = torch.meshgrid(torch.arange(PAD_CELLS, device=env.device),
                                            torch.arange(PAD_CELLS, device=env.device), indexing="ij")
                    w = p / p.sum(dim=(1, 2), keepdim=True).clamp(min=1e-12)
                    cy = (w * yy).sum(dim=(1, 2)); cx = (w * xx).sum(dim=(1, 2))
                    radius = torch.sqrt(area / math.pi)
                    r = torch.sqrt((yy - cy[:, None, None]) ** 2 + (xx - cx[:, None, None]) ** 2) / radius[:, None, None]
                    base = 10 + pad * 3 * cells
                    gimg = g[:, base:base + 3 * cells].reshape(-1, 3, PAD_CELLS, PAD_CELLS)
                    idx = torch.bucketize(r, bins) - 1
                    for b in range(len(bins) - 1):
                        m = (idx == b) & ok[:, None, None]
                        if m.any():
                            radial[:, b] += (gimg * m[:, None]).sum(dim=(0, 2, 3))
                            radial_n[b] += m.sum()
        obs, r, done, info = env.step(net.actor(norm(obs)).squeeze(-1).detach())
    sal /= max(samples, 1)
    ablate(env, net, norm)
    print(f"mean |d grip action / d normalised input| during the first second of the lift ({samples} states)\n")
    for name, idx in groups.items():
        print(f"  {name:30s} {sal[idx].mean().item():.4f} per input")
    for c, name in enumerate(("deflection", "displacement x", "displacement y")):
        chan = torch.cat([sal[10 + pad * 3 * cells + c * cells: 10 + pad * 3 * cells + (c + 1) * cells] for pad in (0, 1)])
        print(f"  image: {name:23s} {chan.mean().item():.4f} per pixel (max {chan.max().item():.4f})")
    print("\n  image saliency by distance from the contact centre (r / contact radius):")
    print("   r      deflection  displ. x  displ. y")
    per = radial / radial_n.clamp(min=1)
    for b in range(len(bins) - 1):
        print(f"  {bins[b]:.2f}-{bins[b + 1]:.2f}   {per[0, b]:.4f}     {per[1, b]:.4f}    {per[2, b]:.4f}")


def ablate(env, net, norm):
    cells = PAD_CELLS * PAD_CELLS
    def image(c):
        return [10 + pad * 3 * cells + c * cells + i for pad in (0, 1) for i in range(cells)]
    groups = {"(nothing)": [], "pad normal forces": [6, 7], "pad shear forces": [8, 9],
              "all pad forces": [6, 7, 8, 9], "deflection images": image(0),
              "displacement x images (along the lift)": image(1), "displacement y images (across)": image(2),
              "all images": image(0) + image(1) + image(2)}
    print("ablation: success with an input group held at its training mean\n")
    for name, idx in groups.items():
        idx = torch.tensor(idx, dtype=torch.long, device=env.device)
        obs = env.reset()
        for k in range(EPISODE):
            o = norm(obs)
            if len(idx):
                o[:, idx] = 0.0
            with torch.no_grad():
                a = net.actor(o).squeeze(-1)
            obs, r, done, info = env.step(a)
        print(f"  {name:40s} success {info['success'].float().mean().item():.3f}  "
              f"broken {info['broken'].float().mean().item():.3f}")
    print()


if __name__ == "__main__":
    main()
