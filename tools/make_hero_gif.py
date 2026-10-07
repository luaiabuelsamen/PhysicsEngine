"""Hero GIF for the README: the median-seed pad-force policy (top row) and the
median-seed gel-image policy (bottom row) on the same randomly drawn hidden
balls, three episodes, half speed. A ball that breaks turns red.

    PYTHONPATH=python:examples:$PYTHONPATH python3 tools/make_hero_gif.py --out docs/media/hero.gif
"""

import argparse
import os

import torch
from PIL import Image, ImageDraw

from grasp_env import EPISODE, FINGER_FRICTION, GraspEnv
from train_grasp import ActorCritic, RunningNorm
from libphys import viz

ROOT = os.path.join(os.path.dirname(__file__), "..")
POLICIES = [("force", "results/grasp/policies/force_s0.pt", "pad force readings"),
            ("markers", "results/grasp/policies/markers_s1.pt", "gel images (deflection + marker displacement)")]
BALLS, EPISODES = 4, 3
TILE_W, TILE_H, LABEL_H, HEAD = 230, 180, 22, 30


def load(path, env):
    ck = torch.load(os.path.join(ROOT, path), map_location=env.device)
    net = ActorCritic(env.obs_size).to(env.device)
    net.load_state_dict(ck["net"])
    norm = RunningNorm(env.obs_size, env.device)
    norm.load_state_dict(ck["norm"])
    return lambda obs: net.actor(norm(obs)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/hero.gif")
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()
    envs = [GraspEnv(num_envs=BALLS, obs=kind, seed=args.seed) for kind, _, _ in POLICIES]
    policies = [load(path, env) for (_, path, _), env in zip(POLICIES, envs)]
    cam = viz.Camera(eye=(0.17, -0.25, 0.16), target=(0.0, 0.0, 0.065), fov=36, size=(TILE_W, TILE_H))
    width = BALLS * TILE_W
    height = HEAD + len(POLICIES) * (LABEL_H + TILE_H)
    frames = []
    for episode in range(EPISODES):
        obs = [env.reset() for env in envs]
        assert torch.equal(envs[0].variant, envs[1].variant)  # same hidden balls for both policies
        for k in range(EPISODE):
            infos = []
            for i, (env, pol) in enumerate(zip(envs, policies)):
                with torch.no_grad():
                    a = pol(obs[i])
                obs[i], _, _, info = env.step(a)
                infos.append(info)
            frame = Image.new("RGB", (width, height), (250, 249, 246))
            d = ImageDraw.Draw(frame)
            d.text((8, 9), f"Lift a fragile ball of hidden mass and friction   episode {episode + 1}/{EPISODES}   "
                           f"t = {k / 30:.2f} s (half speed)", fill=(25, 25, 25))
            for row, ((kind, _, label), env, info) in enumerate(zip(POLICIES, envs, infos)):
                y0 = HEAD + row * (LABEL_H + TILE_H)
                d.text((8, y0 + 5), f"Policy sees: {label}", fill=(20, 20, 20))
                colors = {env.fingers[0]: (60, 72, 88), env.fingers[1]: (60, 72, 88), env.table: (226, 223, 215)}
                for e in range(BALLS):
                    broken = bool(info["broken"][e])
                    for b in env.balls:
                        colors[b] = (205, 55, 55) if broken else (233, 160, 70)
                    img = viz.render(env.world, e, cam, colors=dict(colors), hidden=(env.carriage,),
                                     plane_extent=0.12)
                    frame.paste(img, (e * TILE_W, y0 + LABEL_H))
                    m = env.ball_mass[env.variant[e]].item()
                    mu = env.ball_mu[env.variant[e]].item()
                    d.text((e * TILE_W + 8, y0 + LABEL_H + 6), f"{m * 1000:.0f} g, mu {mu:.1f}", fill=(40, 40, 40))
                    if k == EPISODE - 1 or broken:
                        status, col = ("broken", (190, 40, 40)) if broken else (
                            ("lifted", (30, 120, 60)) if info["height"][e] > 0.08 else ("dropped", (150, 110, 20)))
                        if k == EPISODE - 1 or broken:
                            d.text((e * TILE_W + 8, y0 + LABEL_H + 20), status, fill=col)
            frames.append(frame)
        frames += [frames[-1]] * 8  # hold the outcome for half a second
    viz.save_gif(frames, args.out, fps=15)
    print(f"wrote {args.out}: {len(frames)} frames, {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
