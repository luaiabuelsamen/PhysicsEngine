"""Throughput of the parallel-jaw grasp env (examples/grasp_env.py) on this
GPU, as a basis for the expected steps/s of a libphys PufferLib env.

Random grip actions; resets at episode ends are included in the timing.
Reports control steps/s (one control step = 2 physics steps at 1/60 s,
10 substeps each), 3 repeats per configuration.

    PYTHONPATH=python:examples:$PYTHONPATH python3 tools/bench_grasp_env.py --out results/pufferlib_spike
"""

import argparse
import json
import os
import time

import torch

from grasp_env import EPISODE, PHYSICS_STEPS, GraspEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pufferlib_spike")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()
    rows = []
    for n in (1024, 4096):
        for obs in ("force", "markers"):
            env = GraspEnv(num_envs=n, obs=obs, seed=0)
            gen = torch.Generator(device=env.device).manual_seed(0)
            env.reset()
            for _ in range(EPISODE):                     # warm-up episode
                env.step(torch.rand(n, device=env.device, generator=gen) * 2 - 1)
            rates = []
            for rep in range(3):
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(args.episodes):
                    env.reset()
                    for _ in range(EPISODE):
                        obs_t, r, done, info = env.step(torch.rand(n, device=env.device, generator=gen) * 2 - 1)
                torch.cuda.synchronize()
                rates.append(n * EPISODE * args.episodes / (time.time() - t0))
            row = {"num_envs": n, "obs": obs, "obs_size": int(obs_t.shape[1]),
                   "control_steps_per_s": rates, "physics_steps_per_control_step": PHYSICS_STEPS}
            rows.append(row)
            print(f"{n:5d} envs  obs {obs:8s} ({row['obs_size']:4d})  "
                  + "  ".join(f"{x:,.0f}" for x in rates) + " control steps/s", flush=True)
            del env
            torch.cuda.empty_cache()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "libphys_grasp_env_sps.json"), "w") as f:
        json.dump({"device": torch.cuda.get_device_name(0), "episodes_per_repeat": args.episodes,
                   "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
