"""PPO on the fragile-grasp task (examples/grasp_env.py), for one observation
set. Logs the learning curve as JSON and saves the policy.

    PYTHONPATH=python:$PYTHONPATH python3 examples/train_grasp.py --obs tactile --out runs/tactile
    PYTHONPATH=python:$PYTHONPATH python3 examples/train_grasp.py --obs proprio --out runs/proprio
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from grasp_env import EPISODE, GraspEnv


class RunningNorm:
    def __init__(self, size, device):
        self.mean = torch.zeros(size, device=device)
        self.var = torch.ones(size, device=device)
        self.count = 1e-4

    def update(self, x):
        b_mean, b_var, b_count = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        self.var = (self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total) / total
        self.count = total

    def __call__(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-5, 5)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


class ActorCritic(nn.Module):
    def __init__(self, obs_size, hidden=128):
        super().__init__()
        def head(out):
            return nn.Sequential(nn.Linear(obs_size, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, out))
        self.actor, self.critic = head(1), head(1)
        self.log_std = nn.Parameter(torch.full((1,), -1.2))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def train(args):
    torch.manual_seed(args.seed)
    env = GraspEnv(num_envs=args.envs, obs=args.obs, seed=args.seed)
    dev = env.device
    net = ActorCritic(env.obs_size).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    norm = RunningNorm(env.obs_size, dev)
    log, t0 = [], time.time()
    for update in range(args.updates):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []
        obs = env.reset()
        for _ in range(EPISODE):
            norm.update(obs)
            o = norm(obs)
            with torch.no_grad():
                d = net.dist(o)
                a = d.sample()
                v = net.value(o)
            obs, r, done, info = env.step(a)
            obs_buf.append(o); act_buf.append(a); logp_buf.append(d.log_prob(a).sum(-1))
            rew_buf.append(r); val_buf.append(v)
        rewards, values = torch.stack(rew_buf), torch.stack(val_buf)
        # GAE; episodes end together, with no bootstrap.
        adv = torch.zeros_like(rewards)
        last = torch.zeros(env.n, device=dev)
        for t in reversed(range(EPISODE)):
            next_v = values[t + 1] if t + 1 < EPISODE else torch.zeros(env.n, device=dev)
            delta = rewards[t] + args.gamma * next_v - values[t]
            last = delta + args.gamma * args.lam * last
            adv[t] = last
        ret = adv + values
        O = torch.cat(obs_buf); A = torch.cat(act_buf); LP = torch.cat(logp_buf)
        ADV = adv.flatten(); RET = ret.flatten()
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = O.shape[0]
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=dev)
            for mb in perm.split(n // args.minibatches):
                d = net.dist(O[mb])
                ratio = (d.log_prob(A[mb]).sum(-1) - LP[mb]).exp()
                pg = -torch.min(ratio * ADV[mb], ratio.clamp(1 - args.clip, 1 + args.clip) * ADV[mb]).mean()
                vl = 0.5 * (net.value(O[mb]) - RET[mb]).pow(2).mean()
                loss = pg + args.vf * vl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        entry = {"update": update, "steps": (update + 1) * env.n * EPISODE, "return": rewards.sum(0).mean().item(),
                 "success": info["success"].float().mean().item(), "broken": info["broken"].float().mean().item(),
                 "time": time.time() - t0}
        log.append(entry)
        if update % 10 == 0 or update == args.updates - 1:
            print(f"[{args.obs}] update {update:4d}  return {entry['return']:.3f}  success {entry['success']:.3f}  "
                  f"broken {entry['broken']:.3f}  ({entry['time']:.0f} s)", flush=True)
            save(args, net, norm, log)


def save(args, net, norm, log):
    os.makedirs(args.out, exist_ok=True)
    torch.save({"net": net.state_dict(), "norm": norm.state_dict(), "obs": args.obs}, os.path.join(args.out, "policy.pt"))
    with open(os.path.join(args.out, "log.json"), "w") as f:
        json.dump(log, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="tactile", choices=["proprio", "force", "tactile", "markers"])
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/grasp")
    train(ap.parse_args())
