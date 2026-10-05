"""Vectorized CartPole on libphys: the typical RL loop, entirely on the GPU.

Actions go into world.ctrl, observations come from world.joint_q / joint_qd,
and finished envs are reset by writing into the state tensors through a mask
- no host round-trips. Runs a random policy and reports throughput.

    PYTHONPATH=python:$PYTHONPATH python3 examples/cartpole.py --envs 4096
"""

import argparse
import math
import time

import torch

import libphys as lp

POLE_HALF = 0.5          # pole half length (m)
FORCE = 10.0             # max cart force (N)
X_LIMIT, THETA_LIMIT = 2.4, 12 * math.pi / 180
DT = 1 / 60


class CartPole:
    def __init__(self, num_envs, device):
        model = lp.Model(gravity=(0, -9.81, 0), substeps=8)
        self.cart = model.add_body(lp.Body.none(1.0, (0.1, 0.1, 0.1)))
        self.pole = model.add_body(lp.Body.box((0.02, POLE_HALF, 0.02), 0.1))
        model.add_joint(lp.Joint.slider(-1, self.cart, (0, 0, 0), (0, 0, 0), (1, 0, 0)),
                        actuator="torque", max_force=FORCE)
        model.add_joint(lp.Joint.hinge(self.cart, self.pole, (0, 0, 0), (0, -POLE_HALF, 0), (0, 0, 1)))
        self.world = lp.World(model, num_envs=num_envs, device=device)
        self.n, self.device = num_envs, device
        self.obs = torch.zeros(num_envs, 4, device=device)
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    def reset(self, mask):
        """Re-initialise the envs selected by `mask` by writing their state."""
        k = int(mask.sum())
        if k == 0:
            return
        s = self.world.state
        state = torch.empty(k, 4, device=self.device).uniform_(-0.05, 0.05)  # x, theta, x', theta'
        x, th, xd, thd = state.unbind(1)
        zero = torch.zeros(k, device=self.device)
        c, p = self.cart, self.pole
        # Cart on the rail; pole hinged at the cart, rotated by theta about z.
        s.px[mask, c], s.py[mask, c], s.vx[mask, c] = x, zero, xd
        s.px[mask, p] = x - POLE_HALF * torch.sin(th)
        s.py[mask, p] = POLE_HALF * torch.cos(th)
        s.vx[mask, p] = xd - POLE_HALF * torch.cos(th) * thd
        s.vy[mask, p] = -POLE_HALF * torch.sin(th) * thd
        s.qw[mask, p], s.qz[mask, p] = torch.cos(th / 2), torch.sin(th / 2)
        s.wz[mask, p] = thd
        for field in (s.vy, s.vz, s.pz, s.qx, s.qy, s.wx, s.wy):
            field[mask, c] = 0.0
        s.qw[mask, c], s.qz[mask, c], s.wz[mask, c] = 1.0, 0.0, 0.0
        for field in (s.vz, s.pz, s.qx, s.qy, s.wx, s.wy):
            field[mask, p] = 0.0
        self.obs[mask] = state

    def step(self, action):
        """action in [-1, 1] per env -> (obs, reward, done)."""
        self.world.ctrl[:, 0] = action.clamp(-1, 1) * FORCE
        self.world.step(DT)
        q, qd = self.world.joint_q, self.world.joint_qd
        self.obs = torch.stack((q[:, 0], q[:, 1], qd[:, 0], qd[:, 1]), dim=1)
        done = (self.obs[:, 0].abs() > X_LIMIT) | (self.obs[:, 1].abs() > THETA_LIMIT)
        reward = torch.ones(self.n, device=self.device)
        self.reset(done)
        return self.obs, reward, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    env = CartPole(args.envs, args.device)
    episodes, lengths = 0, torch.zeros(args.envs, device=args.device)
    finished = []
    if args.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.steps):
        action = torch.rand(args.envs, device=args.device) * 2 - 1
        obs, reward, done = env.step(action)
        lengths += 1
        if done.any():
            finished.append(lengths[done].clone())
            lengths[done] = 0
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    ep = torch.cat(finished) if finished else torch.zeros(1)
    print(f"{args.envs} envs on {args.device}: {args.envs * args.steps / elapsed:,.0f} env-steps/s; "
          f"{ep.numel()} episodes, random-policy mean length {ep.float().mean():.1f} steps")


if __name__ == "__main__":
    main()
