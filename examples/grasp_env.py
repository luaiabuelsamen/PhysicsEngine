"""Fragile grasp: a batched RL environment for tactile grasping.

A parallel-jaw gripper with a gel pad on each finger (two-way coupled
tactile sensors) must lift a ball whose mass and friction are hidden and
vary per episode. Each ball is fragile: it breaks if a pad presses on it
harder than `fragility` (1.5x - 2.5x) times the least force that holds it,
m g / (2 mu). Squeeze too little and it slips out; too much and it breaks.

The lift is scripted (the gripper closes for 0.5 s, then rises 10 cm); the
policy adjusts the grip force every control step, by a factor of up to
e^{+-0.3}, starting from 0.4 N - as a grip reflex does. Observation sets:

  "proprio"  time, finger positions and velocities, the last grip command
  "force"    + each pad's normal and shear force
  "tactile"  + each pad's stick fraction (sticking / touching cells)
  "markers"  "force" + each pad's gel deflection and tangential displacement
             images (12 x 12), what a vision-based sensor with markers sees;
             no stick map
  "depth"    "force" + deflection images only (a sensor without markers)
  "shear"    "force" + displacement images only (markers, no depth)

Pad forces tell a policy the ball's weight but not its friction; the stick
fraction - which shrinks as the grip nears slip, Mindlin's
(1 - Q / mu W) - tells it how close to slipping it is, without knowing mu.

    env = GraspEnv(num_envs=2048, obs="tactile")
    obs = env.reset()
    obs, reward, done, info = env.step(action)   # action in [-1, 1]: grip force change
"""

import math

import torch

import libphys as lp

R = 0.03                     # ball radius (m)
MASSES = (0.05, 0.1, 0.2, 0.3, 0.4)
BALL_FRICTION = (0.0, 0.4, 1.0)  # with the fingers' 0.6: contact mu = 0.3, 0.5, 0.8
FINGER_FRICTION = 0.6
FINGER = (0.004, 0.015, 0.015)   # finger half extents
OPEN = 0.012                 # finger gap to the ball at reset (m)
MIN_GRIP, MAX_GRIP = 0.2, 13.0  # N, actions -1 and +1 (log scale)
CONTROL_DT = 1 / 30
PHYSICS_STEPS = 2            # per control step
EPISODE = 60                 # control steps (2 s)
GRASP_TIME = 0.5             # s before lifting
LIFT_SPEED, LIFT_HEIGHT = 0.1, 0.1
PAD_CELLS = 12
IMAGE_CHANNELS = 3           # deflection, displacement x, displacement y
# Image channels per observation set: deflection is channel 3 of a tactile
# reading, tangential displacement channels 4-5.
IMAGE_SETS = {"markers": (3, 6), "depth": (3, 4), "shear": (4, 6)}
OBS_SIZES = {"proprio": 6, "force": 10, "tactile": 12}
OBS_SIZES.update({k: 10 + 2 * (hi - lo) * PAD_CELLS ** 2 for k, (lo, hi) in IMAGE_SETS.items()})


GRIP_RATE = 0.3            # log grip change per control step at |action| = 1
START_GRIP = 0.4           # N


def grip_after(grip, a):
    """Grip force after action a in [-1, 1]: a relative change, so that
    exploration explores smooth grip profiles over the ~20x range of forces
    that matter."""
    return (grip * torch.exp(GRIP_RATE * a)).clamp(MIN_GRIP, MAX_GRIP)


def action_towards(grip, target):
    """The action that moves `grip` towards `target` as fast as allowed."""
    return (torch.log(target / grip) / GRIP_RATE).clamp(-1.0, 1.0)


class GraspEnv:
    def __init__(self, num_envs=1024, obs="tactile", device="cuda", seed=0):
        self.n, self.obs_kind, self.device = num_envs, obs, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        m = lp.Model(gravity=(0, 0, -9.81), substeps=10)
        self.table = m.add_body(lp.Body.plane(), friction=0.5)
        self.carriage = m.add_body(lp.Body.none(0.5, (1e-3, 1e-3, 1e-3)))
        self.fingers = [m.add_body(lp.Body.box(FINGER, 0.05), friction=FINGER_FRICTION, restitution=0.0)
                        for _ in range(2)]
        self.balls, mass, mu = [], [], []
        for mb in MASSES:
            for fb in BALL_FRICTION:
                self.balls.append(m.add_body(lp.Body.sphere(R, mb), friction=fb, restitution=0.0))
                mass.append(mb)
                mu.append(0.5 * (FINGER_FRICTION + fb))
        self.ball_mass = torch.tensor(mass, device=device)
        self.ball_mu = torch.tensor(mu, device=device)
        m.add_joint(lp.Joint.slider(-1, self.carriage, (0, 0, R), (0, 0, 0), (0, 0, 1)), actuator="position",
                    kp=3000.0, kd=150.0, max_force=200.0)
        for f, side in zip(self.fingers, (-1, 1)):
            m.add_joint(lp.Joint.slider(self.carriage, f, (side * (R + FINGER[0] + OPEN), 0, 0), (0, 0, 0),
                                        (1, 0, 0)), actuator="torque", limits=(-0.015, 0.015), damping=20.0)
        q = math.sqrt(0.5)
        for f, side in zip(self.fingers, (-1, 1)):   # pads on the inner faces, normals towards the ball
            m.add_tactile_sensor(f, origin=(-side * FINGER[0], 0, 0), frame=(q, 0, -side * q, 0),
                                 width=2 * FINGER[1], height=2 * FINGER[2], resolution=(12, 12))
        self.world = lp.World(m, num_envs=num_envs, device=device)
        s = self.world.state
        s.qw[:, self.table] = s.qx[:, self.table] = q    # table faces +z
        self.variant = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.fragility = torch.ones(num_envs, device=device)
        self.t = 0
        self.grip = torch.zeros(num_envs, device=device)
        self.broken = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.balls_t = torch.tensor(self.balls, device=device)

    @property
    def obs_size(self):
        return OBS_SIZES[self.obs_kind]

    def reset(self):
        """Reset every env (episodes are aligned): new hidden ball and fragility."""
        s, n, dev = self.world.state, self.n, self.device
        self.variant = torch.randint(len(self.balls), (n,), generator=self.gen, device=dev)
        self.fragility = 1.5 + torch.rand(n, generator=self.gen, device=dev)
        for field in (s.vx, s.vy, s.vz, s.wx, s.wy, s.wz, s.qx, s.qy, s.qz):
            field[:, 1:] = 0.0
        s.qw[:, 1:] = 1.0
        s.px[:, self.carriage] = 0.0
        s.pz[:, self.carriage] = R
        for f, side in zip(self.fingers, (-1, 1)):
            s.px[:, f] = side * (R + FINGER[0] + OPEN)
            s.pz[:, f] = R
        # Park every ball away from the gripper, then bring in each env's own.
        for k, b in enumerate(self.balls):
            s.px[:, b] = 1.0 + 0.1 * k
            s.pz[:, b] = R
            s.enabled[:, b] = 0
        rows = torch.arange(n, device=dev)
        chosen = self.balls_t[self.variant]
        s.enabled[rows, chosen] = 1
        s.px[rows, chosen] = 0.0
        self.t = 0
        self.grip.fill_(START_GRIP)
        self.broken.zero_()
        self.world.ctrl.zero_()
        self.world.step(CONTROL_DT, 1)  # settle the balls on the table
        self.z0 = self.ball_z().clone()
        return self.observe()

    def ball_z(self):
        rows = torch.arange(self.n, device=self.device)
        return self.world.state.pz[rows, self.balls_t[self.variant]]

    def ball_vz(self):
        rows = torch.arange(self.n, device=self.device)
        return self.world.state.vz[rows, self.balls_t[self.variant]]

    def min_grip(self):
        """The least pad force that holds the ball: m g / (2 mu)."""
        return self.ball_mass[self.variant] * 9.81 / (2.0 * self.ball_mu[self.variant])

    def observe(self):
        w = self.world
        time = torch.full((self.n,), self.t / EPISODE, device=self.device)
        fq = torch.stack([w.joint_q[:, 1], w.joint_q[:, 2], w.joint_qd[:, 1], w.joint_qd[:, 2]], dim=1)
        parts = [time[:, None], fq * torch.tensor([50.0, 50.0, 2.0, 2.0], device=self.device),
                 (torch.log(self.grip / START_GRIP) / 3.0)[:, None]]
        if self.obs_kind != "proprio":
            f = w.tactile_force                      # [n, 2, 3]: shear x, shear y, normal
            shear = torch.linalg.norm(f[:, :, :2], dim=2)
            parts += [f[:, :, 2] / 5.0, shear / 5.0]
        if self.obs_kind == "tactile":
            fracs = []
            for t in w.tactile:
                contact = (t[:, 0] > 0).sum(dim=(1, 2)).float()
                stick = (t[:, 6] > 0.5).sum(dim=(1, 2)).float()
                fracs.append(torch.where(contact > 0, stick / contact.clamp(min=1), torch.ones_like(contact)))
            parts.append(torch.stack(fracs, dim=1))
        if self.obs_kind in IMAGE_SETS:
            lo, hi = IMAGE_SETS[self.obs_kind]
            for t in w.tactile:  # [n, 7, 12, 12], in mm
                parts.append((t[:, lo:hi] * 1e3).reshape(self.n, -1))
        obs = torch.cat(parts, dim=1)
        if obs.shape[1] < OBS_SIZES[self.obs_kind]:  # pad to the documented size
            obs = torch.cat([obs, torch.zeros(self.n, OBS_SIZES[self.obs_kind] - obs.shape[1], device=self.device)], 1)
        return obs

    def step(self, action):
        """action: [n] or [n, 1] in [-1, 1], the grip force command.
        Returns obs, reward, done (all envs at the episode's end), info."""
        a = action.reshape(self.n).clamp(-1, 1)
        self.grip = grip_after(self.grip, a)
        t_s = self.t * CONTROL_DT
        lift = min(max(t_s - GRASP_TIME, 0.0) * LIFT_SPEED, LIFT_HEIGHT)
        w = self.world
        w.ctrl[:, 0] = lift
        w.ctrl[:, 1] = torch.where(self.broken, torch.zeros_like(self.grip), self.grip)
        w.ctrl[:, 2] = -w.ctrl[:, 1]
        w.step(CONTROL_DT / PHYSICS_STEPS, PHYSICS_STEPS)
        self.t += 1

        pad_force = w.tactile_force[:, :, 2].max(dim=1).values
        newly_broken = (~self.broken) & (pad_force > self.fragility * self.min_grip())
        self.broken |= newly_broken
        height = (self.ball_z() - self.z0).clamp(min=0.0)
        carriage_v = LIFT_SPEED if GRASP_TIME < t_s < GRASP_TIME + LIFT_HEIGHT / LIFT_SPEED else 0.0
        slip = (self.ball_vz() - carriage_v).abs() * (t_s > GRASP_TIME)
        # Lifting pays every step; slipping costs a little (capped); a broken
        # ball pays nothing more and costs a one-off penalty, so breaking it is
        # worse than dropping it.
        reward = height / LIFT_HEIGHT - 0.1 * (slip / LIFT_SPEED).clamp(max=1.0)
        reward = torch.where(self.broken, torch.zeros_like(height), reward)
        reward = reward - 3.0 * newly_broken.float()
        done = self.t >= EPISODE
        success = (height > 0.8 * LIFT_HEIGHT) & ~self.broken
        info = {"broken": self.broken.clone(), "height": height, "success": success}
        obs = self.observe()
        return obs, reward / EPISODE, done, info
