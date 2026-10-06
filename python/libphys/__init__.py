"""libphys: batched rigid-body physics for robot learning, with validated
tactile contact.

    import libphys as lp

    model = lp.Model(substeps=10)
    model.add_body(lp.Body.plane())
    box = model.add_body(lp.Body.box((0.1, 0.1, 0.1), mass=1.0))
    world = lp.World(model, num_envs=4096, device="cuda")

    world.state.py[:, box] = 0.5      # zero-copy torch views of the state
    world.step(1 / 60)
    heights = world.state.py[:, box]

State, controls and joint readings are torch tensors that alias the
simulator's own buffers (on the GPU for device="cuda"): writing to them sets
the state, and they reflect every step without copies. Kernels and torch
share CUDA's default stream, so reads and writes are ordered with steps.
"""

import ctypes

from ._libphys import (Actuator, Broadphase, BodyDesc, Device, ElasticPatch, ElasticPatchDesc, Indenter,
                       JointDesc, JointType, ModelDesc, Quat, Shape, Solver, TactileSensorDesc, Vec3)
from . import _libphys

# Optional parts, present when built with tinyxml2 / the motion planner.
UrdfModel = getattr(_libphys, "UrdfModel", None)
Planner = getattr(_libphys, "Planner", None)


def load_urdf(path, **options):
    """Robot model from a URDF: returns a UrdfModel whose .model can be passed
    to World. Options: base_position, base_rotation, gravity (default z-up),
    actuator ("position", "torque", "velocity"), kp, kd, effort_limits,
    joint_damping, sphere_radius."""
    if not hasattr(_libphys, "load_urdf"):
        raise RuntimeError("libphys was built without URDF support (needs tinyxml2)")
    opts = _libphys.UrdfOptions()
    for name, value in options.items():
        if name == "actuator":
            value = _ACTUATORS[value]
        elif name in ("base_position", "gravity"):
            value = Vec3(value)
        elif name == "base_rotation":
            value = Quat(value)
        if not hasattr(opts, name):
            raise AttributeError(f"unknown URDF option {name!r}")
        setattr(opts, name, value)
    return _libphys.load_urdf(path, opts)


__all__ = ["Model", "World", "Body", "Joint", "Actuator", "Broadphase", "Device", "Shape", "Solver",
           "JointType", "ElasticPatch", "ElasticPatchDesc", "Indenter", "Vec3", "Quat", "load_urdf",
           "UrdfModel", "Planner", "TactileSensorDesc", "TACTILE_CHANNELS"]

Body = BodyDesc
Joint = JointDesc

_ACTUATORS = {"none": Actuator.None_, "torque": Actuator.Torque, "position": Actuator.Position,
              "velocity": Actuator.Velocity}


TACTILE_CHANNELS = ("pressure", "shear_x", "shear_y", "deflection", "displacement_x", "displacement_y", "stick")


class Model:
    """Scene description shared by every env: bodies, joints and solver
    settings. Bodies and joints are numbered in the order they are added."""

    def __init__(self, gravity=(0.0, -9.81, 0.0), solver="rigid", **settings):
        self.desc = ModelDesc()
        self.desc.gravity = Vec3(gravity)
        self.desc.solver = {"rigid": Solver.Rigid, "particle": Solver.Particle}[solver]
        for name, value in settings.items():
            if not hasattr(self.desc, name):
                raise AttributeError(f"unknown model setting {name!r}")
            setattr(self.desc, name, value)
        self.bodies = []
        self.joints = []
        self.sensors = []

    def add_body(self, body, friction=None, restitution=None):
        """Add a body (lp.Body.sphere / capsule / box / plane / none); returns its index."""
        if friction is not None:
            body.friction = friction
        if restitution is not None:
            body.restitution = restitution
        self.bodies.append(body)
        return len(self.bodies) - 1

    def add_joint(self, joint, actuator=None, kp=None, kd=None, max_force=None, limits=None, damping=None):
        """Add a joint (lp.Joint.hinge / slider / ball / fixed); returns its index.
        actuator: "torque", "position" (PD with kp, kd) or "velocity";
        limits: (lower, upper)."""
        if actuator is not None:
            joint.actuator = _ACTUATORS[actuator]
        for name, value in (("kp", kp), ("kd", kd), ("max_force", max_force), ("damping", damping)):
            if value is not None:
                setattr(joint, name, value)
        if limits is not None:
            joint.limited = True
            joint.lower, joint.upper = limits
        self.joints.append(joint)
        return len(self.joints) - 1

    def add_tactile_sensor(self, body, origin=(0.0, 0.0, 0.0), frame=(1.0, 0.0, 0.0, 0.0), width=0.02,
                           height=0.02, resolution=(16, 16), youngs_modulus=3e5, poisson=0.5, dome_radius=0.0,
                           **settings):
        """Add a tactile pad of elastic gel to a body; returns its index.

        The pad is a width x height rectangle centred at `origin` in the body
        frame, in the x-y plane of `frame` (a w, x, y, z quaternion in the body
        frame), with its outward normal along the frame's +z. Place it on the
        face of the body's shape that touches objects. resolution = (nx, ny)
        cells. After every step the World reports, per cell, pressure, shear,
        the gel's deflection and displacement, and stick / slip
        (World.tactile)."""
        t = TactileSensorDesc()
        t.body = body
        t.origin = Vec3(origin)
        t.frame = Quat(frame)
        t.width, t.height = width, height
        t.nx, t.ny = resolution
        t.youngs_modulus, t.poisson, t.dome_radius = youngs_modulus, poisson, dome_radius
        for name, value in settings.items():
            if not hasattr(t, name):
                raise AttributeError(f"unknown tactile sensor setting {name!r}")
            setattr(t, name, value)
        self.sensors.append(t)
        return len(self.sensors) - 1

    def build(self):
        self.desc.bodies = self.bodies
        self.desc.joints = self.joints
        self.desc.tactile_sensors = self.sensors
        return self.desc


class _Buffer:
    """Exposes one simulator buffer through __cuda_array_interface__ so that
    torch.as_tensor wraps it without copying; holds the World alive."""

    def __init__(self, owner, ptr, shape, typestr):
        self._owner = owner
        self.__cuda_array_interface__ = {"shape": shape, "typestr": typestr, "data": (ptr, False),
                                         "version": 3, "strides": None}


def _view(owner, ptr, shape, typestr, cuda):
    import torch
    if cuda:
        return torch.as_tensor(_Buffer(owner, ptr, shape, typestr), device="cuda")
    import numpy as np
    count = shape[0] * shape[1]
    ctype = ctypes.c_float if typestr == "<f4" else ctypes.c_uint8
    raw = (ctype * count).from_address(ptr)
    raw._owner = owner  # keep the World alive as long as the array
    dtype = np.float32 if typestr == "<f4" else np.uint8
    return torch.from_numpy(np.frombuffer(raw, dtype=dtype).reshape(shape))


class State:
    """Per-env body state as [num_envs, num_bodies] tensors: position px, py,
    pz; linear velocity vx, vy, vz; orientation quaternion qw, qx, qy, qz;
    angular velocity wx, wy, wz (world frame); enabled (uint8)."""

    _FIELDS = ("px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz", "enabled")

    def __init__(self, views):
        for name in self._FIELDS:
            setattr(self, name, views[name])

    def positions(self):
        import torch
        return torch.stack((self.px, self.py, self.pz), dim=-1)

    def set_positions(self, p):
        self.px.copy_(p[..., 0]); self.py.copy_(p[..., 1]); self.pz.copy_(p[..., 2])

    def velocities(self):
        import torch
        return torch.stack((self.vx, self.vy, self.vz), dim=-1)

    def set_velocities(self, v):
        self.vx.copy_(v[..., 0]); self.vy.copy_(v[..., 1]); self.vz.copy_(v[..., 2])

    def orientations(self):
        import torch
        return torch.stack((self.qw, self.qx, self.qy, self.qz), dim=-1)

    def set_orientations(self, q):
        self.qw.copy_(q[..., 0]); self.qx.copy_(q[..., 1]); self.qy.copy_(q[..., 2]); self.qz.copy_(q[..., 3])

    def angular_velocities(self):
        import torch
        return torch.stack((self.wx, self.wy, self.wz), dim=-1)

    def set_angular_velocities(self, w):
        self.wx.copy_(w[..., 0]); self.wy.copy_(w[..., 1]); self.wz.copy_(w[..., 2])


class World:
    """num_envs independent copies of a Model, stepped together on "cuda" or
    "cpu".

    world.state      State of torch views, [num_envs, num_bodies]
    world.ctrl       actuator inputs, [num_envs, num_joints] (writable)
    world.joint_q    joint positions after the last step, [num_envs, num_joints]
    world.joint_qd   joint velocities after the last step
    world.tactile    per tactile sensor, [num_envs, 7, ny, nx] readings after
                     the last step; channels (TACTILE_CHANNELS): pressure (Pa),
                     shear x, y (Pa), normal deflection of the gel (m),
                     tangential displacement x, y (m), stick (1) / slip (0)
    world.tactile_force  [num_envs, num_sensors, 3] total (shear x, shear y,
                     normal) force on each pad, in the pad frame
    """

    def __init__(self, model, num_envs=1, device="cuda"):
        desc = model.build() if isinstance(model, Model) else model
        self.device = device
        dev = {"cuda": Device.CUDA, "cpu": Device.CPU}[device]
        self._world = _libphys.World(desc, num_envs, dev)
        self._bodies = list(desc.bodies)
        cuda = dev == Device.CUDA
        views = {name: _view(self, ptr, tuple(shape), typestr, cuda)
                 for name, (ptr, shape, typestr) in self._world._buffers().items()}
        self.state = State(views)
        self.ctrl = views.get("ctrl")
        self.joint_q = views.get("joint_q")
        self.joint_qd = views.get("joint_qd")
        self.tactile = []
        self.tactile_force = None
        if self._world.nsensor > 0:
            cells = views["tactile"]
            offset = 0
            for i in range(self._world.nsensor):
                t = self._world.sensor(i)
                size = len(TACTILE_CHANNELS) * t.nx * t.ny
                self.tactile.append(cells[:, offset:offset + size].view(num_envs, len(TACTILE_CHANNELS), t.ny, t.nx))
                offset += size
            self.tactile_force = views["tactile_force"].view(num_envs, self._world.nsensor, 3)

    @property
    def num_envs(self):
        return self._world.nenv

    @property
    def num_bodies(self):
        return self._world.nbody

    @property
    def num_joints(self):
        return self._world.njoint

    def step(self, dt, steps=1):
        """Advance every env by `steps` steps of length dt (asynchronous on CUDA)."""
        self._world.step(dt, steps)

    def synchronize(self):
        self._world.synchronize()

    def contact_counts(self):
        """Contacts found per env in the last substep (rigid solver)."""
        return self._world.contact_counts()

    def set_configuration(self, robot, q, envs=None):
        """Place a URDF robot (built into this World's model) at joint positions
        q, at rest, in the given envs (default: all). Bodies are assumed to be
        the robot's, in order, starting at body 0."""
        import torch
        poses = torch.as_tensor(robot.body_poses(list(map(float, q))), device=self.state.px.device)
        n = poses.shape[0]
        idx = slice(None) if envs is None else envs
        s = self.state
        for k, field in enumerate((s.px, s.py, s.pz, s.qw, s.qx, s.qy, s.qz)):
            field[idx, :n] = poses[:, k]
        for field in (s.vx, s.vy, s.vz, s.wx, s.wy, s.wz):
            field[idx, :n] = 0.0
