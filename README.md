# libphys

GPU-batched robot simulation with physically grounded touch. Thousands of
environments step together on one GPU, and **tactile sensor pads** on any
body report what a gel fingertip would feel:
- pressure;
- shear;
- the gel's deformation;
- where the contact sticks or slips.

The pads solve real contact mechanics rather than penalty springs. Every
result is checked against exact solutions, MuJoCo, or real GelSight / DIGIT
data.

![A gel fingertip pressed onto and dragged across three objects](docs/media/tactile_demo.gif)

*One env per row, simulated together. Left: the scene. Middle: the gel as a
GelSight camera would see it, shaded from the simulated deflection. Right:
pressure, shear (arrows) and the sticking zone (white outline). It sticks
while pressed, then slips fully once dragged.*

**What's in it**

- **Batched rigid bodies.**
  - Shapes: spheres, capsules, boxes, planes.
  - Contacts: friction and restitution.
  - Joints: hinge, slider, ball and fixed, with limits and torque / position /
    velocity actuators.
  - Solver: XPBD with substeps.
  - State is exposed as zero-copy PyTorch tensors.
- **Tactile sensor pads.**
  - The gel is an elastic half-space. It gives Hertz pressure under curved
    objects, Mindlin partial slip under shear, and the deflection map that
    vision-based sensors image.
  - About 140k env-steps/s with a 16 × 16 pad on a Jetson Orin NX.
- **Robots.**
  - URDF import.
  - A C++ motion planner (forward / inverse kinematics, MoveJ, MoveL) whose
    trajectories run in thousands of envs at once.
- **Deterministic.** The CPU and CUDA backends produce bit-identical results,
  so the CPU doubles as a reference.
- **Batteries for looking at things.** `libphys.viz` is a dependency-light
  renderer for scenes and simulated sensor images, used for every GIF here.

C++ / CUDA core, Python front end. Developed on a Jetson Orin NX; any CUDA
GPU works.

## Build

Requirements:
- CUDA toolkit, CMake ≥ 3.18 and a C++17 compiler.
- Python 3 with `torch`, `numpy` and `pybind11`.
- Optional:
  - `libtinyxml2-dev` and `libeigen3-dev`: URDF import and the motion planner.
  - `ffmpeg`: GIF output.
  - `scipy`, `matplotlib`, `pillow`: visualization and tools.

```bash
git clone --recursive https://github.com/luaiabuelsamen/PhysicsEngine   # or: git submodule update --init
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=87      # 87 = Orin; use your GPU's
cmake --build build -j
export PYTHONPATH=$PWD/python:$PYTHONPATH          # the module is built into python/libphys
ctest --test-dir build                             # C++ and Python tests
```

## Quick start

A fingertip with a tactile pad presses a ball into the floor in 1,024 envs,
a little harder in each:

```python
import torch
import libphys as lp

model = lp.Model(gravity=(0, 0, -9.81), substeps=20)
model.add_body(lp.Body.plane(), friction=0.8)                    # floor
finger = model.add_body(lp.Body.box((0.01, 0.01, 0.005), 0.05))
ball = model.add_body(lp.Body.sphere(0.012, 0.02))
model.add_joint(lp.Joint.slider(-1, finger, (0, 0, 0.03), (0, 0, 0), (0, 0, 1)),
                actuator="position", kp=200.0, kd=5.0)            # finger moves up / down
model.add_tactile_sensor(finger, origin=(0, 0, -0.005), frame=(0, 1, 0, 0),  # pad on the -z face
                         width=0.02, height=0.02, resolution=(16, 16))

world = lp.World(model, num_envs=1024, device="cuda")
s = world.state                                  # [num_envs, num_bodies] tensors, zero copy
s.qw[:, 0] = s.qx[:, 0] = 0.5 ** 0.5             # turn the floor to face +z (planes face local +y)
s.pz[:, finger], s.pz[:, ball] = 0.03, 0.012
world.ctrl[:, 0] = torch.linspace(-0.004, -0.008, 1024, device="cuda")   # finger targets

for _ in range(240):
    world.step(1 / 240)
pressure = world.tactile[0][:, 0]                # [1024, 16, 16] Pa, on the GPU
force = world.tactile_force[:, 0]                # [1024, 3]: shear x, shear y, normal (N)
```

More in `examples/`:

| Example | |
|---|---|
| `tactile_demo.py` | The GIF above: a fingertip pressed and dragged across three objects |
| `tactile_gallery.py` | Simulated sensor images of five objects under a rising load |
| `ur5e_planner.py` / `.cpp` | Plan MoveL / MoveJ for a UR5e per env, track the plans under gravity |
| `cartpole.py` | Vectorized CartPole from Python |
| `rigid_pile.cpp`, `particle_pour.cpp`, `particle_bounce.cpp` | Rigid and particle solver demos (write GIFs via ffmpeg) |

## Tactile sensors

![Simulated sensor images](docs/media/tactile_gallery.gif)

`model.add_tactile_sensor(body, origin, frame, width, height, resolution,
youngs_modulus, poisson, dome_radius)` puts a rectangular gel pad on a body,
on the face of its collision shape that touches things. The pad's normal is
the +z axis of `frame`.

After every step, `world.tactile[i]` holds sensor `i`'s cells as
`[num_envs, 7, ny, nx]`. The channels are:

| # | Channel | |
|---|---|---|
| 0 | `pressure` | Pa |
| 1, 2 | `shear_x`, `shear_y` | Pa, in the pad frame |
| 3 | `deflection` | m, normal deflection of the gel: the depth map a GelSight / DIGIT camera sees |
| 4, 5 | `displacement_x`, `displacement_y` | m, tangential gel displacement: what markers show |
| 6 | `stick` | 1 where the contact sticks, 0 where it slips |

`world.tactile_force` holds each pad's total force, `[num_envs, num_sensors,
3]`.

How it works (details in [docs/TACTILE.md](docs/TACTILE.md)):

1. **The load comes from the rigid solver.** It is the contact force on the
   pad's face, averaged over the step's substeps.
2. **The gap comes from the touching objects.** Rays are cast from every cell
   along the pad normal against their shapes.
3. **Pressure is solved for that load.** The pad solves the elastic
   half-space contact problem, so the pressure is consistent with both the
   shape and the load (Polonsky–Keer conjugate gradient, warm-started across
   steps).
4. **Shear and the stick zone use the Ciavarella–Jäger construction.** It is
   the exact partial-slip solution under monotonic shear.
5. **One CUDA block per (env, sensor).** The CPU backend reproduces the GPU
   bit for bit.

![Validation against Hertz and Mindlin](docs/media/tactile_validation.png)

*A pad pressed onto a fixed sphere by slider forces, against closed-form
contact mechanics (`tools/plot_tactile_validation.py`;
`tests/test_tactile_sensor.cpp`).*

`libphys.viz.gelsight(reading, cell)` and `viz.tactile_map(reading)` turn a
reading into the images above.

**Toward real sensors.** `tools/` fits contact laws to Meta's Sparsh datasets
(~2,800 GelSight Mini and DIGIT trajectories with forces). So far:
- **Pressing:** the gels follow elastic half-space laws, not Winkler /
  hydroelastic ones.
- **Shear from force traces:** dominated by test-rig compliance, so the
  traces can't tell the contact models apart.
- **Contact radius from images:** lies between the two models, an open
  question.

The analysis is in [docs/TACTILE.md](docs/TACTILE.md).

## Robots: URDF and the motion planner

![UR5e envs tracking planned trajectories](docs/media/ur5e_planner.gif)

```python
robot = lp.load_urdf("ur5e.urdf", kp=2e5, kd=2e3)   # bodies, inertias, joint and effort limits
planner = lp.Planner("ur5e.urdf", dt=0.04)          # kinematics and trajectories
plan, ok = planner.move_l(home, goal_pos, goal_quat, 60)     # [waypoints, dof]; ok: IK converged
world = lp.World(robot.model, num_envs=1024, device="cuda")
world.set_configuration(robot, home)
world.ctrl[:] = torch.tensor(plan[i], device="cuda")         # joint targets per step
```

**URDF import.**
- Each link with mass becomes a body at its centre of mass, along the
  principal axes of its inertia.
- Revolute, continuous and prismatic joints become hinges and sliders, with
  the URDF's limits.
- Fixed joints are merged or welded.
- Collision geometry is not imported yet.

**Motion planner.** The planner
([MotionPlanning](https://github.com/luaiabuelsamen/MotionPlanning),
`extern/`) is header-only C++:
- forward kinematics along the URDF tree;
- damped-least-squares IK with step limiting and convergence reporting;
- minimum-jerk MoveJ;
- MoveL with a success flag.

In `examples/ur5e_planner.py`, 512 envs track random MoveL plans with a
median joint error of 0.0007 rad and end within 0.1 mm of the goal, at
~100k env-steps/s from Python.

## Rigid bodies

![Rigid pile](docs/media/rigid_pile.gif)

| | |
|---|---|
| `Model` | The scene shared by every env: bodies, joints, tactile sensors, gravity, solver settings. Bodies and joints are numbered in the order added; `parent=-1` attaches a joint to the world. |
| `World(model, num_envs, device)` | `num_envs` independent copies on `"cuda"` or `"cpu"`. Bodies in different envs never interact. |
| `world.state` | `[num_envs, num_bodies]` tensors aliasing the simulator's buffers: `px, py, pz`, `vx, vy, vz`, quaternion `qw, qx, qy, qz`, angular velocity `wx, wy, wz`, `enabled`. Writing them sets the state; resets are masked writes. |
| `world.ctrl` | Actuator inputs, `[num_envs, num_joints]`: torque / force, target position or target velocity. |
| `world.joint_q`, `joint_qd` | Joint positions and velocities after the last step. |
| `world.step(dt, steps)` | Asynchronous on CUDA. Torch and the simulator share the default stream, so tensor reads and writes are ordered with steps. |

**Conventions and semantics.**
- Shapes are centred on their body; capsules run along local y, and planes
  face local +y.
- A body with `mass <= 0` is static.
- Position actuators are compliant constraints (compliance `1/kp`), so stiff
  gains on light links stay stable.
- `Model(solver="particle")` selects a separate solver for very large single
  scenes of frictionless spheres.

**C++.** The same API is in `src/phys/phys.h` (`phys::ModelDesc`,
`phys::World`, `phys::HostState`, `phys::load_urdf`).

## Validation

Every number is a test or tool in this repo.

| Area | Check | Result |
|---|---|---|
| Contacts | Elastic bounce height; rolling sphere (5/7 v₀); sliding box stops at v²/2μg | 0.1%, 0.01%, 0.03% |
| Contacts | Box resting / dropped, 5-box stack, tipping box, capsule at rest | no drift > 0.1 mm |
| Joints | Pendulum period | 0.02% |
| Joints | Double pendulum, cart-pole vs exact equations of motion (references checked against MuJoCo to 1e-13) | 0.009 rad, 0.006 rad / 0.4 mm at 40 substeps |
| Actuators | PD steady state under gravity; force-limited velocity drive | 2e-5 rad; 0.01% |
| Tactile pads | Pad load vs applied; Hertz radius / peak pressure / indentation; Mindlin stick radius; full slip | 0.02%; 0.5% / 1% / 0.6%; within one cell; exact |
| Tactile patch | Hertz, flat punch, Cattaneo–Mindlin, Masing hysteresis (standalone `ElasticPatch`) | 0.07–2% |
| Robots | UR5e: URDF tree FK vs the planner's FK; arm held under gravity | 7e-16 m; 4e-4 rad |
| Backends | CPU vs CUDA, run-to-run, env independence | bit-identical |

## Performance (Jetson Orin NX, 16 GB)

| Workload | Throughput |
|---|---|
| CartPole from Python, 4,096 envs | 438k env-steps/s |
| 10 falling shapes per env, 4,096 envs (`bench_envs pile`) | 450k env-steps/s (22× one CPU core) |
| 12-joint actuated hand + cube, 1,024 envs (`bench_envs hand`) | 51k env-steps/s |
| Pad pressed and dragged over a sphere, 4,096 envs: no sensor / 16 × 16 pad / 32 × 32 pad | 965k / 139k / 15k env-steps/s |
| UR5e tracking planned trajectories, 512 envs, from Python | 100k env-steps/s |
| 100,000 spheres in one scene, particle solver (`benchmark`) | 373 ms / 50 steps |

## Limitations

- **One-way tactile coupling.** The gel's compliance doesn't feed back into
  the rigid contact. So the stick-to-slip transition takes a step or two
  instead of the ~1 mm of slide real gels show, and shear has no hysteresis
  under reversal (the standalone `ElasticPatch` has both). This is the next
  thing to fix.
- **Stiff rigid contacts chatter.** When pushed through joints, the pad force
  scatters step to step:

  | Setting | Scatter |
  |---|---|
  | 10 substeps, under shear | ~8% |
  | 40 substeps, loads ≥ 3 N | ~2% |

  The means are exact.
- **Large pads are slow.** The pad's influence sums are dense; 32 × 32 is
  about 10× slower than 16 × 16.
- **Damped joints.** Position-based joints are damped at first order in the
  substep length: a 1.5 rad pendulum loses 9% of its energy in 10 s at 40
  substeps.
- **Not yet supported:**
  - mesh collision;
  - torsional or rolling friction;
  - URDF collision geometry;
  - MJCF loading.

## Repository layout

```
src/phys/      the library: World, rigid / particle solvers, joints, tactile sensors, URDF import
python/        Python package libphys (bindings, viz) and its tests
examples/      demos (tactile, UR5e, CartPole, rigid / particle GIFs)
tests/         C++ tests
benchmarks/    bench_envs (batched RL envs), benchmark (particle solver vs CPU baselines)
tools/         validation figures, MuJoCo reference check, Sparsh tactile data analysis
extern/        MotionPlanning (submodule)
docs/          TACTILE.md (tactile model and real-data findings), media
legacy/        earlier experiments, not built
```

## License

MIT. See [LICENSE](LICENSE).
