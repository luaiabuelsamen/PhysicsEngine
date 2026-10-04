# GPU-Accelerated Physics Engine

A rigid body physics engine in C++ and CUDA, built as a batched library (`libphys`) that steps thousands of independent environments at once — the foundation for a GPU reinforcement-learning physics engine.

- **Rigid solver** (XPBD with substeps): rotating spheres, capsules, boxes and planes with friction and restitution, plus **hinge, slider, ball and fixed joints** with limits and torque / PD-position / velocity actuators. On the GPU it runs **~450,000 env-steps/s** for 4,096 envs of 10 falling shapes, and **~51,000 env-steps/s** for 1,024 envs of a 12-joint actuated hand holding a cube.
- Joint dynamics match the exact equations of motion (checked against MuJoCo) and converge at first order in the substep length.
- **Particle solver**: 100,000+ frictionless spheres in one scene; at 100,000 bodies the GPU runs **9x faster than a hand-optimized single-threaded CPU implementation** (373 ms vs 3,393 ms).
- The CPU and GPU backends are deterministic and produce **bit-identical** trajectories.

All numbers are in the benchmark tables below.

![Rigid Pile](static/rigid_pile.gif)

*Rigid solver: spheres, capsules and boxes rain onto a pyramid of boxes (46 bodies, friction, restitution).*

![Rigid Body Simulation](static/rigid_body_sim.gif)

*Particle solver: 200 spheres with gravity and collisions. Color indicates velocity: teal (slow) to red (fast).*

![Particle Pour](static/particle_pour.gif)

*Particle solver: 2,000 spheres pouring from two streams, simulated on GPU.*

![Pendulum Chaos](static/pendulum_chaos.gif)

*500 double pendulums (0.02 rad spread) integrated in parallel on CUDA, showing chaotic divergence with rainbow trails.*

## Architecture

### Library layout

`libphys` separates a shared, read-only **model** (the bodies: shape, size, mass, friction, restitution; gravity; solver settings) from the batched **state** of `nenv` independent environments, stored as flat structure-of-arrays indexed by `env * nbody + body`. Bodies in different environments never interact. The same `World` API runs on either backend:

```cpp
phys::ModelDesc desc;
desc.bodies = {phys::BodyDesc::plane(),
               phys::BodyDesc::box({0.5f, 0.25f, 0.5f}, /*mass=*/1.0f),
               phys::BodyDesc::capsule(/*radius=*/0.1f, /*half_length=*/0.3f, 1.0f)};

phys::World world(desc, /*nenv=*/4096, phys::Device::CUDA);
world.set_state(initial);        // positions, orientations, velocities of every env
world.step(1.0f / 60, /*nsteps=*/10);  // asynchronous, no host round-trips
world.get_state(out);            // or world.state() for raw device pointers
```

Both backends execute the same `__host__ __device__` routines (`source/phys/rigid.h`, `source/phys/particle.h`), so the CPU backend is a bit-exact reference for the GPU.

### Rigid solver (`Solver::Rigid`, the default)

Extended Position Based Dynamics with substeps, after Macklin et al. 2020, *Detailed Rigid Body Simulation with Extended Position Based Dynamics*. Each step is split into substeps (10 by default); each substep runs:

1. **Integrate** (one thread per body): semi-implicit Euler for position, quaternion integration with the gyroscopic term for orientation
2. **Detect contacts** (one thread per env): every pair in the env, with bounding-sphere culling. Narrowphase covers sphere/capsule/box/plane; box-box uses a separating-axis test over 15 axes and clips the incident face against the reference face (Sutherland-Hodgman) to build a contact manifold
3. **Position solve** (one thread per env): penetration corrections over symmetric Gauss-Seidel sweeps, then static friction against the friction cone, then a final penetration sweep
4. **Update velocities** (one thread per body) from the change in pose
5. **Velocity solve** (one thread per env): restitution sweeps, then alternating dynamic-friction and restitution sweeps, with friction impulses accumulated and clamped to the cone

RL environments hold few bodies, so the parallelism comes from running many envs at once; each env's contacts are solved in sequence, which keeps the result stable, deterministic and identical on CPU and GPU.

Velocities are computed from each body's accumulated displacement and rotation within the substep, never from the difference of two absolute poses, so float32 cancellation doesn't limit accuracy at high substep counts.

### Joints and actuators

```cpp
desc.joints.push_back(phys::JointDesc::hinge(/*parent=*/0, /*child=*/1,
                                             /*parent_anchor=*/{0, -0.5f, 0},
                                             /*child_anchor=*/{0, 0.5f, 0}, /*axis=*/{0, 0, 1}));
desc.joints.back().actuator = phys::Actuator::Position;  // control = target angle
desc.joints.back().kp = 50;  desc.joints.back().kd = 5;
...
world.set_controls(ctrl);              // [nenv * njoint], or write state().ctrl on the device
world.step(1.0f / 60);
world.get_joint_state(q, qd);          // [nenv * njoint], computed on the device
```

- **Types**: hinge, slider, ball, fixed; parent `-1` is the world. Each joint is a frame (anchor + orientation, x = axis) in both bodies, solved as XPBD position constraints (anchor, axis alignment or orientation lock, limits) in the same Gauss-Seidel sweeps as contacts.
- **Actuators** (hinge and slider), one control value per joint:
  - `Torque` — torque / force, applied explicitly, clamped to `max_force`
  - `Position` — PD: a compliant XPBD constraint towards the target (compliance `1/kp`) plus velocity-level damping `kd`. Unlike explicit PD it is unconditionally stable, which matters for light, stiff finger links: a 10 g link with `kp = 50` (natural period ~0.2 ms, below the 1.7 ms substep) tracks its target to 4e-5 rad
  - `Velocity` — target joint velocity, with force clamped to `max_force`
- **Passive damping**, **limits**, and **collision filtering**: a joint's two bodies don't collide by default, and MuJoCo-style group / mask bits filter other pairs. `Shape::None` bodies (with an explicit inertia) have no collision geometry.
- Hinge angles use a polynomial `atan2` so CPU and GPU stay bit-identical.

Shapes are centred on their body. Capsules run along the local y axis; planes face along local +y and must be static. A body with mass <= 0 is static.

### Particle solver (`Solver::Particle`)

Non-rotating, frictionless spheres bouncing off axis-aligned walls — for very large single scenes.

1. **Semi-implicit Euler integration** with wall reflection — one thread per body
2. **Broadphase**, chosen per model:
   - **All-pairs within each env** for small scenes (≤32 bodies by default)
   - **Uniform spatial hash** for large scenes: each body gets a key `env * cells + cell`, keys are radix-sorted with CUB (stable, preallocated scratch), and cell start/end offsets are recorded
3. **Contact response** — each body sums the impulses and positional corrections from all its contacts into a separate delta buffer. Each pair is evaluated in canonical (lower-index-first) order, so both bodies see exactly opposite impulses and momentum is conserved
4. **Apply** the deltas

### Determinism

There are no atomics: every phase either writes only its own body (per-body phases) or processes one env sequentially in a fixed order (per-env phases). Runs are bit-for-bit repeatable. With `PHYS_STRICT_FP` (on by default; disables FMA contraction, measured at no cost) the CPU and CUDA backends agree bit for bit — the tests and the particle benchmark check this.

### Key optimizations

- **Structure-of-Arrays (SoA) layout** for coalesced GPU memory access
- **Spatial hash broadphase** (particle solver): O(n) collision detection; each body only checks the 27 surrounding cells
- **Resident state**: all buffers, including contact buffers and sort scratch space, are allocated once; stepping launches a fixed sequence of kernels with no host transfers or allocations
- **Narrow radix sort**: only the key bits that can be used are sorted

### Current limitations

- **Position-based joints are numerically damped at first order in the substep length**: a pendulum swinging at 1.5 rad loses 32% of its energy over 10 s at 10 substeps, 18% at 20, 9% at 40. This is inherent to (X)PBD projection. Reduced-coordinate articulations (Featherstone, as in MuJoCo and PhysX articulations) don't have it, and are the planned fix for articulated hands.
- One GPU thread solves each env, with bodies laid out env-major, so a warp's memory accesses are strided. The hand benchmark slows down beyond ~1,024 envs (51k → 23k env-steps/s), most likely from uncoalesced access once the working set leaves L2 (not yet confirmed with a profiler). An env-interleaved layout or one thread block per env is planned; a single large scene (dozens of bodies) currently runs faster on the CPU backend.
- Kernel launch overhead dominates small batches (40 launches per step at 10 substeps); CUDA graphs are planned.
- Box-box manifolds are built per substep from scratch (no persistent contacts or warm starting). An elastic box bouncing exactly flat can twist slightly on its second landing.
- No convex meshes, torsional or rolling friction, tendons or MJCF/URDF loading yet.

## Benchmark Results

Tested on NVIDIA Jetson Orin NX (1024 CUDA cores, Ampere architecture).

### Rigid solver: batched environments (`./bench_envs`)

Each env: a ground plane and 10 mixed shapes (spheres, capsules, boxes) falling and settling. 10 substeps, 1/60 s steps; one env-step advances one env by one step.

```
Envs      CPU env-steps/s  GPU env-steps/s    GPU/CPU
----------------------------------------------------------
1                   15448              617       0.0x
16                  19948             5721       0.3x
64                  21239            22056       1.0x
256                 21272            70894       3.3x
1024                  ---           288963        ---
4096                  ---           469910        ---
16384                 ---           464024        ---
```

CPU is a single core. The GPU overtakes it at ~64 envs and saturates around 4,096.

`./bench_envs hand` — a static palm, four fingers of three capsule links on 12 position-driven hinges (limits, `max_force`), and a 5 cm cube; every joint follows its own target trajectory, uploaded every step:

```
Envs      CPU env-steps/s  GPU env-steps/s    GPU/CPU
----------------------------------------------------------
1                    1492               95       0.1x
16                   1719             1442       0.8x
64                   1830             4897       2.7x
256                  1834            19585      10.7x
1024                  ---            50976        ---
4096                  ---            22899        ---
16384                 ---            22308        ---
```

The drop past 1,024 envs is the memory-layout limitation described above.

### Particle solver: one large scene (`./benchmark`)

Dense random scene at ~20% packing, times in ms:

```
Bodies    Steps  CPU Naive    CPU Opt     CPU Ref     GPU       vs Naive  vs Opt   vs Ref   GPU==Ref
----------------------------------------------------------------------------------------------------
1000      50     77.5         20.9        22.7        11.3      6.8x      1.8x     2.0x     yes
5000      50     1771.3       119.5       137.0       27.6      64.3x     4.3x     5.0x     yes
10000     50     7088.9       251.0       305.7       37.4      189.7x    6.7x     8.2x     yes
25000     50     44283.3      673.6       938.6       84.7      522.7x    8.0x     11.1x    yes
50000     50     177501.0     1478.2      2164.4      150.5     1179.6x   9.8x     14.4x    yes
100000    50     ---          3392.5      6707.5      372.5     ---       9.1x     18.0x    yes
200000    20     ---          ---         ---         433.2     ---       ---      ---      ---
500000    10     ---          ---         ---         696.6     ---       ---      ---      ---
```

- **CPU Naive**: hand-written single-threaded O(n²) brute force
- **CPU Opt**: hand-written single-threaded spatial hash that visits each pair once and updates both bodies in place — the fastest single-core approach, but not numerically identical to the GPU
- **CPU Ref**: `libphys` CPU backend — exactly the GPU algorithm, single-threaded
- **GPU**: `libphys` CUDA backend
- GPU and CPU Ref time the steps only, with state already resident (the long-simulation / RL use case). **GPU==Ref** reports whether the two final states are bit-identical.

## Building

### Prerequisites

- CUDA Toolkit 11.0+ (tested with 12.6)
- CMake 3.18+
- g++ with C++14 support

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=87  # adjust for your GPU
make -j$(nproc)
```

### Run Tests

```bash
ctest --output-on-failure     # or ./test_phys
```

Each test runs on both backends. They cover:

- **Rigid solver** — resting and dropped spheres, boxes and capsules settle at the right height without drifting; an elastic ball bounces back to its drop height; a tilted box tips onto a face; a 5-box stack stays standing; a sliding box stops after v²/2μg (within 0.1%); torque-free tumbling conserves angular momentum; static obstacles; disabled bodies
- **Particle solver** — elastic collisions against analytic results, momentum conservation, walls, disabled bodies
- **Joints** — compound pendulum period within 0.02% of analytic, energy loss converging at first order, anchor drift < 1e-6; double pendulum and force-driven cart-pole against their exact equations of motion (integrated with RK4 in the test), with errors that shrink with the substep length (40 substeps: 0.009 rad and 0.006 rad / 0.4 mm); PD drives reach the analytic steady state under gravity; a stiff drive on a 10 g link stays stable; force-limited velocity drives accelerate at `max_force / I`; hinge and slider limits; ball and fixed joints; collision filtering
- **Both** — GPU run-to-run determinism, bit-exact CPU/GPU parity (including actuated, contact-rich joint chains), independence of batched environments, input validation

`tools/check_references_mujoco.py` checks that the exact reference models used by the joint tests agree with MuJoCo (to ~1e-13), so a wrong reference cannot make a wrong engine look right.

### Run Benchmark

```bash
# Full suite
./benchmark

# Custom: ./benchmark <num_bodies> [num_steps] [run_naive] [run_cpu_opt_and_ref]
./benchmark 100000 50 0 1

# Rigid solver, batched envs: ./bench_envs [pile|hand|all] [substeps]
./bench_envs
```

### Docker

```bash
docker build -t physics-engine .
docker run --gpus all physics-engine
```

## Project Structure

```
source/
  phys/
    phys.h                  # Public API: BodyDesc, ModelDesc, HostState, World
    common.h                # Plain-data types shared by solvers and backends
    rigid.h                 # Rigid solver: XPBD, narrowphase, friction, joints
    particle.h              # Particle solver: spheres, walls, spatial hash
    world.cpp               # Model validation, inertia, grid setup, dispatch
    backend_cpu.cpp         # CPU backend (bit-exact reference)
    backend_cuda.cu         # CUDA backend
  cpu_rigid_body.cpp/h      # Hand-written CPU spatial hash baseline
  cpu_rigid_body_naive.cpp/h # Hand-written CPU O(n^2) baseline
  benchmark.cpp             # Particle solver: CPU vs GPU on one large scene
  bench_envs.cpp            # Rigid solver: env-steps/s (falling pile, actuated hand)
  visualize_rigid.cpp       # Rigid solver demo - software-rendered GIF
  visualize.cpp             # 2D particle renderer - outputs GIF via ffmpeg
  visualize_pendulum.cpp    # GPU particle pour visualization
  visualize_chaos.cpp       # CUDA-parallel double pendulum chaos
  cuda_pendulum.cu          # CUDA kernel for parallel pendulum RK4 integration
  main.cpp                  # Interactive spring-mass visualization (OpenGL)
  MechanicalSystem.cpp/h    # Single-DOF mechanical system solver
  MultiMechanicalSystem.cpp/h # Multi-DOF coupled system solver
  cuda_mass_spring.cu/h     # CUDA kernel for mass-spring systems
  ode.h                     # RK4 ODE solver
tests/
  test_phys.cpp             # libphys tests (run with ctest)
tools/
  check_references_mujoco.py # Validates the tests' exact reference models with MuJoCo
CMakeLists.txt              # Build system for rigid body engine
SConstruct                  # Build system for spring-mass visualization
Dockerfile                  # CUDA-enabled container
```

## License

MIT License. See [LICENSE](LICENSE) for details.
