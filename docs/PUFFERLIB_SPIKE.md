# PufferLib on the Jetson Orin

**Question.** Does PufferLib work on this Jetson (Orin NX 16 GB, aarch64,
CUDA 12.6, sm_87, gcc 11.4)?

**Answer: yes, with local build workarounds (compiler, raylib, NCCL).**
- The CUDA trainer trains both a CPU env and a GPU-resident env on the Orin
  at 0.9-2.1M steps/s.
- The current release has no CPU trainer. The previous release's PyTorch CPU
  fallback trains at 27-44k steps/s.

All numbers below come from `results/pufferlib_spike/summary.json`. The
cleaned logs are in `results/pufferlib_spike/logs/`.
- **Mean steps/s:** the logged step count divided by the logged uptime, read
  from the final dashboard frame.
- **Shared machine:** other jobs were using the GPU during the runs, so
  steps/s are lower bounds and vary between repeats.

## Versions

| | Branch | Commit |
|---|---|---|
| PufferLib 5.0 | `5.0` (upstream default branch) | `6ffa5b1` (2026-09-13) |
| PufferLib 4.0 | `4.0` | `42f70d6` (2026-08-20) |

5.0 is the native rewrite:
- Ocean envs are C headers, or `.cu` files for GPU-resident envs.
- `build.sh` compiles one env together with the trainer into a single
  binary.
- The trainer (`src/pufferl.cu`) is CUDA-only.

4.0 is a Python package with a compiled `_C` extension and a `--slowly`
PyTorch backend.

## Results

| Test | Result | Steps/s | Learning (3 runs) |
|---|---|---|---|
| (a) Build C Ocean envs on aarch64 | **pass with workarounds** | 5.0 CPU eval, cartpole, one core, random policy: 20.6M env-steps/s | n/a |
| (b) Train a toy Ocean env on CPU | **fail on 5.0** (no CPU trainer); **pass on 4.0** | 4.0 `--slowly`, cartpole: 27k / 44k / 44k | perf 0.86 / 0.99 / 0.99 after 8.4M steps |
| (c) CUDA trainer on the Orin (sm_87) | **pass with workarounds** | 5.0, cartpole (CPU env): 1.91M / 1.98M / 2.12M | perf 0.995 / 0.995 / 0.985 after 14.9M steps (7-8 s) |
| (c) same, GPU-resident env | **pass** for `breakout --cu`; **fail** for `robot_arm` (build) | 5.0, breakout: 1.27M / 0.93M / 0.93M | perf 0.998 / 0.983 / 0.964 after 54.9M steps (43-59 s) |

Notes on the table:
- **perf** is PufferLib's normalised score: 1.0 is the env's solved score.
- **Seeds.** 5.0 runs used `base.seed` 73 (the default), 1 and 2.
- **4.0 runs are unseeded repeats.** 4.0's `--seed` parses, but nothing on
  the CPU path reads it.
- **Contention.** The first 4.0 run (27k steps/s, perf 0.86 after the fixed
  step budget) ran while other GPU work was active. Breakout runs 2 and 3
  overlapped 4.0 CPU training.

### Failures and workarounds

**(a) Building on aarch64.** The stock build fails:
```
./build.sh: line 311: clang: command not found
```
Workarounds:
- `build.sh` also downloads raylib's `linux_amd64` archive regardless of
  architecture. Fix: raylib 5.5 built from source for aarch64, with the
  archive placed where `build.sh` expects it.
- No clang on the machine. Fix: a `clang` shim that calls gcc and drops
  clang-only flags (`-ferror-limit=*`,
  `-W*incompatible-pointer-types-discards-qualifiers`).
- On 4.0 it also drops `-mavx2` / `-mfma`, which gcc on aarch64 rejects.
- No ccache. Fix: a pass-through shim.

Envs that link prebuilt binaries have the same problem. For example,
`impulse_wars` downloads `box2d-linux-amd64` on Linux; it was not
attempted.

**(b) CPU training.**
- **5.0:** `build.sh --cpu` builds an eval/play binary only. There is no
  CPU training path.
- **4.0:** the default compiled backend fails on a CPU build:
  ```
  AttributeError: module 'pufferlib._C' has no attribute 'create_pufferl'
  ```
  `--slowly` (PyTorch backend) works.
- **4.0 config:** `config/cartpole.ini` sets `num_layers = 2.11327`, which
  crashes the PyTorch model:
  ```
  TypeError: 'float' object cannot be interpreted as an integer
  ```
  A CLI override does not help, because it is parsed with the same float
  type. Fix: set it to 2 in the local config.

**(c) CUDA trainer.**
- **NCCL.** `src/pufferl.cu` includes `nccl.h` unconditionally, and JetPack
  has no NCCL. Fix: the `nvidia-nccl-cu12` 2.32.3 wheel in the spike venv,
  plus a `libnccl.so` symlink and `LD_LIBRARY_PATH`.
- **Architecture.** nvcc's `-arch=native` picks sm_87; no change needed.
- **robot_arm (upstream bug at `6ffa5b1`).** `build.sh` forces robot_arm to
  its `.cu` source, then greps that file for `typedef ... obs_t`. The
  typedef lives in `robot_arm.h`, so the build stops:
  ```
  Error: ocean/robot_arm/robot_arm.cu must typedef obs_t
  ```
  `breakout.cu` defines `obs_t` itself and builds.

All workarounds live in `/tmp`, in an isolated venv. No system packages
were changed.

## Sketch: libphys as a PufferLib 5.0 GPU env (not built)

### Which path

- **GPU-resident (`--cu`), like `robot_arm` and `breakout`.** libphys
  state already lives on the GPU. A CPU-env wrapper would need a
  device synchronisation and a host-side pass over every observation each
  step.
- **Not 4.0's Python path.** It trains at tens of k steps/s. That is no
  faster than the existing PyTorch PPO in `examples/train_grasp.py`.

### Files

**`ocean/libphys_grasp/libphys_grasp.cu`**, built by a new `libphys_grasp`
branch in `build.sh`. The branch would follow the `impulse_wars` / Box2D
pattern:
- `INCLUDES += -I$LIBPHYS/src`
- `LINK_ARCHIVES += $LIBPHYS/build/libphys.a`
- `USE_GPU_ENV=1`

The `.cu` file defines:
```c
typedef float obs_t;                      // in the .cu itself (see the robot_arm build bug)
#define OBS_SIZE  (6 + 4 + 2*3*12*12)     // proprio, pad forces, two 3-channel 12x12 pad images = 874
#define NUM_ATNS  1                       // grasp: grip force rate (lift is scripted); insertion: 4
#define ACT_SIZES {1}                     // continuous

Env* puf_vec_create(int n, Dict* kw, obs_t* obs, float* act, float* rew, float* term);
    // build phys::ModelDesc (table, carriage, two slider fingers with 12x12 pads, object,
    // for insertion: peg held by the fingers + hole fixture); phys::World w(desc, n, Device::CUDA);
    // keep w.state() device pointers and Puffer's obs/act/rew/term pointers in a global.
void puf_bind_stream(cudaStream_t s);      // store s; libphys must launch on it (below)
void puf_reset(Env*);                      // lp_kreset<<<>>>: write StateView px..wz, ctrl for all envs
void puf_step(Env*);
    // lp_kact<<<>>>:  actions -> StateView.ctrl (grip force, lift / approach set-point)
    // w.step(dt, nsub)  on the bound stream
    // lp_kfin<<<>>>:  StateView (+ tactile, tactile_force) -> obs, rewards, terminals;
    //                 envs that terminated are re-initialised in place (as ra_kfin does)
```

### Observation and action

**Grasp (the existing `examples/grasp_env.py` task)**, unchanged so results
compare with the committed 67% vs 36% numbers:

| Group | Size |
|---|---|
| Time | 1 |
| Finger positions and velocities | 4 |
| Grip | 1 |
| Pad forces (normal and shear, two pads) | 4 |
| Pad images: deflection plus two displacement channels per pad, 12 x 12 | 864 |

The action is the grip force rate. The pad-force variant uses only the
first 10 inputs.

**Parallel-jaw insertion variant**:
- Observations add the peg pose relative to the hole (7) and the lateral
  carriage set-point.
- Actions: grip force rate plus a 2-D lateral and 1-D vertical velocity
  set-point for the carriage, so 4 continuous actions.

Per [DECISION.md](DECISION.md), insertion in libphys is blocked on
box-box edge contact (E1). A box-peg insertion env would reproduce the
toppling bug, so only the grasp env is a valid target today.

### Changes libphys needs

1. **A CUDA stream argument.**
   - **Current behaviour.** `Backend::step` launches its kernels on the
     legacy default stream, and `synchronize()` calls
     `cudaDeviceSynchronize()`.
   - **Why it matters.** With `cudagraphs = 1` (the 5.0 default),
     PufferLib captures the whole horizon (`forward` + `puf_step` × H) on
     its own stream with `cudaStreamBeginCapture(...ThreadLocal)`. Launches
     on the legacy stream invalidate the capture.
   - **Why it is feasible.** `step()` itself has no host sync, allocation
     or readback; it only branches on host-side constants. So passing a
     stream to its kernel launches should make it capture-safe.
   - **Fallback.** Run with `cudagraphs = 0`.
2. **A device-side reset.**
   - **Current behaviour.** Python resets by writing torch tensors, and C++
     resets via `set_state` (a host-to-device copy).
   - **Needed.** A reset kernel in the env that writes `StateView` arrays
     for the terminated envs only.
3. **Observation write.** `lp_kfin` packs `tactile` / `tactile_force` into
   Puffer's `obs` buffer. It must reproduce `grasp_env.py`'s image channels
   and normalisation.

### Expected steps/s

The env, not the trainer, sets the rate. Measured on this machine:

| What | Steps/s | Source |
|---|---|---|
| PufferLib 5.0 trainer on toy envs | 0.9-2.1M | above |
| libphys grasp env alone, random actions, 1,024 envs (force / markers obs) | 21-50k / 44-51k | `results/pufferlib_spike/libphys_grasp_env_sps.json` |
| same, 4,096 envs | 14-20k / 16-22k | same |
| Existing PyTorch PPO, grasp, 2,048 envs, end to end (force / markers, 3 seeds) | 6.3-9.5k / 5.7-7.0k | `results/grasp/logs/*.json` (steps / time at the last update) |

The 4,096-env rows are below the 1,024-env rows because the GPU was shared
during the measurement. Treat both as lower bounds.

**Expectation.** A 5.0 libphys env would train at roughly the env-alone
rate, about 20-50k steps/s. That is about 2-9x the current end-to-end
grasp training (5.7-9.5k steps/s), because PyTorch PPO and Python overhead
would leave the loop. It is still roughly 20-100x below the trainer's
toy-env rate.
- **What it would buy.** The 150-update grasp runs (18.4M steps, 32-54 min
  each today) would take about 6-15 min.
- **What it would not buy.** It does not change the decision in
  [DECISION.md](DECISION.md). The gain is training speed on the tactile
  grasp task only.

**Not measured:**
- Physics time with graphs vs without.
- The policy forward/backward cost for an 874-input observation in 5.0's
  trainer.

Building the env would measure both.

## Reproduce

```bash
# 5.0 (CUDA trainer); needs the workarounds above on aarch64
git clone -b 5.0 https://github.com/PufferAI/PufferLib && cd PufferLib
./build.sh cartpole && ./puffer train --base.seed=1
./build.sh breakout puffer_breakout --cu && ./puffer_breakout train --base.seed=1
# 4.0 (CPU, PyTorch backend)
git clone -b 4.0 https://github.com/PufferAI/PufferLib && cd PufferLib
./build.sh cartpole --cpu && puffer train cartpole --slowly
# libphys grasp env throughput
PYTHONPATH=python:examples:$PYTHONPATH python3 tools/bench_grasp_env.py --out results/pufferlib_spike
```
