**Title:** build.sh: build on Linux aarch64 (Jetson); accept obs_t from a .cu env's header

## Problem

On 5.0 (`6ffa5b1`), two problems stop `./build.sh` before training:

1. **Every env fails to link on Linux aarch64.**
   - `build.sh` always downloads `raylib-5.5_linux_amd64`, and raylib
     publishes no Linux arm64 release.
   - On a Jetson Orin, cartpole, breakout (`--cu`) and the others all stop
     with:
     ```
     /usr/bin/ld: raylib-5.5_linux_amd64/lib/libraylib.a: error adding symbols: file in wrong format
     ```
2. **`robot_arm` fails on every platform.**
   - `build.sh` forces robot_arm to its `.cu` source, then greps that file
     for `typedef ... obs_t`.
   - robot_arm.cu gets `obs_t` from `robot_arm.h`, through
     `robot_arm_cuda.cuh`, so the check fails before compiling:
     ```
     Error: ocean/robot_arm/robot_arm.cu must typedef obs_t
     ```

## Fix (build.sh only, +23/-2)

- **raylib on aarch64.** On Linux aarch64, build the same raylib 5.5 tag
  from source (`make -C raylib-5.5/src PLATFORM=PLATFORM_DESKTOP`) into a
  `raylib-5.5_linux_arm64/{include,lib}` directory with the release
  archives' layout. It happens once, like the download.
  - x86_64 Linux, macOS and web still use the prebuilt archives as before.
- **obs_t check.** For `.cu` envs, the native-build check also accepts the
  typedef in the env's `.h`. Other checks and other envs are unchanged.

## Tested

Jetson Orin NX 16 GB (aarch64, JetPack CUDA 12.6, sm_87), Ubuntu 22.04,
clang 14:

| Build | Before | After |
|---|---|---|
| `./build.sh robot_arm` | obs_t error | builds (4 min 46 s, including the raylib build) |
| `./build.sh cartpole` | raylib link error | builds |
| `./build.sh cartpole --cpu` | raylib link error | builds |
| `./build.sh breakout --cu` | raylib link error | builds |

Training throughput on the Orin, measured as logged steps divided by
uptime:
- The GPU was shared with other jobs throughout, so these are lower
  bounds.
- Configs are unchanged, except a shorter robot_arm run.
- The robot_arm run used this branch's clang build. The cartpole and
  breakout runs came from an earlier build of the same commit, made with
  gcc in place of clang plus the same raylib workaround.

| Env | Steps/s | Learning |
|---|---|---|
| cartpole (CPU env) | 1.91M / 1.98M / 2.12M (3 seeds) | perf 0.985-0.995 in 7-8 s |
| breakout (`--cu`) | 1.27M / 0.93M / 0.93M (3 seeds) | perf 0.964-0.998 at 54.9M steps |
| robot_arm | 11.0K mean (7.3K-16.5K), 1 seed | 200M steps (20% of `total_timesteps` in robot_arm.ini) in 5.1 h; no successful episodes yet (peak grasp rate 23%, lift 3.7%) |

The robot_arm row says nothing about the env's learning at its full
budget. It only shows that the env builds, runs and trains on aarch64.

## Notes for Jetson users (not changed in this PR)

- **clang.** `build.sh` needs clang: `apt install clang`.
- **NCCL.** JetPack has no NCCL. The `nvidia-nccl-cu12` wheel works with
  `build.sh`'s existing wheel fallback, but it ships only `libnccl.so.2`,
  so `-lnccl` needs a `libnccl.so -> libnccl.so.2` symlink in the wheel's
  `lib/` directory. This also applies on x86 machines without system NCCL;
  I can send it as a separate change if wanted.
