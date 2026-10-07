**Title:** build.sh: fix robot_arm build on all platforms; support Linux aarch64

## What this fixes

1. **`robot_arm` does not build on any platform.**
   - `build.sh` forces robot_arm to its `.cu` source, then greps that file
     for `typedef ... obs_t`.
   - robot_arm.cu gets `obs_t` from `robot_arm.h` (through
     `robot_arm_cuda.cuh`), so the build stops before compiling:
     ```
     Error: ocean/robot_arm/robot_arm.cu must typedef obs_t
     ```
   - **Fix:** for `.cu` envs, the native-build check also accepts the
     typedef in the env's `.h`. Other checks and other envs are unchanged.
2. **No env links on Linux aarch64** (Jetson, Graviton, Raspberry Pi).
   - `build.sh` always downloads `raylib-5.5_linux_amd64`, and raylib
     publishes no Linux arm64 release:
     ```
     /usr/bin/ld: raylib-5.5_linux_amd64/lib/libraylib.a: error adding symbols: file in wrong format
     ```
   - **Fix:** on Linux aarch64, build the same raylib 5.5 tag from source
     (`make -C raylib-5.5/src PLATFORM=PLATFORM_DESKTOP`) into
     `raylib-5.5_linux_arm64/{include,lib}`, the release archives' layout.
     This happens once, like the download.
   - x86_64 Linux, macOS and web still use the prebuilt archives.

`build.sh` only, +23/-2.

## Tested

Jetson Orin NX 16 GB (aarch64, CUDA 12.6, sm_87), Ubuntu 22.04, clang 14,
on 5.0 at `6ffa5b1`:

| Build | Before | After |
|---|---|---|
| `./build.sh robot_arm` | obs_t error | builds and trains |
| `./build.sh cartpole` | raylib link error | builds and trains |
| `./build.sh cartpole --cpu` | raylib link error | builds |
| `./build.sh breakout --cu` | raylib link error | builds and trains |

I have not tested x86_64. The raylib change only applies when `uname -m`
is `aarch64` or `arm64`.

## Notes

- **robot_arm on a shared Orin.** It ran at about 11K steps/s. A 200M-step
  run (one seed, 20% of the configured `total_timesteps`) had no
  successful episodes. That is a smoke test that it builds and trains on
  aarch64, not a measure of what it learns.
- **Jetson setup outside this PR.**
  - `build.sh` needs clang (`apt install clang`).
  - JetPack has no NCCL. The `nvidia-nccl-cu12` wheel works with
    `build.sh`'s existing fallback, but it ships only `libnccl.so.2`, so
    `-lnccl` needs a `libnccl.so -> libnccl.so.2` symlink. This also
    applies to x86 machines without system NCCL; I can send it as a
    separate change.
