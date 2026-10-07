# Draft upstream comments (NOT posted; Blocked for Luai)

Evidence for every number: `results/pufferlib_robot_arm/pr692/` (build logs, `smoke_cpu.txt`).

---

## Comment on #692 (raylib 6.0 + linux arm64 builds)

Tested `640eeb0` on a Jetson Orin NX 16 GB:
- JetPack 6.2 (L4T R36.4.3), Ubuntu 22.04.5
- glibc 2.35
- clang 14.0.0, gcc 11.4, CUDA 12.6 (sm_87), cmake 3.22.1
- NCCL from the `nvidia-nccl-cu12` wheel

All compiles were CPU-only, and the CPU smoke test used the `--cpu`
binaries.

**As is, every env fails to link on this board.**
- The prebuilt `raylib-6.0_linux_arm64` archive references
  `__isoc23_sscanf`, `__isoc23_strtol`, `__isoc23_strtoll` and
  `__isoc23_strtoul`. Its `libraylib.so` requires `GLIBC_2.38`.
- Ubuntu 22.04, which JetPack 6 ships, has glibc 2.35:
  ```
  /usr/bin/ld: raylib-6.0_linux_arm64/lib/libraylib.a(rtext.o): undefined reference to `__isoc23_sscanf'
  ```
- This affects `cartpole`, `breakout` and `breakout --cu` (native), plus
  `cartpole`, `breakout` and `impulse_wars` (`--cpu`).
- The `raylib-6.0_linux_amd64` archive has the same symbols and
  `GLIBC_2.38`. So x86_64 Ubuntu 22.04 is probably affected too. I checked
  this with `nm` and `objdump`, not with a build on x86.

**With raylib 6.0 built from source on the host, everything in the PR works.**
I built it with `make -C raylib-6.0/src PLATFORM=PLATFORM_DESKTOP`
(11 s); its headers are identical to the release's.

| Build | Result |
|---|---|
| `cartpole`, `breakout`, `breakout --cu` (native) | pass |
| `cartpole --cpu`, `breakout --cpu` | pass |
| `impulse_wars --cpu` | pass (your box2d-from-source step works) |
| `nmmo3 --cpu` | pass |

CPU smoke test (headless eval, one process):

| Run | Result |
|---|---|
| cartpole, trained checkpoint | perf 0.995, 182K steps/s |
| breakout, trained checkpoint | perf 0.941, 209K steps/s |
| impulse_wars, untrained | runs |
| nmmo3, untrained | runs |

There is no CPU training path in 5.0, so I ran no CPU training.

**Suggestion.** When the host's glibc is older than 2.38, build raylib
from the 6.0 source tag instead of downloading the prebuilt archive. Two
ways to do it:
- always on Linux arm64 (it only takes ~11 s);
- or detect it, e.g. with
  `ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$'`.

A minimal version, mirroring your box2d branch:
```bash
build_raylib() {  # $1 = raylib-6.0_linux_<arch>
    [ -d "$1" ] && return
    download raylib-6.0 "https://github.com/raysan5/raylib/archive/refs/tags/6.0.tar.gz"
    make -C raylib-6.0/src PLATFORM=PLATFORM_DESKTOP RAYLIB_LIBTYPE=STATIC -j"$(nproc)" > /dev/null
    mkdir -p "$1/include" "$1/lib"
    cp raylib-6.0/src/{raylib.h,raymath.h,rlgl.h} "$1/include/"
    cp raylib-6.0/src/libraylib.a "$1/lib/"
}
```

**Two notes for Jetson users, outside this PR:**
- **robot_arm.** `./build.sh robot_arm` still stops on the `obs_t` check
  (the typedef is in `robot_arm.h`). #695 removes that check.
- **NCCL wheel.** It ships only `libnccl.so.2`, so `-lnccl` needs a
  `libnccl.so -> libnccl.so.2` symlink in the wheel's `lib/`.

---

## Comment on #695 (robot_arm rewrite; removes the obs_t check)

On 5.0 (`6ffa5b1`), `./build.sh robot_arm` fails on every platform before
compiling:
```
Error: ocean/robot_arm/robot_arm.cu must typedef obs_t
```
- **Cause.** The check greps only the `.cu`, while `robot_arm.cu` gets
  `obs_t` from `robot_arm.h` (via `robot_arm_cuda.cuh`).
- **Your PR fixes it.** It removes the check and lets the compiler
  validate the type.
- **What I verified.** With only the check relaxed, the current
  robot_arm builds and trains on a Jetson Orin NX (aarch64, sm_87).
  - 200M steps in 5.1 h, about 11K steps/s on a GPU shared with other
    jobs.
  - No successful episodes at that budget, which is 20% of
    `total_timesteps` in `robot_arm.ini`.
  - This is a build/run confirmation only, not a learning result.
- **Not tested.** I have not tested #695's rewritten robot_arm itself.
