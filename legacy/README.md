# Legacy code

Earlier experiments from before `libphys`, kept for reference. None of this is
built by the top-level CMake project or used by the library.

| Path | What it was |
|---|---|
| `source/main.cpp`, `MechanicalSystem.*`, `MultiMechanicalSystem.*`, `ode.*` | Interactive spring-mass-damper visualizer (OpenGL / GLFW, Eigen, matplotlib-cpp), built with `SConstruct` |
| `source/cuda_mass_spring.*`, `source/cuda_scons_tool.py` | CUDA kernel for that visualizer. Its coupling term uses the neighbours' *initial* positions, so coupled masses do not actually interact |
| `source/cuda_pendulum.cu`, `source/visualize_chaos.cpp` | Standalone CUDA RK4 double-pendulum chaos demo (`media/pendulum_chaos.gif`) |
| `PINN.py` | Physics-informed neural network for a damped oscillator (needs PyTorch) |
| `runDocker.sh`, `requirements.txt` | Environment for the visualizer |
| `external/matplotlib-cpp/` | Vendored header used by the visualizer |

The chaos demo builds on its own:

```bash
nvcc -O3 -o visualize_chaos source/visualize_chaos.cpp source/cuda_pendulum.cu
./visualize_chaos   # writes /tmp/pendulum_chaos.gif via ffmpeg
```

The `SConstruct` build expects Eigen at `~/eigen-3.4.0` and OpenCV / GLFW / GLUT
system packages; it predates the move to `legacy/` and may need its paths
updated.
