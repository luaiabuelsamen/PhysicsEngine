"""Check the exact reference dynamics used by tests/test_phys.cpp against MuJoCo.

The joint tests compare libphys against equations of motion integrated with
RK4 inside the test. This script integrates the same equations (in Python)
and runs the same systems in MuJoCo (RK4, 200 steps per 1/60 s), so a mistake in a
reference model cannot make a wrong engine look right.

    python3 tools/check_references_mujoco.py
"""

import math

import mujoco
import numpy as np

G = 9.81
DT = 1.0 / 60.0


def rk4(f, x, h, n):
    for _ in range(n):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def double_pendulum():
    m, l1, lc = 1.0, 1.0, 0.5
    inertia = (0.0025 + 0.25) / 3.0  # box rod, half extents (0.05, 0.5, 0.05)

    def f(s):
        q1, q2, w1, w2 = s
        m11 = 2 * inertia + m * (lc * lc + l1 * l1 + lc * lc + 2 * l1 * lc * math.cos(q2))
        m12 = inertia + m * (lc * lc + l1 * lc * math.cos(q2))
        m22 = inertia + m * lc * lc
        h = m * l1 * lc * math.sin(q2)
        c1, c2 = -h * (2 * w1 * w2 + w2 * w2), h * w1 * w1
        g1 = (m * lc + m * l1) * G * math.sin(q1) + m * G * lc * math.sin(q1 + q2)
        g2 = m * G * lc * math.sin(q1 + q2)
        acc = np.linalg.solve([[m11, m12], [m12, m22]], [-c1 - g1, -c2 - g2])
        return np.array([w1, w2, acc[0], acc[1]])

    xml = f"""
    <mujoco>
      <option gravity="0 -{G} 0" timestep="{DT / 200}" integrator="RK4"/>
      <worldbody>
        <body>
          <joint type="hinge" axis="0 0 1"/>
          <inertial pos="0 -0.5 0" mass="1" diaginertia="{inertia} {inertia} {inertia}"/>
          <body pos="0 -1 0">
            <joint type="hinge" axis="0 0 1"/>
            <inertial pos="0 -0.5 0" mass="1" diaginertia="{inertia} {inertia} {inertia}"/>
          </body>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = [1.0, 0.5]
    ref = np.array([1.0, 0.5, 0.0, 0.0])
    worst = 0.0
    for _ in range(90):  # 1.5 s, as in the test
        ref = rk4(f, ref, DT / 100, 100)
        mujoco.mj_step(model, data, nstep=200)
        worst = max(worst, np.max(np.abs(ref[:2] - data.qpos)))
    return worst


def cartpole():
    cart_mass, pole_mass, l = 1.0, 0.1, 0.5
    inertia = pole_mass / 3.0 * (0.0004 + l * l)  # box pole, half extents (0.02, 0.5, 0.02)

    def force(t):
        return 2.0 * math.sin(3.0 * t)

    xml = f"""
    <mujoco>
      <option gravity="0 -{G} 0" timestep="{DT / 200}" integrator="RK4"/>
      <worldbody>
        <body>
          <joint name="rail" type="slide" axis="1 0 0"/>
          <inertial pos="0 0 0" mass="{cart_mass}" diaginertia="0.1 0.1 0.1"/>
          <body>
            <joint name="hinge" type="hinge" axis="0 0 1"/>
            <inertial pos="0 0.5 0" mass="{pole_mass}" diaginertia="{inertia} {inertia} {inertia}"/>
          </body>
        </body>
      </worldbody>
      <actuator><motor joint="rail" gear="1"/></actuator>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[:] = [0.0, 0.2]
    ref = np.array([0.0, 0.2, 0.0, 0.0])
    worst = 0.0
    t = 0.0
    for _ in range(60):  # 1 s, as in the test; force held over each step
        u = force(t)

        def f(s):
            th, w = s[1], s[3]
            c, sn = math.cos(th), math.sin(th)
            a = [[cart_mass + pole_mass, -pole_mass * l * c],
                 [-pole_mass * l * c, inertia + pole_mass * l * l]]
            b = [u - pole_mass * l * sn * w * w, pole_mass * G * l * sn]
            acc = np.linalg.solve(a, b)
            return np.array([s[2], w, acc[0], acc[1]])

        ref = rk4(f, ref, DT / 100, 100)
        data.ctrl[0] = u
        mujoco.mj_step(model, data, nstep=200)
        t += DT
        worst = max(worst, np.max(np.abs(ref[:2] - data.qpos)))
    return worst


if __name__ == "__main__":
    ok = True
    for name, check in [("double pendulum", double_pendulum), ("cart-pole", cartpole)]:
        err = check()
        passed = err < 1e-4
        ok &= passed
        print(f"{name:16s} max |reference - MuJoCo| = {err:.2e}  {'ok' if passed else 'MISMATCH'}")
    raise SystemExit(0 if ok else 1)
