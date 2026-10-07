"""Trace one libphys peg-in-hole drop (clearance 0.5 mm, tilt 2 deg, no lateral
offset): peg position, tilt and contact count every 8 ms. Used in
docs/DECISION.md to show where libphys's box-box contact fails.

    PYTHONPATH=python:$PYTHONPATH python3 tools/peg_hole_trace.py > results/peg_hole/topple_trace.txt
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import peg_hole_compare as E  # noqa: E402
import libphys as lp  # noqa: E402

c, th = 0.0005, math.radians(2.0)
m = lp.Model(gravity=(0, 0, -E.PUSH), substeps=10)
floor = m.add_body(lp.Body.plane(), friction=E.MU, restitution=0.0)
wb = E.walls(c)
wid = [m.add_body(lp.Body.box(h, 0.0), friction=E.MU, restitution=0.0) for _, h in wb]
peg = m.add_body(lp.Body.box((E.A, E.A, E.HL), E.MASS), friction=E.MU, restitution=0.0)
w = lp.World(m, num_envs=1, device="cpu")
s = w.state
q = math.sqrt(0.5)
s.qw[:, floor] = q
s.qx[:, floor] = q
s.pz[:, floor] = -E.DEPTH
for b, (cc, _) in zip(wid, wb):
    s.px[:, b], s.py[:, b], s.pz[:, b] = cc
s.pz[:, peg] = 0.001 + E.HL * math.cos(th) + E.A * math.sin(th)
s.qw[:, peg] = math.cos(th / 2)
s.qy[:, peg] = math.sin(th / 2)
print("# clearance 0.5 mm per side, hole half width 10.5 mm, initial tilt 2 deg about y, peg centre over the hole")
print("# t_ms  x_mm  z_mm  tilt_deg  contacts")
for k in range(60):
    w.step(E.DT)
    if k % 4 == 0:
        tilt = math.degrees(2 * math.atan2(s.qy[0, peg].item(), s.qw[0, peg].item()))
        print(f"{k * 2:4d} {s.px[0, peg].item() * 1e3:7.2f} {s.pz[0, peg].item() * 1e3:7.2f} {tilt:7.2f} "
              f"{w.contact_counts()[0]}")
