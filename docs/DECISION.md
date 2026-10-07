# Decision: libphys as the contact simulator for peg insertion

**Question.** For the contact-insertion flagship (rigid peg and hole,
friction, jamming, 2 mm to 0.5 mm clearance, SO-101 and Franka), does
libphys give anything that MuJoCo does not?

**Recommendation: stop.**
- Use MuJoCo for the flagship, with contact settings checked against the
  penetration test below.
- Freeze libphys. Revisit only if the flagship adds a vision-based tactile
  sensor, and then only after the tactile validation in
  [VALIDATION.md](VALIDATION.md) passes.

## Evidence

### E1: matched peg-in-hole, libphys vs MuJoCo 3.8

The pass/fail criteria were written before the run (STATUS.md, commit
`8657481`). Script: `tools/peg_hole_compare.py`. Results:
`results/peg_hole/e1_results.json`.

**Setup.**
- A square box peg (20 x 20 x 60 mm, 0.1 kg) is pushed by a 30 m/s² body
  load into a square hole made of four box walls, 30 mm deep.
- Friction is 0.3 everywhere.
- 50 initial poses per cell: lateral offset |dx| up to the clearance, fixed
  tilt per cell, random tilt direction.
- Simulators compared:
  - libphys at a 2 ms step with 10 substeps;
  - MuJoCo with its default contact settings;
  - MuJoCo with stiffer contacts (`solref 0.004`);
  - a near-rigid MuJoCo reference: 0.2 ms step, `solref 0.0004`,
    `solimp 0.99 0.999 0.0001`, elliptic cones, 30 no-slip iterations.

**Metrics.**
- Success: the peg bottom reaches 90% of the depth within 1 s.
- Peak interpenetration: deepest peg corner inside a wall or the floor
  during the run. This includes landing on the hole floor at about 1.3 m/s.

Each cell shows success / peak interpenetration (mm):

| clearance per side | tilt | libphys | MuJoCo near-rigid | MuJoCo stiff | MuJoCo default |
|---|---|---|---|---|---|
| 2 mm | 0 deg | 100% / 0.00 | 100% / 0.00 | 100% / 0.38 | 100% / 9.44 |
| 2 mm | 1 deg | 94% / 0.00 | 100% / 0.04 | 100% / 1.79 | 100% / 9.59 |
| 2 mm | 2 deg | 82% / 0.00 | 100% / 0.01 | 96% / 0.51 | 100% / 9.44 |
| 2 mm | 4 deg | 52% / 0.00 | 100% / 0.13 | 88% / 2.05 | 100% / 9.44 |
| 1 mm | 0 deg | 100% / 0.00 | 100% / 0.00 | 100% / 0.38 | 100% / 9.44 |
| 1 mm | 1 deg | 86% / 0.00 | 100% / 0.02 | 100% / 0.94 | 100% / 9.44 |
| 1 mm | 2 deg | 62% / 0.00 | 100% / 0.16 | 100% / 3.04 | 100% / 9.48 |
| 1 mm | 4 deg | 12% / 0.00 | 100% / 0.28 | 100% / 2.75 | 100% / 8.40 |
| 0.5 mm | 0 deg | 100% / 0.00 | 100% / 0.00 | 100% / 0.38 | 100% / 9.44 |
| 0.5 mm | 1 deg | 64% / 0.00 | 100% / 0.17 | 100% / 1.56 | 100% / 9.44 |
| 0.5 mm | 2 deg | 32% / 0.00 | 100% / 0.23 | 100% / 3.46 | 100% / 8.78 |
| 0.5 mm | 4 deg | 12% / 0.00 | 100% / 0.12 | 100% / 2.28 | 100% / 8.08 |
| 0.2 mm | 0 deg | 100% / 0.00 | 100% / 0.00 | 100% / 0.38 | 100% / 9.44 |
| 0.2 mm | 1 deg | 12% / 0.00 | 100% / 0.10 | 100% / 3.39 | 100% / 8.41 |
| 0.2 mm | 2 deg | 22% / 0.00 | 100% / 0.17 | 100% / 2.69 | 100% / 6.50 |
| 0.2 mm | 4 deg | 22% / 0.00 | 100% / 0.14 | 100% / 2.08 | 100% / 4.91 |

**Throughput** (env-steps per wall-clock second):

| Simulator | Env-steps/s | Simulated env-seconds per wall second |
|---|---|---|
| libphys (200 envs batched on the GPU, 2 ms steps) | 20k-25k | about 45 |
| MuJoCo (one CPU core, 2 ms steps) | 70k-74k | about 140 |
| MuJoCo near-rigid (0.2 ms steps) | 72k-80k | about 15 |

**Where libphys fails.** In every tilted case libphys loses pegs, while the
near-rigid reference inserts all of them.
- The failures are not jams. The pegs topple outward and come to rest lying
  on a wall top: `results/peg_hole/topple_trace.txt`.
- That trace (0.5 mm clearance, 2° tilt) starts with the peg's centre of
  mass over the hole and its low corner inside it. The peg then rotates
  away from the hole and travels 21 mm sideways onto the wall top.
- Gravity acting on a centre of mass over the hole cannot produce that
  motion. libphys's box-box contact is resolving the corner-on-rim contact
  with the wrong normal: it lifts the peg over the edge. That is the
  contact that decides insertion.

**Pass / fail against the pre-registered criteria.**

| Criterion | Result |
|---|---|
| (a) libphys interpenetration < c/4 at every clearance | **pass** (0.00 mm) |
| (b) success within 10 points of MuJoCo, or closer to the quasi-static expectation where they differ | **fail**: 6-88 points below the near-rigid reference at every tilt above 0, and the difference is unphysical toppling |
| (c) something the flagship needs that MuJoCo lacks | **fail**: lower throughput at this batch size, and no needed capability (table below) |

### Capabilities the flagship needs

| Need (flagship M1-M4) | MuJoCo 3.8 | libphys |
|---|---|---|
| SO-101 and Franka models (MuJoCo Menagerie), with mesh collision | yes | no: URDF import only, no collision geometry, no meshes |
| Cylindrical peg, chamfered hole | yes: cylinder primitive, convex meshes, SDF plugins | no: spheres, capsules, boxes, planes only |
| Correct rigid edge/corner contact at the hole rim | yes (near-rigid settings, E1) | no (E1) |
| Torsional / rolling friction | yes (condim 4/6) | no |
| Force-torque and touch sensors | yes | pad forces only, on tactile pads |
| Impedance control, mocap / weld targets | yes | position, velocity and torque joint drives only |
| Batched GPU simulation for RL | MJX / MuJoCo Warp (not installed here) | yes |
| Partial-slip tactile pads (Hertz / Mindlin) | no | yes, validated against closed-form solutions only |

The one capability unique to libphys is the tactile pad, and the flagship's
contact signal is tracking error, not touch.

### Making libphys usable for insertion

Missing pieces, in order:
1. Correct box-box edge and corner contact.
2. Cylinders and convex meshes, then SDF collision for the hole.
3. Collision geometry from URDF for the robot links.
4. Torsional friction.
5. A Franka / SO-101 model with working contacts.

That is weeks of work to reach what MuJoCo already does, before any
flagship experiment.

## What the flagship should take from this

MuJoCo's default contact settings are not usable at tight clearance:
- **Defaults:** the peg sinks up to 9.4 mm into the floor on landing.
- **`solref 0.004`:** it still passes 1-3.5 mm into the walls, more than
  the clearance itself.

Both report 100% insertion at 0.2 mm clearance whatever the tilt. A
tight-tolerance result in MuJoCo therefore means nothing unless the
contact settings are checked.

Even the near-rigid settings (0.2 ms step, `solref 0.0004`,
`solimp 0.99 0.999 0.0001`, elliptic cones, no-slip iterations) do not keep
peak interpenetration below c/4 at 1 mm clearance or tighter:

| Clearance per side | c/4 | Near-rigid peak interpenetration |
|---|---|---|
| 1 mm | 0.25 mm | up to 0.28 mm |
| 0.5 mm | 0.125 mm | up to 0.23 mm |
| 0.2 mm | 0.05 mm | up to 0.17 mm |

They meet it only at 2 mm. So at 0.5 mm the reference may also make
insertion slightly easier than reality. Recommended M1 checks for the
flagship:
- Report peak wall interpenetration per tolerance.
- Tighten the step or contact settings until it is below c/4. If that is not
  affordable, state the margin next to every tight-tolerance number.

## Revisit conditions

- **Tactile sensing.** Reconsider libphys only if the flagship adds a
  vision-based tactile sensor, *and* the tactile pad passes V2 and V3 in
  [VALIDATION.md](VALIDATION.md): agreement with TacSL on matched setups and
  with measured data.
- **Even then, as an add-on.** The likely form would be the pad model as a
  sensor on a MuJoCo scene, not libphys as the simulator.
