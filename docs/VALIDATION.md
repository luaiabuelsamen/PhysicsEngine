# Validation plan

What has to hold before any libphys output is trusted for the flagship, with
pass/fail criteria fixed in advance. Status as of this commit:

| Item | What it checks | Status |
|---|---|---|
| V0 | Against closed-form mechanics | done, passing (`tests/`) |
| V1 | Rigid contact for insertion | done, **failed**; see [DECISION.md](DECISION.md) |
| V2 | Tactile pad against TacSL | planned |
| V3 | Tactile pad against measured data | planned; one criterion already fails |

Per [DECISION.md](DECISION.md), V2 and V3 run only if the flagship adds a
vision-based tactile sensor.

## V0: analytic references (done)

`tests/test_phys.cpp`, `tests/test_tactile_sensor.cpp`,
`tools/plot_tactile_validation.py`. Covered:
- **Rigid bodies:** bounce, rolling, sliding, pendulum and double-pendulum
  periods, cart-pole against exact equations of motion (references checked
  against MuJoCo).
- **Tactile pads:** Hertz contact radius, peak pressure and indentation;
  Mindlin stick radius and shear displacement; force balance.

These only show that the code solves its own model correctly; they say
nothing about whether the model matches a real gel.

## V1: rigid contact for insertion (done, failed)

`tools/peg_hole_compare.py`, `results/peg_hole/e1_results.json`.
- **Criterion:** success within 10 points of a near-rigid MuJoCo reference
  on a matched peg-in-hole sweep, and peak interpenetration below c/4.
- **Result:** fails at every tilt above 0, because box-box edge contacts at
  the hole rim are resolved with the wrong normal.

**Precondition for any further libphys insertion work:** fix the
edge/corner contact and re-run E1 to a pass.

## V2: tactile pad against TacSL on matched setups

TacSL (NVIDIA's visuotactile simulation library, built on Isaac Gym / Isaac
Lab) is the strongest recent fast tactile simulator. Both are models, so V2
measures where they differ, not which is right. V3 decides that.

**Where it runs.** TacSL needs an x86 machine with an RTX GPU, which the
Jetson is not, so V2 needs paid cloud compute. It does not run until Luai
approves an explicit budget (job, GPU type, count, hours, dollars). libphys
runs on the Jetson.

**Matched setups.** Same indenter geometry, gel size and modulus (E fitted
so the two agree on the normal force-depth curve at 1 mm), and the same
prescribed indenter trajectory:

| Setup | Motion | Compare |
|---|---|---|
| V2a: sphere press | R = 10 mm sphere into a 20 x 20 mm pad, 0 to 1.5 mm depth | normal force vs depth; contact area vs force |
| V2b: sphere drag | At 1 N constant normal load, drag 3 mm tangentially | shear force vs drag distance; the drag distance at which full slip sets in |
| V2c: shear field | Snapshots of V2b at 25%, 50% and 90% of the full-slip shear | tangential displacement (marker) field |

**Pass/fail.** These criteria record agreement; they are not a verdict on
either model.

| Setup | Criterion |
|---|---|
| V2a | normal force within 10% over the shared range (contact area reported, no threshold) |
| V2b | full-slip onset distances within a factor of 1.5 |
| V2c | Pearson r ≥ 0.8 between displacement fields after resampling to a common grid |

**Expected outcome**, to check against: V2a agrees. V2b and V2c differ,
because a model whose contact has a single friction state slips all at
once rather than from the rim inwards.

## V3: tactile pad against measured data

Any claim that libphys predicts real sensor output needs V3. Per criterion:

| Criterion | Data | Pass if | Status |
|---|---|---|---|
| V3a: normal force vs depth exponent | Sparsh DIGIT sphere presses (324 trajectories) | model exponent within the 95% CI of the fitted exponent, after modelling rig compliance as a series spring fitted on held-out trajectories | open: measured 1.58 vs model 1.5 at small depth; rig compliance not yet separated |
| V3b: contact radius vs force exponent | Sparsh GelSight sphere images (81 trajectories, batch 1) | model's synthetic-image exponent within the measured CI | **fails**: measured 0.326 ± 0.007 (batch 1); the half-space model's images give 0.366 |
| V3c: shear / marker field under partial slip | Needs a dataset with marker fields and synchronised force / torque. Not on disk; candidate public datasets to be identified | full-slip onset distance within a factor of 1.5; marker-field r ≥ 0.8 | blocked on data |

**Stop rule.** If V3a or V3b still fail after one calibration pass
(modulus, thickness, rig spring), do not use the pad model as a predictor of
real sensor signals.

**Results trace.**
- V3a: `results/sparsh/contact_laws.txt` (`tools/sparsh_contact_laws.py`).
- V3b: `results/sparsh/contact_area_b1.txt` (`tools/sparsh_contact_area.py
  --batch 1`).
