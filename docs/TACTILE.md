# Tactile contact: models and validation against real sensors

**Status: research in progress.** `phys::ElasticPatch` is validated against
contact-mechanics theory and is being compared with real sensor data. It is
not yet coupled into `World` (the rigid solver's contacts are still rigid).

## Why

Tactile RL policies are trained in simulators whose contact models were
built for speed rather than fidelity. Recent work reports the consequences:
simulated shear from Isaac Lab's contact sensors was too unreliable to use on
an Allegro hand with XELA skins (*Beyond Binary*, 2026), and some groups now
avoid tactile simulation altogether. The goal here is a tactile contact model
whose behaviour is *measured* against real sensors, then made fast enough for
batched RL.

## Data

Meta's [Sparsh](https://sparsh-ssl.github.io/) force datasets
([DIGIT](https://huggingface.co/datasets/facebook/digit-force-estimation),
[GelSight Mini](https://huggingface.co/datasets/facebook/gelsight-force-estimation),
CC BY-NC 4.0): a robot presses a probe (hemisphere, flat or sharp) into the
gel, then slides it a few millimetres, while an ATI Nano17 records 3-axis
force. About 2,800 trajectories with probe poses. Only the small
`dataset_slip_forces.pkl` files are needed (137 MB in total); put them in
`data/sparsh/` named `<sensor>_<probe>_batch_<n>.pkl`.

## Model

`src/phys/tactile.h`. The gel is a linear elastic half-space sampled on a
grid of square cells; cells are coupled through the Boussinesq (normal) and
angle-averaged Cerruti (tangential) kernels. Pressure comes from projected
Gauss-Seidel with p >= 0; shear is path dependent (Kalker): each cell keeps a
slip state, sticks while its traction is inside the friction cone, and slides
at |q| = mu p otherwise. Options: a Winkler mode (independent springs - the
hydroelastic / brush family) as a baseline, a tangential-compliance scale, and
a holder spring in series (robot, mount and force-sensor compliance).

Checked against exact solutions (`tests/test_tactile.cpp`):

| Check | Error |
|---|---|
| Hertz force, sphere on half-space | 0.3% |
| Hertz contact radius / pressure profile | 0.2% / 2% |
| Flat-punch force (Boussinesq) | 0.07% |
| Cattaneo-Mindlin load-displacement / stick radius | 0.17% / 0.8% |
| Masing (Mindlin-Deresiewicz) hysteresis on reversal | 0.1% |
| Winkler sphere force / brush shear curve | 1.3% / 0.2% |
| Holder in series with a Cattaneo-Mindlin contact | 0.4% |

## Tactile sensors in World

`src/phys/tactile_sensor.h` (the contact solve) and the "tactile pads"
section of `src/phys/rigid.h` (the coupling); configured through
`phys::TactileSensorDesc` / `Model.add_tactile_sensor`. These are the same
half-space gel as the standalone patch, as pads on rigid bodies, solved for
every env on the GPU after every step.

**Two-way coupling** (`coupled = true`, the default). Contacts on a pad's
face take the gel's compliance inside the rigid solve:

- **Normal.** A pad contact is a compliant XPBD constraint. Its stiffness is
  the gel's secant stiffness W / delta from the previous step's tactile
  solve (delta: the deflection under the deepest point), shared among the
  pad's contacts. Before the first solve it is the stiffness of a circular
  contact a quarter of the pad wide. An object therefore sinks into the pad
  as far as the gel model says, at most half the gel's `thickness` (a
  sharp corner is a point load on a linear half-space, which would sink
  without limit; a real gel bottoms out on its backing). Restitution is
  replaced by critical damping
  of the gel's normal motion, which never makes the contact pull.
- **Tangential.** Rigid static and dynamic friction are replaced by a
  persistent shear deflection u of the gel, driven by the object's motion
  relative to the pad. Its force follows Mindlin's law
  Q = mu W (1 - (1 - |u| / u*)^(3/2)), with u* = 3 mu W / (2 k_t). Here
  k_t = 2 E* sqrt(A / pi) / tangential_ratio is the initial tangential
  stiffness of a contact of area A, taken from the last tactile solve. The
  deflection is critically damped while the contact sticks; beyond u* the
  pad slides at mu W. The shear is applied last in the velocity solve.
  Otherwise the restitution step of an object's other contacts (a ball
  resting on a table) zeroes the velocity the gel gives it, and the object
  can never be lifted off.
- **Coverage.** The gel is taken to cover the face around the pad: contacts
  up to one pad size outside it are coupled too. Otherwise an indented
  contact sliding off the sensing area would turn rigid and be ejected.

The tactile solve then works as follows:

1. **The pad's load** is the contact force on its face, averaged over the
   step's substeps, including the gel's shear.
2. **The gap field** comes from rays cast along the pad normal, from every
   cell, against the shapes of the bodies touching the pad.
3. **Pressure.** The solve finds P >= 0 with sum P = W such that the deformed
   gap h + K P equals the approach over the contact and is no smaller
   elsewhere (Polonsky & Keer 1999, conjugate gradient). It is warm-started
   from the last step and converges in a few iterations once contacts are
   steady.
4. **Shear: q = mu (P - P*) Q / |Q|.** P* solves the same problem at the
   reduced load W - |Q| / mu (Ciavarella 1998; Jaeger 1998). Cells where
   P* > 0 stick. This is exact for monotonic shear here, because the normal
   and the angle-averaged tangential kernels are proportional; it is the
   same Mindlin theory the coupling uses. Within 2% of the friction cone the
   whole contact slides.

Checked in `tests/test_tactile_sensor.cpp`. A pad on a box is pressed onto a
fixed sphere by slider forces, then sheared:

| Check | Result |
|---|---|
| Pad force vs applied load (0.5, 1, 2 N) | 0.02% |
| Hertz contact radius / peak pressure at 1 N | 0.5% / 1% |
| The pad body's indentation into the sphere vs Hertz (1, 2 N) | 0.6%, 0.0% |
| The pad's shear displacement vs Mindlin, Q / mu W = 0.3, 0.6 | < 0.5% |
| Mindlin stick radius at Q / mu W = 0.3, 0.6, 0.8 | within one cell (0.375 mm) |
| Full slip: every cell slides, abs(q) proportional to p, Q = mu W | exact |
| Step-to-step scatter of the pad force at 10 substeps, under shear | < 1e-5 N |
| Two pads lift a ball off a table: grip 3 N holds, 1 N (< m g / 2 mu) slips out | yes |
| A corner pressed at 3 N: force, indentation limit | exact; 2.00 mm |
| Flat pad on a floor: load, full contact, edge pressure rise | 0.01%; 400 / 400 cells; 3.9x the centre |
| CPU vs CUDA | bit-identical |

**Not modelled yet.**

- Shear hysteresis: partial-slip unloading follows the loading curve back
  (no Masing rule).
- A pad's single rigid shift is shared by everything touching it; separate
  objects on one pad shear together.
- Pads larger than about 32 x 32 cells are slow (dense influence sums).
- Float32 positions round the solver's corrections, leaving about 0.03%
  in the force balances.

## Findings so far

**1. Real gels follow elastic half-space laws when pressed, not Winkler ones**
(`tools/sparsh_contact_laws.py`, model-free fits with a fitted contact onset):

| Probe | Trajectories | Fitted exponent n (F ~ d^n) | Winkler (n=2) error | Hertz (n=1.5) error |
|---|---|---|---|---|
| DIGIT, sphere | 324 | 1.58 | 1.33% | 0.66% |
| GelSight Mini, sphere | 718 | 1.36 | 2.31% | 1.25% |
| DIGIT, sharp | 307 | 1.40 | 1.99% | 0.68% |

**2. Full sliding takes 0.8-1.1 mm of probe travel to develop** (DIGIT, all
probes), and the transition is closer in shape to Cattaneo-Mindlin (fitted
exponent 1.3-1.55) than to the brush model (2). But see finding 3: most of
that travel turns out to be compliance of the test rig, not of the gel.

**3. Slide-phase force traces cannot tell the contact models apart; rig
compliance dominates** (`tools/sparsh_validate.py`, GelSight sphere, 30 train /
150 held-out trajectories; per-trajectory contact onset fitted on the press
phase only; metric: RMS error of shear / normal during the slide):

| Run | Elastic | Winkler | Rigid Coulomb |
|---|---|---|---|
| Elastic: radius only; Winkler: radius + shear ratio; Coulomb: nothing | 0.153 | **0.101** | 0.244 |
| Elastic and Winkler: radius + one shear-compliance knob | 0.123 | **0.101** | 0.244 |
| All three + a fitted holder spring in series | 0.108 | **0.102** | 0.122 |

In the last run the elastic model's own tangential knob fits to 1.0 (a pure
half-space) with a holder of 931 N/m; Winkler picks an effectively rigid
holder; rigid Coulomb picks 444 N/m and then has the lowest raw shear-force
error of the three (10.5% vs 16.9-18.3% of peak). In other words, once the
robot / mount / force-sensor compliance is modelled, a rigid contact explains
these traces about as well as either compliant one: the recorded probe pose
and a wrist force sensor do not see the gel's partial slip clearly. An
earlier version of this note read the first row as "rigid engines get shear
wrong" - that compared against a rigid model without a holder spring, which
was not a fair baseline.

What survives: the press-phase contact law (finding 1). What this data cannot
settle: the shear behaviour at the contact itself. That needs measurements
taken *at* the gel - the tactile images (contact area, and the slip region on
marker-based gels), or a sensor with a stiff, known mount.

**4. Measured at the gel, the contact radius grows between the two models'
predictions** (`tools/sparsh_contact_area.py`, GelSight Mini, hemisphere
probe). For a sphere the contact radius scales as a ~ F^beta with beta = 1/3
on an elastic half-space and 1/4 for independent springs; the exponent does
not depend on probe size, stiffness or rig compliance. The images show the
imprint as a ring (the camera sees surface slope, which peaks at the contact
edge for both models), so the ring's radius is measured in every frame of
the press and fitted against the measured force.

The rings are small (7 -> 12 px), and blur biases the measured exponent
upward, so the tool also runs the identical measurement on synthetic images
rendered from each model's exact surface shape at the same scale, contrast
and noise:

| | Batch 1 (81 trajectories) | Batch 2 (82 trajectories) |
|---|---|---|
| Real data | **0.326 +- 0.007** | **0.315 +- 0.010** |
| Hertz images, same measurement | 0.366 | 0.366 |
| Winkler images, same measurement | 0.297 | 0.298 |

The gel sits between a semi-infinite elastic solid and independent springs,
somewhat closer to the springs; both pure models are off by several standard
errors.

**5. The gel's finite thickness does not explain finding 4: these contacts
are too small.** A gel layer of thickness h bonded to a rigid backing does
lower the radius exponent below 1/3, but only once the contact radius is
comparable to h. We solved the exact linear-elastic layer problem (bonded
and frictionless backing, checked against the published series of Garcia &
Garcia 2018 and Dimitriadis et al. 2002 to 0.1-0.6%):

| a / h | bonded, nu = 0.5: n / beta | frictionless backing: n / beta |
|---|---|---|
| 0.3 | 1.71 / 0.327 | 1.63 / 0.331 |
| 0.5 | 1.85 / 0.315 | 1.71 / 0.325 |
| 1.0 | 2.16 / 0.275 | 1.88 / 0.299 |

(n: F ~ d^n, beta: a ~ F^beta, local exponents.) The GelSight Mini gel is
4.25 mm thick (datasheet). The rings in finding 4 end at about 12 px.

The 320 x 240 images cover at most the sensor's 18.6 x 14.3 mm field of view,
which bounds the scale at >= 16.8 px/mm. Tracking the imprint while the probe
slides gives 12.3 px/mm (median of 137 slides, IQR 11.1-13.8). That figure
is probably low, because the gel recovers slowly behind the probe and its
trail drags the imprint centroid back.

Either way the contact radius stays below 1 mm, so a/h <= 0.23. At that
ratio a layer is indistinguishable from a half-space (beta >= 0.327,
n <= 1.71). The layer also moves the two exponents the wrong way relative to
each other: whenever it lowers beta it raises n above 1.5, while the press
fits give n = 1.36. No linear-elastic layer gives n < 1.5 for a sphere.

Consequences:

- For probe-sized contacts (a ~ 1 mm on a 4 mm gel) the half-space kernel
  in `ElasticPatch` is the right one. A layer kernel matters for large
  contacts (flat or broad objects, a/h ~ 1), so it is worth having as an
  option, but this data cannot validate it.
- The low press exponent (n = 1.36 < 1.5) is a property of the depth axis,
  not of the gel. The candidates are compliance of the robot, mount and
  force sensor in series (which fits finding 3), a force offset / preload
  traded off against the fitted onset, and gel viscoelasticity. The presses
  here are light (about 0.15 -> 0.3 N, starting from a preload).
- What remains of finding 4 (measured 0.32 vs 0.37 for synthetic Hertz
  images) comes from outside linear elasticity and the image model. The
  candidates are adhesion (silicone is tacky, and at sub-newton loads JKR
  contact grows more slowly than Hertz) and a nonlinear or saturating
  photometric response, which the synthetic images (signal proportional to
  slope) do not include.

## Reproducing

```bash
cmake -B build && cmake --build build -j
python3 tools/sparsh_contact_laws.py --data data/sparsh
PHYS_TACTILE_LIB=build/libphys_tactile.so python3 tools/sparsh_validate.py --sensor gelsight --probe sphere
# needs a batch's tactile images (dataset_gelsight_0*.pkl, ~230 MB) in data/sparsh/images
python3 tools/sparsh_contact_area.py --batch 1
```

## Open questions

- Adhesion (JKR) versus photometric nonlinearity as the cause of the
  remaining contact-radius gap. Fitting F against a^3 (Hertz) versus
  a^3 - c a^1.5 (JKR) needs radii corrected for blur.
- A layer kernel for large contacts, validated on data with a/h ~ 1 (flat
  probes, or a thin gel).
- Verify the image model (signal proportional to surface slope) with a
  photometric model of the sensor, or with depth reconstructions.
- Shear at the gel itself: stick / slip regions from marker displacement
  fields on marker-based gels.
- Compare the coupled stick-to-slip transition with real slide data, once
  the rig compliance is separated out (finding 3).
- All fits push the probe radius to the top of its range (30 mm): the probe
  geometry is unknown, and the data may also reflect gel thickness effects
  that a half-space ignores.
- Coupling `ElasticPatch` into `World` as a fingertip skin, on the GPU, at RL
  throughput (the half-space solve is O(n^2) per contact patch).
- Taxel-array skins (e.g. XELA uSkin), e.g. with the UniTac-NV dataset.
