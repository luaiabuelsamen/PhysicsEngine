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

## Reproducing

```bash
cmake -B build && cmake --build build -j
python3 tools/sparsh_contact_laws.py --data data/sparsh
PHYS_TACTILE_LIB=build/libphys_tactile.so python3 tools/sparsh_validate.py --sensor gelsight --probe sphere
```

## Open questions

- Validate shear at the gel itself: predicted contact area and stick / slip
  regions against the tactile images in the Sparsh datasets, or marker
  displacement fields on marker-based gels.
- All fits push the probe radius to the top of its range (30 mm): the probe
  geometry is unknown, and the data may also reflect gel thickness effects
  that a half-space ignores.
- Coupling `ElasticPatch` into `World` as a fingertip skin, on the GPU, at RL
  throughput (the half-space solve is O(n^2) per contact patch).
- Taxel-array skins (e.g. XELA uSkin), e.g. with the UniTac-NV dataset.
