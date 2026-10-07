# Changelog

## 2026-10 (tactile coupling and grasp RL)

### Fixed
- **Slider joints drifted along their own axis.** The lateral correction
  was computed as delta - ex (ex . delta) with a float axis that is unit only
  to ~1e-7, leaving an axial push of ~1e-7 times the slider's travel every
  iteration. This biased force balances along slider axes by up to 5x
  (5 kg carriage, 8 iterations). The axial part is now removed exactly in
  the joint frame.
- **Sliding friction exceeded the friction cone.** Dynamic friction was
  capped by mu times the position solve's normal impulse alone; when
  restitution removed part of that impulse (contacts pushed through joints),
  sliding friction exceeded mu N by up to 20%. It is now capped by the net
  normal impulse.
- **Coupled pads could not lift an object off a table.** The table
  contact's restitution step cancelled the gel's shear impulse; the shear is
  now applied last in the velocity solve.

### Corrected claims
- The README previously said results were checked against real GelSight /
  DIGIT data. Only analytic and MuJoCo references are checked; the Sparsh
  analysis fits contact laws and is not a validation of the simulator.
- The fragile-grasp result was first reported for one seed (75% for the
  stick-fraction policy); over 3 seeds it is 59% (25-75%).
- Benchmark throughputs were re-measured on 2026-10-05; earlier figures were
  taken under different thermal / clock conditions.
