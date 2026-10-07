Goal: Decide, with evidence, whether libphys should be the contact simulator for the contact-insertion flagship (rigid peg/hole, friction, jamming) or be stopped.
Finish line: docs/DECISION.md (continue/stop recommendation with evidence) + docs/VALIDATION.md (TacSL / measured-data comparison plan with pass/fail) + honest README scope; all numbers traced to committed results files.
Deadline: 2026-10-13 (alongside flagship M0)
Milestone: L1 honesty fix (done) + validation plan + decision memo — in progress
Last result: README scope fixed (commit c8beb04); no experiment results yet
Blocked: no
Next: experiment E1 (peg-in-hole, libphys vs MuJoCo 3.8), then docs/VALIDATION.md and docs/DECISION.md
Experiment E1 (pre-registered): square box peg (20x20x60 mm, 0.1 kg, mu 0.3) pushed into a hole of 4 box walls (depth 30 mm), clearance per side c in {2, 1, 0.5, 0.2} mm, initial tilt in {0, 1, 2, 4} deg, random lateral offset |dx| <= c, 50 initial poses per cell, identical in both sims. Metrics: success (bottom reaches 90% depth in 1 s), jam (no success, peg speed < 1 mm/s at end), peak corner interpenetration, env-steps/s.
E1 pass/fail for "libphys adds something for insertion": libphys must (a) keep peak interpenetration < c/4 at every clearance, (b) agree with MuJoCo's success rate within 10 points per cell or be closer to a quasi-static expectation where they differ, AND (c) offer something the flagship needs that MuJoCo does not (>=5x throughput at matched behaviour at the flagship's batch size, or a needed capability). If (c) fails -> recommend stop.
Updated: 2026-10-06 21:20
