Goal: Decide, with evidence, whether libphys should be the contact simulator for the contact-insertion flagship (rigid peg/hole, friction, jamming) or be stopped.
Finish line: docs/DECISION.md (continue/stop recommendation with evidence) + docs/VALIDATION.md (TacSL / measured-data comparison plan with pass/fail) + honest README scope; all numbers traced to committed results files.
Deadline: 2026-10-13 (alongside flagship M0)
Milestone: L1 done: README scope fixed, VALIDATION.md written, DECISION.md recommends STOP (use MuJoCo with checked contact settings; freeze libphys)
Last result: README headline: gel-image policy 67% (65-68%) vs pad-force policy 36% (36-36%) held-out grasp success, 3 seeds -> results/grasp/eval.json. E1 peg-in-hole: libphys success 12-94% vs near-rigid MuJoCo 100% at tilt >= 1 deg (unphysical toppling at the rim); E1 criteria (b) and (c) fail -> results/peg_hole/e1_results.json, results/peg_hole/topple_trace.txt
Blocked: needs Luai decision: accept STOP recommendation in docs/DECISION.md (freeze libphys) or ask for the box-box edge fix + E1 re-run.
Pushed (2026-10-06, plain git push to origin/main, approved by Luai): 7eb169e..18ecff7 = 2ba94e3, c8beb04, 8657481, a81657f, 3aba6b4, 18ecff7 (README claim + hero GIF + headline 67% vs 36% from results/grasp/eval.json + figure; DECISION.md; VALIDATION.md; results/).
Paid compute: none launched from this repo (all work so far ran on the Jetson). No Modal or cloud jobs without an approved budget; V2 (TacSL on an x86 RTX GPU) would need one and is not requested while the STOP recommendation stands.
Next: if STOP accepted: pass the MuJoCo contact-settings finding (docs/DECISION.md, "What the flagship should take from this") to the flagship M1 and freeze this repo. No new libphys features until then.
Updated: 2026-10-06 22:40
