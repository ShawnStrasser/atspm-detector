# 115dx Fix of the three weak spots found by 115dv (2026-10-07)
Package `dc_work/final_v7_prod` (7.0.0). Backup of the pre-fix tree (src, dist, README, pyproject):
`dc_work/final_v7_prod_bak115dv`. CPU, 4 threads. Scripts `research/code/final115dv/f6dead.py`, `permfix.py`.

| weak spot (115dv) | verdict | change |
|---|---|---|
| 3. healthy busy detector called "bad, choppy 21-130x" when its phase-mates die mid-sample | real defect (health only) | fixed |
| 1. DataFrame without start / end: window = first to last event | not a defect (no other window is known) | documented |
| 2. a 90-min hole in a 3-h sample still gives minutes_of_data 180 | not a defect (window length; the health text already says "in 1.6 h") | documented |

**Fix (health_core.py).** A detector whose silent-run statistic reaches the dropout suspect level
(drop_lam >= 30 expected ONs, already computed) is no longer used as a yardstick by the choppy check
(shape_stats reference and the spike reference). With < 2 phase-mates left the existing fall-back to the rest
of the signal applies. Nothing else changes; phase / function / lanes / setback / speed never read health.
Docs: README + pipeline docstring now say both of the points in rows 1 and 2.

**Evidence**
- Stress (f6dead.py, n07 3 h, n09 24 h; phase-mates cut to their first k = 6-100 ONs): busy det was bad 21-130x
  (n07) / 30-46x (n09) in v7 and ref114; now ok at every k. The dead phase-mates are still bad. Phase-mates
  that are only quiet (k ONs spread over the window) were and are ok.
- Parity vs ref114, 115dv set (30 cases, 606 detectors), both profiles: every column identical.
- Parity vs ref114, note-115 set (132 cases, 2,627 detectors), both profiles: phase / function / lanes / setback /
  speed / probabilities as before (v7 = v7f exactly; the known ref114 prob noise <= 1.3e-4 on 3 detectors is
  unchanged). Health changes on 7 detectors in 4 cases, each on a phase with a detector that went silent
  (night miss or stuck):
  - s25 24 h det 25: P8 Advance: bad -> suspect. The choppy 13x call is gone; only the shared stuck-on event remains.
  - s25 24 h det 26: P8 Presence: still bad. Choppy 8x -> 9x.
  - s32 24 h det 26: P8 Presence and det 48: P8 Other: still bad (chatter); the spike periods are no longer listed.
  - b_typical_r8 3 h det 7: P7 Advance: choppy 16x -> 15x.
  - b_typical_r8 24 h det 23 and det 25: P8 Advance: choppy 6x -> 8x and 10x -> 9x.
- Channel + phase renumbering (permfix.py; s25, s32, r8 3 h / 24 h; 2 seeds each): every column follows. PASS.
- check from src: 6/6. Wheel rebuilt (same 76 entries). Fresh-venv smoke: 71 files = src, CLI both profiles, check 6/6.
- 7 days n08 from the new wheel: = ref114 in every column; peak 886 / 861 MB (full / le2h).

**Not changed:** the version stays 7.0.0 (never shipped). The old wheel is in the backup.
