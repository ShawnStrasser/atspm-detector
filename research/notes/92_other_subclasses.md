# 92 — Other split into non-ATSPM subclasses (training labels only) on the v4f function recipe (2026-10-05)
Brief (orchestrator): user rule, non-ATSPM labels may be added, ATSPM labels never touched. Split Other by print/config
subtype into subclasses with >= ~50 trained detectors; retrain v4f's function chain; subclasses output as themselves but
score as non-ATSPM. Code `research/code/final92/sub92.py` (census / repro / fit / stack / score); work `%DC_WORK%/x92/`
(census.json, trees/, oof/sub92.json, ok92.npz). CPU 2 x 3 threads, GPU / model/ / v4f untouched, locked_v2 absent.
## Classes (pre-registered map; detectors whose v4o TRAINING label is Other; >= 50 trained detectors, decided before scoring)
| subclass | print subtypes | trained det. / signals / rows |
|---|---|---|
| Upstream_presence | advance_presence, eta, radar_advance_presence, advance_presence_long/_trucks, advance_long_presence | 405 / 135 / 11,306 |
| Setback_presence | presence_20_75, setback, setback_presence, presence_setback, presence_setback_75 | 274 / 85 / 6,685 |
| Long_zone | long_zone, long_presence_20_75ft / _0_75, radar_long_presence, presence_setback_long, long_phase(_call)_zone | 206 / 54 / 4,177 |
| Departure | departure, count_loop_past_stopbar | 51 / 17 / 1,238 |
| (left in Other) Superseded_loop | superseded_by_radar, old_call_loop, abandoned_loop | 37 / 13 -> too few |
Other keeps 33,874 training rows ('explained' whole-intersection channels 1,055 detectors, rare subtypes). 11 classes.
## Recipe (= note 90 v4f, only the class set changes)
* Trees 229 features arm c, 3 seeds x 6 folds, v4o labels, K = 11. Harness: base seed-0 refit == f76/function_c_v4o, |dP| 0.0.
* Stacker = mean3 note-86/90 recipe (x86_siba4l filtered 3-seed mean, PRM0 150 rounds, seeds 0/1/2, six-fold OOF): the 47
  columns built on the trees' 7-class view (subclasses summed into Other) + the 4 tree subclass probabilities; target =
  v4l truth with Other -> subclass; K = 11. Siba net unchanged (7 classes).
* Decode, gate .9 + D lanes + pick + twin as v4f. kway (PRIMARY, pre-registered): every subclass a full non-ATSPM class,
  like Mid / Bike. sum (secondary): subclasses summed into Other for the decode, a detector decoded Other takes its best
  Other subclass. Control shuf: the subclass labels permuted over all Other detectors (detector level, seed 92), trees AND
  stacker. v4f = note 90 P_v4f, reproduced .9197 E / .9291 R.
## Result (six-fold OOF, v4l truth, ATSPM-only stack-aware score, paired signal bootstrap, pt [95 % CI])
| vs v4f | >= 30 min E (187,444) | >= 30 min R (182,828) | 10 min E | 5 min E |
|---|---|---|---|---|
| sub kway (primary) | .9190, -0.07 [-0.18,+0.04] | .9283, -0.08 [-0.19,+0.04] | -0.17 [-0.33,+0.01] | -0.17 [-0.36,+0.01] |
| sub sum | .9193, -0.04 [-0.13,+0.06] | .9287, -0.04 [-0.13,+0.05] | -0.09 [-0.23,+0.07] | -0.10 [-0.28,+0.08] |
| shuf kway (control) | .9183, -0.13 [-0.23,-0.03] | .9277, -0.14 [-0.25,-0.03] | -0.30 [-0.48,-0.13] | -0.20 [-0.39,-0.02] |
| shuf sum (control) | .9188, -0.09 [-0.15,-0.03] | .9283, -0.08 [-0.14,-0.02] | -0.11 | -0.06 |
sub kway - shuf kway >= 30 E +0.06 [-0.04,+0.17]. By class >= 30 E, sub kway: Advance +0.07, Presence +0.14 (n.s.), Count -0.12,
Yellow_Red -0.40 [-0.87,-0.01], non-ATSPM -0.59 [-1.05,-0.14]; sub sum: non-ATSPM +0.25 [-0.12,+0.62], YR -0.41 [-0.88,-0.02]
(YR loses in the control too: a class-set refit effect, not the subclasses). Folds (kway, E) -0.28..+0.07.
## Definitional bucket (notes 70 / 78 subtypes, Other truth; >= 30 min, 13,320 rows / 761 det.; error = called ATSPM)
| | v4f | sub kway | sub sum | shuf kway | shuf sum |
|---|---|---|---|---|---|
| error rows (rate) | 3,120 (23.4 %) | 3,198 (24.0 %) | 3,040 (22.8 %) | 3,384 (25.4 %) | 3,096 (23.2 %) |
| pt of all rows E | 1.66 | 1.71 | 1.62 | 1.81 | 1.65 |
kway: advance_presence 1,084 -> 1,197 errors (more radar upstream zones called Advance), long_zone / departure slightly fewer.
## Subclass own accuracy (reported, not optimised; >= 30 min, exact subclass / any non-ATSPM; kway)
Upstream_presence (7,166 rows) .712 / .829 (v4f non-ATSPM .846); Setback_presence (3,412) .731 / .849 (.849); Long_zone (1,960)
.361 / .505 (.504); Departure (543) .400 / .536 (.475); Other rest (3,692) exact .448 (v4f .633) / .723 (.739). Main confusions:
Upstream -> Setback 451 rows, Upstream -> Advance 1,032. Shuffled control: exact ~0 for every subclass (labels are learnable).
## Verdict: NOT ADOPTED (v4f stays)
* The subclasses are learnable (71-73 % exact for the two big ones vs 0 % shuffled) but buy no ATSPM accuracy: primary kway
  -0.07 n.s. with MORE definitional errors (+78 rows), fails the rule. sum is neutral (-0.04 n.s.) with 80 fewer definitional
  rows (-0.04 pt), but it is the secondary decode and the gain is noise-sized; the control shows the class-set change alone
  costs ~0.1 pt (YR -0.4), which the real subclasses only win back. Short windows -0.09..-0.17.
* For the orchestrator: no package change. The definitional bucket (~1.7 pt) does not move by labelling these zones; it
  needs the open user definition question (score as nearest class), as note 78 said.
