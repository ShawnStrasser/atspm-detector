# 56b — In-stack loser -> best non-ATSPM class (2026-09-30)
Orchestrator follow-up to note 56: the user said for STACKS "the highest-probability one gets it, the other is
Other"; "next free class" was agreed for the general (non-stacked) one-per-lane contest. Test: a detector that loses its
FIRST-choice ATSPM class to a partner in its hi-res stack (ln6 co-location / span cover) takes its best non-ATSPM class;
all other losers keep next-free-class. Base = note 56's default (pick_h2: stack-relative health + span1 + track).
Code: `lanes/atspm_decode.py` pick["stack_loser"] = "next_free" (default) | "nonatspm" | "nonatspm_ap" (Advance /
Presence claims only); eval `evaluation/atspm_pick56.py` -> `%DC_WORK%/trackA/atspm56/v3s/pick56b.json`,
`pick56b_ap.json`. OOF six folds, run v3s, locked_v2 absent, CPU only.
| decode (step-4 rows) | ATSPM E | vs greedy | vs default (pick_h2_nf) | ATSPM R | vs default R | stacked members >= 30 min (A->wrongA) |
|---|---|---|---|---|---|---|
| greedy (note 54) | .8910 | - | | .9036 | | .8055 (53) |
| default pick_h2, next free | .8904 | -0.06 [-.12,-.01] | - | .9031 | - | .7869 (71) |
| loser -> non-ATSPM, all classes | .8859 | -0.51 [-.65,-.39] | -0.45 [-.58,-.33] | .8986 | -0.45 [-.59,-.33] | .8041 (45) |
| loser -> non-ATSPM, Advance/Presence only | .8901 | -0.09 [-.15,-.04] | -0.03 [-.07,-.00] | .9029 | -0.02 [-.06,-.00] | .8048 (45) |
| all classes, inputs shuffled | .8796 | -1.14 | | .8922 | | .8089 |
  Per seed E (default / AP-only): .8894 / .8891, .8892 / .8888, .8898 / .8894 (-0.03 to -0.04 every seed).
* All-class version: A->nonA +1,800 rows. Hi-res co-location is not a same-role stack: Count and Yellow_Red zones sit
  on the same spot, and a Count-claimant that loses to its co-located partner is often a true Yellow_Red (next free
  class was right). Dropped.
* AP-only version: the stacked-member damage of the pick is undone (.7869 -> .8048, greedy .8055; A->wrongA 71 -> 45)
  and the contest outcomes are unchanged (true-stack span contests lane-by-lane 767 / 868, loop-over-radar 166 / 262,
  healthy-member-only .780); lane count exact .8025 both. But it costs -0.03 pt overall (CI upper bound -0.00, all
  three seeds): Advance/Presence claimants that lose to a co-located partner outside a labelled stack were, slightly
  more often than not, right to move to their next ATSPM class.
* Signal 13025 det 39: next-free gives Yellow_Red in 8 windows >= 30 min (an ATSPM error); AP-only gives Other in all
  9 windows where it loses (correct under stack credit). Det 28 holds Advance in the same windows either way.
## Verdict
* Not neutral-or-better by the rule (-0.03 pt, CI just below 0) -> default stays next free class (note 56). The AP-only
  rule is the user's stated stack behaviour and fixes stacked members (+1.8 pt on them) - a judgement call for the
  orchestrator: option `stack_loser="nonatspm_ap"`, one line to switch.
