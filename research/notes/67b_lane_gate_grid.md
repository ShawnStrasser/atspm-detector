# 67b — Lane-confidence gate: wider grid on the context stacker (2026-10-01; follow-up to note 67)
Orchestrator adopted context stacker + lane gate; gate chose .9 (grid top) in 5/6 folds. Re-run `s67_decider.py fix --base ctx
--grid 1.0 --gates 0,.9,.93,.95,.97,.99` (nested per held-out fold on the other folds' E >= 30 rows) and `--gates 0,1.01`
(decode off = every detector unconstrained, argmax; twin decode unchanged). Out `%DC_WORK%/s67/fix_gatewide_ctx.json`,
`fix_decodeoff_ctx.json`. vs shipped decode (gate 0) on the context stacker, >= 30 min, pt [95 % paired signal CI].
## Full-pool >= 30 min E by gate (not nested)
gate 0 .9092 | .90 .9136 | .93 .9136 | .95 .9136 | .97 .9136 | .99 .9130 | decode off .9127. Flat .90-.97; the edge is not hiding a gain.
## Nested
| | >= 30 E | >= 30 R | by fold E |
|---|---|---|---|
| gate .90-.99 (chosen .9 x4, .97 x2) | +0.42 [+0.23,+0.61] | +0.40 [+0.21,+0.59] | +.16 +.76 +.01 +.36 +.30 +.76 |
| decode off | +0.36 [+0.04,+0.66] | +0.32 [+0.01,+0.63] | +.01 +.48 -.04 +.36 -.10 +1.21 |
By class E, gate / decode off vs shipped:
* Count +1.74 / +2.37, Presence +0.75 / +1.18, Advance +1.12 / +1.44.
* Yellow_Red -1.24 / -3.52, non-ATSPM -2.55 / -4.43.
R: Count +1.77 / +2.41, Presence +0.76 / +1.20, Advance +1.03 / +1.30, YR -1.24 / -3.52, non-ATSPM -2.42 / -4.25.
## Verdict
Keep gate .9: same result as note 67 (+0.42), plateau .90-.97, steadier across folds than decode off. On the stacker the lane
decode is now worth only ~+0.09 pt over argmax (.9136 vs .9127). It mainly protects YR / non-ATSPM: decode off costs YR 3.5 and
non-ATSPM 4.4 pt for +2.4 Count.
