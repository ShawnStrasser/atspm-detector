# 134 Health 4-h persistence rule adopted in the package; ship stage re-synced and re-simulated (2026-10-08)
User decision (Oct 8): note 121's proposed rule P is the default. Package `dc_work/final_v7_prod` (1.0.0, version
kept); backup before: `dc_work/final_v7_prod_bak134`. Scripts `research/code/final134/` (cmp134, par134, cmp_par134);
outputs `dc_work/s134/`. CPU, 4 threads.

**Change.** `health_v4.severity_v4d` adds rule P after A, before C, exactly as `score_v4d.severity_v4d(persist=True)`:
a detector the resolver calls suspect, not already bad by A, with a finding lasting >= 4 h of the day -> bad
(`prof` busy at night 01-05 h; `shape24` with run `bd_ev_h` >= 4; `dropout` with (drop_b1 - drop_b0) x 5 min >= 4 h;
`occspk` with `n3_n` >= 16 15-min periods). `sev_rule` = "P". In `health_categories` the persisting categories of a
P detector read "(bad)" (e.g. "Busy at night (bad)"); other text unchanged. Docs: health_v4 docstring, README,
pipeline docstring, model card key `health_severity`; staged README row and REPORT.md health bullet (cites 121, 134).

**Equality with the research scorer** (note-124 method: 24 signals behind the 40 sheet rows, every detector,
h3_a/b, h24_a/b, m30_a-d = 4,180 detector-windows, research classifier inputs; truth `s121/resolved_v4d`
st_v4d_persist / sev_rule_persist):
- research mode (research stage 1 + out-of-fold refs): 4,180 / 4,180 statuses and rules equal; P fires 21 times
  (h24_a 12, h24_b 9), never at 30 min or 3 h. Sheet rows 40 / 40 equal; rows 25, 27, 28, 30 suspect -> bad (P;
  25 "Erratic time ON (bad)", 27 / 28 "Busy at night (bad)", 30 "Unusual daily pattern (bad)").
- package mode (shipped refs): 4,178 / 4,180; the 2 status differences (+1 rule label) are note 124's three
  all-data night-model cases (53893b2e h24_a det 7, be117825 h24_a det 48, fb4a3824 h24_b det 37). 40 / 40 rows.

**Rates** (research population, rates121, per 100 detector-windows; flagged = bad + suspect unchanged):
| length | bad before -> after | suspect before -> after |
|---|---|---|
| 30 min | .062 -> .062 | .304 -> .304 |
| 3 h | .533 -> .533 | .629 -> .629 |
| 24 h | 1.648 -> 2.175 (+32 %) | 2.263 -> 1.736 |
Review signals, package mode: 24 h bad 5.25 -> 7.31 per 100 (22 moves: 19 busy at night, 2 erratic time ON,
1 unusual day); 30 min / 3 h no change. Parity set below: no detector moved (none has a busy-at-night,
unusual-day, 4-h silent or 16-period time-ON finding).

**Parity** (note-115 set, 132 cases, 2,627 detectors, bak134 vs new): candidate (15,705) and phase (688) tables
identical; all 32 non-health detector columns identical (phase, function, lanes, setback, night speed, ...);
health columns also 0 differences (s134/cmp_par134.log). Bad / suspect per 100: 30 min 0 / .357, 3 h .673 / .785,
24 h 2.573 / 2.796, before and after.
**Check** 7 / 7 PASS from src; reference answers unchanged, no --freeze. Wheel rebuilt (5.66 MB); smoke_install.py
in a fresh venv (pandas 3.0.6, numpy 2.5.3, duckdb 1.5.6, ort 1.30): wheel = src, CLI + --out-atspm, check 7 / 7.

**Ship stage.** sync_from_prod.py re-run: only health_v4.py, pipeline.py, model_card.json changed, plus README and
REPORT.md text by hand (REPORT cites 121 / 134 for severity; its [118] became 118a / 118b / 118c / 118d, which exist).
apply_ship.py unchanged (REPORT.md is replaced by the staged final-model one; research notes and scripts are
committed). Simulation on a clone of the working tree + its .git/info/exclude (dc_work/tmp/repo_sim3): tag
beta-final_v2, apply --apply, `git add -A`, commit (sim only), clean clone, build, twine check --strict, fresh venv,
atspm-detector-check, pytest, CLI --out-atspm. Every REPORT citation resolves to a committed note; the excluded
TRB draft folder is not committed and no committed file outside STATUS.md / ledger.md names it.
Effect to know: `to_atspm_config` drops "bad" detectors, so a 24-h sample now leaves out a few more (busy at night).
