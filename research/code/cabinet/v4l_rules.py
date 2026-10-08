"""Note 80: last label step of v4l (after cab_final + stack_labels_v3s in store mode), then the change list.

    python v4l_rules.py apply      research/labels/function_labels_v4l.parquet: + the rules below (in place, idempotent)
    python v4l_rules.py diff       research/labels/function_label_changes_v4l.csv (every changed label, priority-sorted)
                                   + %DC_WORK%/lab80/diff_summary.json

Rules (cleansing layer only; nothing here is a model input):
  R1 loop_config_count  user (review sheet v1, row 1, 2026-10-01): "a loop detector cannot be labelled as a stop-bar
     count, so the label is false". Hard rule since 2026-09-23: loops are never Count / Yellow_Red. Where the chosen label
     is a CONFIG Count / Yellow_Red and the print lands the channel as a loop: the print's own reading wins at any
     confidence (source print_<conf>, flag); with no print reading the row is left out of training AND scoring.
  R2 user_excluded_signal  user (same sheet, row 2): 2B091's labels are not trustworthy -> the whole signal is left
     out of function training and scoring.
  R4 user_correction  (note 81) the user's answers on review sheet v1, listed in
     research/labels/function_label_corrections_v4l.csv: function rows answered "label wrong" -> out of training AND
     scoring (column user_review = answer + action); phase rows answered "label right" -> phase_user_confirmed = True
     (the timing phase stays the truth; the row is never put to the user again).
  R5 behaviour_tied_low  (orchestrator 2026-10-03, note 81) a LOW-confidence print reading whose channel tie or class
     was decided from hi-res behaviour (hold-through-red share, ON lengths, count pairing / sums, lead / follow, liveness)
     is circular as a label -> out of training AND scoring. Matched on the reader's confidence_reason (BEH_TIE); rows
     where behaviour is only a supporting remark next to a print position / config code are exempt (BEH_EXEMPT).
  R6 noise_input  (orchestrator 2026-10-03, note 81) an input not on the print and without a config function
     (whole-intersection Other) with < 15 ONs in the newest window is noise -> unclassified (out of training AND scoring).
The column names of v3s are kept (`truth_v3s` = the scoring truth of THIS table) so every scorer can take the path.
locked_v2 asserted absent.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from cab_common import DC_WORK, REPO

V3S = REPO / "research/labels/function_labels_v3s.parquet"
V4L = REPO / "research/labels/function_labels_v4l.parquet"
OUT = REPO / "research/labels/function_label_changes_v4l.csv"
REREAD_DONE = DC_WORK / "lab80/reread/done"
EXCLUDE_SIGNALS = {"2B091": "user 2026-10-01 review sheet v1: labels of this signal not trustworthy, throw it out"}
ATSPM = {"Advance", "Presence", "Count", "Yellow_Red"}
CORR = REPO / "research/labels/function_label_corrections_v4l.csv"
BEH_TIE = (r"hold(?:s|ing)? through red|hold-through-red|red occ|(?:long|short|longer|shorter|long-tail) ONs|count pairing"
           r"|counts? (?:track|match)|tied? by (?:phase and )?counts|tie by phase and counts|lane-level counts|count structure"
           r"|volume match|\(volume |(?:leads|follows) det|by liveness|low counts suggest|very low counts|sum of advances"
           r"|~ sum of|~ det\d|pairs with [^;]*\(\d+ vs \d+\)|only this channel counts|queue sits on it|ONs \(p90"
           r"|ONs \(median|ONs \(0\.")
# behaviour mentioned only as support; the class comes from the print position or the config code (checked by hand)
BEH_EXEMPT = {("2C050", 2), ("10063", 2), ("10063", 5), ("10063", 17), ("10063", 19), ("10002", 50), ("10002", 52)}
NOISE_ON = 15


def locked() -> set[str]:
    return set(pd.read_csv(DC_WORK / "official/locked_v2.csv").DeviceId.astype(str).str.lower())


def apply():
    d = pd.read_parquet(V4L)
    assert not d.DeviceId.str.lower().isin(locked()).any(), "locked_v2 in v4l"
    for c in ("rule_v4l",):
        d[c] = pd.Series(pd.NA, index=d.index, dtype="string")
    # R1 -- config Count / YR on a print loop
    loop = d.print_source.eq("print").fillna(False) & d.technology.eq("loop").fillna(False)
    # cab_final (store mode) already moved the training label (so the label check saw it); here: flag, truth, exclusion
    notprint = ~d.source.astype("string").isin(["user_ruling", "print_high"]).fillna(False)
    r1 = (loop & notprint & d.func7_v2.isin(["Count", "Yellow_Red"]).fillna(False)
          & ~d.print_function.isin(["Count", "Yellow_Red"]).fillna(False))
    has_p = r1 & d.print_function.notna()
    bad = has_p & d.function.ne(d.print_function) & ~d.stack_relabel.astype(bool)
    assert not bad.any(), f"run cab_final in store mode first ({int(bad.sum())} rows)"
    no_p = r1 & d.print_function.isna()
    d.loc[r1, "rule_v4l"] = np.where(has_p[r1], "R1_loop_config_count->print", "R1_loop_config_count->excluded")
    # truth: a config Count on a loop is never scored as Count; the print reading is not high (else it would have won)
    tr = r1 & d.truth_v3s.isin(["Count", "Yellow_Red"])
    d.loc[tr, "truth_v3s"] = None
    d.loc[no_p, "exclude_train_score"] = True
    # R2 -- whole signals out
    r2 = d.DeviceName.astype(str).isin(EXCLUDE_SIGNALS)
    d.loc[r2, "exclude_train_score"] = True
    d.loc[r2, "rule_v4l"] = "R2_user_excluded_signal"
    # R3 -- a re-read YR zone the reader found count-identical to a Count zone (AGENTS.md: left out of training and scoring)
    r3 = d.print_flags.astype("string").str.contains("yr_identical_to_count").fillna(False) & d.function.eq("Yellow_Red")
    d.loc[r3, "exclude_train_score"] = True
    d.loc[r3, "yr_count_identical"] = True
    d.loc[r3 & d.rule_v4l.isna(), "rule_v4l"] = "R3_yr_identical_to_count"
    # R4 -- the user's answers on review sheet v1 (note 81)
    c = pd.read_csv(CORR, dtype={"DeviceName": str})
    d["user_review"] = pd.Series(pd.NA, index=d.index, dtype="string")
    d["phase_user_confirmed"] = False
    key = d.DeviceName.astype(str) + "_" + d.detector.astype(int).astype(str)
    r4f = pd.Series(False, index=d.index)
    for r in c.itertuples():
        m = key.eq(f"{r.DeviceName}_{int(r.detector)}")
        assert m.sum() == 1, (r.DeviceName, r.detector, int(m.sum()))
        d.loc[m, "user_review"] = f"v1 row {r.sheet_row}: {r.answer} ({r.action})"
        if r.kind == "function":
            assert r.answer == "M" and r.action in ("exclude_train_score", "exclude_signal"), r
            d.loc[m, "exclude_train_score"] = True
            d.loc[m, "truth_v3s"] = None
            r4f |= m
        else:
            assert r.kind == "phase" and r.answer == "L" and r.action == "confirm_phase", r
            shown = int(str(r.label_shown).split()[-1])
            assert d.loc[m, "phase_target"].iloc[0] == f"P{shown}", (r.DeviceName, r.detector, d.loc[m, "phase_target"].iloc[0])
            d.loc[m, "phase_user_confirmed"] = True
    d.loc[r4f & d.rule_v4l.isna(), "rule_v4l"] = "R4_user_correction"
    # R5 -- low-confidence print readings tied / classed by hi-res behaviour (orchestrator rule, note 81)
    nm_det = list(zip(d.DeviceName.astype(str), d.detector.astype(int)))
    exempt = pd.Series([k in BEH_EXEMPT for k in nm_det], index=d.index)
    r5 = (d.source.astype("string").eq("print_low").fillna(False) & d.print_confidence.eq("low").fillna(False)
          & d.confidence_reason.astype("string").str.contains(BEH_TIE, case=False, regex=True).fillna(False) & ~exempt)
    d.loc[r5, "exclude_train_score"] = True
    d.loc[r5, "truth_v3s"] = None
    d.loc[r5 & d.rule_v4l.isna(), "rule_v4l"] = "R5_behaviour_tied_low"
    # R6 -- unused inputs (whole-intersection Other: not on the print, no config function) with < 15 ONs = noise
    r6 = (d.source.astype("string").eq("whole_intersection_other").fillna(False) & d.config_function.isna()
          & d.n_on_new.lt(NOISE_ON).fillna(False))
    d.loc[r6, "exclude_train_score"] = True
    d.loc[r6, "truth_v3s"] = None
    d.loc[r6 & d.rule_v4l.isna(), "rule_v4l"] = "R6_noise_input"
    d.loc[d.exclude_train_score.astype(bool), "train_use"] = False
    d["train_use_validated"] = d.train_use.astype(bool) & d.validated.eq("pass").fillna(False).astype(bool)
    d["label_version"] = "v4l"
    d.to_parquet(V4L, index=False)
    s = dict(R1_rows=int(r1.sum()), R1_to_print=int(has_p.sum()), R1_excluded=int(no_p.sum()),
             R1_truth_cleared=int(tr.sum()), R2_rows=int(r2.sum()), R3_rows=int(r3.sum()),
             R4_function_rows=int(r4f.sum()), R4_phase_confirmed=int(d.phase_user_confirmed.sum()), R5_rows=int(r5.sum()),
             R6_rows=int(r6.sum()),
             exclude_train_score=int(d.exclude_train_score.sum()), train_use_validated=int(d.train_use_validated.sum()))
    print(json.dumps(s, indent=1))
    return s


def _reread() -> set[str]:
    return {p.stem for p in REREAD_DONE.glob("*.txt")} if REREAD_DONE.exists() else set()


def diff():
    a = pd.read_parquet(V3S)
    b = pd.read_parquet(V4L)
    lk = locked()
    assert not a.DeviceId.str.lower().isin(lk).any() and not b.DeviceId.str.lower().isin(lk).any()
    rel = set(pd.read_csv(DC_WORK / "official/newtest_released.csv").DeviceName.astype(str))
    rr = _reread()
    for x in (a, b):
        x["dev"] = x.DeviceId.str.lower()
        x["train_label"] = x.function.where(x.train_use_validated.astype(bool))
        x["score_truth"] = x.truth_v3s.where(~x.exclude_train_score.astype(bool))
    k = ["dev", "detector"]
    cols = ["DeviceName", "function", "source", "train_label", "score_truth", "validated", "print_function",
            "print_confidence", "technology", "tier", "phase_target", "description"]
    m = a[k + cols].merge(b[k + cols + ["rule_v4l"]], on=k, how="outer", suffixes=("_v3s", "_v4l"), indicator=True)
    m["DeviceName"] = m.DeviceName_v4l.fillna(m.DeviceName_v3s)
    f = lambda c: m[f"{c}_v3s"].astype("string").fillna("-").ne(m[f"{c}_v4l"].astype("string").fillna("-"))
    m["chg_label"], m["chg_train"], m["chg_truth"] = f("function"), f("train_label"), f("score_truth")
    m = m[m.chg_label | m.chg_train | m.chg_truth].copy()
    nm = m.DeviceName.astype(str)
    sw = DC_WORK / "lab80" / "expl_removed.csv"
    swk = set(pd.read_csv(sw, dtype=str).apply(lambda r: f"{r.DeviceName}_{int(r.detector)}", axis=1)) if sw.exists() else set()
    key = nm + "_" + m.detector.astype(int).astype(str)
    rule = m.rule_v4l.astype("string").fillna("")
    m["cause"] = np.select(
        [rule.str.startswith("R2"), rule.str.startswith("R1"), rule.str.startswith("R3"), rule.str.startswith("R4"),
         rule.str.startswith("R5"), rule.str.startswith("R6"), key.isin(swk), nm.isin(rel), nm.isin(rr)],
        ["user_excluded_signal", "loop_config_count", "yr_identical_to_count", "user_correction", "behaviour_tied_low",
         "noise_input", "explained_other_sweep", "released_newtest_pipeline", "print_reread"], "pipeline_rerun")
    atspm = lambda s: s.astype("string").isin(list(ATSPM)).fillna(False)
    m["atspm_change"] = (atspm(m.score_truth_v3s) | atspm(m.score_truth_v4l)) & m.chg_truth
    m["priority"] = (3 * m.atspm_change + 2 * (m.chg_truth & m.score_truth_v4l.notna()) + m.chg_train).astype(int)
    m = m.sort_values(["priority", "cause", "DeviceName", "detector"], ascending=[False, True, True, True])
    out = m[["priority", "cause", "DeviceName", "detector", "phase_target_v4l", "score_truth_v3s", "score_truth_v4l",
             "train_label_v3s", "train_label_v4l", "function_v3s", "function_v4l", "source_v3s", "source_v4l",
             "validated_v3s", "validated_v4l", "print_function_v4l", "print_confidence_v4l", "technology_v4l", "tier_v3s",
             "tier_v4l", "rule_v4l", "description_v4l"]]
    out.to_csv(OUT, index=False)
    s = dict(rows_v3s=len(a), rows_v4l=len(b), changed=len(out),
             by_cause=out.cause.value_counts().to_dict(),
             truth_changed=int(m.chg_truth.sum()), truth_changed_by_cause=m[m.chg_truth].cause.value_counts().to_dict(),
             truth_atspm_changed=int(m.atspm_change.sum()),
             train_changed=int(m.chg_train.sum()), train_changed_by_cause=m[m.chg_train].cause.value_counts().to_dict(),
             truth_added=int((m.score_truth_v3s.isna() & m.score_truth_v4l.notna()).sum()),
             truth_removed=int((m.score_truth_v3s.notna() & m.score_truth_v4l.isna()).sum()),
             truth_relabelled=int((m.score_truth_v3s.notna() & m.score_truth_v4l.notna()).sum()),
             train_added=int((m.train_label_v3s.isna() & m.train_label_v4l.notna()).sum()),
             train_removed=int((m.train_label_v3s.notna() & m.train_label_v4l.isna()).sum()),
             train_relabelled=int((m.train_label_v3s.notna() & m.train_label_v4l.notna()).sum()),
             truth_pairs={f"{x}->{y}": int(n) for (x, y), n in m[m.chg_truth].groupby(
                 [m.score_truth_v3s.astype("string").fillna("-"), m.score_truth_v4l.astype("string").fillna("-")]).size()
                 .sort_values(ascending=False).head(25).items()},
             tiers_v3s=a.drop_duplicates("dev").tier.fillna("none").value_counts().to_dict(),
             tiers_v4l=b.drop_duplicates("dev").tier.fillna("none").value_counts().to_dict(),
             reread_signals=len(rr))
    (DC_WORK / "lab80").mkdir(exist_ok=True)
    (DC_WORK / "lab80/diff_summary.json").write_text(json.dumps(s, indent=1, default=str), encoding="utf-8")
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    {"apply": apply, "diff": diff}[sys.argv[1]]()
