"""Consolidation of the cabinet-print labels: one command, in order (guide section 6; note 27).

    python cab_final.py                  every finished batch (all its signals visual_done); applies the fixes
    python cab_final.py --dry-run        fixes as dry runs, no record written (tables still rebuilt)
    python cab_final.py --skip-dq        reuse cabinet/dq_print.parquet (DQ is the slow step)
    python cab_final.py --review-only    only the review/ lists, from the saved print_labels + v3 table

1. Finished batches = batches whose every signal is `visual_done`; a batch still being written is never touched.
2. Rulings (`%DC_WORK%/cabinet/final_rulings.csv`: DeviceName, detector, rule, reason):
   `unusual_layout` -> signal flag written into the record (logged to sweep_changes.csv);
   `keep_print`     -> the channel must be a print detector (a config "dummy" text never makes it derived).
3. Bulk fixes over the finished batches: fix_long_presence (--config-desc: a config description saying
   Presence/Pres is a Presence code), fix_video_yr and fix_radar_over_loops (a LIVE loop under a radar zone that is
   itself Other keeps its own role, medium), all logging to sweep_changes.csv.
4. cab_build.build -> print_labels.parquet (print rows + whole-intersection rows) and print_tiers.csv.
5. DQ re-run (dq_core) with the PRINT location (stop-bar Presence/Count/YR -> stopbar, Advance -> advance,
   Mid -> mid, Bike -> bike, else other), newest window; dq_score / dq_flags / suspect attached to print_labels.
6. Training label table research/labels/function_labels_v3.parquet (README there), two label variants:
   label_print_first  user ruling > print high > v2 > print medium/low > whole-intersection Other (complete tiers only)
   label_print_agree  as above, but a high print label replaces a DIFFERENT v2 label only where the six-fold
                      out-of-fold 7-class prediction (note 25, variant b7, seed mean, full window) agrees.
7. Label-behaviour validation (label_check.py, note 29): every labelled row checked against the engineer's rules on the
   hi-res data (thresholds calibrated once on trusted rows, frozen in cabinet/label_check_thresholds.json); columns
   validated / failed_checks / validation_reason / validation_numbers / train_use_validated written into v3;
   review/label_check_failures.xlsx. v3 of the check (2026-09-28): label-wrong vs detector / configuration problem;
   field_issue column; no fault events; protected-permissive phases exempt from the red-time rules.
10. (note 51b) Dec-2024 role drift: cabinet/dec_role_changed.parquet - channels whose Dec-2024 config export gives a
   different function / phase than the current label, or omits an Advance / Presence / Count channel; their Dec rows
   are dropped from training and scoring by v3_retrain.py --dec-role. Alone: `python cab_final.py --dec-role`.
8. User lists in review/: print_label_overrides.xlsx, dead_detectors_from_prints.csv, field_issues_for_staff.xlsx; summary -> cabinet/final_summary.json.
9. Released NEWTEST signals (note 36): their rows of function_labels_locked_v1 are appended to v3
   (`released_from_newtest` = True). Since 2026-09-28 this runs BEFORE step 7, so the label check validates them like
   every other training row. On its own (`python cab_final.py --append-released`) they stay validated = pending.
Locked hold-out signals never enter any output (asserted; locked = locked_v2.csv once the split exists).

LOCKED PATH (note 36; labels only — no DQ, no OOF / model output, no label_check, no user lists):
    python cab_final.py --locked-consolidate --locked-labels-only
runs on the separate store %DC_WORK%/cabinet_locked/: rulings (its own final_rulings.csv, seeded with the training
file's '*' rulings), fixes (the three above + fix_uncoded_zones), cab_build.build, label rulings, then
research/labels/function_labels_locked_v1.parquet (v3 schema where applicable + set / released_from_newtest /
func5_config; label = the label_print_first rule; validated = pending).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import cab_build
from cab_common import CAB, DC_WORK, LOCKED_FLAG, LOCKED_MODE, REPO, SIG_DIR, STORE, signals, forbid_locked_mode
import os

LOCKED_PATH = "--locked-consolidate" in sys.argv
if not LOCKED_PATH:
    forbid_locked_mode(__name__)  # training-label / model-facing: never on the locked-label store
elif not LOCKED_MODE:
    raise PermissionError(f"--locked-consolidate needs {LOCKED_FLAG} (it writes only the locked store)")

HERE = Path(__file__).resolve().parent
LOG = CAB / "sweep_changes.csv"
RULINGS = CAB / "final_rulings.csv"
V2 = REPO / "research/labels/function_labels_v2.parquet"
# note 80: with DC_CAB_STORE set (store copy incl. the released NEWTEST records) the table is versioned and the user
# lists stay in the store (<store>/lists/), never review/
LABELS_TAG = os.environ.get("DC_LABELS_TAG", "v4l")
V3 = REPO / (f"research/labels/function_labels_{LABELS_TAG}_base.parquet" if STORE else "research/labels/function_labels_v3.parquet")
LOCKED_V1 = REPO / "research/labels/function_labels_locked_v1.parquet"
TRAIN_RULINGS = DC_WORK / "cabinet" / "final_rulings.csv"
RELEASED = DC_WORK / "official/newtest_released.csv"
LOCKED_V2 = DC_WORK / "official/locked_v2.csv"
REVIEW = (CAB / "lists") if STORE else REPO / "review"
W1 = DC_WORK / "trackA/w1"
CLASSES7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]  # w1 b7 column order
LOC = {"Presence": "stopbar", "Count": "stopbar", "Yellow_Red": "stopbar", "Advance": "advance", "Mid": "mid",
       "Bike": "bike"}
WIN_NAME = {"staging": "Sep 2026", "dec2024": "Dec 2024"}
NOT_MAINT = {"superseded_by_radar", "radar_not_commissioned"}  # dead by design, not a maintenance item


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ 1. finished batches
def finished_batches() -> tuple[list[int], set[str], dict]:
    nums, dns, state = [], set(), {}
    for p in sorted((CAB / "batches").glob("batch_[0-9][0-9].txt")):
        n = int(p.stem.split("_")[1])
        b = [ln.split("\t")[0].strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        st = collections.Counter(
            json.loads((SIG_DIR / f"{d}.json").read_text(encoding="utf-8")).get("status")
            if (SIG_DIR / f"{d}.json").exists() else "missing" for d in b)
        state[n] = dict(st)
        if set(st) == {"visual_done"}:
            nums.append(n)
            dns |= set(b)
    return nums, dns, state


# ------------------------------------------------------------------ 2. rulings
RCOLS = ["DeviceName", "detector", "subtype", "rule", "value", "reason"]


def rulings() -> pd.DataFrame:
    if not RULINGS.exists():
        return pd.DataFrame(columns=RCOLS)
    return pd.read_csv(RULINGS, dtype=str, keep_default_na=False).reindex(columns=RCOLS, fill_value="")


def _match(pl: pd.DataFrame, x) -> pd.Series:
    """Rows a ruling selects: DeviceName ('*' = all, 'a|b' = several), detector (blank = all), subtype ('a|b').
    Print rows only, except a ruling that names both the signal and the detector(s): it also reaches a data-only
    (not on the print) channel (2026-09-29: 2B143 det 8)."""
    named = x.DeviceName not in ("", "*") and bool(x.detector)
    m = pl.source.isin(["print", "data_only"]) if named else pl.source.eq("print")
    if x.DeviceName not in ("", "*"):
        m &= pl.DeviceName.isin(x.DeviceName.split("|"))
    if x.detector:
        m &= pl.detector.isin([int(v) for v in x.detector.split("|")])
    if x.subtype:
        m &= pl.subtype.isin(x.subtype.split("|"))
    return m


def apply_label_rulings(pl: pd.DataFrame, R: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label questions open with the user, applied on the consolidated table (the records stay as read).
    `pending_user` rows tag the selected detectors with the question (column pending_user); an answer is ONE extra
    line `relabel` with the same selector and value = the new function: it relabels those rows (flag user_ruling,
    function_as_read keeps the reading) and clears their pending tag; or `resolved` (label kept as read, tag
    cleared, flag user_ruling). `confidence` (value high/medium/low) sets the reading's confidence."""
    pl = pl.copy()
    pl["function_as_read"] = pl.function
    pl["pending_user"] = pd.Series(pd.NA, index=pl.index, dtype="string")
    out = {}
    for _, x in R[R.rule == "pending_user"].iterrows():
        m = _match(pl, x)
        pl.loc[m, "pending_user"] = x.value
        out[x.value] = int(m.sum())
    for _, x in R[R.rule == "relabel"].iterrows():
        m = _match(pl, x)
        pl.loc[m, "function"] = x.value
        pl.loc[m, "flags"] = pl.loc[m, "flags"].fillna("").map(lambda s: ",".join(sorted(set(filter(None, s.split(","))) | {"user_ruling"})))
        pl.loc[m, "pending_user"] = pd.NA
        out[f"relabel->{x.value}"] = out.get(f"relabel->{x.value}", 0) + int(m.sum())
    for _, x in R[R.rule == "resolved"].iterrows():  # answered, label kept as read
        m = _match(pl, x)
        pl.loc[m, "flags"] = pl.loc[m, "flags"].fillna("").map(lambda s: ",".join(sorted(set(filter(None, s.split(","))) | {"user_ruling"})))
        pl.loc[m, "pending_user"] = pd.NA
        out[f"resolved:{x.value}"] = int(m.sum())
    for _, x in R[R.rule == "confidence"].iterrows():  # the user's ruling settles the reading's confidence
        m = _match(pl, x)
        pl.loc[m, "confidence"] = x.value
        out[f"confidence->{x.value}"] = out.get(f"confidence->{x.value}", 0) + int(m.sum())
    return pl, out


def apply_unusual(R: pd.DataFrame, done: set[str], apply: bool) -> list[str]:
    out, rows = [], []
    for _, x in R[R.rule == "unusual_layout"].iterrows():
        dn = x.DeviceName
        if dn not in done:
            log(f"ruling unusual_layout {dn}: batch not finished, left for the final pass")
            continue
        fp = SIG_DIR / f"{dn}.json"
        r = json.loads(fp.read_text(encoding="utf-8"))
        fl = r.get("signal_flags") or []
        if "unusual_layout" in fl:
            continue
        r["signal_flags"] = fl + ["unusual_layout"]
        r["signal_flags_reason"] = x.reason
        rows.append(dict(DeviceName=dn, detector="", field="signal_flags", old=",".join(fl),
                         new=",".join(r["signal_flags"]), rule="unusual_layout"))
        out.append(dn)
        if apply:
            fp.write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    if apply and rows:
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, ["DeviceName", "detector", "field", "old", "new", "rule"]).writerows(rows)
    return out


# ------------------------------------------------------------------ 3. bulk fixes
def run_fix(script: str, args: list[str]) -> str:
    r = subprocess.run([sys.executable, str(HERE / script)] + args, cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{script} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


# ------------------------------------------------------------------ 5. DQ with print locations
def dq_print(pl: pd.DataFrame, skip: bool) -> pd.DataFrame:
    f = CAB / "dq_print.parquet"
    pr = pl[pl.source == "print"].copy()
    if STORE and f.exists() and not skip:
        # note 80: recompute DQ only for signals whose print rows changed (location / lane / phase) since the cache
        old = pd.read_parquet(CAB / "print_labels_prev.parquet") if (CAB / "print_labels_prev.parquet").exists()             else pd.read_parquet(CAB / "print_labels.parquet")
        key = ["DeviceId", "detector", "function", "lane_index", "lanes_spanned", "phase_timing", "technology"]
        a = pr[key].astype(str).assign(k=1)
        b = old[old.source == "print"][key].astype(str).assign(k=1)
        m = a.merge(b, on=key, how="left", suffixes=("", "_o"))
        cached = pd.read_parquet(f)
        chg = set(m[m.k_o.isna()].DeviceId) | (set(pr.DeviceId) - set(cached.DeviceId))
        log(f"dq incremental: {len(chg)} signals to (re)compute")
        keep = cached[~cached.DeviceId.isin(chg) & cached.DeviceId.isin(set(pr.DeviceId))]
        if chg:
            import dq_core
            q = pr[pr.DeviceId.isin(chg)]
            d = pd.DataFrame(dict(
                DeviceId=q.DeviceId, detector=q.detector.astype(int),
                phase=q.phase_timing.fillna(q.phase_diagram.astype("string")).fillna("?").astype(str),
                location=q.function.map(LOC).fillna("other"), lane_index=q.lane_index.astype("float"),
                lanes_spanned=q.lanes_spanned.astype("float").fillna(1), technology=q.technology, function=q.function))
            new = dq_core.run(d, period="auto", threads=4)
            new.attrs.pop("pairs", None)
            new = new[["DeviceId", "detector", "source", "dq_score", "flags", "reasons", "suspect_config_or_health"]]
            new.columns = ["DeviceId", "detector", "dq_source", "dq_score", "dq_flags", "dq_reasons", "dq_suspect"]
            keep = pd.concat([keep, new], ignore_index=True)
        dq = keep
        dq.to_parquet(f, index=False)
    elif skip and f.exists():
        dq = pd.read_parquet(f)
    else:
        import dq_core
        d = pd.DataFrame(dict(
            DeviceId=pr.DeviceId, detector=pr.detector.astype(int),
            phase=pr.phase_timing.fillna(pr.phase_diagram.astype("string")).fillna("?").astype(str),
            location=pr.function.map(LOC).fillna("other"), lane_index=pr.lane_index.astype("float"),
            lanes_spanned=pr.lanes_spanned.astype("float").fillna(1), technology=pr.technology,
            function=pr.function))
        dq = dq_core.run(d, period="auto", threads=8)
        pairs = dq.attrs.pop("pairs", pd.DataFrame())
        if len(pairs):
            pairs.astype({c: str for c in pairs.columns if pairs[c].dtype == object}).to_parquet(
                CAB / "dq_print_pairs.parquet", index=False)
        dq = dq[["DeviceId", "detector", "source", "dq_score", "flags", "reasons", "suspect_config_or_health"]]
        dq.columns = ["DeviceId", "detector", "dq_source", "dq_score", "dq_flags", "dq_reasons", "dq_suspect"]
        dq.to_parquet(f, index=False)
    pl = pl.drop(columns=[c for c in ("dq_score", "dq_source", "dq_flags", "dq_reasons", "dq_suspect") if c in pl])
    pl = pl.merge(dq, on=["DeviceId", "detector"], how="left")
    pl.to_parquet(CAB / "print_labels.parquet", index=False)
    return pl


# ------------------------------------------------------------------ 6. OOF + v3
def oof7() -> pd.DataFrame | None:
    """Seed-mean 7-class out-of-fold function probabilities (note 25 b7), full window, mean over periods."""
    files = sorted(W1.glob("Praw_b7_s*.npy"))
    if not files or not (W1 / "keys.parquet").exists():
        return None
    k = pd.read_parquet(W1 / "keys.parquet", columns=["DeviceId", "Detector", "wgroup"])
    P = np.mean([np.load(x) for x in files], axis=0)
    if P.shape != (len(k), len(CLASSES7)):
        return None
    m = (k.wgroup == "full").to_numpy()
    df = pd.DataFrame(P[m], columns=CLASSES7)
    df["DeviceId"], df["Detector"] = k.DeviceId[m].str.lower().to_numpy(), k.Detector[m].to_numpy()
    g = df.groupby(["DeviceId", "Detector"])[CLASSES7].mean()
    return pd.DataFrame({"oof_pred": g.idxmax(axis=1), "oof_p": g.max(axis=1), "oof_n_seeds": len(files)}).reset_index()


def v2_func7(v2: pd.DataFrame) -> pd.Series:
    """v2 label in the 7 classes: an Other whose config text names mid / bike -> Mid / Bike (note 25 b7)."""
    raw = v2.config_function.astype("string").str.strip().str.lower().fillna("")
    sub = np.where(raw.str.contains("mid"), "Mid", np.where(raw.str.contains("bike"), "Bike", "Other"))
    return pd.Series(np.where(v2.func5 == "Other", sub, v2.func5), index=v2.index).where(v2.func5.notna())


def build_v3(pl: pd.DataFrame, tiers: pd.DataFrame, locked: set[str]) -> tuple[pd.DataFrame, bool]:
    v2 = pd.read_parquet(V2)
    v2["DeviceId"] = v2.DeviceId.str.lower()
    # v2 marks the 186 ORIGINAL hold-outs in `locked`; in the store copy (note 80) the released NEWTEST are training
    v2 = v2[~v2.DeviceId.isin(locked) & (True if STORE else v2.locked.fillna("").eq(""))].copy()
    v2["func7_v2"] = v2_func7(v2)
    v2["v2_source"] = np.where(v2.label_source.str.startswith("review"), "review_round1", "config")
    v2 = v2.rename(columns={"func5": "func5_v2", "Detector": "detector"})[
        ["DeviceId", "DeviceName", "detector", "func5_v2", "func7_v2", "v2_source", "config_function", "description"]]
    p = pl.copy()
    p["DeviceId"] = p.DeviceId.str.lower()
    p = p.rename(columns={"function": "print_function", "subtype": "print_subtype", "confidence": "print_confidence",
                          "flags": "print_flags", "tier": "print_tier", "source": "print_source"})
    keep = ["DeviceId", "DeviceName", "detector", "print_source", "print_function", "print_subtype", "print_confidence",
            "confidence_reason", "technology", "phase_diagram", "lane_index", "lanes_spanned", "lane_type",
            "n_lanes_phase", "print_flags", "n_on_new", "window_new", "dq_score", "dq_flags", "dq_suspect", "crop",
            "pending_user"]
    df = v2.merge(p[keep], on=["DeviceId", "detector"], how="outer", suffixes=("", "_p"))
    df["DeviceName"] = df.DeviceName.fillna(df.pop("DeviceName_p"))
    t = tiers.assign(DeviceId=tiers.DeviceId.str.lower())[["DeviceId", "tier", "unusual_layout"]]
    df = df.merge(t, on="DeviceId", how="left")
    df["unusual_layout"] = df.unusual_layout.astype("boolean").fillna(False).astype(bool)
    off = pd.read_parquet(DC_WORK / "official/labels_official.parquet",
                          columns=["DeviceId", "Detector", "target", "target_type", "switch_phase", "additional_call_phases"])
    off = off.rename(columns={"Detector": "detector", "target": "phase_target", "target_type": "phase_target_type"})
    df = df.merge(off.assign(DeviceId=off.DeviceId.str.lower()), on=["DeviceId", "detector"], how="left")
    o = oof7()
    has_oof = o is not None
    if has_oof:
        df = df.merge(o.rename(columns={"Detector": "detector"}), on=["DeviceId", "detector"], how="left")
    else:
        df["oof_pred"], df["oof_p"], df["oof_n_seeds"] = None, np.nan, 0

    pf, pc = df.print_function, df.print_confidence
    is_print = df.print_source.eq("print") & pf.notna()
    high = is_print & pc.eq("high")
    # the user's answer beats every source (also on a data-only channel he named, 2026-09-29)
    user = pf.notna() & df.print_flags.fillna("").str.contains(r"\buser_ruling\b")
    v2ok = df.func7_v2.notna()
    wio = df.print_source.eq("data_only") & df.tier.isin(["complete_high", "complete_mixed"])

    def pick(high_ok):
        lab = pd.Series(None, index=df.index, dtype="object")
        src = pd.Series(None, index=df.index, dtype="object")
        steps = [(user, pf, "user_ruling"), (high & high_ok, pf, "print_high"), (v2ok, df.func7_v2, None),
                 (is_print & pc.eq("medium"), pf, "print_medium"), (is_print & pc.eq("low"), pf, "print_low"),
                 (wio, pd.Series("Other", index=df.index), "whole_intersection_other")]
        for m, val, s in steps:
            m = m & lab.isna()
            lab[m] = val[m]
            src[m] = df.v2_source[m] if s is None else s
        return lab, src

    agree_ok = ~v2ok | df.func7_v2.eq(pf) | df.oof_pred.eq(pf)
    df["label_print_first"], df["source"] = pick(pd.Series(True, index=df.index))
    df["label_print_agree"], df["source_agree"] = pick(agree_ok)
    df["function"] = df.label_print_first
    if STORE:  # note 80 rule R1 (user 2026-10-01 + hard rule 2026-09-23): a CONFIG Count / Yellow_Red on a print loop is
        # never the label: the print's own reading wins at any confidence, else no label (v4l_rules.py also clears truth)
        r1 = (df.print_source.eq("print") & df.technology.eq("loop") & df.source.isin(["config", "review_round1"])
              & df.function.isin(["Count", "Yellow_Red"])).fillna(False)
        hp = r1 & pf.notna()
        df.loc[hp, "source"] = "print_" + pc[hp].astype(str)
        df.loc[r1, "function"] = pf[r1]
        df.loc[r1, "label_print_first"] = pf[r1]
        df.loc[r1 & ~hp, "source"] = None
    df["print_overrides_v2"] = high & v2ok & df.func7_v2.ne(pf)
    df["oof_agrees_print"] = np.where(df.oof_pred.isna(), None, df.oof_pred.eq(pf))
    dead = df.print_flags.fillna("").str.contains(r"\bdead\b|no_data")
    df["dead"] = dead
    df["train_use"] = df.function.notna() & ~df.unusual_layout & ~dead
    assert not df.DeviceId.isin(locked).any(), "locked signal in v3"
    assert not df.duplicated(["DeviceId", "detector"]).any()
    cols = ["DeviceId", "DeviceName", "detector", "phase_target", "phase_target_type", "switch_phase",
            "additional_call_phases", "function", "source", "label_print_first", "label_print_agree", "source_agree",
            "print_overrides_v2", "func5_v2", "func7_v2", "v2_source", "print_source", "print_function", "print_subtype",
            "print_confidence", "confidence_reason", "technology", "phase_diagram", "lane_index", "lanes_spanned",
            "lane_type", "n_lanes_phase", "print_flags", "tier", "unusual_layout", "dead", "n_on_new", "window_new",
            "dq_score", "dq_flags", "dq_suspect", "oof_pred", "oof_p", "oof_agrees_print", "train_use", "pending_user",
            "config_function", "description", "crop"]
    df = df[cols].sort_values(["DeviceName", "detector"]).reset_index(drop=True)
    for c in ("oof_agrees_print", "dq_suspect"):
        df[c] = df[c].astype("boolean")
    for c in ("additional_call_phases", "config_function", "description", "crop", "confidence_reason", "dq_flags"):
        df[c] = df[c].astype("string")
    df.to_parquet(V3, index=False)
    return df, has_oof


# ------------------------------------------------------------------ 7. label-behaviour validation
def label_check_step(v3: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """label_check.run on the fresh v3 table; writes the table back with the validation columns + the review list."""
    import label_check
    out, d, cal = label_check.run(v3)
    out.to_parquet(V3, index=False)
    lab = out[out.function.notna()]
    chk = lab[lab.validated.isin(["pass", "fail"])]
    rate = lambda g: {k: round(float((x.validated == "fail").mean()), 4) for k, x in g}
    summ = dict(validated=lab.validated.value_counts(dropna=False).to_dict(),
                fail_rate_checked=round(float((chk.validated == "fail").mean()), 4),
                fail_rate_by_class=rate(chk.groupby("function")), fail_rate_by_source=rate(chk.groupby("source")),
                rules_kept={k: [r.get("stat"), r.get("auc")] for k, r in cal["rules"].items() if r["kept"]},
                rules_dropped={k: [r.get("stat"), r.get("auc")] for k, r in cal["rules"].items() if not r["kept"]},
                review_rows=int((out.validated == "fail").sum()),
                train_use_validated=int(out.train_use_validated.sum()))
    log(f"label check: {summ['validated']}; fail rate {summ['fail_rate_checked']:.1%} of checked")
    return out, summ


# ------------------------------------------------------------------ 8. user lists
ATSPM = {"Advance", "Presence", "Count", "Yellow_Red"}
REFINE = {"Other", "Mid", "Bike"}  # a change inside these only refines "not a performance-measure detector"
WRITTEN = r"table|code|labell?ed|written"  # print reading backed by a zone table / written code


def override_priority(o: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Review priority of a print-over-config change (user 2026-09-23), a product of simple factors:
    model sides with the config x(1 + 2p) | model agrees with the print x0.5 | otherwise x1;
    an ATSPM class on either side x2; busy detector x(1 + log10(1 + actuations)/4);
    print reading cites a zone table / written code x0.6; unusual_layout signal (not used for training) x0.3;
    change only among Other/Mid/Bike x0.1 (and always last).
    Scaled so the top row = 100. Returns (weight, is_refinement, plain-English reason)."""
    cfg, prt, g, pp = o.func7_v2, o.print_function, o.oof_pred, o.oof_p.fillna(0)
    n = o.n_on_new.fillna(0)
    side_cfg, side_prt = g.eq(cfg), g.eq(prt)
    atspm = cfg.isin(ATSPM) | prt.isin(ATSPM)
    written = o.confidence_reason.fillna("").str.contains(WRITTEN, case=False)
    refine = cfg.isin(REFINE) & prt.isin(REFINE)
    unusual = o.unusual_layout.fillna(False).astype(bool)
    w = (np.where(side_cfg, 1 + 2 * pp, np.where(side_prt, 0.5, 1.0)) * np.where(atspm, 2.0, 1.0)
         * (1 + np.log10(1 + n) / 4) * np.where(written, 0.6, 1.0) * np.where(unusual, 0.3, 1.0) * np.where(refine, 0.1, 1.0))
    w = pd.Series(100 * w / w.max(), index=o.index).round(1)
    why = []
    for sc, sp, gg, p_, a, wr, rf, k, un in zip(side_cfg, side_prt, g, pp, atspm, written, refine, n, unusual):
        r = []
        if sc:
            r.append(f"model sides with the config ({p_:.0%} sure)")
        elif sp:
            r.append("model agrees with the print")
        elif isinstance(gg, str):
            r.append(f"model says neither ({gg})")
        else:
            r.append("no model guess")
        if rf:
            r.append("only Other/Mid/Bike detail changes")
        elif a:
            r.append("performance-measure function changes")
        r.append(f"{'busy' if k >= 5000 else 'quiet' if k < 500 else 'moderate'} ({int(k):,} actuations)")
        if wr:
            r.append("print has a zone table or written code")
        if un:
            r.append("unusual site, not used for training")
        why.append("; ".join(r))
    return w, refine, pd.Series(why, index=o.index)


def user_lists(v3: pd.DataFrame, pl: pd.DataFrame):
    REVIEW.mkdir(exist_ok=True)
    o = v3[v3.print_overrides_v2].copy()
    crop = o.crop.where(o.crop.notna(), None)
    guess = [f"{dn}_d{d}.png" for dn, d in zip(o.DeviceName, o.detector)]
    crop = [c if isinstance(c, str) and c else (g if (CAB / "crops" / g).exists() else "") for c, g in zip(crop, guess)]
    w, refine, why = override_priority(o)
    ov = pd.DataFrame({
        "Priority": w, "Signal": o.DeviceName, "Detector": o.detector, "Phase": o.phase_target,
        "Config label": o.func7_v2, "Print label": o.print_function,
        "Model's guess": [f"{g} ({p:.0%})" if isinstance(g, str) else "none"
                          for g, p in zip(o.oof_pred, o.oof_p.fillna(0))],
        "Actuations (Sep 2026)": o.n_on_new.fillna(0).astype(int),
        "Print evidence": o.confidence_reason.fillna("").str.slice(0, 160),
        "Crop file": [Path(c).name if c else "" for c in crop],
        "Why this priority": why, "_r": refine})
    ov = ov.sort_values(["_r", "Priority", "Signal", "Detector"], ascending=[True, False, True, True]).drop(columns="_r")
    ov.insert(0, "Rank", range(1, len(ov) + 1))
    with pd.ExcelWriter(REVIEW / "print_label_overrides.xlsx", engine="openpyxl") as xw:
        ov.to_excel(xw, sheet_name="overrides", index=False)
        ws = xw.sheets["overrides"]
        for col, wd in zip("ABCDEFGHIJKL", (6, 8, 8, 8, 7, 12, 12, 16, 11, 70, 22, 70)):
            ws.column_dimensions[col].width = wd
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
    dead, stale, why = dead_list(v3, pl)
    dead.to_csv(REVIEW / "dead_detectors_from_prints.csv", index=False)
    return ov, dead, stale, why


def dead_list(v3: pd.DataFrame, pl: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Maintenance list: a detector is listed only if it is (a) drawn on the print, (b) not abandoned (superseded / not-commissioned zones, flags
    superseded*, replaced*, print_outdated, print_stale, moved_to_other_channel, not_configured), (d) silent in the
    NEWEST window (Sep 2026) while the same device has other actuating channels there (a silent cabinet is a
    communications loss, not a dead detector), (e) not on a stale-print signal (every print detector silent) or an
    unusual_layout signal. Drawn-but-not-programmed detectors are kept (user 2026-09-23) and marked in_timing = no
    (programmed = official timing target or a timing phase, and no not_configured / not_in_config flag).
    Column in_use_for = the detector's label (ATSPM functions first)."""
    pr = pl[pl.source == "print"].copy()
    fl = pr["flags"].fillna("")
    isdead = fl.str.contains(r"\bdead\b")
    stale = (set(isdead.groupby(pr.DeviceName).all().loc[lambda s: s].index)
             | set(pr.DeviceName[pr.unusual_layout.astype(bool)]))
    tgt = v3[["DeviceId", "detector", "phase_target", "function"]].assign(DeviceId=v3.DeviceId.str.lower())
    pr = pr.assign(_id=pr.DeviceId.str.lower()).merge(tgt.rename(columns={"DeviceId": "_id", "function": "v3_function"}),
                                                       on=["_id", "detector"], how="left").set_index(pr.index)
    fl = pr["flags"].fillna("")
    live = pl[pl.n_on_new.fillna(0) > 0].assign(_id=pl.DeviceId.str.lower())
    n_live = live.groupby("_id").detector.nunique()
    n_live_other = pr._id.map(n_live).fillna(0) - (pr.n_on_new.fillna(0) > 0)
    newest = pr.window_new.eq("staging")
    programmed = (pr.phase_target.notna() | pr.phase_timing.notna()) & ~fl.str.contains(r"\bnot_configured\b|\bnot_in_config\b")
    abandoned = pr.subtype.isin(NOT_MAINT) | fl.str.contains(
        r"superseded|replaced|print_outdated|print_stale|moved_to_other_channel")
    steps = collections.OrderedDict()
    m = isdead
    steps["dead_print_detectors"] = int(m.sum())
    for name, keep in (("not_stale_signal", ~pr.DeviceName.isin(stale)),
                       ("not_abandoned", ~abandoned), ("newest_window_sep2026", newest),
                       ("device_has_other_live_channels", n_live_other > 0)):
        m = m & keep
        steps[name] = int(m.sum())
    steps["of_which_not_in_timing"] = int((m & ~programmed).sum())
    d = pr[m]
    use = d.v3_function.fillna(d.function).fillna("unclassified")
    order = {c: i for i, c in enumerate(["Advance", "Presence", "Count", "Yellow_Red", "Bike", "Mid", "Other"])}
    dead = pd.DataFrame({"signal": d.DeviceName, "detector": d.detector,
                         "phase": d.phase_target.astype("string").fillna(("P" + d.phase_timing.astype("string")))
                         .fillna("P" + d.phase_diagram.astype("Int64").astype("string") + " (print)"),
                         "in_timing": np.where(programmed[m], "yes", "no"),
                         "in_use_for": use, "atspm_function": use.isin(["Advance", "Presence", "Count", "Yellow_Red"]),
                         "technology": d.technology, "lane": d.lane_type, "print reading": d.confidence.fillna("low"),
                         "no actuations in": d.window_new.map(WIN_NAME),
                         "other live channels on the device": n_live_other[m].astype(int),
                         "last window with actuations": np.where(d.n_on_dec2024.fillna(0) > 0, "Dec 2024", "none")})
    dead = (dead.assign(_o=dead.in_use_for.map(order).fillna(99))
            .sort_values(["_o", "signal", "detector"]).drop(columns="_o").reset_index(drop=True))
    return dead, sorted(stale), dict(steps)


# ------------------------------------------------------------------ locked path + released signals (note 36)
LOCKED_EXTRA = ["set", "released_from_newtest", "func5_config"]


def locked_sets() -> tuple[set[str], set[str], set[str]]:
    """(TEST ids, NEWTEST ids, released NEWTEST ids), lower case. The locked store holds all 186 original hold-outs."""
    te = set(pd.read_csv(DC_WORK / "data/splits/test_config.csv").DeviceId.astype(str).str.lower())
    nt = set(pd.read_csv(DC_WORK / "official/newtest_signals.csv").DeviceId.astype(str).str.lower())
    rel = set(pd.read_csv(RELEASED).DeviceId.astype(str).str.lower()) if RELEASED.exists() else set()
    assert rel <= nt and not te & nt
    return te, nt, rel


def locked_now() -> set[str]:
    """The hold-out as it stands: locked_v2.csv once the split exists, else the two original files."""
    if LOCKED_V2.exists():
        return set(pd.read_csv(LOCKED_V2).DeviceId.astype(str).str.lower())
    te, nt, _ = locked_sets()
    return te | nt


def locked_rulings() -> pd.DataFrame:
    """The locked store's own rulings file, seeded once with the training file's general ('*') rulings."""
    if not RULINGS.exists():
        tr = pd.read_csv(TRAIN_RULINGS, dtype=str, keep_default_na=False).reindex(columns=RCOLS, fill_value="")
        tr[tr.DeviceName == "*"].to_csv(RULINGS, index=False)
        log(f"seeded {RULINGS} with {int((tr.DeviceName == '*').sum())} general rulings from the training store")
    return rulings()


def _pick(df: pd.DataFrame, high_ok: pd.Series) -> tuple[pd.Series, pd.Series]:
    """build_v3's label order: user ruling > print high > v2 > print medium > print low > whole-intersection Other."""
    pf, pc = df.print_function, df.print_confidence
    is_print = df.print_source.eq("print") & pf.notna()
    user = is_print & df.print_flags.fillna("").str.contains(r"\buser_ruling\b")
    wio = df.print_source.eq("data_only") & df.tier.isin(["complete_high", "complete_mixed"])
    lab = pd.Series(None, index=df.index, dtype="object")
    src = pd.Series(None, index=df.index, dtype="object")
    steps = [(user, pf, "user_ruling"), (is_print & pc.eq("high") & high_ok, pf, "print_high"),
             (df.func7_v2.notna(), df.func7_v2, None), (is_print & pc.eq("medium"), pf, "print_medium"),
             (is_print & pc.eq("low"), pf, "print_low"), (wio, pd.Series("Other", index=df.index), "whole_intersection_other")]
    for m, val, s in steps:
        m = m & lab.isna()
        lab[m] = val[m]
        src[m] = df.v2_source[m] if s is None else s
    return lab, src


def build_locked_table(pl: pd.DataFrame, tiers: pd.DataFrame) -> pd.DataFrame:
    """function_labels_locked_v1: every labelled / print detector of the 186 original hold-out signals. Labels only:
    no DQ, no OOF, no validation (validated = pending). label_print_agree has no model input here: where a high print
    label differs from v2, v2 stays in that variant."""
    te, nt, rel = locked_sets()
    v2 = pd.read_parquet(V2)
    v2["DeviceId"] = v2.DeviceId.str.lower()
    v2 = v2[v2.DeviceId.isin(te | nt)].copy()
    v2["func7_v2"] = v2_func7(v2)
    v2["v2_source"] = np.where(v2.label_source.str.startswith("review"), "review_round1", "config")
    v2 = v2.rename(columns={"func5": "func5_v2", "Detector": "detector"})[
        ["DeviceId", "DeviceName", "detector", "func5_v2", "func7_v2", "v2_source", "func5_config", "config_function",
         "description"]]
    p = pl.copy()
    p["DeviceId"] = p.DeviceId.str.lower()
    p = p.rename(columns={"function": "print_function", "subtype": "print_subtype", "confidence": "print_confidence",
                          "flags": "print_flags", "source": "print_source"})
    keep = ["DeviceId", "DeviceName", "detector", "print_source", "print_function", "print_subtype", "print_confidence",
            "confidence_reason", "technology", "phase_diagram", "lane_index", "lanes_spanned", "lane_type",
            "n_lanes_phase", "print_flags", "n_on_new", "window_new", "crop", "pending_user"]
    df = v2.merge(p[keep], on=["DeviceId", "detector"], how="outer", suffixes=("", "_p"))
    df["DeviceName"] = df.DeviceName.fillna(df.pop("DeviceName_p"))
    t = tiers.assign(DeviceId=tiers.DeviceId.str.lower())[["DeviceId", "tier", "unusual_layout"]]
    df = df.merge(t, on="DeviceId", how="left")
    df["unusual_layout"] = df.unusual_layout.astype("boolean").fillna(False).astype(bool)
    off = pd.read_parquet(DC_WORK / "official/labels_official.parquet",
                          columns=["DeviceId", "Detector", "target", "target_type", "switch_phase", "additional_call_phases"])
    off = off.rename(columns={"Detector": "detector", "target": "phase_target", "target_type": "phase_target_type"})
    df = df.merge(off.assign(DeviceId=off.DeviceId.str.lower()), on=["DeviceId", "detector"], how="left")
    pf = df.print_function
    high = df.print_source.eq("print") & pf.notna() & df.print_confidence.eq("high")
    v2ok = df.func7_v2.notna()
    df["label_print_first"], df["source"] = _pick(df, pd.Series(True, index=df.index))
    df["label_print_agree"], df["source_agree"] = _pick(df, ~v2ok | df.func7_v2.eq(pf))
    df["function"] = df.label_print_first
    df["print_overrides_v2"] = high & v2ok & df.func7_v2.ne(pf)
    df["dead"] = df.print_flags.fillna("").str.contains(r"\bdead\b|no_data")
    df["label_use"] = df.function.notna() & ~df.unusual_layout & ~df.dead
    df["set"] = np.where(df.DeviceId.isin(te), "TEST", "NEWTEST")
    df["released_from_newtest"] = df.DeviceId.isin(rel)
    df["validated"] = "pending"
    assert df.DeviceId.isin(te | nt).all(), "non-locked signal in the locked table"
    assert not df.duplicated(["DeviceId", "detector"]).any()
    cols = ["DeviceId", "DeviceName", "set", "released_from_newtest", "detector", "phase_target", "phase_target_type",
            "switch_phase", "additional_call_phases", "function", "source", "label_print_first", "label_print_agree",
            "source_agree", "print_overrides_v2", "func5_v2", "func7_v2", "v2_source", "func5_config", "print_source",
            "print_function", "print_subtype", "print_confidence", "confidence_reason", "technology", "phase_diagram",
            "lane_index", "lanes_spanned", "lane_type", "n_lanes_phase", "print_flags", "tier", "unusual_layout", "dead",
            "n_on_new", "window_new", "label_use", "pending_user", "config_function", "description", "crop", "validated"]
    df = df[cols].sort_values(["set", "DeviceName", "detector"]).reset_index(drop=True)
    for c in ("additional_call_phases", "config_function", "description", "crop", "confidence_reason", "validated"):
        df[c] = df[c].astype("string")
    df.to_parquet(LOCKED_V1, index=False)
    return df


def append_released() -> dict:
    """Append the released NEWTEST signals' rows of function_labels_locked_v1 to v3 (idempotent: earlier released rows
    are replaced). validated = pending here; the training path of main() then runs the label check over them (step 7).
    No OOF / DQ for them (dq_suspect NA)."""
    if not (RELEASED.exists() and LOCKED_V1.exists()):
        return {}
    _, _, rel = locked_sets()
    v3 = pd.read_parquet(V3)
    if "released_from_newtest" in v3:
        v3 = v3[~v3.released_from_newtest.astype(bool)]
    base = v3.assign(released_from_newtest=False).reset_index(drop=True)
    r = pd.read_parquet(LOCKED_V1)
    r = r[r.DeviceId.isin(rel)].copy()
    r["train_use"] = r.label_use
    r["oof_pred"], r["oof_p"], r["oof_agrees_print"] = None, np.nan, pd.NA
    r["dq_score"], r["dq_flags"], r["dq_suspect"] = np.nan, pd.NA, pd.NA
    r["validated"] = "pending"
    r["failed_checks"], r["validation_numbers"], r["field_issue"] = pd.NA, pd.NA, pd.NA
    r["validation_reason"] = "pending: released NEWTEST row, not yet through the label check (cab_final step 7)"
    r["train_use_validated"] = False
    r = r[list(base.columns)]
    for c in base.columns:
        try:
            r[c] = r[c].astype(base[c].dtype)
        except (TypeError, ValueError):
            pass
    out = pd.concat([base, r], ignore_index=True)
    assert not out.DeviceId.str.lower().isin(locked_now()).any(), "locked_v2 signal in v3"
    assert not out.duplicated(["DeviceId", "detector"]).any()
    out.to_parquet(V3, index=False)
    back = pd.read_parquet(V3)
    assert back.iloc[:len(base)].reset_index(drop=True).equals(base), "existing v3 rows changed"
    s = dict(released_signals=len(rel), released_rows=len(r), released_signals_with_rows=int(r.DeviceId.nunique()),
             released_labelled=int(r.function.notna().sum()), released_train_use=int(r.train_use.sum()),
             released_source=r.source.value_counts().to_dict(), released_function=r.function.value_counts().to_dict(),
             released_tier_signals=r.drop_duplicates("DeviceId").tier.fillna("no_print").value_counts().to_dict(),
             v3_rows=len(out))
    log(f"appended released NEWTEST rows to v3: {s}")
    return s


def locked_main(dry: bool):
    """Locked-store consolidation: label steps only (see the module docstring)."""
    nums, done, state = finished_batches()
    log(f"[locked] finished batches {nums} ({len(done)} signals); not finished: "
        f"{ {k: v for k, v in state.items() if k not in nums} }")
    spec = ",".join(map(str, nums))
    R = locked_rulings()
    ch = apply_unusual(R, done, not dry)
    ap_ = [] if dry else ["--apply"]
    fx = {"long_presence": run_fix("fix_long_presence.py", ["--batches", spec, "--skip", "", "--config-desc"] + ap_),
          "video_yr": run_fix("fix_video_yr.py", ["--batches", spec] + ap_),
          "radar_over_loops": run_fix("fix_radar_over_loops.py", ["--batches", spec] + ap_),
          "uncoded_zones": run_fix("fix_uncoded_zones.py", ["--batches", spec] + ap_)}
    for k, v in fx.items():
        log(f"{k}: {v.splitlines()[0]}")
    kp = {dn: set(g.detector.astype(int)) for dn, g in R[R.rule == "keep_print"].groupby("DeviceName")}
    pl, tiers = cab_build.build(only=done, keep_print=kp)
    pl, lab_r = apply_label_rulings(pl, R)
    pl.to_parquet(CAB / "print_labels.parquet", index=False)
    log(f"[locked] build: {len(tiers)} signals, {len(pl)} rows; label rulings {lab_r}")
    lk = build_locked_table(pl, tiers)
    pr = pl[pl.source == "print"]
    tt = tiers.assign(DeviceId=tiers.DeviceId.str.lower())
    _, _, rel = locked_sets()
    tt["group"] = np.where(tt.DeviceId.isin(rel), "released", "stays_locked")
    lab = lk[lk.function.notna()]
    summ = dict(
        finished_batches=nums, n_signals_print=len(tiers), table_signals=int(lk.DeviceId.nunique()),
        fixes={k: v.splitlines()[0] for k, v in fx.items()}, unusual_layout_ruling=ch, label_rulings=lab_r,
        tiers=tiers.groupby("tier").size().to_dict(),
        tiers_by_group=tt.groupby(["group", "tier"]).size().unstack(fill_value=0).to_dict("index"),
        n_unusual=int(tiers.unusual_layout.sum()), unusual=sorted(tiers[tiers.unusual_layout].DeviceName),
        print_detectors=len(pr),
        function_x_confidence=pd.crosstab(pr.function.fillna("unclassified"), pr.confidence.fillna("none")).to_dict("index"),
        whole_intersection_rows=pl[pl.source == "data_only"].subtype.value_counts().to_dict(),
        rows=len(lk), labelled=len(lab), label_use=int(lk.label_use.sum()),
        source=lab.source.value_counts().to_dict(), function=lab.function.value_counts().to_dict(),
        by_set={k: dict(rows=len(g), labelled=int(g.function.notna().sum()), label_use=int(g.label_use.sum()),
                        signals=int(g.DeviceId.nunique()))
                for k, g in lk.assign(k=lk.set + np.where(lk.released_from_newtest, "_released", "")).groupby("k")},
        print_overrides_v2=int(lk.print_overrides_v2.sum()), dead=int(lk.dead.sum()),
        pending_user=int(lk.pending_user.notna().sum()))
    (CAB / "final_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=str))


# ------------------------------------------------------------------ main
def review_only():
    """Rebuild review/print_label_overrides.xlsx and review/dead_detectors_from_prints.csv from the existing
    outputs (no record, label table or DQ is touched); refresh their keys in final_summary.json."""
    S = signals()
    pl = pd.read_parquet(CAB / "print_labels.parquet")
    v3 = pd.read_parquet(V3)
    ov, dead, stale, steps = user_lists(v3, pl)
    import label_check  # released NEWTEST rows are validated like every training row (2026-09-28)
    label_check.run(v3.drop(columns=[c for c in label_check.OUT_COLS + ["train_use_validated"] if c in v3]))
    fi = label_check.field_issues(label_check.run.last["d"], label_check.run.last["flags"], dead)
    lk = set(pd.read_csv(LOCKED_V2).DeviceName.astype(str)) if LOCKED_V2.exists() else set(S.DeviceName[S.locked])
    assert not fi.signal.isin(lk).any(), "locked signal in the field-issues list"
    assert not ov.Signal.isin(lk).any() and not dead.signal.isin(lk).any()
    f = CAB / "final_summary.json"
    summ = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    summ.update(overrides=len(ov), overrides_model_agrees=int(v3[v3.print_overrides_v2].oof_agrees_print.fillna(False).sum()),
                dead_for_maintenance=len(dead), dead_signals=int(dead.signal.nunique()),
                dead_not_in_timing=int((dead.in_timing == "no").sum()),
                dead_atspm_function=int(dead.atspm_function.sum()),
                dead_by_use=dead.in_use_for.value_counts().to_dict(), dead_filter_steps=steps,
                dead_list_skipped_stale_signals=stale)
    f.write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    log(f"overrides {len(ov)}; dead {len(dead)} ({summ['dead_not_in_timing']} not in timing); steps {steps}")
    print(ov.head(10).to_string(index=False))


# ------------------------------------------------------------------ 10. Dec-2024 role drift (note 51b)
DEC_CFG = DC_WORK / "data/raw/detector-configs.csv"   # config export of Dec 2024, contemporaneous with the Dec events
DEC_ROLE = CAB / "dec_role_changed.parquet"
DEC_VOCAB = {"Advance", "Presence", "Count"}          # the Dec export lists performance-measure detectors only


def _phase_set(r) -> set[int] | None:
    """Phases the current label accepts for a channel: call phase + switch phase + additional call phases (the
    A1 phase-scorer rule); None where the label has no timing phase (overlap / none) -> phase not comparable."""
    if r.phase_target_type != "phase" or not isinstance(r.phase_target, str) or not r.phase_target.startswith("P"):
        return None
    s = {int(r.phase_target[1:])}
    if pd.notna(r.switch_phase) and r.switch_phase > 0:
        s.add(int(r.switch_phase))
    for x in str(r.additional_call_phases or "").replace(";", ",").split(","):
        if x.strip().isdigit():
            s.add(int(x))
    return s


def dec_role_changed(v3: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    """Per labelled channel: does the Dec-2024 config export give it the same role as `label` (the current label)?
    Orchestrator rule 2026-09-30 (labels must describe the data they label): where the Dec config gives a DIFFERENT
    function or phase, or the channel is absent from it while the current label is a class the export lists
    (Advance / Presence / Count) at a signal the export covers, the channel's Dec-2024 rows are dropped from
    training AND scoring; the Sept-2026 rows keep the current label. Independent config evidence only - never the
    model's right / wrong pattern. dec_status: same | function | phase | absent | no_signal (export lacks the
    signal) | not_listed (current class outside the export's vocabulary, channel not listed = consistent)."""
    c = pd.read_csv(DEC_CFG)
    c["DeviceId"] = c.DeviceId.str.lower()
    assert not c.duplicated(["DeviceId", "Detector"]).any()
    c = c.rename(columns={"Detector": "detector", "Function": "dec_function", "Phase": "dec_phase"})
    x = v3[["DeviceId", "detector", "phase_target", "phase_target_type", "switch_phase", "additional_call_phases"]
           ].assign(DeviceId=v3.DeviceId.str.lower(), label=label.to_numpy())
    x = x.merge(c[["DeviceId", "detector", "dec_function", "dec_phase"]], on=["DeviceId", "detector"], how="left")
    ps = x.apply(_phase_set, axis=1)
    listed = x.dec_function.notna()
    f_diff = listed & x.label.ne(x.dec_function).fillna(True)
    p_diff = listed & pd.Series([s is not None and int(p) not in s if pd.notna(p) else False
                                 for s, p in zip(ps, x.dec_phase)], index=x.index)
    sig_in = x.DeviceId.isin(set(c.DeviceId))
    absent = ~listed & sig_in & x.label.isin(DEC_VOCAB)
    x["dec_status"] = np.select([x.label.isna(), ~sig_in, f_diff, p_diff, absent, listed],
                                ["unlabelled", "no_signal", "function", "phase", "absent", "same"], "not_listed")
    x["dec_role_changed"] = x.dec_status.isin(["function", "phase", "absent"]) & x.label.notna()
    return x[["DeviceId", "detector", "label", "dec_function", "dec_phase", "dec_status", "dec_role_changed"]]


def dec_role_step(v3: pd.DataFrame) -> dict:
    """Write DEC_ROLE (per channel, two flags: vs the training label `function` and vs the scoring truth of the
    honest set - user ruling > high print > config label where there is no print; notes 45 / 49)."""
    pf = v3.print_function
    user = v3.source.astype("string").eq("user_ruling").fillna(False)
    high = v3.print_source.eq("print") & v3.print_confidence.eq("high") & pf.isin(CLASSES7)
    noprint = ~v3.print_source.eq("print").fillna(False) | pf.isna()
    truth = pd.Series(np.where(user, v3.function, np.where(high, pf, np.where(noprint & v3.func7_v2.notna(),
                                                                                v3.func7_v2, None))), index=v3.index)
    a = dec_role_changed(v3, v3.function)
    b = dec_role_changed(v3, truth)
    out = a.rename(columns={"label": "label_train", "dec_status": "dec_status_train"}).drop(columns="dec_role_changed")
    out["dec_role_changed_train"] = a.dec_role_changed.to_numpy()
    out["label_truth"], out["dec_status_truth"] = b.label.to_numpy(), b.dec_status.to_numpy()
    out["dec_role_changed_truth"] = b.dec_role_changed.to_numpy()
    assert not out.DeviceId.isin(locked_now()).any(), "locked_v2 signal in dec_role table"
    out.to_parquet(DEC_ROLE, index=False)
    s = {"train": out.dec_status_train.value_counts().to_dict(), "truth": out.dec_status_truth.value_counts().to_dict(),
         "changed_train": int(out.dec_role_changed_train.sum()), "changed_truth": int(out.dec_role_changed_truth.sum())}
    log(f"dec_role_changed -> {DEC_ROLE.name}: {s}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="fixes / rulings not written to the records")
    ap.add_argument("--skip-dq", action="store_true", help="reuse cabinet/dq_print.parquet")
    ap.add_argument("--review-only", action="store_true",
                    help="rebuild only the two review/ lists from the saved print_labels.parquet and v3 table")
    ap.add_argument("--locked-consolidate", action="store_true",
                    help="label-only consolidation of the locked store (needs --locked-labels-only; note 36)")
    ap.add_argument("--append-released", action="store_true",
                    help="only append the released NEWTEST rows of function_labels_locked_v1 to v3")
    ap.add_argument("--dec-role", action="store_true", dest="dec_role",
                    help="only (re)write cabinet/dec_role_changed.parquet from the saved v3 table (note 51b)")
    a = ap.parse_args()
    if a.locked_consolidate:
        return locked_main(a.dry_run)
    if a.append_released:
        return print(json.dumps(append_released(), indent=1, default=str))
    if a.review_only:
        return review_only()
    if a.dec_role:
        return print(json.dumps(dec_role_step(pd.read_parquet(V3)), indent=1, default=str))
    S = signals()
    locked = set(S.DeviceId[S.locked].str.lower())  # + ids the hold-out files list but the plans table lacks
    if STORE:  # note 80: the hold-out as it stands (locked_v2); released NEWTEST records are ordinary training records
        locked |= locked_now()
        _, _, rel_ids = locked_sets()
        assert not locked & rel_ids and len(locked) == 115, "store mode: hold-out must be exactly locked_v2"
    else:
        for f in (DC_WORK / "data/splits/test_config.csv", DC_WORK / "official/newtest_signals.csv"):
            locked |= set(pd.read_csv(f).DeviceId.astype(str).str.lower())
    nums, done, state = finished_batches()
    log(f"finished batches {nums[0]:02d}..{nums[-1]:02d} ({len(nums)}), {len(done)} signals; "
        f"not finished: { {k: v for k, v in state.items() if k not in nums} }")
    spec = ",".join(map(str, nums))
    R = rulings()
    ch = apply_unusual(R, done, not a.dry_run)
    log(f"unusual_layout ruling applied to {ch}")
    ap_ = [] if a.dry_run else ["--apply"]
    s1 = run_fix("fix_long_presence.py", ["--batches", spec, "--skip", "", "--config-desc"] + ap_)
    s2 = run_fix("fix_video_yr.py", ["--batches", spec] + ap_)
    s3 = run_fix("fix_radar_over_loops.py", ["--batches", spec] + ap_)
    log("fix_long_presence: " + s1.splitlines()[0] + " | " + next((x.strip() for x in s1.splitlines() if "config description only" in x), ""))
    log("fix_video_yr: " + s2.splitlines()[0])
    log("fix_radar_over_loops: " + s3.splitlines()[0])
    kp = {dn: set(g.detector.astype(int)) for dn, g in R[R.rule == "keep_print"].groupby("DeviceName")}
    pl, tiers = cab_build.build(only=done, keep_print=kp)
    for dn, dets in kp.items():
        if dn in done:
            x = pl[(pl.DeviceName == dn) & pl.detector.isin(dets)]
            assert (x.source == "print").all() and len(x) == len(dets), f"{dn} keep_print failed"
    log(f"build: {len(tiers)} signals, {len(pl)} rows")
    pl, lab_r = apply_label_rulings(pl, R)
    log(f"label rulings: {lab_r}")
    pl = dq_print(pl, a.skip_dq)
    log(f"dq: {int(pl.dq_score.notna().sum())} scored, {int(pl.dq_suspect.astype("boolean").fillna(False).sum())} suspect")
    v3, has_oof = build_v3(pl, tiers, locked)
    for d in (pl, tiers, v3):
        assert not d.DeviceId.str.lower().isin(locked).any()
    # step 9 before step 7 (2026-09-28): the released NEWTEST rows are training rows now, so the label check
    # validates them like every other row. The locked_v2 signals never enter (asserted in append_released).
    if STORE:  # released records are in the store: no append from locked_v1 (never read in this mode)
        _, _, rel_ids = locked_sets()
        v3["released_from_newtest"] = v3.DeviceId.str.lower().isin(rel_ids)
        v3.to_parquet(V3, index=False)
        released = dict(released_signals=len(rel_ids), released_rows=int(v3.released_from_newtest.sum()),
                        released_signals_with_rows=int(v3[v3.released_from_newtest].DeviceId.nunique()))
    else:
        released = append_released()
    v3 = pd.read_parquet(V3)
    v3, lcheck = label_check_step(v3)
    rel_m = v3.released_from_newtest.astype(bool)
    rl = v3[rel_m & v3.function.notna()]
    released["validated"] = rl.validated.value_counts(dropna=False).to_dict()
    released["train_use_validated"] = int(v3[rel_m].train_use_validated.sum())
    log(f"released NEWTEST rows validated: {released['validated']}")
    lk_names = set(pd.read_csv(LOCKED_V2).DeviceName.astype(str)) if LOCKED_V2.exists() else set(S.DeviceName[S.locked])
    assert not v3.DeviceId.str.lower().isin(locked_now()).any(), "locked_v2 signal in v3"
    # overrides / dead lists: training-store prints only (the released signals' prints are in the locked review list)
    ov, dead, stale, dead_steps = user_lists((v3 if STORE else v3[~rel_m]).drop(columns="released_from_newtest"), pl)
    import label_check  # review/field_issues_for_staff.xlsx (engineer, 2026-09-28): after the dead list exists
    fi = label_check.field_issues(label_check.run.last["d"], label_check.run.last["flags"], dead)
    assert not fi.signal.isin(lk_names).any(), "locked signal in the field-issues list"
    lcheck["field_issues_rows"], lcheck["field_issues_signals"] = len(fi), int(fi.signal.nunique())
    assert not ov.Signal.isin(set(S.DeviceName[S.locked])).any() and not dead.signal.isin(set(S.DeviceName[S.locked])).any()

    pr = pl[pl.source == "print"]
    summ = dict(
        finished_batches=nums, n_signals=len(tiers),
        fixes={"long_presence": s1.splitlines()[0], "video_yr": s2.splitlines()[0], "radar_over_loops": s3.splitlines()[0],
               "unusual_layout_ruling": ch, "label_rulings": lab_r},
        tiers=tiers.groupby("tier").size().to_dict(),
        tiers_unusual=tiers[tiers.unusual_layout].groupby("tier").size().to_dict(),
        n_unusual=int(tiers.unusual_layout.sum()),
        print_detectors=len(pr),
        function_x_confidence=pd.crosstab(pr.function.fillna("unclassified"), pr.confidence.fillna("none")).to_dict("index"),
        whole_intersection_rows=pl[pl.source == "data_only"].subtype.value_counts().to_dict(),
        dq_suspect_print=int(pr.dq_suspect.astype("boolean").fillna(False).sum()),
        overrides=len(ov), overrides_model_agrees=int(v3[v3.print_overrides_v2].oof_agrees_print.fillna(False).sum()),
        overrides_by_pair={f"{a}->{b}": int(n) for (a, b), n in
                           ov.groupby(["Config label", "Print label"]).size().sort_values(ascending=False).items()},
        dead_for_maintenance=len(dead), dead_signals=int(dead.signal.nunique()),
        dead_atspm_function=int(dead.atspm_function.sum()),
        dead_by_use=dead.in_use_for.value_counts().to_dict(), dead_filter_steps=dead_steps,
        pending_user=int(pl.pending_user.notna().sum()), dead_list_skipped_stale_signals=stale, dead_all_print=int(pr["flags"].fillna("").str.contains(r"\bdead\b").sum()),
        v3_rows=len(v3), v3_signals=int(v3.DeviceId.nunique()), oof_available=has_oof,
        v3_source=v3.source.value_counts().to_dict(), v3_source_agree=v3.source_agree.value_counts().to_dict(),
        v3_function=v3.function.value_counts().to_dict(),
        v3_train_use=int(v3.train_use.sum()),
        v3_first_vs_agree_differ=int((v3.label_print_first.fillna("-") != v3.label_print_agree.fillna("-")).sum()),
        label_check=lcheck, released_newtest=released, dec_role_changed=dec_role_step(v3))
    (CAB / "final_summary.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
