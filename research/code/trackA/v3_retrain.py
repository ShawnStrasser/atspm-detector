"""Note 28 -- function head retrained on the cabinet-print labels (label table v3).

One command for the final run (after the open user questions are answered and
`cab_final.py` has been re-run, so `pending_user` is empty):

    python v3_retrain.py all                          # prep + fit the grid (6 folds x 3 seeds) + report

Pieces:
    python v3_retrain.py prep                         # cache the feature frame (label-independent)
    python v3_retrain.py fit --variants grid --seeds 0,1,2 [--folds 0,1,2,3,4,5] [--threads 12]
    python v3_retrain.py report [--allow-partial]     # score every saved OOF + the note-25 baseline
    --frame v5 (default) | v6 | v4: v6 = v5 + the released NEWTEST half (note 36, folds_v4.csv); v5 = funcframe_v5 + feat_expert_*_v5 + folds_v3.csv (every v3
    signal with data, `v5_frame.py`); v4 = the original 399-signal frame + folds.csv/folds_v4.
    v5 writes its cache / predictions under trackA/v3/frame_v5/ so v4 runs are never mixed in.
    Row filters (training AND scoring; each adds a run-directory suffix; counts in report.json
    coverage.excluded): --min-on 5 (production --min-actuations), --clean (dq_suspect detectors
    out; EXTRA_UNUSUAL signals -> unusual_layout), --require-validated (v3 `validated` == pass),
    --allow-not-checkable-high (+ not_checkable rows with a high print label, not dead /
    card-suspect, no field_issue; run suffix _valnc).

A variant is `label.train.wi`:
    label  first = label_print_first | agree = label_print_agree | v2 = func7_v2 (v2 labels
           retrained in this harness, the same-code baseline)
    train  high = complete_high signals | mixed = complete_high + complete_mixed |
           all = + incomplete + no-print signals (config labels)
    wi     wi = whole-intersection Other rows in training | nowi = left out
unusual_layout signals are never trained on (train_use) and are scored apart. `--pending
exclude` (default) drops rows with a non-null `pending_user` from training AND scoring.

Model: note 14 T head (production 388 + 54 expert features, same LightGBM params), 7 classes
(A, P, C, YR, Mid, Bike, Other), all window lengths of the frame, both periods, six
signal-grouped folds, inner fold (k+1) % 6 for early stopping. Fixed evaluation sets, identical
for every variant (all out-of-fold):
    FIX  complete_high signals, print label at high confidence (the headline set)
    V2   the note-14 set: every v2-labelled frame row, 5 classes (continuity)
    WI   whole-intersection Other rows (Sept-2026 period, where "active" was judged)
    UNU  unusual_layout signals, label_print_first
Baseline: the saved note-25 b7 OOF (3 seeds; = the current head with Mid/Bike) on the same rows.
Locked signals (locked_v2.csv once it exists, note 36) are asserted absent from the frame and the label table.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

import a2_model as A2

REPO = A2.REPO
DCW = A2.DCW
OUT = DCW / "trackA" / "v3"
FEATS = OUT / "feat_frame.parquet"
LABELS_V3 = REPO / "research" / "labels" / "function_labels_v3.parquet"
LABELS_V2 = A2.LABELS
FOLDS = DCW / "folds.csv"
FRAMES = {"v4": (A2.FRAME, "", DCW / "folds.csv"),
          "v5": (DCW / "function_v4" / "funcframe_v5.parquet", "_v5", DCW / "folds_v3.csv"),
          "v6": (DCW / "function_v4" / "funcframe_v6.parquet", "_v6", DCW / "folds_v4.csv"),   # note 36
          # note 49: v6 with the phase-derived columns rebuilt from the phase_v3 blend OOF (v6e_frame.py)
          "v6e": (DCW / "function_v4" / "funcframe_v6e.parquet", "_v6e", DCW / "folds_v4.csv"),
          "v6t": (DCW / "function_v4" / "funcframe_v6t.parquet", "_v6t", DCW / "folds_v4.csv")}  # trees-only phase_v3
FRAME, EXP_SFX = A2.FRAME, ""
MIN_ON = 0   # --min-on: detector-windows with fewer actuations are neither trained nor scored
CLEAN = False  # --clean: dq_suspect detectors neither trained nor scored (reported apart), and
#               the signals below treated as unusual_layout (flagged in a record note only)
EXTRA_UNUSUAL = {"01064", "04073", "10086", "12032", "13025", "2C023", "2C042"}  # DeviceName
REQ_VAL = False  # --require-validated: only rows whose v3 `validated` == "pass" are trained / scored
NC_HIGH = False  # --allow-not-checkable-high (with --require-validated): also `not_checkable` rows whose label is a
#                  high-confidence print label (source print_high), not dead, not on a suspect card (note 34),
#                  no field_issue (orchestrator 2026-09-28)
CARDS = DCW / "cabinet" / "card_channels.parquet"
LC_FEATS = False   # --lc-feats: + the label-check statistics as features (lc_features.py, note 45); run suffix _lc
LC_SHUFFLE = False  # --lc-shuffle: those columns permuted across rows (noise-column control); run suffix _lcshuf
LC_DIR = DCW / "trackA" / "lc"
LC_SEED = 20260929
PM_FEATS = False   # --pm-feats: + permissive-phase features and the OOF phase score (pm_features.py / pm_score.py,
#                    note 50); run suffix _pm
PM_SHUFFLE = False  # --pm-shuffle: those columns permuted across rows (control); run suffix _pmshuf
PM_DIR = DCW / "trackA" / "pm"
PM_COLS = ["pm_score", "pm_ph_leave_sb", "pm_ph_leave_wmean", "pm_ph_drop43", "pm_det_leave", "pm_det_leave_minus_sb",
           "pm_det_is_sb", "pm_det_wait_share", "pm_score_other_max"]
HEALTH3 = False    # --health3 (orchestrator 2026-09-29): training rows dropped where the detector is `bad` in that
#                    period (health v3, note 43), or `suspect` with a listed bad period overlapping the window; run
#                    suffix _h3. Scoring rows are NOT changed (same evaluation sets as without).
HEALTH3_FILE = DCW / "health" / "health_v3.parquet"
DEC_ROLE = False   # --dec-role (orchestrator 2026-09-30, note 51b): Dec-2024 rows of channels whose Dec-2024 config
#                    export gives a different function / phase than the current label (or omits an A / P / C channel)
#                    are dropped from training AND scoring (cab_final.py dec_role_step); run suffix _dr
DEC_ROLE_FILE = DCW / "cabinet" / "dec_role_changed.parquet"
DEC_RULE = "all"   # --dec-role-rule fp (orchestrator 2026-09-30, note 54): only the function / phase-differs statuses,
#                    not 'absent'; run suffix _drfp
LABEL_SETS = {"v3": (REPO / "research" / "labels" / "function_labels_v3.parquet", DCW / "cabinet" / "dec_role_changed.parquet"),
              # note 54: v3 + stacked-group relabel + YR==Count exclusion + radar-over-loops signals re-admitted
              # (cabinet/stack_labels_v3s.py); its Dec-role table is recomputed on the v3s labels
              # note 81 (orchestrator 2026-10-03): the "v3s" slot now carries the v4l table (note 80 + note 81
              # rules / user corrections) so every trainer that asks for "v3s" trains on v4l; the old table is "v3s_orig"
              # note 89 (orchestrator 2026-10-04): the default "v3s" slot now carries v4o (final training labels =
              # v4n + not_checkable restored); the v4l table is "v4l" below
              # note 95 (2026-10-05): the default "v3s" slot carries v4q (v4p + rulings of 2026-10-05 16:20 + 3 prints)
              "v3s": (REPO / "research" / "labels" / "function_labels_v4q.parquet",
                      DCW / "lab95" / "dec_role_changed_v4q.parquet"),
              "v4l": (REPO / "research" / "labels" / "function_labels_v4l.parquet",
                      DCW / "cabinet_v4l" / "dec_role_changed_v4l.parquet"),
              # note 88: v4l with dq_suspect recomputed WITHOUT detector fault events 83-88 (final88/dq88.py)
              "v4m": (REPO / "research" / "labels" / "function_labels_v4m.parquet",
                      DCW / "cabinet_v4l" / "dec_role_changed_v4l.parquet"),
              # note 89: v4m with the eight tested exclusion groups restored to training (g89_groups.py labels)
              "v4n": (REPO / "research" / "labels" / "function_labels_v4n.parquet",
                      DCW / "x89" / "dec_role_changed_v4n.parquet"),
              "v4o": (REPO / "research" / "labels" / "function_labels_v4o.parquet",
                      DCW / "x89" / "dec_role_changed_v4o.parquet"),
              # note 94: v4o + the user's review-v1 answers, the 05999 print, print-vs-hand rule N1, dummy 40/41 (N2)
              "v4p": (REPO / "research" / "labels" / "function_labels_v4p.parquet",
                      DCW / "lab93" / "dec_role_changed_v4p.parquet"),
              # note 95: v4p + loop 'advance presence' = Advance, channel text vs Function (model agreement), 3 prints
              "v4q": (REPO / "research" / "labels" / "function_labels_v4q.parquet",
                      DCW / "lab95" / "dec_role_changed_v4q.parquet"),
              "v3s_orig": (REPO / "research" / "labels" / "function_labels_v3s.parquet",
                           DCW / "cabinet" / "dec_role_changed_v3s.parquet")}
G89_TRAIN_COLS = ("validated", "dq_suspect", "exclude_train_score", "train_use", "unusual_layout", "label_print_first",
                  "tier")
BASE_DIR = DCW / "trackA" / "w1"         # note 25 b7 OOF (Praw_b7_s*.npy + keys.parquet)
LOCKED = [DCW / "data" / "splits" / "test_config.csv",
          DCW / "official" / "newtest_signals.csv"]
LOCKED_V2 = DCW / "official" / "locked_v2.csv"   # note 36: 43 TEST + the NEWTEST half that stays locked
KEY = ["DeviceId", "Detector", "period", "win"]
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
C5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
CORE4 = ["Advance", "Presence", "Count", "Yellow_Red"]
APC = ["Advance", "Presence", "Count"]
TO5 = {"Mid": "Other", "Bike": "Other"}
WGROUPS = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]
GRID = ["first.high.wi", "first.high.nowi", "first.mixed.wi", "first.mixed.nowi",
        "agree.high.wi", "agree.high.nowi", "agree.mixed.wi", "agree.mixed.nowi",
        "first.all.wi", "v2.all.nowi"]
TIERS = {"high": {"complete_high"}, "mixed": {"complete_high", "complete_mixed"},
         "all": {"complete_high", "complete_mixed", "incomplete", None}}
BOOK = {"fold", "health_flag", "wgroup", "is_full", "pred_phase", "period_n_on"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def set_frame(tag: str) -> None:
    """Point the harness at frame v4 or v5 (paths only)."""
    global FRAME, EXP_SFX, FOLDS, OUT, FEATS
    FRAME, EXP_SFX, FOLDS = FRAMES[tag]
    if tag != "v4":
        OUT = DCW / "trackA" / "v3" / f"frame_{tag}"
        FEATS = OUT / "feat_frame.parquet"


def run_dir(pending: str) -> Path:
    """Predictions live under a fingerprint of the label table + pending mode, so a re-built
    v3 table (user answers folded in) never reuses stale fold files."""
    import hashlib
    h = hashlib.sha1(LABELS_V3.read_bytes()).hexdigest()[:10]
    return OUT / (f"run_{h}_{pending}" + (f"_min{MIN_ON}" if MIN_ON else "")
                  + ("_clean" if CLEAN else "") + ("_val" if REQ_VAL else "")
                  + ("nc" if REQ_VAL and NC_HIGH else "")
                  + ("_h3" if HEALTH3 else "")
                  + (("_drfp" if DEC_RULE == "fp" else "_dr") if DEC_ROLE else "")
                  + ("_lcshuf" if LC_FEATS and LC_SHUFFLE else "_lc" if LC_FEATS else "")
                  + ("_pmshuf" if PM_FEATS and PM_SHUFFLE else "_pm" if PM_FEATS else ""))


def locked_signals() -> set:
    """The hold-out: locked_v2.csv once the NEWTEST split exists (note 36), else the two original files."""
    if LOCKED_V2.exists():
        return set(pd.read_csv(LOCKED_V2).DeviceId.astype(str).str.lower())
    s = set()
    for f in LOCKED:
        s |= set(pd.read_csv(f).DeviceId.astype(str).str.lower())
    return s


# ------------------------------------------------------------------ prep: features only
def stage_prep(a) -> None:
    t0 = time.time()
    fr = pd.read_parquet(FRAME)
    fr["DeviceId"] = fr.DeviceId.str.lower()
    lock = locked_signals()
    assert not fr.DeviceId.isin(lock).any(), "locked signal in the frame"
    # folds: dc_work/folds.csv where it has the signal (asserted equal to the frame's own
    # fold), else the frame's folds_v4 fold -- the two agree on every shared signal
    own = fr[fr.fold.notna()].groupby("DeviceId").fold.first().astype(int)
    fc = pd.read_csv(FOLDS)
    fc = fc.assign(DeviceId=fc.DeviceId.str.lower()).set_index("DeviceId").fold
    both = own.index.intersection(fc.index)
    assert (own[both] == fc[both]).all(), "folds.csv disagrees with the frame's folds"
    fold = pd.concat([fc, own[~own.index.isin(fc.index)]])
    fr["fold"] = fr.DeviceId.map(fold)
    fr = fr[fr.fold.notna()].reset_index(drop=True)
    fr["fold"] = fr.fold.astype(int)
    for tag in ("det", "cyc", "pair"):
        e = pd.concat([pd.read_parquet(A2.WORK / f"feat_expert_{tag}_{p}{EXP_SFX}.parquet")
                       for p in ("dec", "stg")], ignore_index=True)
        e["DeviceId"] = e.DeviceId.str.lower()
        e["Detector"] = e.Detector.astype(fr.Detector.dtype)
        fr = fr.merge(e, on=KEY, how="left")
    cols = [c for c in A2.feat_cols(fr, A2.ALL_FAM) if c not in BOOK]
    assert len(cols) == 442, len(cols)
    assert not fr.duplicated(KEY).any()
    keep = KEY + ["wgroup", "fold", "health_flag", "pred_phase"] + cols
    fr = fr[keep]
    OUT.mkdir(parents=True, exist_ok=True)
    fr.to_parquet(FEATS, index=False)
    json.dump({"rows": len(fr), "signals": int(fr.DeviceId.nunique()), "features": cols,
               "frame": str(FRAME), "folds": str(FOLDS),
               "built": time.strftime("%Y-%m-%d %H:%M")}, open(OUT / "feat_meta.json", "w"),
              indent=1)
    log(f"feature frame {fr.shape}, {fr.DeviceId.nunique()} signals -> {FEATS} "
        f"({time.time()-t0:.0f}s)")


def load_feats() -> tuple[pd.DataFrame, list[str]]:
    meta = json.load(open(OUT / "feat_meta.json"))
    fr = pd.read_parquet(FEATS)
    assert len(fr) == meta["rows"]
    cols = list(meta["features"])
    if LC_FEATS:
        lc = pd.concat([pd.read_parquet(LC_DIR / f"feat_lc_{p}.parquet") for p in ("dec", "stg")], ignore_index=True)
        lc["DeviceId"] = lc.DeviceId.str.lower()
        lc["Detector"] = lc.Detector.astype(fr.Detector.dtype)
        assert not lc.duplicated(KEY).any()
        new = [c for c in lc.columns if c.startswith("lc_")]
        fr = fr.merge(lc[KEY + new], on=KEY, how="left")
        assert len(fr) == meta["rows"]
        if LC_SHUFFLE:
            rng = np.random.default_rng(LC_SEED)
            for c in new:
                fr[c] = fr[c].to_numpy()[rng.permutation(len(fr))]
        log(f"+ {len(new)} label-check features{' (SHUFFLED control)' if LC_SHUFFLE else ''}; rows with any: "
            f"{int(fr[new].notna().any(axis=1).sum()):,} of {len(fr):,}")
        cols += new
    if PM_FEATS:
        fr = add_pm(fr, meta["rows"])
        cols += PM_COLS
    return fr, cols


def add_pm(fr: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """note 50: the detector's permissive relation features + its PREDICTED phase's features and OOF score."""
    d = pd.concat([pd.read_parquet(PM_DIR / f"pm_det_{p}.parquet") for p in ("dec", "stg")], ignore_index=True)
    ph = pd.concat([pd.read_parquet(PM_DIR / f"pm_ph_{p}.parquet") for p in ("dec", "stg")], ignore_index=True)
    sc = pd.read_parquet(PM_DIR / "pm_score.parquet")
    ph = ph.merge(sc, on=["DeviceId", "period", "win", "p"], how="left")
    for t in (d, ph):
        t["DeviceId"] = t.DeviceId.str.lower()
        t["p"] = t.p.astype(int)
    d["Detector"] = d.Detector.astype(fr.Detector.dtype)
    # the largest score among the window's OTHER candidate phases (a right-turn lane sits on a through phase)
    g = ph.groupby(["DeviceId", "period", "win"]).pm_score
    top = ph.assign(r=g.rank(ascending=False, method="first"))
    m1 = top[top.r == 1].set_index(["DeviceId", "period", "win"])[["p", "pm_score"]]
    m2 = top[top.r == 2].set_index(["DeviceId", "period", "win"]).pm_score
    ph = ph.join(m1.rename(columns={"p": "p1", "pm_score": "s1"}), on=["DeviceId", "period", "win"])
    ph = ph.join(m2.rename("s2"), on=["DeviceId", "period", "win"])
    ph["pm_score_other_max"] = np.where(ph.p == ph.p1, ph.s2, ph.s1)
    x = fr[KEY + ["pred_phase"]].rename(columns={"pred_phase": "p"}).astype({"p": int})
    x = x.merge(d[KEY + ["pm_det_leave", "pm_det_leave_minus_sb", "pm_det_is_sb", "pm_det_wait_share"]], on=KEY,
                how="left")
    x = x.merge(ph[["DeviceId", "period", "win", "p", "pm_score", "pm_ph_leave_sb", "pm_ph_leave_wmean",
                    "pm_ph_drop43", "pm_score_other_max"]], on=["DeviceId", "period", "win", "p"], how="left")
    assert len(x) == len(fr) == n_rows
    if PM_SHUFFLE:
        rng = np.random.default_rng(LC_SEED + 48)
        for c in PM_COLS:
            x[c] = x[c].to_numpy()[rng.permutation(len(x))]
    for c in PM_COLS:
        fr[c] = x[c].to_numpy(np.float32)
    log(f"+ {len(PM_COLS)} permissive features{' (SHUFFLED control)' if PM_SHUFFLE else ''}; non-null "
        f"{ {c: int(fr[c].notna().sum()) for c in PM_COLS} }")
    return fr


# ------------------------------------------------------------------ labels, per row
def load_labels(fr: pd.DataFrame, pending: str) -> pd.DataFrame:
    """Label columns aligned to the frame rows (same order). v3 is per detector -> both periods."""
    v3 = pd.read_parquet(LABELS_V3)
    v3["DeviceId"] = v3.DeviceId.str.lower()
    assert not v3.DeviceId.isin(locked_signals()).any(), "locked signal in label table v3"
    if "g89_restored" in v3:             # v4n+ (note 89): TRAINING values of the restored exclusion groups live in
        for c in G89_TRAIN_COLS:         # <col>_train; the plain columns keep the v4m values (scoring sets unchanged)
            v3[c] = v3[f"{c}_train"]
        log(f"g89 training overrides: {int(v3.g89_restored.notna().sum())} label rows restored")
    v3 = v3.rename(columns={"detector": "Detector"})
    v3["Detector"] = v3.Detector.astype(fr.Detector.dtype)
    v3["tier"] = v3.tier.where(v3.tier.notna(), None)
    v3["pend"] = v3.pending_user.notna()
    if pending == "include":
        v3["pend"] = False
    if "exclude_train_score" in v3:      # v3s (note 54): YR identical to a Count zone, stacked groups answered '?'
        log(f"exclude_train_score: {int(v3.exclude_train_score.sum())} label rows out of training AND scoring")
        v3["pend"] |= v3.exclude_train_score.eq(True)
    v3["dq"] = v3.dq_suspect.eq(True) if CLEAN else False
    if REQ_VAL:
        assert "validated" in v3.columns, "--require-validated: no `validated` column in v3"
        v3["notval"] = ~v3.validated.astype(str).str.lower().eq("pass")
        if NC_HIGH:
            v3["notval"] &= ~nc_high_mask(v3)
    else:
        v3["notval"] = False
    if CLEAN:
        extra = v3.DeviceName.astype(str).str.upper().isin(EXTRA_UNUSUAL)
        assert v3.loc[extra, "DeviceName"].nunique() == len(EXTRA_UNUSUAL), "extra unusual missing"
        if "readmit_radar_over_loops" in v3:   # v3s: radar-over-loops signals are re-admitted (note 54)
            extra &= ~v3.readmit_radar_over_loops.eq(True)
        if "g89_restored" in v3:               # v4n (note 89): the unusual-signal group is trained on
            extra &= ~v3.g89_restored.astype("string").fillna("").str.contains("unusual_signal")
        v3.loc[extra, "unusual_layout"] = True
        v3.loc[extra, "train_use"] = False
    lab = fr[KEY].merge(v3[["DeviceId", "Detector", "label_print_first", "label_print_agree",
                            "source", "source_agree", "func7_v2", "tier", "unusual_layout",
                            "train_use", "dead", "pend", "dq", "notval", "print_function",
                            "print_confidence"]], on=["DeviceId", "Detector"], how="left")
    if REQ_VAL:                          # frame rows with no v3 row are not validated either
        lab["notval"] = lab.notval.ne(False)
    v2 = pd.read_parquet(LABELS_V2)[["DeviceId", "Detector", "func5", "drop_from_use"]]
    v2["DeviceId"] = v2.DeviceId.str.lower()
    v2["Detector"] = v2.Detector.astype(fr.Detector.dtype)
    v2 = v2[v2.func5.notna() & ~v2.drop_from_use.fillna(False).astype(bool)]
    lab = lab.merge(v2[["DeviceId", "Detector", "func5"]], on=["DeviceId", "Detector"],
                    how="left").rename(columns={"func5": "func5_v2t"})
    assert len(lab) == len(fr)
    lab["decchg_train"] = dec_role_rows(fr, "train") if DEC_ROLE else False
    lab["decchg_truth"] = dec_role_rows(fr, "truth") if DEC_ROLE else False
    for c in ("unusual_layout", "train_use", "dead", "pend", "dq", "notval", "decchg_train", "decchg_truth"):
        lab[c] = lab[c].eq(True).to_numpy(dtype=bool, na_value=False)
    return lab


def dec_role_rows(fr: pd.DataFrame, which: str = "truth") -> np.ndarray:
    """Frame rows in the Dec-2024 period of a channel flagged dec_role_changed_{which} (train = vs the training label,
    truth = vs the honest-set scoring truth); cab_final.py --dec-role writes the table (note 51b)."""
    d = pd.read_parquet(DEC_ROLE_FILE, columns=["DeviceId", "detector", f"dec_role_changed_{which}",
                                                f"dec_status_{which}"])
    if DEC_RULE == "fp":
        d = d[d[f"dec_status_{which}"].isin(["function", "phase"])]
    else:
        d = d[d[f"dec_role_changed_{which}"]]
    bad = set(zip(d.DeviceId.str.lower(), d.detector.astype(int)))
    k = zip(fr.DeviceId.str.lower(), fr.Detector.astype(int))
    return np.fromiter((x in bad for x in k), bool, len(fr)) & (fr.period == "dec").to_numpy()


def nc_high_mask(v3: pd.DataFrame) -> pd.Series:
    """not_checkable rows admitted by --allow-not-checkable-high: high-confidence print label, not dead,
    not on a card whose two outputs are both dead / erratic (dq_suspect goes separately via --clean)."""
    c = pd.read_parquet(CARDS, columns=["DeviceName", "detector", "card_suspect", "card_mate", "dead", "reasons"])
    # note 88: a card counts only when both outputs are dead or both erratic BY ACTUATIONS (stuck-on / chatter);
    # vd_audit's card_suspect also called a card erratic on detector fault events 83-88 (user ban 2026-09-28)
    act = dict(zip(zip(c.DeviceName.astype(str), c.detector.astype(int)),
                   c.reasons.fillna("").str.contains("stuck-on|chatter")))
    cs = c[c.card_suspect.eq(True)]
    ok = [bool(d) or (act.get((str(n), int(k)), False) and pd.notna(m) and act.get((str(n), int(m)), False))
          for n, k, m, d in zip(cs.DeviceName, cs.detector, cs.card_mate, cs.dead)]
    cs = cs[ok]
    bad = set(zip(cs.DeviceName.astype(str), cs.detector.astype(int)))
    card = pd.Series([(str(a), int(b)) in bad for a, b in zip(v3.DeviceName, v3.Detector)], index=v3.index)
    # orchestrator 2026-09-28: a not_checkable row carrying a field_issue (e.g. pulse-set presence) is NOT admitted
    fi = v3.field_issue.astype("string").fillna("").str.strip().ne("") if "field_issue" in v3 else False
    return (v3.validated.astype(str).str.lower().eq("not_checkable") & v3.source.eq("print_high")
            & v3.print_confidence.eq("high") & ~v3.dead.eq(True) & ~card & ~fi)


_H3 = {}


def health3_drop(fr: pd.DataFrame) -> np.ndarray:
    """Frame rows to keep out of TRAINING under --health3 (cached per frame length)."""
    if len(fr) in _H3:
        return _H3[len(fr)]
    import a2_features as A2F
    h = pd.read_parquet(HEALTH3_FILE, columns=["period", "DeviceId", "detector", "status", "bad_periods"])
    h["DeviceId"] = h.DeviceId.str.lower()
    k = fr[KEY].merge(h.rename(columns={"detector": "Detector"}).astype({"Detector": fr.Detector.dtype}),
                      on=["DeviceId", "Detector", "period"], how="left")
    assert len(k) == len(fr)
    bad = k.status.eq("bad").to_numpy()
    win = {(per, n): (t0, t0 + pd.Timedelta(seconds=sec)) for per, ws in A2F.WINDOWS.items() for n, t0, sec in ws}
    sus = np.zeros(len(fr), bool)
    m = k.status.eq("suspect") & k.bad_periods.fillna("[]").ne("[]")
    for i in np.flatnonzero(m.to_numpy()):
        w0, w1 = win[(k.period.iat[i], k.win.iat[i])]
        for b in json.loads(k.bad_periods.iat[i]):
            if pd.Timestamp(b["start"]) < w1 and pd.Timestamp(b["end"]) > w0:
                sus[i] = True
                break
    out = bad | sus
    log(f"health3: {int(bad.sum()):,} rows of `bad` detector-periods, {int(sus.sum()):,} `suspect` rows overlapping a "
        f"bad period -> kept out of training")
    _H3[len(fr)] = out
    return out


def variant_target(lab: pd.DataFrame, fr: pd.DataFrame, v: str):
    """(target 7-class string array, training mask) for a variant `label.train.wi`."""
    lbl, trn, wi = v.split(".")
    col, src = {"first": ("label_print_first", "source"),
                "agree": ("label_print_agree", "source_agree"),
                "v2": ("func7_v2", "source")}[lbl]
    y = lab[col].to_numpy(object)
    ok = pd.notna(y) & lab.train_use.to_numpy() & ~lab.pend.to_numpy()
    if lbl == "v2":                      # v2 labels are not gated by the print's train_use
        ok = pd.notna(y) & ~lab.unusual_layout.to_numpy() & ~lab.pend.to_numpy()
    ok &= lab.tier.isin(TIERS[trn]).to_numpy() if trn != "all" else True
    if wi == "nowi":
        ok &= (lab[src] != "whole_intersection_other").to_numpy()
    ok &= (fr.health_flag != "failed").to_numpy()
    ok &= (fr.det_n_on >= MIN_ON).to_numpy() & ~lab.dq.to_numpy() & ~lab.notval.to_numpy()
    if HEALTH3:
        ok &= ~health3_drop(fr)
    if DEC_ROLE:
        ok &= ~lab.decchg_train.to_numpy()
    assert set(pd.unique(y[ok])) <= set(C7), set(pd.unique(y[ok]))
    return y, ok


# ------------------------------------------------------------------ fit
def fit_fold(fr, cols, yi, ok, k, seed, threads):
    import lightgbm as lgb
    folds = fr.fold.to_numpy()
    inner = (k + 1) % A2.N_FOLDS
    trm = ok & (folds != k) & (folds != inner)
    vam = ok & (folds == inner)
    prm = dict(A2.FUNC_PARAMS, n_jobs=threads, num_class=len(C7), seed=seed,
               bagging_seed=seed + 1, feature_fraction_seed=seed + 2,
               data_random_seed=seed + 3)
    n = prm.pop("n_estimators")
    m = lgb.LGBMClassifier(n_estimators=n, **prm)
    m.fit(fr.loc[trm, cols], yi[trm], eval_set=[(fr.loc[vam, cols], yi[vam])],
          eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
    return m.predict_proba(fr.loc[folds == k, cols]), int(m.best_iteration_ or 0), \
        int(trm.sum()), int(vam.sum())


def parse_variants(s: str) -> list[str]:
    vs = GRID if s == "grid" else s.split(",")
    for v in vs:
        lbl, trn, wi = v.split(".")
        assert lbl in ("first", "agree", "v2") and trn in TIERS and wi in ("wi", "nowi"), v
    return vs


def stage_fit(a) -> None:
    fr, cols = load_feats()
    lab = load_labels(fr, a.pending)
    threads = min(int(a.threads), 12)
    folds = [int(x) for x in a.folds.split(",")]
    rd = run_dir(a.pending)
    log(f"run directory {rd}")
    for v in parse_variants(a.variants):
        y, ok = variant_target(lab, fr, v)
        yi = pd.Series(y).map({c: i for i, c in enumerate(C7)}).fillna(-1).astype(int
                                                                                ).to_numpy()
        log(f"{v}: {int(ok.sum()):,} training rows (all folds), "
            f"{lab.loc[ok, 'DeviceId'].nunique()} signals, "
            f"classes {pd.Series(y[ok]).value_counts().to_dict()}")
        for s in [int(x) for x in a.seeds.split(",")]:
            for k in folds:
                f = rd / f"P_{v}_s{s}_f{k}.npy"
                if f.exists() and not a.overwrite:
                    log(f"  {f.name} exists, skipped")
                    continue
                t0 = time.time()
                Pk, it, ntr, nva = fit_fold(fr, cols, yi, ok, k, s, threads)
                f.parent.mkdir(parents=True, exist_ok=True)
                np.save(f, Pk.astype(np.float32))
                sec = time.time() - t0
                with open(rd / "timing.jsonl", "a") as fh:
                    fh.write(json.dumps({"variant": v, "seed": s, "fold": k, "trees": it,
                                         "n_train": ntr, "n_valid": nva, "secs": round(sec, 1),
                                         "threads": threads,
                                         "when": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
                log(f"  {v} s{s} fold {k}: {it} trees, {ntr:,} train rows, {sec:.0f}s")


# ------------------------------------------------------------------ scoring
def prf(yt, yp, classes) -> dict:
    d = {}
    for c in classes:
        tp = int(((yt == c) & (yp == c)).sum())
        np_, nt = int((yp == c).sum()), int((yt == c).sum())
        p = tp / np_ if np_ else np.nan
        r = tp / nt if nt else np.nan
        f1 = 2 * p * r / (p + r) if np_ and nt and (p + r) > 0 else np.nan
        d[c] = {"n": nt, "P": _r(p), "R": _r(r), "F1": _r(f1)}
    return d


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def metrics(yt7, p7, wg, sig) -> dict:
    """7-class, 5-class collapse and core-four scores; all windows, by window, per class."""
    yt7, p7 = np.asarray(yt7, object), np.asarray(p7, object)
    yt5 = np.array([TO5.get(c, c) for c in yt7], object)
    p5 = np.array([TO5.get(c, c) for c in p7], object)
    core = np.isin(yt7, CORE4)
    apc = np.isin(yt7, APC)

    def block(m):
        if not m.any():
            return None
        pr7, pr5 = prf(yt7[m], p7[m], C7), prf(yt5[m], p5[m], C5)
        f = lambda d, cs: _r(np.nanmean([d[c]["F1"] if d[c]["F1"] is not None else np.nan
                                          for c in cs if d[c]["n"] > 0]))
        return {"n": int(m.sum()), "signals": int(pd.unique(sig[m]).size),
                "acc7": _r((yt7[m] == p7[m]).mean()), "acc5": _r((yt5[m] == p5[m]).mean()),
                "acc_core4": _r((yt7[m & core] == p7[m & core]).mean()) if (m & core).any()
                else None,
                "acc_APC": _r((yt7[m & apc] == p7[m & apc]).mean()) if (m & apc).any()
                else None,
                "macroF1_7": f(pr7, C7), "macroF1_5": f(pr5, C5), "macroF1_core4": f(pr7, CORE4),
                "per_class7": pr7}
    allm = np.ones(len(yt7), bool)
    d = {"all": block(allm)}
    d["by_window"] = {g: {k: v for k, v in (block(wg == g) or {}).items() if k != "per_class7"}
                      for g in WGROUPS if (wg == g).any()}
    d["full_per_class7"] = (block(wg == "full") or {}).get("per_class7")
    return d


def eval_sets(fr, lab) -> dict:
    ok = (~lab.pend.to_numpy() & (fr.det_n_on >= MIN_ON).to_numpy() & ~lab.dq.to_numpy()
          & ~lab.notval.to_numpy() & ~lab.decchg_truth.to_numpy())
    usual = ~lab.unusual_layout.to_numpy()
    pc = lab.print_confidence.to_numpy(object)
    fix = (ok & usual & (lab.tier == "complete_high").to_numpy() & (pc == "high")
           & lab.print_function.isin(C7).to_numpy() & ~lab.dead.to_numpy())
    wi = (ok & usual & (lab.source == "whole_intersection_other").to_numpy()
          & (fr.period == "stg").to_numpy())
    unu = ok & ~usual & lab.label_print_first.isin(C7).to_numpy()
    v2 = lab.func5_v2t.notna().to_numpy() & ok
    return {"FIX": (fix, lab.print_function.to_numpy(object)),
            "V2": (v2, lab.func5_v2t.to_numpy(object)),
            "WI": (wi, None),
            "UNU": (unu, lab.label_print_first.to_numpy(object))}


def score_all(fr, lab, P7, have) -> dict:
    """Every evaluation set for one 7-class probability matrix (rows without a prediction
    -- folds not run -- are dropped via `have`)."""
    pred = np.array(C7, object)[P7.argmax(1)]
    wg, sig = fr.wgroup.to_numpy(), fr.DeviceId.to_numpy()
    out = {}
    for name, (m, yt) in eval_sets(fr, lab).items():
        m = m & have
        if name == "WI":
            p = pred[m]
            full = wg[m] == "full"
            out[name] = {"n": int(m.sum()), "signals": int(pd.unique(sig[m]).size),
                         "pred_Other": _r((p == "Other").mean()) if m.any() else None,
                         "pred_nonPM": _r(np.isin(p, ["Other", "Mid", "Bike"]).mean())
                         if m.any() else None,
                         "pred_nonPM_full": _r(np.isin(p[full], ["Other", "Mid", "Bike"]
                                                       ).mean()) if full.any() else None,
                         "pred_share": pd.Series(p).value_counts(normalize=True).round(4
                                                                                     ).to_dict()}
            continue
        if name == "V2":          # the note-14 set is five-class truth
            yt5, p5 = yt[m], np.array([TO5.get(c, c) for c in pred[m]], object)
            out[name] = {"n": int(m.sum()), "signals": int(pd.unique(sig[m]).size),
                         "acc5": _r((yt5 == p5).mean()) if m.any() else None,
                         "acc_APC": _r((p5[np.isin(yt5, APC)] == yt5[np.isin(yt5, APC)]).mean())
                         if m.any() else None,
                         "acc5_by_window": {g: _r((yt5[wg[m] == g] == p5[wg[m] == g]).mean())
                                            for g in WGROUPS if (wg[m] == g).any()}}
            continue
        out[name] = metrics(yt[m], pred[m], wg[m], sig[m])
    return out


def load_oof(fr, rd, v, seed, folds_req) -> tuple[np.ndarray, np.ndarray, list[int]]:
    P = np.zeros((len(fr), len(C7)), np.float32)
    have = np.zeros(len(fr), bool)
    got = []
    fo = fr.fold.to_numpy()
    for k in folds_req:
        f = rd / f"P_{v}_s{seed}_f{k}.npy"
        if f.exists():
            Pk = np.load(f)
            assert len(Pk) == int((fo == k).sum()), f"{f.name}: frame changed since fit"
            P[fo == k] = Pk
            have[fo == k] = True
            got.append(k)
    return P, have, got


def load_baseline(fr) -> tuple[np.ndarray, np.ndarray]:
    """Note-25 b7 OOF (seed mean of the saved seeds), 7-class, aligned to the frame rows."""
    k = pd.read_parquet(BASE_DIR / "keys.parquet", columns=KEY)
    fs = sorted(BASE_DIR.glob("Praw_b7_s*.npy"))
    Pb = np.mean([np.load(f) for f in fs], axis=0)
    assert Pb.shape == (len(k), 7)
    k["DeviceId"] = k.DeviceId.str.lower()
    k["_i"] = np.arange(len(k))
    k["Detector"] = k.Detector.astype(fr.Detector.dtype)
    idx = fr[KEY].merge(k, on=KEY, how="left")._i.to_numpy()
    have = ~np.isnan(idx)
    P = np.zeros((len(fr), 7), np.float32)
    P[have] = Pb[idx[have].astype(int)]          # note-25 column order == C7
    return P, have


def stage_report(a) -> None:
    fr, _ = load_feats()
    lab = load_labels(fr, a.pending)
    folds_req = list(range(A2.N_FOLDS))
    rd = run_dir(a.pending)
    res = {"pending": a.pending, "run_dir": str(rd), "coverage": coverage(fr, lab), "variants": {}}
    fixm = eval_sets(fr, lab)["FIX"][0]
    Pb, hb = load_baseline(fr)
    oks = {}
    for v in parse_variants(a.variants):
        seeds, Ps, have = [], [], None
        for s in [int(x) for x in a.seeds.split(",")]:
            P, h, got = load_oof(fr, rd, v, s, folds_req)
            if not got or (len(got) < A2.N_FOLDS and not a.allow_partial):
                continue
            seeds.append({"seed": s, "folds": got, **score_all(fr, lab, P, h)})
            Ps.append(P)
            have = h if have is None else have & h
        if not Ps:
            continue
        Pm = np.mean(Ps, axis=0)
        r = {"seeds": [x["seed"] for x in seeds], "folds": seeds[0]["folds"],
             "seed_mean": score_all(fr, lab, Pm, have), "per_seed": seeds}
        if len(seeds) > 1:
            accs = [x["FIX"]["all"]["acc7"] for x in seeds]
            r["seed_sd_FIX_acc7_pt"] = _r(np.std(accs, ddof=1) * 100, 3)
        # paired, signal-grouped bootstrap vs the baseline on FIX (rows both models score)
        yt = lab.print_function.to_numpy(object)
        okv = np.array(C7, object)[Pm.argmax(1)] == yt
        okb = np.array(C7, object)[Pb.argmax(1)] == yt
        both = fixm & have & hb
        full = both & (fr.wgroup == "full").to_numpy()
        frl = fr[["DeviceId"]]
        r["vs_baseline_FIX_acc7"] = {
            "allwin": A2.paired_bootstrap(frl, both, okb, okv) if both.any() else None,
            "full": A2.paired_bootstrap(frl, full, okb, okv) if full.any() else None}
        res["variants"][v] = r
        oks[v] = have
        f = r["seed_mean"]["FIX"]["all"]
        log(f"{v}: seeds {r['seeds']} folds {r['folds']} | FIX n {f['n']:,} acc7 {f['acc7']} "
            f"acc5 {f['acc5']} core4 {f['acc_core4']} | V2 acc5 "
            f"{r['seed_mean']['V2']['acc5']} | WI nonPM {r['seed_mean']['WI']['pred_nonPM']}")
    # baseline on exactly the rows the variants were scored on (union of folds present)
    cover = np.zeros(len(fr), bool)
    for h in oks.values():
        cover |= h
    if not oks:
        cover[:] = True
    res["baseline_b7_note25"] = score_all(fr, lab, Pb, hb & cover)
    b = res["baseline_b7_note25"]["FIX"]["all"]
    log(f"baseline b7 (note 25, saved OOF): FIX n {b['n']:,} acc7 {b['acc7']} acc5 "
        f"{b['acc5']} core4 {b['acc_core4']} | V2 acc5 {res['baseline_b7_note25']['V2']['acc5']}"
        f" | WI nonPM {res['baseline_b7_note25']['WI']['pred_nonPM']}")
    tim = rd / "timing.jsonl"
    if tim.exists():
        t = pd.read_json(tim, lines=True)
        res["timing"] = {"fits": int(len(t)), "median_secs_per_fold": _r(t.secs.median(), 1),
                         "total_secs": _r(t.secs.sum(), 0)}
    json.dump(res, open(rd / a.out, "w"), indent=1, default=str)
    log(f"-> {rd / a.out}")


def coverage(fr, lab) -> dict:
    """What the frame can see of the label table (the frame predates the print labels)."""
    v3 = pd.read_parquet(LABELS_V3, columns=["DeviceId", "tier", "pending_user",
                                             "unusual_layout"])
    v3["DeviceId"] = v3.DeviceId.str.lower()
    inf = v3.DeviceId.isin(set(fr.DeviceId))
    g = v3.assign(inf=inf).groupby(v3.tier.fillna("none"))
    return {"label_signals_by_tier": g.DeviceId.nunique().to_dict(),
            "label_signals_in_frame_by_tier": g.apply(
                lambda x: int(x[x.inf].DeviceId.nunique())).to_dict(),
            "frame_signals": int(fr.DeviceId.nunique()), "frame_rows": int(len(fr)),
            "pending_rows_in_frame_signals": int(v3[inf].pending_user.notna().sum()),
            "fixed_set_rows": int(eval_sets(fr, lab)["FIX"][0].sum()),
            "excluded": excluded_counts(fr, lab)}


def excluded_counts(fr, lab) -> dict:
    """Labelled frame rows (label_print_first or v2) removed by each filter, counted apart
    (a row can fall under several filters)."""
    has = (lab.label_print_first.notna() | lab.func5_v2t.notna()).to_numpy()
    low = (fr.det_n_on < MIN_ON).to_numpy()
    d = {"labelled_rows": int(has.sum()),
         f"lt{MIN_ON}_actuations": int((has & low).sum()),
         "dq_suspect": int((has & lab.dq.to_numpy()).sum()),
         "not_validated": int((has & lab.notval.to_numpy()).sum()),
         "unusual_layout": int((has & lab.unusual_layout.to_numpy()).sum())}
    d["not_validated_by_label"] = (lab.loc[has & lab.notval.to_numpy(), "label_print_first"]
                                   .fillna("none").value_counts().to_dict())
    return d


def stage_all(a) -> None:
    if not FEATS.exists():
        stage_prep(a)
    stage_fit(a)
    stage_report(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "fit", "report", "all"])
    ap.add_argument("--variants", default="grid")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--folds", default="0,1,2,3,4,5")
    ap.add_argument("--threads", default=12, type=int)
    ap.add_argument("--pending", default="exclude", choices=["exclude", "include"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", dest="allow_partial")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--frame", default="v5", choices=list(FRAMES))
    ap.add_argument("--min-on", default=0, type=int, dest="min_on",
                    help="drop detector-windows with fewer actuations from training AND "
                         "scoring (production --min-actuations is 5)")
    ap.add_argument("--clean", action="store_true",
                    help="drop dq_suspect detectors from training AND scoring; treat "
                         "EXTRA_UNUSUAL signals as unusual_layout")
    ap.add_argument("--require-validated", action="store_true", dest="require_validated",
                    help="train and score only rows whose v3 `validated` == 'pass'; the rest "
                         "are counted apart in report.json (coverage.excluded)")
    ap.add_argument("--allow-not-checkable-high", action="store_true", dest="allow_nc_high",
                    help="with --require-validated: also admit `not_checkable` rows with a high-confidence "
                         "print label that are not dead / card-suspect (run-directory suffix _valnc)")
    ap.add_argument("--lc-feats", action="store_true", dest="lc_feats",
                    help="add the label-check statistics as features (lc_features.py; run suffix _lc)")
    ap.add_argument("--lc-shuffle", action="store_true", dest="lc_shuffle",
                    help="with --lc-feats: permute those columns across rows (control; run suffix _lcshuf)")
    ap.add_argument("--pm-feats", action="store_true", dest="pm_feats",
                    help="add the permissive-phase features + OOF phase score (pm_features.py; run suffix _pm)")
    ap.add_argument("--pm-shuffle", action="store_true", dest="pm_shuffle",
                    help="with --pm-feats: permute those columns across rows (control; run suffix _pmshuf)")
    ap.add_argument("--health3", action="store_true",
                    help="keep health-v3 `bad` detector-periods and `suspect` bad-period windows out of training (_h3)")
    ap.add_argument("--labels", default="v3", choices=list(LABEL_SETS),
                    help="label table: v3 (default) or v3s (note 54: stacked relabel, YR==Count out, radar-over-loops "
                         "signals re-admitted)")
    ap.add_argument("--dec-role-rule", default="all", choices=["all", "fp"], dest="dec_role_rule",
                    help="with --dec-role: 'fp' = only the function / phase-differs statuses (run suffix _drfp)")
    ap.add_argument("--dec-role", action="store_true", dest="dec_role",
                    help="drop Dec-2024 rows of channels whose Dec config gives another role, from training AND "
                         "scoring (note 51b; run suffix _dr)")
    a = ap.parse_args()
    set_frame(a.frame)
    DEC_ROLE, DEC_RULE = a.dec_role, a.dec_role_rule
    LABELS_V3, DEC_ROLE_FILE = LABEL_SETS[a.labels]
    LC_FEATS, LC_SHUFFLE, HEALTH3 = a.lc_feats, a.lc_feats and a.lc_shuffle, a.health3
    PM_FEATS, PM_SHUFFLE = a.pm_feats, a.pm_feats and a.pm_shuffle
    assert not a.allow_nc_high or a.require_validated, "--allow-not-checkable-high needs --require-validated"
    MIN_ON, CLEAN, REQ_VAL, NC_HIGH = a.min_on, a.clean, a.require_validated, a.allow_nc_high
    globals()[f"stage_{a.stage}"](a)
