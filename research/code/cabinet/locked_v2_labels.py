"""Note 94: the exam answer key under the TRAINING label rules (user 2026-10-05, USER_INPUT q5: "yes").

    python locked_v2_labels.py pairs     %DC_WORK%/lab93/locked/pairs.parquet  (actuation pair statistics, labels only)
    python locked_v2_labels.py build     research/labels/function_labels_locked_v2.parquet
                                         + research/labels/function_label_changes_locked_v2.csv

LABELS ONLY. No model is run on a locked signal, no prediction is read, no metric is computed. The only hi-res data
read are detector ON intervals, for the two actuation-based cleansing rules (stacked groups, Yellow_Red identical to
Count), exactly as for training: TEST signals from the Dec-2024 interval cache, NEWTEST from the Sept-2026 one.
Scope: the 115 locked_v2 signals (the 71 released NEWTEST signals live in the training table v4p).
Rules (training names): R1 loop config Count / YR -> print reading (truth cleared) or out; R3 re-read YR flagged
identical to Count -> out; STACK stacked groups (stacked_review.build_groups) -> every member carries the group's role
(no user answers exist for locked groups: all applied, flagged stack_unreviewed, as blank answers are in training);
YRC Yellow_Red zone identical to a Count zone (stacked_review.yr_count) -> out; SWEEP explained_other entries that
could hide a performance-measure detector (v4l_expl_sweep regexes) -> the whole-intersection Other label of that
channel is removed; R5 low-confidence behaviour-tied print readings -> out; R6 unused inputs < 15 ONs -> out;
READMIT signals unusual only because of radar over loops -> not unusual; N2 channels 40/41 without phase / text /
function -> out. N1 (print vs hand, needs model agreement) is NOT applied: conflict rows are flagged
`n1_pending_exam` and left out of `truth_v2` (their old-rule truth is kept in `truth_v2_oldrule`).
"""
from __future__ import annotations

import itertools
import json
import re
import sys
import time

import duckdb
import numpy as np
import pandas as pd

from cab_common import DC_WORK, REPO

LOCKED_V1 = REPO / "research/labels/function_labels_locked_v1.parquet"
LOCKED_V2 = REPO / "research/labels/function_labels_locked_v2.parquet"
CHG = REPO / "research/labels/function_label_changes_locked_v2.csv"
OUT = DC_WORK / "lab93" / "locked"
IVF = {"TEST": (DC_WORK / "cache/det_intervals.parquet").as_posix(),
       "NEWTEST": (DC_WORK / "official/stg/cache/det_intervals.parquet").as_posix()}
SIG = DC_WORK / "cabinet_locked" / "signals"
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
ATSPM = {"Advance", "Presence", "Count", "Yellow_Red"}
TOL, TIGHT, MIN_N = 1.5, 0.3, 50
# same regexes as v4l_expl_sweep.py (that module refuses to import outside the v4l store)
POSSIBLE_PM = re.compile(r"added after|renumbered|upgrade|new radar|probably a new|likely (a|the|one|VD)|plausibly one of|"
                         r"radar data channel|radar advance|relanded|replacement|re-?landed|new channel since|template channel", re.I)
KEEP = re.compile(r"bike|ped|truck|\bcars\b|classification|derived|dummy|pass|fail|extend|logic|copy|mirror|call zone|phase-call", re.I)


def locked_v2() -> pd.DataFrame:
    return pd.read_csv(DC_WORK / "official/locked_v2.csv")


def load() -> pd.DataFrame:
    lk = locked_v2()
    d = pd.read_parquet(LOCKED_V1)
    d["dev"] = d.DeviceId.str.lower()
    d = d[d.dev.isin(set(lk.DeviceId.str.lower()))].copy()
    return d


def pairs():
    """stacked_pairs.py's statistics, per locked signal, from its own window (labels only)."""
    from stacked_pairs import onset_match
    OUT.mkdir(parents=True, exist_ok=True)
    lab = load()
    lab = lab[lab.phase_target.notna()]
    cn = duckdb.connect()
    cn.execute("SET memory_limit='10GB'")
    cn.execute("SET threads=8")
    cn.execute(f"SET temp_directory='{(DC_WORK / 'tmp').as_posix()}'")
    rows, t0 = [], time.time()
    for st, ivf in IVF.items():
        devs = sorted(lab[lab.set.eq(st)].dev.unique())
        cn.execute("DROP TABLE IF EXISTS iv")
        cn.execute(f"""CREATE TEMP TABLE iv AS SELECT lower(DeviceId) dev, Detector det, t_on, t_off, dur FROM '{ivf}'
                       WHERE lower(DeviceId) IN ({",".join(f"'{x}'" for x in devs)}) AND Detector <= 64""")
        for k, dev in enumerate(devs):
            L = lab[lab.dev == dev]
            iv = cn.execute("SELECT DISTINCT det, t_on, t_off, dur FROM iv WHERE dev = ? ORDER BY det, t_on", [dev]).df()
            if iv.empty:
                continue
            base = iv.t_on.min().floor("15min")
            iv["s_on"] = (iv.t_on - base).dt.total_seconds().to_numpy()
            iv["s_off"] = (iv.t_off - base).dt.total_seconds().to_numpy()
            tmax = float(np.nanmax(iv[["s_on", "s_off"]].to_numpy())) + 1
            span_h = tmax / 3600
            nb = int(tmax // 900) + 1
            per = {}
            for det, g in iv.groupby("det"):
                if len(g) < MIN_N:
                    continue
                on = g.s_on.to_numpy()
                off = np.where(np.isfinite(g.s_off.to_numpy()), g.s_off.to_numpy(), on)
                cnt = np.bincount((on // 900).astype(int), minlength=nb)
                grid = np.zeros(int(tmax * 10) + 2, np.int32)
                np.add.at(grid, (on * 10).astype(int), 1)
                np.add.at(grid, np.maximum((off * 10).astype(int), (on * 10).astype(int) + 1), -1)
                occ = np.cumsum(grid) > 0
                dur = g.dur.to_numpy()
                per[int(det)] = dict(on=on, cnt=cnt, occ=occ, n=len(on), dmed=float(np.nanmedian(dur)),
                                     pulse=float(np.nanmean(dur <= 0.15)), occ_share=float(occ.mean()))
            for ph, P in L.groupby("phase_target"):
                dets = [int(x) for x in P.detector if int(x) in per]
                for a, b in itertools.combinations(sorted(dets), 2):
                    A, B = per[a], per[b]
                    ab, ab_t, lag_ab = onset_match(A["on"], B["on"])
                    ba, ba_t, _ = onset_match(B["on"], A["on"])
                    ca, cb = A["cnt"], B["cnt"]
                    act = (ca + cb) > 0
                    corr = (float(np.corrcoef(ca[act], cb[act])[0, 1])
                            if act.sum() > 8 and ca[act].std() > 0 and cb[act].std() > 0 else np.nan)
                    inter = float((A["occ"] & B["occ"]).sum())
                    rows.append(dict(dev=dev, DeviceName=L.DeviceName.iloc[0], phase=ph, a=a, b=b, n_a=A["n"], n_b=B["n"],
                                     match_ab=ab, match_ba=ba, tight_ab=ab_t, tight_ba=ba_t, lag_ab=lag_ab,
                                     chance_ab=min(1.0, B["n"] / (span_h * 3600) * 2 * TOL),
                                     chance_ba=min(1.0, A["n"] / (span_h * 3600) * 2 * TOL), corr15=corr,
                                     occ_a=A["occ_share"], occ_b=B["occ_share"],
                                     ovl_a=inter / max(A["occ"].sum(), 1), ovl_b=inter / max(B["occ"].sum(), 1),
                                     dmed_a=A["dmed"], dmed_b=B["dmed"], pulse_a=A["pulse"], pulse_b=B["pulse"],
                                     hours=span_h, set=st))
            if k % 20 == 0:
                print(st, k, len(devs), len(rows), f"{time.time() - t0:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(OUT / "pairs.parquet", index=False)
    print("pairs", len(out), "signals", out.dev.nunique())


def honest_truth(v: pd.DataFrame) -> pd.Series:
    """stack_labels_v3s.honest_truth: user ruling > high-confidence print > config label where no print reading."""
    pf, src = v.print_function, v.source.astype("string")
    user = src.eq("user_ruling").fillna(False)
    high = v.print_source.eq("print") & v.print_confidence.eq("high") & pf.isin(C7)
    noprint = ~v.print_source.eq("print").fillna(False) | pf.isna()
    return pd.Series(np.where(user, v.function, np.where(high, pf, np.where(noprint & v.func7_v2.notna(), v.func7_v2,
                                                                                 None))), index=v.index)


def build():
    import stacked_review as SR
    from stack_labels_v3s import unusual_reasons
    from v4l_rules import BEH_EXEMPT, BEH_TIE, NOISE_ON
    d = load().reset_index(drop=True)
    n_sig = d.dev.nunique()
    d0 = d.copy()
    d["truth_v1"] = honest_truth(d)
    for c in ("stack_relabel", "stack_unreviewed", "yr_count_identical", "readmit_radar_over_loops",
              "exclude_score", "n1_pending_exam", "sweep_unlabelled"):
        d[c] = False
    d["stack_group"], d["stack_role"] = None, None
    d["rule_v2"] = pd.Series(pd.NA, index=d.index, dtype="string")

    def tag(m, r):
        d.loc[m & d.rule_v2.isna(), "rule_v2"] = r
    # R1 -- config Count / YR on a print loop: print reading at any confidence, else out (truth never Count / YR)
    loop = d.print_source.eq("print").fillna(False) & d.technology.eq("loop").fillna(False)
    notprint = ~d.source.astype("string").isin(["user_ruling", "print_high"]).fillna(False)
    r1 = (loop & notprint & d.func7_v2.isin(["Count", "Yellow_Red"]).fillna(False)
          & ~d.print_function.isin(["Count", "Yellow_Red"]).fillna(False))
    has_p = r1 & d.print_function.notna()
    for c in ("function", "label_print_first"):
        d.loc[has_p, c] = d.loc[has_p, "print_function"]
    d.loc[has_p, "source"] = "print_" + d.loc[has_p, "print_confidence"].astype(str)
    d.loc[r1 & ~has_p, "exclude_score"] = True
    tag(r1 & has_p, "R1_loop_config_count->print")
    tag(r1 & ~has_p, "R1_loop_config_count->excluded")
    # STACK + YRC -- actuation pair statistics (labels only)
    p = pd.read_parquet(OUT / "pairs.parquet")
    lab = d.assign(detector=d.detector.astype(int))
    pe = SR.edges(p, lab)
    G = SR.build_groups(pe, lab)
    Y = SR.yr_count(pe)
    G.drop(columns=["best"]).to_parquet(OUT / "groups.parquet", index=False)
    Y.to_parquet(OUT / "yr_count_pairs.parquet", index=False)
    idx = {k: i for i, k in enumerate(zip(d.dev, d.detector.astype(int)))}
    th = honest_truth(d)
    for g in G.itertuples():
        mem = sorted(int(m) for m in g.members)
        rows = [idx[(g.dev, m)] for m in mem if (g.dev, m) in idx]
        gid = f"{g.DeviceName}_{g.phase}_{g.role}_{'-'.join(map(str, mem))}"
        for i in rows:
            d.at[i, "stack_group"], d.at[i, "stack_role"], d.at[i, "stack_unreviewed"] = gid, g.role, True
            if d.at[i, "function"] != g.role or th.iat[i] != g.role:
                d.at[i, "stack_relabel"] = True
                for c in ("function", "label_print_first", "print_function"):
                    d.at[i, c] = g.role
    tag(d.stack_relabel, "STACK_relabel")
    for dev, yr in zip(Y[Y.identical].dev, Y[Y.identical].yr.astype(int)):
        if (dev, yr) in idx:
            i = idx[(dev, yr)]
            d.at[i, "yr_count_identical"] = True
            d.at[i, "exclude_score"] = True
    tag(d.yr_count_identical, "YRC_yr_identical_to_count")
    # R3 -- re-read YR flagged identical by the reader
    r3 = d.print_flags.astype("string").str.contains("yr_identical_to_count").fillna(False) & d.function.eq("Yellow_Red")
    d.loc[r3, ["exclude_score", "yr_count_identical"]] = True
    tag(r3, "R3_yr_identical_to_count")
    # SWEEP -- explained_other entries that may hide a PM detector: the channel loses its whole-intersection Other
    sw = []
    for dn in sorted(d.DeviceName.dropna().astype(str).unique()):
        f = SIG / f"{dn}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        on = {int(x["detector"]) for x in r.get("detectors", [])}
        for k, v in (r.get("explained_other") or {}).items():
            if int(k) not in on and POSSIBLE_PM.search(str(v)) and not KEEP.search(str(v)):
                sw.append((dn, int(k), str(v)[:200]))
    swk = {(a, b) for a, b, _ in sw}
    ms = pd.Series([(str(a), int(b)) in swk for a, b in zip(d.DeviceName, d.detector)], index=d.index)
    msw = ms & d.source.astype("string").eq("whole_intersection_other").fillna(False)
    for c in ("function", "label_print_first"):
        d.loc[msw, c] = None
    d.loc[msw, "sweep_unlabelled"] = True
    tag(msw, "SWEEP_explained_other_removed")
    pd.DataFrame(sw, columns=["DeviceName", "detector", "explained_other"]).to_csv(OUT / "sweep_entries.csv", index=False)
    # R5 -- low-confidence print readings tied / classed by behaviour
    nm_det = list(zip(d.DeviceName.astype(str), d.detector.astype(int)))
    exempt = pd.Series([k in BEH_EXEMPT for k in nm_det], index=d.index)
    r5 = (d.source.astype("string").eq("print_low").fillna(False) & d.print_confidence.eq("low").fillna(False)
          & d.confidence_reason.astype("string").str.contains(BEH_TIE, case=False, regex=True).fillna(False) & ~exempt)
    d.loc[r5, "exclude_score"] = True
    tag(r5, "R5_behaviour_tied_low")
    # R6 -- unused inputs with < 15 ONs
    r6 = (d.source.astype("string").eq("whole_intersection_other").fillna(False) & d.config_function.isna()
          & d.n_on_new.lt(NOISE_ON).fillna(False) & ~msw)
    d.loc[r6, "exclude_score"] = True
    tag(r6, "R6_noise_input")
    # N2 -- channels 40 / 41 without phase, text or config function (user q4)
    desc = d.description.astype("string").fillna("").str.strip()
    n2 = d.detector.isin([40, 41]) & d.phase_target.isna() & desc.eq("") & d.config_function.isna()
    d.loc[n2, "exclude_score"] = True
    tag(n2, "N2_dummy_40_41")
    # READMIT -- unusual only because live radar sits over live loops (stack_labels_v3s step 3)
    un = d[d.unusual_layout.astype(bool)].DeviceName.dropna().astype(str).unique()
    R = unusual_reasons(set(un))
    re_names = set(R.DeviceName[R.readmit])
    mr = d.DeviceName.astype(str).isin(re_names)
    d.loc[mr, "readmit_radar_over_loops"] = True
    d["unusual_layout_v1"] = d.unusual_layout
    d.loc[mr, "unusual_layout"] = False
    # truth after the rules; R1 never scores Count / YR on a loop
    d["truth_v2_oldrule"] = honest_truth(d)
    d.loc[d.stack_relabel, "truth_v2_oldrule"] = d.loc[d.stack_relabel, "stack_role"]
    d.loc[r1 & d.truth_v2_oldrule.isin(["Count", "Yellow_Red"]), "truth_v2_oldrule"] = None
    d.loc[d.exclude_score | msw, "truth_v2_oldrule"] = None
    # N1 -- print vs hand conflicts: pending (rule needs model agreement = scoring locked signals)
    conflict = (d.print_source.eq("print") & d.print_function.notna() & d.func7_v2.notna()
                & d.print_function.ne(d.func7_v2)).fillna(False)
    skip = (d.source.astype("string").isin(["user_ruling", "review_round1"]).fillna(False)
            | d.v2_source.astype("string").str.startswith("review").fillna(False) | r1 | d.stack_relabel)
    n1 = conflict & ~skip & ~d.exclude_score
    d.loc[n1, "n1_pending_exam"] = True
    d["truth_v2"] = d.truth_v2_oldrule.where(~n1, None)
    d["label_use_v2"] = d.function.notna() & ~d.exclude_score & ~d.unusual_layout.astype(bool) & ~d.dead.eq(True)
    d["label_version"] = "locked_v2"
    d = d.drop(columns=["dev"])
    assert d.DeviceId.str.lower().isin(set(locked_v2().DeviceId.str.lower())).all()
    d.to_parquet(LOCKED_V2, index=False)
    # change list (labels only)
    x = d[["DeviceName", "set", "detector", "phase_target", "technology", "func7_v2", "print_function",
           "print_confidence", "rule_v2", "n1_pending_exam", "stack_group"]].copy()
    x["old_label"], x["old_truth"] = d0.function.values, d0.pipe(honest_truth).values
    x["new_label"], x["new_truth"] = d.function.where(~d.exclude_score, None).values, d.truth_v2.values
    ch = ((x.old_label.astype("string").fillna("-") != x.new_label.astype("string").fillna("-"))
          | (x.old_truth.astype("string").fillna("-") != x.new_truth.astype("string").fillna("-")))
    x = x[ch].copy()
    x["atspm_truth_changed"] = x.old_truth.isin(ATSPM) | x.new_truth.isin(ATSPM)
    x["why"] = np.where(x.n1_pending_exam, "N1 pending exam (print vs hand disagree; model agreement not computed)",
                        x.rule_v2.astype("string").fillna("other"))
    x.sort_values(["why", "DeviceName", "detector"]).to_csv(CHG, index=False)
    s = dict(signals=n_sig, rows=len(d), changed=len(x), by_reason=x.why.value_counts().to_dict(),
             stacked_groups=len(G), stack_relabel=int(d.stack_relabel.sum()), yr_identical=int(d.yr_count_identical.sum()),
             sweep_entries=len(sw), sweep_unlabelled=int(msw.sum()), r1=int(r1.sum()), r5=int(r5.sum()), r6=int(r6.sum()),
             n2=int(n2.sum()), readmit_signals=sorted(re_names), n1_pending_exam=int(n1.sum()),
             n1_pending_atspm_truth=int((n1 & d.truth_v2_oldrule.isin(ATSPM)).sum()),
             truth_rows_v1=int(honest_truth(d0).notna().sum()), truth_rows_v2_oldrule=int(d.truth_v2_oldrule.notna().sum()),
             truth_rows_v2=int(d.truth_v2.notna().sum()))
    (OUT / "locked_v2_summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    {"pairs": pairs, "build": build}[sys.argv[1]]()
