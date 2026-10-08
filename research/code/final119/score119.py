"""Note 119 (LOCKED EXAM, user-authorised 2026-10-07): score the saved exam predictions of v7 (dc_work/final_v7_prod) and
the beta (model/ = final_v2) on identical locked-signal windows.  Nothing is tuned; this only reads predictions + truth.

    python score119.py      -> %DC_WORK%/s119/score119.json, rows_phase.parquet, rows_func.parquet, n1_decided.csv

Phase truth  = official controller timing (labels_official, target_type 'phase'; overlap targets are not scored).
               E exact; R accepts the timing's switch phase / additional call phases and drops print-phase disagreements.
Scorable     = >= 5 actuations in the window AND the labelled phase has a Begin Green in the window (OOF 'everything').
Function     = locked key v3 (research/labels/function_labels_locked_v3.parquet, truth_v3) + the 149 n1_pending_exam rows
               decided now with the exam model (AGENTS.md print-vs-hand rule, note-94 agreement: v7 majority class over
               the >= 30-min windows, share >= .6 and >= 2 windows; hand = func7_v2, print = print_function).
               ATSPM-only score with stack credit (atspm_score.credit rule); >= 5 actuations; exclude_score rows out.
Lengths      = OOF anchors (windows_stg): m5 x4, m30 x4, h1 x3, h3 x2, h24 x2; headline >= 30 min = m30 + h1 + h3 + h24.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
REPO = Path(__file__).resolve().parents[3]
S = W / "s119"
LAB = REPO / "research" / "labels"
FAMS = {"m5": ["m5"], "m30": ["m30"], "h1": ["h1"], "h3": ["h3"], "h24": ["h24"], "ge30": ["m30", "h1", "h3", "h24"]}
GE30 = FAMS["ge30"]
ATS = ["Advance", "Presence", "Count", "Yellow_Red"]
C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Other", "Mid", "Bike"]
PCOL = {"Advance": "p_advance", "Presence": "p_presence", "Count": "p_count", "Yellow_Red": "p_yellow_red"}
MODELS = ["v7", "beta"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def acc_ci(ok, sig, n=2000, seed=64):
    if len(ok) == 0:
        return [None, None, None]
    u, inv = np.unique(sig, return_inverse=True)
    s, c = np.bincount(inv, ok, len(u)), np.bincount(inv, minlength=len(u))
    idx = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return [round(float(ok.mean()), 4), round(float(np.quantile(b, .025)), 4), round(float(np.quantile(b, .975)), 4)]


def delta_ci(a, b, sig, n=2000, seed=64):
    if len(a) == 0:
        return [None, None, None]
    u, inv = np.unique(sig, return_inverse=True)
    sa, sb, c = np.bincount(inv, a, len(u)), np.bincount(inv, b, len(u)), np.bincount(inv, minlength=len(u))
    idx = np.random.default_rng(seed).integers(0, len(u), (n, len(u)))
    d = (sb[idx].sum(1) - sa[idx].sum(1)) / c[idx].sum(1)
    return [round(100 * float(b.mean() - a.mean()), 3), round(100 * float(np.quantile(d, .025)), 3),
            round(100 * float(np.quantile(d, .975)), 3)]


def load_preds(m):
    fs = sorted(p for p in (S / "pred" / m).glob("*.parquet") if p.name.count(".") == 1)
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["DeviceId"] = d.DeviceId.str.lower()
    d["Detector"] = d.Detector.astype(int)
    d["ph"] = d.phase_pred.fillna(d.phase_guess).astype(float)
    d["fn"] = d.function_pred.fillna(d.function_guess)
    keep = ["DeviceId", "Detector", "win", "ph", "fn", "n_actuations", "phase_pred", "function_pred"] + \
        [c for c in PCOL.values()] + (["distance_ft"] if "distance_ft" in d.columns else [])
    return d[keep], len(fs)


def candidates(ids, wins):
    c = duckdb.connect()
    c.execute(f"SET threads=4; SET memory_limit='10GB'; SET temp_directory='{(W / 'tmp').as_posix()}'")
    parts = []
    for tag, start, mins in wins:
        end = str(pd.Timestamp(start) + pd.Timedelta(minutes=mins))
        q = c.sql(f"""SELECT DISTINCT DeviceId, Parameter AS cand_phase FROM read_parquet('{(S / 'ev' / '*.parquet').as_posix()}')
                      WHERE EventId = 1 AND Parameter BETWEEN 1 AND 16
                        AND Timestamp >= TIMESTAMP '{start}' AND Timestamp < TIMESTAMP '{end}'""").df()
        parts.append(q.assign(win=tag))
    return pd.concat(parts, ignore_index=True)


def alt_set(sw, add):
    s = set()
    if pd.notna(sw) and int(sw) > 0:
        s.add(int(sw))
    s |= {int(x) for x in re.findall(r"\d+", str(add or ""))}
    return s


def credit(d, mdl):
    """ATSPM ok with stack credit (atspm_score.credit), rows d of one model (columns t, p_<mdl>, P_<role>_<mdl>, ...)."""
    t = d.t.to_numpy(object)
    p = d[f"fn_{mdl}"].to_numpy(object)
    ta, pa = np.isin(t, ATS), np.isin(p, ATS)
    ok = np.where(ta, p == t, ~pa)
    tag = np.array([""] * len(d), object)
    g = d[d.stack_group.notna()]
    valid = {k for k, gg in g.groupby("stack_group") if (gg.t == gg.stack_role).all()}
    pos = {ix: i for i, ix in enumerate(d.index)}
    for (sg, win), gg in g[g.stack_group.isin(valid)].groupby(["stack_group", "win"]):
        if len(gg) < 2:
            continue
        ii = np.array([pos[x] for x in gg.index])
        r = gg.stack_role.iloc[0]
        pr = gg[f"{PCOL[r]}_{mdl}"].to_numpy(float)
        called = p[ii] == r
        nona = ~np.isin(p[ii], ATS)
        if called.any():
            w = ii[called][np.nanargmax(pr[called])]
            for j in ii[called]:
                ok[j] = j == w
                if j != w:
                    tag[j] = "extra"
            for j in ii[nona]:
                ok[j], tag[j] = True, "credited"
        elif nona.any():
            miss = ii[nona][np.nanargmax(pr[nona])]
            for j in ii[nona]:
                ok[j] = j != miss
                tag[j] = "miss" if j == miss else "credited"
    return ok.astype(float), tag


# same list as run119.py (copied: run119 imports a model at import time)
WINDOWS = [("m5_a", "2026-09-21 07:45:00", 5), ("m5_b", "2026-09-19 12:20:00", 5), ("m5_c", "2026-09-19 22:10:00", 5),
           ("m5_d", "2026-09-18 17:05:00", 5),
           ("m30_a", "2026-09-21 07:30:00", 30), ("m30_b", "2026-09-19 12:00:00", 30),
           ("m30_c", "2026-09-19 21:30:00", 30), ("m30_d", "2026-09-18 17:00:00", 30),
           ("h1_a", "2026-09-20 17:00:00", 60), ("h1_b", "2026-09-20 02:00:00", 60), ("h1_c", "2026-09-21 09:00:00", 60),
           ("h3_a", "2026-09-21 06:00:00", 180), ("h3_b", "2026-09-19 14:00:00", 180),
           ("h24_a", "2026-09-19 00:00:00", 1440), ("h24_b", "2026-09-20 00:00:00", 1440)]


def run():
    T0 = time.time()
    lk = pd.read_csv(W / "official" / "locked_v2.csv")
    lk["DeviceId"] = lk.DeviceId.str.lower()
    ids = set(lk.DeviceId)
    setof = dict(zip(lk.DeviceId, lk["set"]))
    wins = pd.DataFrame(WINDOWS, columns=["win", "start", "mins"])
    wins["fam"] = wins.win.str.split("_").str[0]
    P = {}
    for m in MODELS:
        P[m], nf = load_preds(m)
        assert nf == 115 or os.environ.get("X119_PARTIAL"), (m, nf)
    res = {"n_files": 115}
    # n_actuations agreement between the two runs (same events, own de-dup)
    k = ["DeviceId", "Detector", "win"]
    na = P["v7"][k + ["n_actuations"]].merge(P["beta"][k + ["n_actuations"]], on=k, how="outer", suffixes=("_v7", "_b"))
    res["n_act_rows_differ"] = int((na.n_actuations_v7.fillna(-1) != na.n_actuations_b.fillna(-1)).sum())
    log(f"preds loaded; n_actuations differ on {res['n_act_rows_differ']} of {len(na)} detector-windows")
    cand = candidates(ids, WINDOWS)
    ck = set(cand.DeviceId + "|" + cand.win + "|" + cand.cand_phase.astype(int).astype(str))
    v3 = pd.read_parquet(LAB / "function_labels_locked_v3.parquet")
    v3["DeviceId"] = v3.DeviceId.str.lower()
    v3 = v3.rename(columns={"detector": "Detector"}).astype({"Detector": int})
    assert set(v3.DeviceId) <= ids

    # ------------------------------------------------------------------ phase
    o = pd.read_parquet(W / "official" / "labels_official.parquet")
    o["DeviceId"] = o.DeviceId.str.lower()
    o = o[o.DeviceId.isin(ids)]
    res["phase_label_rows"] = {"all_targets": int(len(o)), "phase": int((o.target_type == "phase").sum()),
                               "overlap_not_scored": int((o.target_type == "overlap").sum())}
    o = o[o.target_type == "phase"].copy()
    o["Detector"] = o.Detector.astype(int)
    o["Phase"] = o.target_num.astype(int)
    o["alt"] = [",".join(map(str, sorted(alt_set(a, b)))) for a, b in zip(o.switch_phase, o.additional_call_phases)]
    tp = v3.phase_target.str.extract(r"P(\d+)")[0].astype(float)
    hi = v3.print_source.eq("print") & v3.print_confidence.eq("high") & v3.phase_diagram.notna() & tp.notna()
    v3["ppd"] = hi & (v3.phase_diagram.astype(float) != tp)
    o = o.merge(v3[["DeviceId", "Detector", "ppd"]], on=["DeviceId", "Detector"], how="left")
    o["ppd"] = o.ppd.fillna(False).astype(bool)
    pr = o[["DeviceId", "Detector", "Phase", "alt", "ppd"]].merge(wins[["win", "fam"]], how="cross")
    for m in MODELS:
        x = pr[k].merge(P[m][k + ["ph", "phase_pred", "n_actuations"]], on=k, how="left")
        pr[f"ph_{m}"] = x.ph.to_numpy()
        pr[f"answered_{m}"] = x.phase_pred.notna().to_numpy()
        pr[f"nact_{m}"] = x.n_actuations.fillna(0).to_numpy()
    pr["nact"] = pr.nact_v7
    pr["has_cand"] = (pr.DeviceId + "|" + pr.win + "|" + pr.Phase.astype(str)).isin(ck)
    pr["E"] = (pr.nact >= 5) & pr.has_cand
    pr["R"] = pr.E & ~pr.ppd
    altl = [set(int(v) for v in s.split(",") if v) for s in pr.alt]
    for m in MODELS:
        okE = (pr[f"ph_{m}"] == pr.Phase).to_numpy()
        pr[f"okE_{m}"] = okE.astype(float)
        pr[f"okR_{m}"] = (okE | np.array([(q == q) and int(q) in s for q, s in zip(pr[f"ph_{m}"], altl)])).astype(float)
    pr["set"] = pr.DeviceId.map(setof)
    pr.to_parquet(S / "rows_phase.parquet", index=False)
    ph = {"coverage": {}, "acc": {}, "delta_v7_minus_beta": {}, "by_set_ge30": {}}
    sig = pr.DeviceId.to_numpy()
    for fam, fl in FAMS.items():
        mf = pr.fam.isin(fl).to_numpy()
        u = pr[mf]
        ph["coverage"][fam] = {"labelled_rows": int(mf.sum()), "zero_act": int((u.nact == 0).sum()),
                               "act_1_4": int(((u.nact > 0) & (u.nact < 5)).sum()),
                               "never_green": int(((u.nact >= 5) & ~u.has_cand).sum()),
                               "scored_E": int(u.E.sum()), "scored_R": int(u.R.sum()),
                               "signals_E": int(u[u.E].DeviceId.nunique()),
                               "answered_share_E_v7": round(float(u[u.E].answered_v7.mean()), 4),
                               "answered_share_E_beta": round(float(u[u.E].answered_beta.mean()), 4)}
        for s_ in ("E", "R"):
            mm = mf & pr[s_].to_numpy()
            for m in MODELS:
                ph["acc"].setdefault(m, {}).setdefault(s_, {})[fam] = acc_ci(pr[f"ok{s_}_{m}"].to_numpy()[mm], sig[mm])
            ph["delta_v7_minus_beta"].setdefault(s_, {})[fam] = delta_ci(pr[f"ok{s_}_beta"].to_numpy()[mm],
                                                                         pr[f"ok{s_}_v7"].to_numpy()[mm], sig[mm])
    for st in ("TEST", "NEWTEST"):
        mm = pr.fam.isin(GE30).to_numpy() & pr.E.to_numpy() & (pr.set == st).to_numpy()
        ph["by_set_ge30"][st] = {m: acc_ci(pr[f"okE_{m}"].to_numpy()[mm], sig[mm]) for m in MODELS} | {
            "n": int(mm.sum()), "d": delta_ci(pr.okE_beta.to_numpy()[mm], pr.okE_v7.to_numpy()[mm], sig[mm])}
    mm = pr.fam.isin(GE30).to_numpy() & pr.E.to_numpy()
    ph["errors_ge30_E"] = {m: int((1 - pr[f"okE_{m}"].to_numpy()[mm]).sum()) for m in MODELS}
    res["phase"] = ph
    log(f"phase E: " + json.dumps({m: ph['acc'][m]['E'] for m in MODELS}))

    # ------------------------------------------------------------------ function: decide the 149 pending rows
    fr7 = P["v7"]
    pend = v3[v3.n1_pending_exam.eq(True)][["DeviceId", "Detector", "func7_v2", "print_function"]].copy()
    x = pend.merge(fr7[fr7.win.str.split("_").str[0].isin(GE30) & (fr7.n_actuations >= 5)][["DeviceId", "Detector", "fn"]],
                   on=["DeviceId", "Detector"], how="left")
    dec = []
    for (d_, det), g in x.groupby(["DeviceId", "Detector"]):
        hand, prn = g.func7_v2.iloc[0], g.print_function.iloc[0]
        f = g.fn.dropna()
        if len(f) < 2:
            dec.append((d_, det, hand, prn, None, np.nan, len(f), None, "out_no_model"))
            continue
        vc = f.value_counts()
        maj, share = vc.index[0], vc.iloc[0] / len(f)
        if share < .6:
            dec.append((d_, det, hand, prn, maj, share, len(f), None, "out_unsure"))
        elif maj == hand:
            dec.append((d_, det, hand, prn, maj, share, len(f), hand, "hand_kept"))
        elif maj == prn:
            dec.append((d_, det, hand, prn, maj, share, len(f), prn, "print_used"))
        else:
            dec.append((d_, det, hand, prn, maj, share, len(f), None, "out_model_third_class"))
    dec = pd.DataFrame(dec, columns=["DeviceId", "Detector", "hand", "print", "model_major", "model_share", "model_nwin",
                                     "truth_decided", "decision"])
    dec.to_csv(S / "n1_decided.csv", index=False)
    res["n1_decision"] = dec.decision.value_counts().to_dict() | {
        "atspm_truth_decided": int(dec.truth_decided.isin(ATS).sum()), "rows": int(len(dec))}
    log(f"n1 pending decided: {res['n1_decision']}")

    # ------------------------------------------------------------------ function truth table + rows
    t = v3[["DeviceId", "Detector", "truth_v3", "truth_v2", "exclude_score", "yr_count_identical", "stack_group",
            "stack_role"]].merge(dec[["DeviceId", "Detector", "truth_decided"]], on=["DeviceId", "Detector"], how="left")
    t["n1"] = t.truth_decided.notna()
    t["t"] = t.truth_v3.where(t.truth_v3.notna(), t.truth_decided)
    t = t[t.t.isin(C7) & ~t.exclude_score.eq(True) & ~t.yr_count_identical.eq(True)].copy()
    fr = t.merge(wins[["win", "fam"]], how="cross")
    for m in MODELS:
        x = fr[k].merge(P[m][k + ["fn", "function_pred", "n_actuations"] + list(PCOL.values())], on=k, how="left")
        fr[f"fn_{m}"] = x.fn.to_numpy(object)
        fr[f"fanswered_{m}"] = x.function_pred.notna().to_numpy()
        for c in PCOL.values():
            fr[f"{c}_{m}"] = x[c].to_numpy(float)
        if m == "v7":
            fr["nact"] = x.n_actuations.fillna(0).to_numpy()
    fr = fr[fr.nact >= 5].reset_index(drop=True)
    for m in MODELS:
        fr[f"ok_{m}"], fr[f"tag_{m}"] = credit(fr, m)
    fr["cls"] = np.where(fr.t.isin(ATS), fr.t, "nonATSPM")
    fr["set"] = fr.DeviceId.map(setof)
    fr.to_parquet(S / "rows_func.parquet", index=False)
    fu = {"acc": {}, "delta_v7_minus_beta": {}, "n": {}, "by_class": {}, "without_n1": {}, "n1_only_ge30": {},
          "secondary_nonatspm_v7_ge30": None, "key_v2_ge30": {}}
    sig = fr.DeviceId.to_numpy()
    n1 = fr.n1.to_numpy()
    for fam, fl in FAMS.items():
        mf = fr.fam.isin(fl).to_numpy()
        fu["n"][fam] = [int(mf.sum()), int(len(set(sig[mf]))), int((mf & n1).sum())]
        for m in MODELS:
            fu["acc"].setdefault(m, {})[fam] = acc_ci(fr[f"ok_{m}"].to_numpy()[mf], sig[mf])
            fu["without_n1"].setdefault(m, {})[fam] = acc_ci(fr[f"ok_{m}"].to_numpy()[mf & ~n1], sig[mf & ~n1])
        fu["delta_v7_minus_beta"][fam] = delta_ci(fr.ok_beta.to_numpy()[mf], fr.ok_v7.to_numpy()[mf], sig[mf])
        fu["without_n1"].setdefault("delta", {})[fam] = delta_ci(fr.ok_beta.to_numpy()[mf & ~n1],
                                                                 fr.ok_v7.to_numpy()[mf & ~n1], sig[mf & ~n1])
    for fam in ("ge30", "m5"):
        mf = fr.fam.isin(FAMS[fam]).to_numpy()
        for c in ATS + ["nonATSPM"]:
            mm = mf & (fr.cls == c).to_numpy()
            fu["by_class"].setdefault(fam, {})[c] = {"n": int(mm.sum()),
                                                     "v7": acc_ci(fr.ok_v7.to_numpy()[mm], sig[mm]),
                                                     "beta": acc_ci(fr.ok_beta.to_numpy()[mm], sig[mm]),
                                                     "d": delta_ci(fr.ok_beta.to_numpy()[mm], fr.ok_v7.to_numpy()[mm], sig[mm])}
    mg = fr.fam.isin(GE30).to_numpy()
    fu["n1_only_ge30"] = {"n": int((mg & n1).sum()), **{m: acc_ci(fr[f"ok_{m}"].to_numpy()[mg & n1], sig[mg & n1])
                                                       for m in MODELS}}
    na_ = mg & ~fr.t.isin(ATS).to_numpy() & ~fr.fn_v7.isin(ATS).to_numpy()
    fu["secondary_nonatspm_v7_ge30"] = {"n": int(na_.sum()), "acc": round(float((fr.t[na_] == fr.fn_v7[na_]).mean()), 4)}
    fu["errors_ge30"] = {m: fr[mg][f"ok_{m}"].eq(0).sum().item() for m in MODELS}
    fu["err_types_v7_ge30"] = {}
    tt, pp = fr.t.to_numpy(object), fr.fn_v7.to_numpy(object)
    e = mg & (fr.ok_v7.to_numpy() == 0)
    fu["err_types_v7_ge30"] = pd.Series(np.where(np.isin(tt[e], ATS) & np.isin(pp[e], ATS), "A->wrongA",
                                        np.where(np.isin(tt[e], ATS), "A->nonA", "nonA->A"))).value_counts().to_dict()
    fu["stack_ge30"] = {m: fr[mg][f"tag_{m}"].value_counts().to_dict() for m in MODELS}
    for st in ("TEST", "NEWTEST"):
        mm = mg & (fr.set == st).to_numpy()
        fu.setdefault("by_set_ge30", {})[st] = {"n": int(mm.sum())} | {m: acc_ci(fr[f"ok_{m}"].to_numpy()[mm], sig[mm])
                                                                       for m in MODELS}
    # locked key v2 truth (truth_v2, its own pending rows left out), for traceability to note 94
    k2 = mg & fr.truth_v2.notna().to_numpy() & fr.truth_v2.isin(C7).to_numpy()
    fr2 = fr[k2].copy()
    fr2["t"] = fr2.truth_v2
    for m in MODELS:
        ok2, _ = credit(fr2.reset_index(drop=True), m)
        fu["key_v2_ge30"][m] = acc_ci(ok2, fr2.DeviceId.to_numpy())
    fu["key_v2_ge30"]["n"] = int(len(fr2))
    res["function"] = fu
    log(f"function: " + json.dumps({m: fu['acc'][m] for m in MODELS}))

    # ------------------------------------------------------------------ lanes (v7 only) on locked print lanes
    # primary truth = ln1_cues.truth rules (complete prints, not unusual, consistent n_lanes, no diagram clash, >= 1 high);
    # 'loose' = mode of n_lanes_phase over every printed row of the phase.
    phs = pd.concat([pd.read_parquet(f) for f in sorted((S / "pred" / "v7").glob("*.phases.parquet"))], ignore_index=True)
    phs["DeviceId"] = phs.DeviceId.str.lower()
    d = v3[v3.tier.isin(["complete_high", "complete_mixed"]) & ~v3.unusual_layout.astype(bool)
           & (v3.phase_target_type == "phase") & v3.phase_target.notna()].copy()
    d["span"] = d.lanes_spanned.fillna(1).astype(int)
    d["end"] = d.lane_index.astype(float) + d.span - 1
    d["ph_num"] = pd.to_numeric(d.phase_target.str[1:], errors="coerce")
    d["diag_bad"] = d.phase_diagram.notna() & (pd.to_numeric(d.phase_diagram, errors="coerce") != d.ph_num)
    g = d.groupby(["DeviceId", "ph_num"]).agg(nl=("n_lanes_phase", "max"), nmin=("n_lanes_phase", "min"),
                                              max_end=("end", "max"), diag_bad=("diag_bad", "any"),
                                              n_high=("print_confidence", lambda s: int((s == "high").sum()))).reset_index()
    keep = g.nl.notna() & (g.nl == g.nmin) & (g.nl >= 1) & ~(g.max_end > g.nl) & ~g.diag_bad & (g.n_high >= 1)
    strict = g[keep].rename(columns={"ph_num": "phase", "nl": "nl_true"})[["DeviceId", "phase", "nl_true"]]
    loose = v3[v3.phase_target.astype(str).str.match(r"^P\d+$") & v3.n_lanes_phase.notna()].copy()
    loose["phase"] = loose.phase_target.str[1:].astype(int)
    loose = loose.groupby(["DeviceId", "phase"]).n_lanes_phase.agg(lambda s: int(s.mode().iloc[0])).rename(
        "nl_true").reset_index()
    res["lanes"] = {}
    for nm, lt in (("strict_ln1", strict), ("loose", loose)):
        lt = lt.astype({"phase": int, "nl_true": int})
        ln = phs.merge(lt, on=["DeviceId", "phase"], how="inner")
        ln = ln[ln.n_lanes.notna()].copy()
        ln["fam"] = ln.win.str.split("_").str[0]
        ln["exact"] = (ln.n_lanes.astype(int) == ln.nl_true).astype(float)
        ln["within1"] = ((ln.n_lanes.astype(int) - ln.nl_true).abs() <= 1).astype(float)
        r_ = {"truth_phases": int(len(lt)), "truth_signals": int(lt.DeviceId.nunique())}
        for fam, fl in FAMS.items():
            u = ln[ln.fam.isin(fl)]
            if len(u):
                r_[fam] = {"n": int(len(u)), "exact": acc_ci(u.exact.to_numpy(), u.DeviceId.to_numpy()),
                           "within1": round(float(u.within1.mean()), 4)}
        res["lanes"][nm] = r_

    # ------------------------------------------------------------------ setback (v7 only) on printed Advance distances
    pl = pd.read_parquet(W / "cabinet_locked" / "print_labels.parquet")
    pl["DeviceId"] = pl.DeviceId.str.lower()
    pl["distance_ft"] = pd.to_numeric(pl.distance_ft, errors="coerce")
    pl = pl[pl.DeviceId.isin(ids) & pl.function.eq("Advance") & (pl.distance_ft > 0)]
    pl = pl.rename(columns={"detector": "Detector"}).astype({"Detector": int}).drop_duplicates(["DeviceId", "Detector"])
    pl = pl.drop(columns=["unusual_layout", "tier"]).merge(v3[["DeviceId", "Detector", "unusual_layout", "tier"]], on=["DeviceId", "Detector"], how="left")
    sb = P["v7"].merge(pl[["DeviceId", "Detector", "distance_ft", "confidence", "unusual_layout"]],
                       on=["DeviceId", "Detector"], suffixes=("", "_print"))
    sb = sb[sb.n_actuations >= 5].copy()
    sb["fam"] = sb.win.str.split("_").str[0]
    sb["err"] = (sb.distance_ft - sb.distance_ft_print).abs()
    res["setback"] = {"printed_advance": int(len(pl))}
    variants = {"all_printed_advance": np.ones(len(sb), bool),
                "pred_advance_usual": (sb.fn == "Advance").to_numpy() & ~sb.unusual_layout.eq(True).to_numpy(),
                "pred_advance_usual_high": (sb.fn == "Advance").to_numpy() & ~sb.unusual_layout.eq(True).to_numpy()
                & (sb.confidence == "high").to_numpy()}
    for vn, vm in variants.items():
        for fam, fl in FAMS.items():
            u = sb[vm & sb.fam.isin(fl).to_numpy()]
            h = u[u.distance_ft.notna()]
            res["setback"].setdefault(vn, {})[fam] = {
                "rows": int(len(u)), "with_estimate": int(len(h)), "signals": int(h.DeviceId.nunique()),
                "median_abs_err_ft": round(float(h.err.median()), 1) if len(h) else None,
                "within_50ft": round(float((h.err <= 50).mean()), 4) if len(h) else None,
                "within_100ft": round(float((h.err <= 100).mean()), 4) if len(h) else None}

    # ------------------------------------------------------------------ OOF comparison (saved six-fold OOF of the same
    # recipe: phase v6_lagt = v6b/v7 phase, note 113; function mean3 'nou' = v7 stacker, note 114)
    p113 = json.load(open(W / "s113" / "score113.json"))
    f114 = json.load(open(W / "s114" / "score114.json"))
    oof = {"phase_E": {f: p113["acc"]["v6_lagt"]["E"][f] for f in ["m5", "m30", "h1", "h3", "h24", "ge30"]},
           "phase_R": {f: p113["acc"]["v6_lagt"]["R"][f] for f in ["m5", "m30", "h1", "h3", "h24", "ge30"]},
           "function_E": {f: f114["acc"]["mean3"]["nou"]["E"][f] for f in ["m5", "m30", "h1", "h3", "h24", "ge30"]}}
    for key, nkey in (("phase_E", p113["n"]["E"]), ("phase_R", p113["n"]["R"]), ("function_E", f114["n"]["mean3"]["E"])):
        nn = np.array([nkey[f][0] for f in GE30], float)
        aa = np.array([oof[key][f][0] for f in GE30], float)
        oof[key]["ge30_same_lengths"] = round(float((nn * aa).sum() / nn.sum()), 4)
    cmp_ = {}
    for key, ex in (("phase_E", ph["acc"]["v7"]["E"]), ("phase_R", ph["acc"]["v7"]["R"]), ("function_E", fu["acc"]["v7"])):
        for f in FAMS:
            o_ = oof[key]["ge30_same_lengths"] if f == "ge30" else oof[key][f][0]
            cmp_.setdefault(key, {})[f] = {"exam": ex[f], "oof": o_, "oof_inside_exam_ci": bool(ex[f][1] <= o_ <= ex[f][2]),
                                          "exam_minus_oof_pt": round(100 * (ex[f][0] - o_), 2)}
    res["oof"] = oof
    res["exam_vs_oof"] = cmp_

    # ------------------------------------------------------------------ run time
    tm = {}
    for m in MODELS:
        tt_ = pd.concat([pd.read_parquet(f) for f in sorted((S / "pred" / m).glob("*.time.parquet"))], ignore_index=True)
        tt_["fam"] = tt_.win.str.split("_").str[0]
        tm[m] = {"calls": int(len(tt_)), "total_s": round(float(tt_.seconds.sum()), 1),
                 "per_call_mean_s": {f: round(float(g.seconds.mean()), 3) for f, g in tt_.groupby("fam")},
                 "per_call_max_s": {f: round(float(g.seconds.max()), 2) for f, g in tt_.groupby("fam")},
                 "first_call_s": round(float(tt_[tt_.first_call].seconds.iloc[0]), 2)}
    res["runtime_predict"] = tm
    res["score_s"] = round(time.time() - T0, 1)
    json.dump(res, open(S / "score119.json", "w"), indent=1, default=str)
    log(f"done in {res['score_s']} s -> {S / 'score119.json'}")


if __name__ == "__main__":
    run()
