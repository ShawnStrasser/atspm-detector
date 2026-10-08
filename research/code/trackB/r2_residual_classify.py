"""Stage 22, step 2: what ARE the stage-13 blend's residual phase errors?

Reads `trackB/resid/blend13_rows.parquet` (step 1: six-fold out-of-fold blend + decoder, 701
training signals, exactly the stage-13 rows) and puts every error at 30 min (4 anchors) and at
the full window into one of four bins, in this order:

  a0  label, already legal: the model's phase is the timing's switch_phase or one of its
      additional_call_phases (the A1 scorer fix accepts these; stage 13 did not).
  a   likely LABEL error: the detector's full-window answer is the same wrong phase with
      p >= P_CONF, that phase is top-1 in >= CONSIST of the detector's scored windows, AND
      external support -- the error is signal-wide (>= 2 detectors of the signal make the
      same label->model move at the full window) or the hand config disagrees with the timing;
      and the pair is NOT green together >= J_CONC of the full window unless hand = model.
      "strong" subset: hand config = model, or >= 3 detectors, or a swap (the signal also has
      model->label errors) -- two siblings alone can be the decoder coupling them.
  b   concurrent-pair ambiguity: labelled and predicted phase are green together for
      >= J_CONC of their joint green time (Jaccard) inside THAT window.
  c   low data: fewer than MIN_ACT actuations in the window.
  d   the rest (genuine model errors), sub-typed by the phase relationship.

Labels are used for evaluation only.  Hand phase labels are retired: here they are only
evidence for a review list.  Writes the note's numbers to `resid/residual_summary.json`,
per-error rows to `resid/residual_errors.parquet` and the review list to
`<repo>/review/phase_label_suspects_round1.xlsx` (git-ignored).

    python research/code/trackB/r2_residual_classify.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import CONCURRENT_PAIRS, DC_WORK, REPO  # noqa: E402
import build_features as bf  # noqa: E402
import windows_stg as ws  # noqa: E402
from a1_rescore import accept_map  # noqa: E402

RES = DC_WORK / "trackB" / "resid"
DET = ["DeviceId", "Detector", "win"]
P_CONF, CONSIST, J_CONC, MIN_ACT = 0.80, 0.75, 0.90, 10
RING_ADJ = {frozenset(p) for p in [(1, 2), (3, 4), (5, 6), (7, 8)]}


def top2(df: pd.DataFrame, col: str) -> pd.DataFrame:
    d = df.sort_values(DET + [col, "cand_phase"], ascending=[1, 1, 1, 0, 1])
    d["r"] = d.groupby(DET).cumcount()
    a = d[d.r == 0][DET + ["cand_phase", col]].rename(columns={"cand_phase": "pred",
                                                                col: "p"})
    b = d[d.r == 1][DET + ["cand_phase", col]].rename(columns={"cand_phase": "pred2",
                                                                col: "p_2"})
    return a.merge(b, on=DET, how="left")


def win_times() -> pd.DataFrame:
    rows = [(w["win"], "dec", w["t0"], w["secs"]) for w in bf.WINDOWS_MIXED + bf.WINDOWS_SHORT_B]
    rows += [(w["win"], "stg", w["t0"], w["secs"])
             for w in ws.WINDOWS_STG_MIXED + ws.WINDOWS_STG_SHORT_B]
    t = pd.DataFrame(rows, columns=["win", "period", "t0", "secs"]).drop_duplicates(
        ["win", "period"])
    t["t1"] = t.t0 + pd.to_timedelta(t.secs, unit="s")
    return t


def green_jaccard(err: pd.DataFrame) -> pd.Series:
    """Jaccard of the green time of (Phase, pred) inside each error's own window."""
    wt = win_times()
    e = err[["DeviceId", "win", "Phase", "pred"]].drop_duplicates().copy()
    e["period"] = np.where(e.DeviceId.str.endswith("@stg"), "stg", "dec")
    e["dev"] = e.DeviceId.str.replace("@stg", "", regex=False)
    e = e.merge(wt, on=["win", "period"], how="left")
    assert e.t0.notna().all(), "window without a time span"
    e["bL"] = (2 ** (e.Phase.astype(int) - 1)).astype("int64")
    e["bM"] = (2 ** (e.pred.astype(int) - 1)).astype("int64")
    con = duckdb.connect()
    con.execute("SET memory_limit='10GB'"); con.execute("SET threads=6")
    con.execute(f"SET temp_directory='{(DC_WORK / 'tmp').as_posix()}'")
    con.register("e", e[["DeviceId", "dev", "period", "win", "Phase", "pred", "t0", "t1",
                         "bL", "bM"]])
    gs = {"dec": (DC_WORK / "cache" / "green_state.parquet").as_posix(),
          "stg": (DC_WORK / "official" / "stg" / "cache" / "green_state.parquet").as_posix()}
    parts = []
    for per, f in gs.items():
        parts.append(con.sql(f"""
            WITH g AS (SELECT lower(DeviceId) dev, t_start, t_end, CAST(mask AS BIGINT) m
                       FROM read_parquet('{f}')
                       WHERE lower(DeviceId) IN (SELECT dev FROM e WHERE period='{per}'))
            , x AS (SELECT e.DeviceId, e.win, e.Phase, e.pred, e.bL, e.bM, g.m,
                           epoch(least(g.t_end, e.t1)) - epoch(greatest(g.t_start, e.t0)) d
                    FROM e JOIN g ON g.dev = e.dev AND g.t_end > e.t0 AND g.t_start < e.t1
                    WHERE e.period = '{per}')
            SELECT DeviceId, win, Phase, pred,
                   sum(CASE WHEN (m & bL) > 0 AND (m & bM) > 0 THEN d ELSE 0 END) both_,
                   sum(CASE WHEN (m & bL) > 0 OR (m & bM) > 0 THEN d ELSE 0 END) any_
            FROM x GROUP BY ALL""").df())
    con.close()
    j = pd.concat(parts, ignore_index=True)
    j["jac"] = np.where(j.any_ > 0, j.both_ / j.any_, np.nan)
    out = err.merge(j[["DeviceId", "win", "Phase", "pred", "jac"]],
                    on=["DeviceId", "win", "Phase", "pred"], how="left")
    return out.jac.fillna(0.0).to_numpy()


def hand_phase() -> pd.DataFrame:
    """Newest hand config (current export), else the Dec-2024 file; evidence only."""
    cur = pd.read_parquet(DC_WORK / "data" / "labels" / "detector_config_current.parquet")
    old = pd.read_csv(DC_WORK / "data" / "raw" / "detector-configs.csv")
    out = []
    for d, tag in ((cur, "current"), (old, "dec2024")):
        d = d[["DeviceId", "Detector", "Phase"]].copy()
        d["DeviceId"] = d.DeviceId.str.lower()
        d["Detector"] = pd.to_numeric(d.Detector, errors="coerce")
        d = d.dropna().astype({"Detector": int, "Phase": int})
        d = d.drop_duplicates(["DeviceId", "Detector"], keep="first")
        d["hand_src"] = tag
        out.append(d)
    h = pd.concat(out).drop_duplicates(["DeviceId", "Detector"], keep="first")
    return h.rename(columns={"DeviceId": "dev", "Phase": "hand_phase"})


def main() -> None:
    q = pd.read_parquet(RES / "blend13_rows.parquet")
    q["dev"] = q.DeviceId.str.replace("@stg", "", regex=False)
    # guard: the locked hold-outs must not be here (ids only, no data read)
    locked = set(pd.read_csv(DC_WORK / "data" / "splits" / "test_config.csv").DeviceId.str.lower())
    locked |= set(pd.read_csv(DC_WORK / "official" / "newtest_signals.csv").DeviceId.str.lower())
    assert not (set(q.dev) & locked), "locked signal in the OOF rows"

    t = top2(q, "p2")
    info = q.groupby(DET).agg(Phase=("Phase", "first"), fold=("fold", "first"),
                              n_act=("n_act", "max"), fam=("fam", "first")).reset_index()
    t = t.merge(info, on=DET)
    for col, tag in (("p_lg", "trees"), ("p_nn", "net"), ("p0", "ranker")):
        a = top2(q, col)[DET + ["pred"]].rename(columns={"pred": f"pred_{tag}"})
        t = t.merge(a, on=DET, how="left")
    t["ok"] = (t.pred == t.Phase)
    t["dev"] = t.DeviceId.str.replace("@stg", "", regex=False)
    t["period"] = np.where(t.DeviceId.str.endswith("@stg"), "stg", "dec")

    # ---- detector-level evidence, from all 22 windows ---------------------------------
    full = t[t.fam == "full"][["DeviceId", "Detector", "pred", "p", "ok"]].rename(
        columns={"pred": "pred_full", "p": "p_full", "ok": "ok_full"})
    t = t.merge(full, on=["DeviceId", "Detector"], how="left")
    cons = t.groupby(["DeviceId", "Detector", "pred"]).size().rename("n_same").reset_index()
    t = t.merge(cons, on=["DeviceId", "Detector", "pred"], how="left")
    t["n_win"] = t.groupby(["DeviceId", "Detector"]).win.transform("size")
    t["consist"] = t.n_same / t.n_win
    # signal-wide: detectors of the signal making the same label->model move at full window
    fe = t[(t.fam == "full") & ~t.ok]
    sw = fe.groupby(["DeviceId", "Phase", "pred"]).size().rename("n_sigwide").reset_index()
    t = t.merge(sw, on=["DeviceId", "Phase", "pred"], how="left").fillna({"n_sigwide": 0})
    # swap: the signal also has a full-window error going the other way (pred -> Phase)
    rev = sw.rename(columns={"Phase": "pred", "pred": "Phase", "n_sigwide": "n_swap"})
    t = t.merge(rev, on=["DeviceId", "Phase", "pred"], how="left").fillna({"n_swap": 0})
    acc = accept_map()
    t["accepted"] = [int(m) in acc.get((d, int(k)), set())
                     for d, k, m in zip(t.dev, t.Detector, t.pred)]
    h = hand_phase()
    t = t.merge(h, on=["dev", "Detector"], how="left")
    lab = pd.read_parquet(DC_WORK / "official" / "labels_official.parquet")
    lab = lab.assign(dev=lab.DeviceId.str.lower(), Detector=lab.Detector.astype(int))
    t = t.merge(lab[["dev", "Detector", "DeviceName", "description"]], on=["dev", "Detector"],
                how="left")

    fams = ("m30", "full")
    e = t[(~t.ok) & t.fam.isin(fams)].copy()
    e["jac"] = green_jaccard(e)
    e["hand_diff"] = e.hand_phase.notna() & (e.hand_phase != e.Phase)
    e["hand_eq_model"] = e.hand_phase.notna() & (e.hand_phase == e.pred)
    # the detector's full-window green Jaccard for the same move: inside a pair that is green
    # together >= J_CONC of the time, model confidence is no evidence -- only hand = model is
    jf = e[e.fam == "full"][["DeviceId", "Detector", "pred", "jac"]].rename(
        columns={"jac": "jac_full"})
    e = e.merge(jf, on=["DeviceId", "Detector", "pred"], how="left")
    e["lab_ok"] = ((e.pred_full == e.pred) & (e.p_full >= P_CONF) & (e.consist >= CONSIST)
                   & ((e.n_sigwide >= 2) | e.hand_diff)
                   & ((e.jac_full < J_CONC) | e.hand_eq_model))
    ph = e.description.fillna("").str.extract(r"(?i)\b(?:ph|phase)\s*(\d{1,2})(?!\d)")[0]
    e["desc_phase"] = pd.to_numeric(ph, errors="coerce")
    conc_pair = np.array([frozenset((int(a), int(b))) in CONCURRENT_PAIRS
                          for a, b in zip(e.Phase, e.pred)])
    ring = np.array([frozenset((int(a), int(b))) in RING_ADJ for a, b in zip(e.Phase, e.pred)])
    # the stronger core of (a): evidence that does not come from decoder-coupled siblings alone
    e["a_strong"] = e.lab_ok & (e.hand_eq_model | (e.n_sigwide >= 3) | (e.n_swap >= 1))
    e["cat"] = np.select(
        [e.accepted, e.lab_ok, e.jac >= J_CONC, e.n_act < MIN_ACT],
        ["a0_switch_additional", "a_label", "b_concurrent", "c_lowdata"], "d_model")
    e["dtype"] = np.select(
        [e.cat != "d_model", (e.p >= P_CONF) & (e.pred_full == e.pred) & (e.consist >= CONSIST),
         conc_pair, ring],
        ["", "confident_unconfirmed", "partial_concurrent", "ring_neighbour"], "other")
    e["none_right"] = ((e.pred_trees != e.Phase) & (e.pred_net != e.Phase)
                       & (e.pred_ranker != e.Phase))
    e.to_parquet(RES / "residual_errors.parquet", index=False)

    # ---- numbers for the note -------------------------------------------------------
    s: dict = {"thresholds": dict(P_CONF=P_CONF, CONSIST=CONSIST, J_CONC=J_CONC,
                                  MIN_ACT=MIN_ACT)}
    for fam in fams:
        a = t[t.fam == fam]
        x = e[e.fam == fam]
        n, ne = len(a), len(x)
        per_win = a.groupby("win").ok.mean().mean()
        cats = x.cat.value_counts().to_dict()
        fix_a = x.cat.isin(["a0_switch_additional", "a_label"]).sum()
        s[fam] = dict(
            n=int(n), errors=int(ne), acc_pooled=round(float(a.ok.mean()), 5),
            acc_per_window=round(float(per_win), 5), cats=cats,
            shares={k: round(v / ne, 3) for k, v in cats.items()},
            ceiling_a_fixed=round(float((a.ok.sum() + fix_a) / n), 5),
            ceiling_a_b_removed=round(float((a.ok.sum() + fix_a) /
                                            (n - (x.cat == "b_concurrent").sum())), 5),
            none_right=int(x.none_right.sum()),
            none_right_by_cat=x.groupby("cat").none_right.sum().astype(int).to_dict(),
            d_types=x[x.cat == "d_model"].dtype.value_counts().to_dict(),
            d_top_moves=(x[x.cat == "d_model"].Phase.astype(int).astype(str) + "->"
                         + x[x.cat == "d_model"].pred.astype(int).astype(str))
            .value_counts().head(8).to_dict(),
            d_median_p=round(float(x[x.cat == "d_model"].p.median()), 3),
            d_median_nact=float(x[x.cat == "d_model"].n_act.median()),
            a_evidence=dict(sigwide=int(((x.cat == "a_label") & (x.n_sigwide >= 2)).sum()),
                            hand_diff=int(((x.cat == "a_label") & x.hand_diff).sum()),
                            hand_eq_model=int(((x.cat == "a_label") & x.hand_eq_model).sum())),
            signals_a=int(x[x.cat == "a_label"].dev.nunique()),
            a_strong=int(((x.cat == "a_label") & x.a_strong).sum()),
            a_desc_eq_model=int(((x.cat == "a_label") & (x.desc_phase == x.pred)).sum()),
            a_desc_eq_label=int(((x.cat == "a_label") & (x.desc_phase == x.Phase)).sum()),
            ceiling_a_strong_fixed=round(float((a.ok.sum() + (x.cat == "a0_switch_additional").sum()
                                                + ((x.cat == "a_label") & x.a_strong).sum()) / n), 5),
            ceiling_a0_only=round(float((a.ok.sum() + (x.cat == "a0_switch_additional").sum()) / n), 5),
            b_sens={str(j): int(((x.cat.isin(["b_concurrent", "c_lowdata", "d_model"]))
                                 & (x.jac >= j)).sum()) for j in (0.8, 0.9, 0.95)},
            c_sens={str(k): int(((x.cat.isin(["c_lowdata", "d_model"])) & (x.n_act < k)).sum())
                    for k in (5, 10, 20)},
            conc_listpair_errors=int(sum(frozenset((int(a), int(b))) in CONCURRENT_PAIRS
                                         for a, b in zip(x.Phase, x.pred))),
            by_fold=x.groupby("fold").size().astype(int).to_dict(),
        )
        # how many of the full-window hand-labelled errors does the hand config back?
        hl = x[x.hand_phase.notna()]
        s[fam]["hand_labelled_errors"] = int(len(hl))
        s[fam]["hand_eq_model_all"] = int(hl.hand_eq_model.sum())
        s[fam]["hand_eq_timing_all"] = int((hl.hand_phase == hl.Phase).sum())
    # hand-vs-timing agreement on all scored full-window detectors (context)
    f = t[(t.fam == "full") & t.hand_phase.notna()]
    s["hand_vs_timing_full"] = dict(n=int(len(f)),
                                    agree=round(float((f.hand_phase == f.Phase).mean()), 4))
    # 30-min errors on detectors whose full-window answer is a label suspect
    m30 = e[e.fam == "m30"]
    s["m30_on_full_suspect_detectors"] = int(m30.merge(
        e[(e.fam == "full") & (e.cat == "a_label")][["DeviceId", "Detector"]],
        on=["DeviceId", "Detector"]).shape[0])
    json.dump(s, open(RES / "residual_summary.json", "w"), indent=1, default=str)
    print(json.dumps(s, indent=1, default=str))

    # ---- review list: category a at the full window ----------------------------------
    r = e[(e.fam == "full") & (e.cat == "a_label")].copy()
    why = []
    for x in r.itertuples():
        w = []
        if x.n_sigwide >= 2:
            w.append(f"{int(x.n_sigwide)} detectors on this signal move {int(x.Phase)}->{int(x.pred)}")
        if x.n_swap >= 1:
            w.append(f"swap: {int(x.n_swap)} detector(s) here also move {int(x.pred)}->{int(x.Phase)}")
        if x.hand_eq_model:
            w.append(f"hand config says {int(x.hand_phase)} (= model)")
        elif x.hand_diff:
            w.append(f"hand config says {int(x.hand_phase)} (neither)")
        elif pd.notna(x.hand_phase):
            w.append("hand config agrees with timing")
        if pd.notna(x.desc_phase):
            dp = int(x.desc_phase)
            w.append(f"description names phase {dp}"
                     + (" (= model)" if dp == x.pred else " (= label)" if dp == x.Phase else ""))
        w.append(f"model phase top-1 in {int(x.n_same)}/{int(x.n_win)} windows")
        w.append(f"label/model phases green together {x.jac:.0%} of their green time")
        why.append("; ".join(w))
    r["why"] = why
    prev = REPO / "review" / "official_vs_model_disagreements.csv"
    if prev.exists():
        pv = pd.read_csv(prev)
        pv = set(zip(pv.DeviceId.str.lower(), pv.Detector.astype(str)))
        r["on_earlier_list"] = [(d, str(k)) in pv for d, k in zip(r.dev, r.Detector)]
    r = r.sort_values(["a_strong", "p", "n_sigwide"], ascending=[False, False, False]).head(60)
    out = pd.DataFrame({
        "DeviceId": r.dev, "DeviceName": r.DeviceName, "data_period": r.period,
        "detector_channel": r.Detector.astype(int), "channel_description": r.description,
        "labelled_phase_timing": r.Phase.astype(int),
        "hand_config_phase": r.hand_phase.astype("Int64"),
        "model_phase": r.pred.astype(int), "model_prob": r.p.round(3),
        "second_phase": r.pred2.astype("Int64"), "second_prob": r.p_2.round(3),
        "n_actuations": r.n_act.astype(int), "why": r.why,
        "evidence": np.where(r.a_strong, "strong", "moderate"),
        "on_earlier_list": r.get("on_earlier_list", False),
        "user_verdict": ""})
    dst = REPO / "review" / "phase_label_suspects_round1.xlsx"
    with pd.ExcelWriter(dst) as xw:
        out.to_excel(xw, sheet_name="suspects", index=False)
        pd.DataFrame({"read_me": [
            "Phase-label suspects, round 1. Training/dev signals only, full data window, out-of-fold "
            "(the model never saw the signal).",
            "labelled_phase_timing = the controller timing database target (call_phase), the "
            "current truth. hand_config_phase = the hand-maintained config (evidence only).",
            f"Listed: model >= {P_CONF} confident, same answer in >= {int(CONSIST*100)}% of the "
            "sample windows, AND signal-wide pattern or hand config disagreeing with the timing.",
            "user_verdict: write MODEL, LABEL, OTHER:<phase> or ? (unsure)."]}).to_excel(
            xw, sheet_name="read_me", index=False)
    print(f"review list: {len(out)} rows (of {int((e.fam == 'full').sum())} full-window errors, "
          f"{int(((e.fam == 'full') & (e.cat == 'a_label')).sum())} in category a) -> {dst}")


if __name__ == "__main__":
    main()
