"""Note 56: the pick rule with a STACK-RELATIVE health key and a tighter spanning flag (inputs lanes/ln7_stackhealth.py).

    python atspm_pick56.py --run <run dir> [--lanes ln5_lanes_v3s] [--pick ln6_pick_v3s] [--sh ln7_stackhealth_v3s]

Keys (hi-res only; set once here, see note 56 for how they were chosen):
  unh2   member of a hi-res stack whose binned counts scatter around the shared independent reference far more than its
         best partner's: chi >= CHI_MIN and chi >= CHI_X * (partner min chi + 0.5)   (replaces health_core's window status)
  span2  note 55's span flag AND the side-by-side cue: the covered lane detectors see clearly more near-simultaneous ONs
         than chance (coinc_x >= COINC_MIN per peer ON) - two lanes, not two pieces of one lane
Variants (all stack-scoped, note 55's decode): greedy | pick55 (note 55: health_core status + span1) | pick_h2 | pick_s2 |
pick_h2s2 | pick_h2s2_shuf (inputs permuted within phase groups). Reports = note 55's (ATSPM E/R + CI, per seed, stacked
members, contests with evaluation-only labels, lane-count exactness) + the health keys' hit / false-flag rates.
OOF six folds, locked_v2 asserted absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import atspm_decode as AD  # noqa: E402
import atspm_pick55 as A  # noqa: E402
import atspm_score as S  # noqa: E402
import v3_retrain as V  # noqa: E402

OUTD = V.DCW / "trackA" / "atspm56"
CHI_MIN, CHI_X = 10.0, 3.0
COINC_MIN = 0.02
COLS = {"pick55": ("unh1", "span1", "span1_peers"), "pick_h2": ("unh2", "span1", "span1_peers"),
        "pick_s2": ("unh1", "span2", "span2_peers"), "pick_h2s2": ("unh2", "span2", "span2_peers"),
        "pick_h2s2_shuf": ("unh2", "span2", "span2_peers"),
        # s12: span2 decides the lane extension (a spanning zone put on its peers' lanes), span1 the in-stack key
        "pick_h2s12": ("unh2", "span2", "span2_peers"), "pick_h2s12_shuf": ("unh2", "span2", "span2_peers"),
        # 56b: in-stack loser rule; "_nf" = next free class (note 55/56), plain = best non-ATSPM class (new default)
        "pick55_nf": ("unh1", "span1", "span1_peers"), "pick_h2_nf": ("unh2", "span1", "span1_peers"),
        "pick_h2_nl": ("unh2", "span1", "span1_peers"), "pick_h2_nl_shuf": ("unh2", "span1", "span1_peers"),
        "pick_h2_nlap": ("unh2", "span1", "span1_peers")}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def attach_sh(fr: pd.DataFrame, tag: str) -> pd.DataFrame:
    H = pd.read_parquet(V.DCW / "lanes" / f"{tag}.parquet").astype({"Detector": fr.Detector.dtype})
    assert not H.DeviceId.isin(V.locked_signals()).any()
    k = ["DeviceId", "period", "win", "clus"]

    def pmin(v):
        a = v.to_numpy(float)
        out = np.full(len(a), np.nan)
        for i in range(len(a)):
            o = np.delete(a, i)
            o = o[np.isfinite(o)]
            if len(o):
                out[i] = o.min()
        return pd.Series(out, index=v.index)
    H["chi_p"] = H.groupby(k, group_keys=False)["chi"].apply(pmin)
    H["unh2"] = (H.chi >= CHI_MIN) & (H.chi >= CHI_X * (H.chi_p + 0.5))
    H["span2"] = H.sp_span1.eq(True) & (H.sp_coinc_x >= COINC_MIN)
    x = fr[["DeviceId", "Detector", "period", "win"]].merge(
        H[["DeviceId", "Detector", "period", "win", "unh2", "span2", "chi", "chi_p", "sp_peers", "sp_coinc_x"]],
        on=["DeviceId", "Detector", "period", "win"], how="left")
    assert len(x) == len(fr)
    fr["unh1"], fr["span1"], fr["span1_peers"] = fr.unhealthy.to_numpy(), fr.span.to_numpy(), fr.span_peers.to_numpy()
    fr["unh2"] = x.unh2.eq(True).to_numpy()
    fr["span2"] = x.span2.eq(True).to_numpy()
    fr["span2_peers"] = np.where(fr.span2, x.sp_peers.fillna("").to_numpy(object), "")
    fr["chi"], fr["chi_p"] = x.chi.to_numpy(float), x.chi_p.to_numpy(float)
    return fr


def use(fr, v):
    """point note 55's pick columns at a variant's inputs (A.pick_inputs reads unhealthy / span / span_peers)."""
    u, s, p = COLS.get(v, COLS["pick55"])
    fr["unhealthy"], fr["span"], fr["span_peers"] = fr[u].to_numpy(), fr[s].to_numpy(), fr[p].to_numpy()


def decode(fr, P, G, v):
    if v == "greedy":
        return A.decode_all(fr, P, G, "greedy")
    use(fr, v)
    if v.endswith("_nf") or "_nl" in v:
        return decode_loser(fr, P, G, "next_free" if v.endswith("_nf") else "nonatspm_ap" if "_nlap" in v else "nonatspm",
                            shuf=v.endswith("_shuf"))
    if "s12" in v:
        return decode_s12(fr, P, G, shuf=v.endswith("_shuf"))
    return A.decode_all(fr, P, G, "pick_shuf" if v.endswith("_shuf") else "pick")


def decode_loser(fr, P, G, loser, shuf=False, seed=55):
    pred = P.argmax(1).copy()
    rng = np.random.default_rng(seed)
    for ix in G:
        ls = A.lane_sets(fr, ix)
        pk = A.pick_inputs(fr, ix, ls, "abc", rng if shuf else None)
        pk["stack_loser"] = loser
        pred[ix] = AD.decode_group(P[ix], ls, "greedy", "strict", pk)
    return pred


def decode_s12(fr, P, G, shuf=False, seed=55):
    pred = P.argmax(1).copy()
    rng = np.random.default_rng(seed)
    s1 = fr.span1.to_numpy()
    for ix in G:
        ls = A.lane_sets(fr, ix)
        pk = A.pick_inputs(fr, ix, ls, "abc")
        sk = s1[ix].copy()
        # in-stack ordering: also link span1 zones to the detectors they cover (as note 55 did)
        det = fr.Detector.to_numpy()[ix].astype(int)
        pos = {d: k for k, d in enumerate(det)}
        for k, (f, peers) in enumerate(zip(sk, fr.span1_peers.to_numpy(object)[ix])):
            if f:
                for q in str(peers).split(","):
                    if q and int(q) in pos:
                        pk["stack"][k].add(pos[int(q)])
                        pk["stack"][pos[int(q)]].add(k)
        if shuf:
            o = rng.permutation(len(ix))
            pk["unhealthy"], pk["track"] = pk["unhealthy"][o], pk["track"][o]
            pk["span_lanes"], sk = [pk["span_lanes"][j] for j in o], sk[o]
        pk["span_key"] = sk
        pred[ix] = AD.decode_group(P[ix], ls, "greedy", "strict", pk)
    return pred


def lane_counts(fr, P, G, preds):
    rows = []
    pred0 = P.argmax(1)
    for ix in G:
        nt = fr.nl_true.to_numpy()[ix[0]]
        if not np.isfinite(nt) or nt < 1:
            continue
        ls = A.lane_sets(fr, ix)
        r = {"true": int(nt), "stk": bool(fr.stack_group.notna().to_numpy()[ix].any()),
             "sf1": bool(fr.span1.to_numpy()[ix].any()), "sf2": bool(fr.span2.to_numpy()[ix].any())}
        for v, p in preds.items():
            if v == "greedy":
                la = ls
            else:
                use(fr, v)
                la = AD._pick_lanes(ls, pred0[ix], A.pick_inputs(fr, ix, ls, "abc"))
            held = set()
            for k, i in enumerate(ix):
                if AD.ATSPM[p[i]] and len(la[k]) == 1:
                    held |= la[k]
            r[v] = max(1, len(held))
        rows.append(r)
    D = pd.DataFrame(rows)
    out = {}
    for scope, m in (("all", np.ones(len(D), bool)), ("stack_phases", D.stk.to_numpy()),
                     ("span1_flagged", D.sf1.to_numpy()), ("span2_flagged", D.sf2.to_numpy())):
        d = D[m]
        out[scope] = {"phase_windows": int(len(d))}
        for v in preds:
            out[scope][v] = [round(float((d[v] == d.true).mean()), 4), round(float((d[v] > d.true).mean()), 4),
                             round(float((d[v] < d.true).mean()), 4)] if len(d) else None
    return out


def health_rates(fr):
    """hit / false-flag of the two health keys on stack members (>= 30 min, in a hi-res stack), evaluation labels."""
    m = fr.chi.notna().to_numpy() & ~fr.wgroup.isin(["m5", "m10"]).to_numpy()
    y = fr.ev_unhealthy.to_numpy()[m]
    out = {"members": int(m.sum()), "label_unhealthy": int(y.sum())}
    for k in ("unh1", "unh2"):
        f = fr[k].to_numpy()[m]
        out[k] = {"flags": int(f.sum()), "hit": round(float(f[y].mean()), 4), "false": round(float(f[~y].mean()), 4),
                  "precision": round(float(y[f].mean()), 4) if f.any() else None}
    return out


def span_rates(fr):
    m = fr.ev_span.notna().to_numpy() & ~fr.wgroup.isin(["m5", "m10"]).to_numpy()
    pr = np.array(V.C7)[fr[[f"P_{c}" for c in V.C7]].to_numpy().argmax(1)]
    out = {}
    for scope, mm in (("all", m), ("pred_APC", m & np.isin(pr, ["Advance", "Presence", "Count"])),
                      ("pred_Advance", m & (pr == "Advance"))):
        y = fr.ev_span.to_numpy()[mm] >= 2
        out[scope] = {}
        for k in ("span1", "span2"):
            f = fr[k].to_numpy()[mm]
            out[scope][k] = {"flags": int(f.sum()), "on_print_span": int((f & y).sum()), "on_print_single": int((f & ~y).sum()),
                             "precision": round(float(y[f].mean()), 3) if f.any() else None}
    return out


VARIANTS = ["greedy", "pick55_nf", "pick_h2_nf", "pick_h2_nlap"]       # 56b second run (first: pick56b.json)
SEEDV = ("greedy", "pick_h2_nf", "pick_h2_nlap")
OUTNAME = "pick56b_ap.json"


def main(run, lanes, pick, sh, tag):
    out = OUTD / tag
    out.mkdir(parents=True, exist_ok=True)
    fr = S.attach_lanes(S.load(run), lanes)
    fr = A.attach_pick(fr, pick)
    fr = A.eval_labels(fr)
    fr = attach_sh(fr, sh)
    Ps = A.seed_probs(fr, run)
    Pm = np.mean(Ps, 0)
    G = A.groups(fr)
    variants = VARIANTS
    preds = {v: decode(fr, Pm, G, v) for v in variants}
    R = A.score_rows(fr)
    res = {"run": run, "keys": {"CHI_MIN": CHI_MIN, "CHI_X": CHI_X, "COINC_MIN": COINC_MIN},
           "health_rates": health_rates(fr), "span_rates": span_rates(fr)}
    for sname, rows in R.items():
        d = {v: S.credit(fr, preds[v], "truth_v3s", rows, True) for v in variants}
        res[sname] = {}
        for v in variants:
            s = S.summ(d[v])
            s["ci_vs_greedy_pt"] = S.boot(d["greedy"], d[v])
            s["ci_vs_pick55_pt"] = S.boot(d["pick55" if "pick55" in d else "pick_h2_nf"], d[v])
            res[sname][v] = s
    res["per_seed"] = {}
    for si, P in enumerate(Ps):
        r = {}
        for v in SEEDV:
            p = decode(fr, P, G, v)
            r[v] = {sn: S.summ(S.credit(fr, p, "truth_v3s", rows, True))["atspm"] for sn, rows in R.items()}
        res["per_seed"][si] = r
    m = R["everything"]
    big = m[~np.isin(fr.wgroup.to_numpy()[m], ["m5", "m10"])]
    st = big[fr.stack_group.notna().to_numpy()[big]]
    res["stacked_members_ge30"] = {v: S.summ(S.credit(fr, preds[v], "truth_v3s", st, True)) for v in variants}
    use(fr, "pick55")                                     # contests defined as in note 55 (same contest set)
    preds["pick55"] = preds.get("pick55", preds.get("pick55_nf"))
    CV = [v for v in variants]
    C = A.contests(fr, Pm, G, {v: preds[v] for v in CV})
    C.to_parquet(out / OUTNAME.replace(".json", "_contests.parquet"), index=False)
    res["contests"] = A.contest_summary(C, CV)
    res["lane_counts"] = lane_counts(fr, Pm, G, {v: preds[v] for v in SEEDV})
    # signal 13025 (user's case): who holds Advance on P6, per window
    k = fr.DeviceId.eq("5feb1117-f119-4e7c-bc6e-0b17ec9136ac").to_numpy() & fr.Detector.isin([28, 39]).to_numpy()
    ix = np.flatnonzero(k)
    res["case_13025"] = [{"period": fr.period.iat[i], "win": fr.win.iat[i], "det": int(fr.Detector.iat[i]),
                          **{v: V.C7[preds[v][i]] for v in SEEDV},
                          "unh1": bool(fr.unh1.iat[i]), "unh2": bool(fr.unh2.iat[i])} for i in ix]
    json.dump(res, open(out / OUTNAME, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "case_13025"}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run_51ba131222_exclude_min5_clean_valnc_h3_drfp")
    ap.add_argument("--lanes", default="ln5_lanes_v3s")
    ap.add_argument("--pick", default="ln6_pick_v3s")
    ap.add_argument("--sh", default="ln7_stackhealth_v3s")
    ap.add_argument("--tag", default="v3s")
    a = ap.parse_args()
    main(a.run, a.lanes, a.pick, a.sh, a.tag)
