"""Note 55: the stacked-group PICK rule inside the per-lane ATSPM decode (atspm_decode.py `pick`, inputs ln6_pick.py).

    python atspm_pick55.py --run <run dir> --lanes ln5_lanes_v3s --pick ln6_pick_v3s --tag v3s

Variants on note 54's primary decode (greedy, strict span, gated at >= 30 min; 5 / 10 min keep argmax):
  greedy   note 54 (most probable member gets the class)
  ff       first-choice ATSPM claims settled before second choices, by probability (control for the ordering)
  pick     + (a) unhealthy loses, (b) spanning zone not a lane of its own / lane-by-lane wins, (c) Advance tracking the
           stop-bar Count total wins - keys act only inside a hi-res stack (co-located / spanning-cover claimants)
           pick_a / pick_b / pick_c = one key at a time; pick_global = keys over ALL first-choice claims (first version)
  pick_shuf  pick with the three inputs permuted among the detectors of each phase group (shuffled control)
Reports, all OOF on the six folds (locked_v2 asserted absent):
  * ATSPM score (stack credit; v3s truth, Dec fp rule = note 54 step 4), everything / realistic, signal-bootstrap CI vs
    greedy, per seed (each seed's OOF decoded on its own = the model noise floor).
  * Pick evaluation with EVALUATION-only labels (never model inputs): contests = >= 2 detectors of one predicted
    phase-window claiming the same ATSPM class as first choice on overlapping lanes (pick-adjusted lanes). Health
    label = dead | dq_suspect | validated 'unhealthy' (prints / DQ); span label = print lanes_spanned >= 2 vs 1;
    technology from the print. Share of contests where the class goes to the healthy / lane-by-lane member.
  * Inferred lane count after the decode (= distinct lanes held by single-lane ATSPM-decoded detectors) vs the print's
    n_lanes_phase, before (greedy) / after (pick), all phase-windows >= 30 min with a print lane count.
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
import atspm_score as S  # noqa: E402
import v3_retrain as V  # noqa: E402

C7 = list(V.C7)
ATS = S.ATS
OUTD = V.DCW / "trackA" / "atspm55"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def seed_probs(fr: pd.DataFrame, run: str) -> list:
    out = []
    for s in (0, 1, 2):
        P, hv, got = V.load_oof(fr, V.OUT / run, "first.all.wi", s, range(6))
        assert len(got) == 6 and hv.all()
        out.append(P)
    return out


def attach_pick(fr: pd.DataFrame, tag: str) -> pd.DataFrame:
    K = pd.read_parquet(V.DCW / "lanes" / f"{tag}.parquet").astype({"Detector": fr.Detector.dtype})
    x = fr[["DeviceId", "Detector", "period", "win"]].merge(K, on=["DeviceId", "Detector", "period", "win"], how="left")
    assert len(x) == len(fr)
    fr["h_status"] = x.health.fillna("not_run").to_numpy()
    fr["unhealthy"] = x.health.isin(["bad", "suspect"]).to_numpy()
    fr["span"] = x.span.eq(True).to_numpy()
    fr["span_peers"] = x.span_peers.fillna("").to_numpy()
    fr["coloc_peers"] = x.coloc_peers.fillna("").to_numpy()
    fr["track"] = x.track.to_numpy(float)
    return fr


def groups(fr: pd.DataFrame, lane_col: str = "lanes5g"):
    sub = fr.loc[(fr.det_n_on >= 5), ["DeviceId", "period", "win", "pred_phase", lane_col]]
    has = sub[lane_col].notna().groupby([sub.DeviceId, sub.period, sub.win, sub.pred_phase]).transform("sum")
    sub = sub[has.to_numpy() >= 2]
    return [sub.index.to_numpy()[g] for g in sub.groupby(["DeviceId", "period", "win", "pred_phase"]).indices.values()]


def lane_sets(fr, ix, lane_col="lanes5g"):
    return [frozenset(int(v) for v in s.split(",")) if isinstance(s, str) and s else frozenset()
            for s in fr[lane_col].to_numpy(object)[ix]]


def pick_inputs(fr, ix, ls, which="abc", shuffle_rng=None, scoped=True):
    det = fr.Detector.to_numpy()[ix].astype(int)
    pos = {d: k for k, d in enumerate(det)}
    unh = fr.unhealthy.to_numpy()[ix].copy()
    tr = fr.track.to_numpy(float)[ix].copy()
    sp = []
    for k, (f, peers) in enumerate(zip(fr.span.to_numpy()[ix], fr.span_peers.to_numpy(object)[ix])):
        if not f:
            sp.append(None)
            continue
        u = frozenset()
        for p in str(peers).split(","):
            if p and int(p) in pos:
                u |= ls[pos[int(p)]]
        sp.append(u or None)
    stack = None
    if scoped:                                  # hi-res stack: co-located, or spanning zone <-> a detector it covers
        stack = [set() for _ in ix]
        cp, spp, spf = fr.coloc_peers.to_numpy(object)[ix], fr.span_peers.to_numpy(object)[ix], fr.span.to_numpy()[ix]
        for k in range(len(ix)):
            for q in [x for x in str(cp[k]).split(",") if x] + ([x for x in str(spp[k]).split(",") if x] if spf[k] else []):
                if int(q) in pos:
                    stack[k].add(pos[int(q)])
                    stack[pos[int(q)]].add(k)
    if shuffle_rng is not None:
        o = shuffle_rng.permutation(len(ix))
        unh, tr, sp = unh[o], tr[o], [sp[j] for j in o]
    return {"stack": stack,"unhealthy": unh if "a" in which else np.zeros(len(ix), bool),
            "span_lanes": sp if "b" in which else None,
            "track": tr if "c" in which else np.full(len(ix), np.nan)}


def decode_all(fr, P, G, variant: str, seed: int = 55):
    pred = P.argmax(1).copy()
    rng = np.random.default_rng(seed)
    for ix in G:
        ls = lane_sets(fr, ix)
        if variant == "greedy":
            pk = None
        elif variant == "ff":
            pk = {"order_only": True}
        elif variant == "pick_shuf":
            pk = pick_inputs(fr, ix, ls, "abc", rng)
        elif variant == "pick_global":
            pk = pick_inputs(fr, ix, ls, "abc", scoped=False)
        else:
            pk = pick_inputs(fr, ix, ls, variant.split("_")[1] if "_" in variant else "abc")
        pred[ix] = AD.decode_group(P[ix], ls, "greedy", "strict", pk)
    return pred


def score_rows(fr):
    out = {}
    for sname, scol in (("everything", "setA_s"), ("realistic", "setR_s")):
        out[sname] = np.flatnonzero(fr[scol].to_numpy() & ~fr.dec_fp_truth.to_numpy())
    return out


def eval_labels(fr):
    lab = pd.read_parquet(S.V3S)
    lab["DeviceId"] = lab.DeviceId.str.lower()
    lab = lab.rename(columns={"detector": "Detector"}).astype({"Detector": fr.Detector.dtype})
    lab["ev_unhealthy"] = lab.dead.eq(True) | lab.dq_suspect.eq(True).fillna(False) | lab.validated.eq("unhealthy").fillna(False)
    lab["ev_span"] = lab.lanes_spanned.astype("Float64")
    x = fr[["DeviceId", "Detector"]].merge(lab[["DeviceId", "Detector", "ev_unhealthy", "ev_span", "technology",
                                                "n_lanes_phase", "phase_target"]], on=["DeviceId", "Detector"], how="left")
    fr["ev_unhealthy"] = x.ev_unhealthy.astype("boolean").fillna(False).to_numpy(dtype=bool)
    fr["ev_span"] = x.ev_span.to_numpy(dtype=float, na_value=np.nan)
    fr["tech"] = x.technology.fillna("?").to_numpy()
    # print lane count of the PREDICTED phase (truth phase number = timing phase number, as in note 42)
    ph = lab[lab.phase_target.astype(str).str.match(r"^P\d+$")].copy()
    ph["pn"] = ph.phase_target.str[1:].astype(int)
    nl = ph.dropna(subset=["n_lanes_phase"]).groupby(["DeviceId", "pn"]).n_lanes_phase.agg(lambda s: int(s.mode().iloc[0]))
    key = pd.MultiIndex.from_arrays([fr.DeviceId, fr.pred_phase.astype(int)])
    fr["nl_true"] = nl.reindex(key).to_numpy(dtype=float)
    return fr


def contests(fr, P, G, preds: dict):
    """Per contest: members' eval labels and which member(s) kept the class under each variant."""
    pred0 = P.argmax(1)
    rows = []
    for ix in G:
        ls = lane_sets(fr, ix)
        pk = pick_inputs(fr, ix, ls, "abc")
        la = AD._pick_lanes(ls, pred0[ix], pk)
        for c in (0, 1, 2, 3):
            m = [k for k in range(len(ix)) if pred0[ix[k]] == c and la[k]]
            if len(m) < 2:
                continue
            # connected components of lane overlap among the claimants
            comp, seen = [], set()
            for k in m:
                if k in seen:
                    continue
                st, cc = [k], []
                seen.add(k)
                while st:
                    a = st.pop()
                    cc.append(a)
                    for b in m:
                        if b not in seen and la[a] & la[b]:
                            seen.add(b)
                            st.append(b)
                if len(cc) >= 2:
                    comp.append(cc)
            for cc in comp:
                ii = ix[cc]
                r = {"DeviceId": fr.DeviceId.iat[ii[0]], "period": fr.period.iat[ii[0]], "win": fr.win.iat[ii[0]],
                     "wgroup": fr.wgroup.iat[ii[0]], "cls": C7[c], "n": len(ii),
                     "truth_n": int((fr.truth_v3s.to_numpy(object)[ii] == C7[c]).sum()),
                     "unh": fr.ev_unhealthy.to_numpy()[ii].tolist(), "evspan": fr.ev_span.to_numpy()[ii].tolist(),
                     "tech": fr.tech.to_numpy()[ii].tolist(), "m_unh": fr.unhealthy.to_numpy()[ii].tolist(),
                     "m_span": [bool(pk["span_lanes"][k]) for k in cc]}
                for v, p in preds.items():
                    r[f"kept_{v}"] = (p[ii] == c).tolist()
                rows.append(r)
    return pd.DataFrame(rows)


def contest_summary(C: pd.DataFrame, variants) -> dict:
    out = {}
    for scope, m in (("all", np.ones(len(C), bool)), ("true_stack", C.truth_n.to_numpy() >= 2)):
        D = C[m]
        res = {"contests": int(len(D))}
        # (a) health: members differ on the evaluation health label
        H = D[D.unh.map(lambda u: 0 < sum(u) < len(u))]
        res["health_differs"] = int(len(H))
        for v in variants:
            k = H[f"kept_{v}"]
            healthy_only = [any(kk and not uu for kk, uu in zip(ke, un)) and not any(kk and uu for kk, uu in zip(ke, un))
                            for ke, un in zip(k, H.unh)]
            res[f"healthy_member_only_{v}"] = round(float(np.mean(healthy_only)), 3) if len(H) else None
        # (b) spanning: one member spans >= 2 lanes in the print, another 1 lane
        B = D[D.evspan.map(lambda s: any(x >= 2 for x in s if x == x) and any(x == 1 for x in s if x == x))]
        res["span_differs"] = int(len(B))
        for v in variants:
            cat = []
            for ke, sp in zip(B[f"kept_{v}"], B.evspan):
                lbl = any(kk and x == 1 for kk, x in zip(ke, sp))
                spn = any(kk and x == x and x >= 2 for kk, x in zip(ke, sp))
                cat.append("lane_by_lane" if lbl and not spn else "spanning" if spn and not lbl else
                           "both (extra lane)" if lbl and spn else "neither")
            res[f"span_outcome_{v}"] = pd.Series(cat).value_counts().to_dict() if cat else {}
        # technology view (radar vs loop members)
        T = D[D.tech.map(lambda t: "loop" in t and "radar" in t)]
        res["radar_vs_loop"] = int(len(T))
        for v in variants:
            res[f"radar_vs_loop_winner_{v}"] = pd.Series(
                ["+".join(sorted({t for kk, t in zip(ke, te) if kk})) or "none" for ke, te in zip(T[f"kept_{v}"], T.tech)]
            ).value_counts().to_dict() if len(T) else {}
        # does the hi-res pick input agree with the evaluation labels?
        res["model_unhealthy_vs_label"] = {
            "members": int(sum(len(u) for u in D.unh)),
            "label_unhealthy": int(sum(sum(u) for u in D.unh)),
            "flag_on_label_unhealthy": int(sum(sum(a and b for a, b in zip(u, mu)) for u, mu in zip(D.unh, D.m_unh))),
            "flag_on_label_healthy": int(sum(sum((not a) and b for a, b in zip(u, mu)) for u, mu in zip(D.unh, D.m_unh)))}
        res["model_span_vs_print"] = {
            "print_span": int(sum(sum(x >= 2 for x in s if x == x) for s in D.evspan)),
            "flag_on_print_span": int(sum(sum(f and x == x and x >= 2 for f, x in zip(ms, s)) for ms, s in zip(D.m_span, D.evspan))),
            "flag_on_print_single": int(sum(sum(f and x == 1 for f, x in zip(ms, s)) for ms, s in zip(D.m_span, D.evspan)))}
        out[scope] = res
    return out


def lane_counts(fr, P, G, preds: dict) -> dict:
    """n_lanes after the decode vs the print's n_lanes_phase, per predicted phase-window (>= 30 min)."""
    rows = []
    pred0 = P.argmax(1)
    for ix in G:
        nt = fr.nl_true.to_numpy()[ix[0]]
        if not np.isfinite(nt) or nt < 1:
            continue
        ls = lane_sets(fr, ix)
        pk = pick_inputs(fr, ix, ls, "abc")
        la = AD._pick_lanes(ls, pred0[ix], pk)
        r = {"wgroup": fr.wgroup.iat[ix[0]], "true": int(nt), "raw": len(set().union(*ls)),
             "stk": bool(fr.stack_group.notna().to_numpy()[ix].any()), "spanflag": any(x is not None for x in pk["span_lanes"])}
        for v, p in preds.items():
            lanes_v = la if v in ("pick", "pick_global", "pick_b") else ls
            held = set()
            for k, i in enumerate(ix):
                if AD.ATSPM[p[i]] and len(lanes_v[k]) == 1:
                    held |= lanes_v[k]
            r[v] = max(1, len(held))
        rows.append(r)
    D = pd.DataFrame(rows)
    out = {}
    for scope, m in (("all", np.ones(len(D), bool)), ("stack_phases", D.stk.to_numpy()),
                     ("span_flagged", D.spanflag.to_numpy())):
        d = D[m]
        out[scope] = {"phase_windows": int(len(d))}
        for v in ["raw"] + list(preds):
            out[scope][v] = {"exact": round(float((d[v] == d.true).mean()), 4), "over": round(float((d[v] > d.true).mean()), 4),
                             "under": round(float((d[v] < d.true).mean()), 4)} if len(d) else None
    return out


def main(run, lanes, pick, tag):
    out = OUTD / tag
    out.mkdir(parents=True, exist_ok=True)
    fr = S.attach_lanes(S.load(run), lanes)
    fr = attach_pick(fr, pick)
    fr = eval_labels(fr)
    Ps = seed_probs(fr, run)
    Pm = np.mean(Ps, 0)
    G = groups(fr)
    log(f"{len(G):,} decode groups")
    variants = ["greedy", "ff", "pick", "pick_global", "pick_a", "pick_b", "pick_c", "pick_shuf"]
    preds = {v: decode_all(fr, Pm, G, v) for v in variants}
    preds["argmax"] = Pm.argmax(1)
    R = score_rows(fr)
    res = {"run": run, "pick_inputs": {"h_status": fr.loc[fr.wgroup.isin(["m30", "h1", "h3", "h6", "h24", "full"]), "h_status"]
                                       .value_counts().to_dict(), "span_flags": int(fr.span.sum()),
                                       "track_known": float(np.isfinite(fr.track).mean())}}
    for sname, rows in R.items():
        d0 = S.credit(fr, preds["greedy"], "truth_v3s", rows, True)
        res[sname] = {"greedy": S.summ(d0)}
        for v in ["argmax"] + variants[1:]:
            d1 = S.credit(fr, preds[v], "truth_v3s", rows, True)
            s = S.summ(d1)
            s["ci_vs_greedy_pt"] = S.boot(d0, d1)
            s["changed_vs_greedy"] = int((preds[v][rows] != preds["greedy"][rows]).sum())
            res[sname][v] = s
    # per-seed: each seed's OOF decoded on its own (noise floor of the model under the rule)
    res["per_seed"] = {}
    for s, P in enumerate(Ps):
        pg, pp = decode_all(fr, P, G, "greedy"), decode_all(fr, P, G, "pick")
        r = {}
        for sname, rows in R.items():
            a = S.summ(S.credit(fr, pg, "truth_v3s", rows, True))
            b = S.summ(S.credit(fr, pp, "truth_v3s", rows, True))
            r[sname] = {"greedy": a["atspm"], "pick": b["atspm"], "extra_greedy": a["stack_extra"], "extra_pick": b["stack_extra"],
                        "acc7_greedy": a["acc7"], "acc7_pick": b["acc7"]}
        res["per_seed"][s] = r
    # stacked members only (>= 30 min)
    m = R["everything"]
    big = m[~np.isin(fr.wgroup.to_numpy()[m], ["m5", "m10"])]
    st = big[fr.stack_group.notna().to_numpy()[big]]
    res["stacked_members_ge30"] = {v: S.summ(S.credit(fr, preds[v], "truth_v3s", st, True)) for v in ["argmax", "greedy", "pick", "pick_global"]}
    CV = ["greedy", "pick", "pick_global", "pick_a", "pick_b", "pick_c", "pick_shuf"]
    C = contests(fr, Pm, G, {v: preds[v] for v in CV})
    C.to_parquet(out / "contests.parquet", index=False)
    res["contests"] = contest_summary(C, CV)
    res["lane_counts"] = lane_counts(fr, Pm, G, {v: preds[v] for v in CV})
    json.dump(res, open(out / "pick.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lanes", default="ln5_lanes_v3s")
    ap.add_argument("--pick", default="ln6_pick_v3s")
    ap.add_argument("--tag", default="v3s")
    a = ap.parse_args()
    main(a.run, a.lanes, a.pick, a.tag)
