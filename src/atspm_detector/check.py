"""Check of the installed model.  Run it after installing, or after any change.

    python -m atspm_detector.check          (or: atspm-detector-check)
    python -m atspm_detector.check --freeze (maintainers only: rewrite the references after a model change)

The checks run on the bundled 30-minute single-signal sample (`data/sample_events.parquet`: raw controller events
including codes the pipeline must filter out and duplicate rows it must drop):

1. **Reproduces the stored answers.**  The network's pair phase probabilities and function probabilities, and the final
   per-detector predictions (on the 30-minute sample and on its first 10 minutes, the short-sample path: no lanes, twin
   decode) must match the references in `reference/` -- phase, function, lanes, setback distance, night speed, health.
   Catches a broken install, a wrong model file or a library upgrade that changes the numbers.  The outputs are also
   checked for internal consistency (stop-bar zones at 0 ft, no lane for Bike, n_lanes per phase = the highest lane).
2. **Phase-number invariance.**  Three random renumberings of the phases (30 min) and one of the 10-minute path: every
   prediction follows the renumbering exactly, every probability (each candidate's before and after the decoder, both
   phase ranks, all seven function classes) within 1e-6; lanes, setback and health unchanged.
3. **Channel-order invariance.**  Detector channels reversed, shifted and both (30 min), reversed (10 min): every output
   follows the detector exactly (1e-6).
3b. **Joint channel + phase permutation.**  Three random bijections of the channels onto 1..64 (any order: channel
   differences are NOT kept) combined with random phase renumberings: every probability and output follows (1e-6).
4. **No hidden dependencies.**  predict() runs in a process where torch, lightgbm, scipy, sklearn and pyarrow cannot be
   imported at all.
5. **Profile argument ignored.**  There is one path: predict(profile=...) and env DC_FAST_PROFILE are accepted for
   compatibility and must give exactly the default answers with one FutureWarning; a plain call must not give it.
6. **Health smoke test.**  The shipped health references load; every detector gets a status and a reason
   with its label; categories parse; a detector held ON for 12 minutes is called "Stuck on" with the period listed.
Health text names detectors and phases ("det 15: P5 Presence"); checks 2-3b map those labels back before comparing.

Exit code 0 = all passed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import blend as gb
from . import pipeline as P
from . import streams as st
from .common import ALLOWED_EVENTS, read_json
from .common import timedelta as _td
from .pipeline import DEFAULT_MODEL_DIR, EXTRA_COLS, OUT_COLS, PHASE_COLS, PROB_COLS, predict

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "data" / "sample_events.parquet"
REF_DIR = HERE / "reference"
NET_PHASE_REF = REF_DIR / "phase_net_reference.parquet"
NET_REF = REF_DIR / "funcnet_reference.parquet"
PRED_REF = REF_DIR / "predict_reference.parquet"
PRED10_REF = REF_DIR / "predict_reference_m10.parquet"
PHASE_REF = REF_DIR / "phase_reference.parquet"
KEY = ["DeviceId", "Detector", "cand_phase"]
CLASSES = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
PHASE_EVENTS = [1, 7, 8, 9, 10, 11, 43, 44]
STOPBAR = ("Presence", "Count", "Yellow_Red")
HEALTH = ("ok", "watch", "suspect", "bad", "not_enough_data")
# outputs that carry no phase number and must not move at all when the phases are renumbered
INVARIANT = ["lanes", "lane_conf", "phase_n_lanes", "phase_n_lanes_conf", "distance_ft",
             "setback_confidence", "night_speed_mph", "night_speed_vehicles", "health_status", "health_score",
             "health_reason", "health_bad_periods", "health_watch", "health_categories", "health_config",
             "health_signal_note"]
HEALTH_TEXT = ("health_reason", "health_watch", "health_bad_periods", "health_categories", "health_config",
               "health_signal_note", "review_reason")
PROB_TOL = 1e-6          # float noise only
PERM_SEEDS = (0, 1, 2)
OUT_PROBS = ["phase_prob", "phase_2nd_prob", "phase_guess_prob", "phase_margin", "function_prob",
             "function_guess_prob"] + PROB_COLS


# ------------------------------------------------------------------ parquet i/o without pyarrow (DuckDB)
def read_parquet(path) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        return con.sql(f"SELECT * FROM read_parquet({P._q(Path(path).as_posix())})").df()
    finally:
        con.close()


def write_parquet(df: pd.DataFrame, path) -> None:
    con = duckdb.connect()
    try:
        con.register("df_out", df)
        con.execute(f"COPY df_out TO {P._q(Path(path).as_posix())} (FORMAT parquet)")
    finally:
        con.close()


def _first_minutes(ev: pd.DataFrame, minutes: float) -> pd.DataFrame:
    t = pd.to_datetime(ev.Timestamp)
    return ev[t < t.min() + _td(minutes=minutes)].reset_index(drop=True)


def _net_pass(ev: pd.DataFrame):
    """The network on every (detector, candidate) pair of the sample's streams: pair phase probabilities and function
    probabilities (one pass)."""
    from . import funcnet
    cfg = gb.config(DEFAULT_MODEL_DIR)
    con = P._connect(4, "4GB")
    try:
        w0, w1, _ = P.load_events(con, ev)
        P.build_chunk_tables(con)
        minutes = round((w1 - w0) / 60.0)
        streams, rel = gb.streams_for(con, int(round(w0 * 1000)), int(round(w1 * 1000)), **gb.piece_plan(cfg, minutes))
    finally:
        con.close()
    net = funcnet.FuncNet(DEFAULT_MODEL_DIR / "funcnet")
    rows, prow = [], []
    for dev, z in streams.items():
        dets = [int(c) for c in z["det_ch"]]
        p, pp = net.both(z, dets, rel)
        for d, r in zip(dets, p):
            rows.append((dev, d, *r.tolist()))
        for i, d in enumerate(dets):
            for k, c in enumerate(z["cand"]):
                prow.append((dev, d, int(c), float(pp[i, k])))
    phn = pd.DataFrame(prow, columns=KEY + ["p_gru"]).astype({"Detector": "int64", "cand_phase": "int64"})
    netp = pd.DataFrame(rows, columns=["DeviceId", "Detector"] + [f"P_{c}" for c in funcnet.C7])
    return phn.sort_values(KEY).reset_index(drop=True), netp.sort_values(["DeviceId", "Detector"]).reset_index(drop=True)


def freeze() -> None:
    """maintainers only: (re)write the references after a deliberate model change."""
    REF_DIR.mkdir(exist_ok=True)
    ev = read_parquet(SAMPLE)
    phn, netp = _net_pass(ev)
    write_parquet(phn, NET_PHASE_REF)
    write_parquet(netp, NET_REF)
    out, ph = predict(SAMPLE, min_actuations=1, return_phases=True)
    write_parquet(out, PRED_REF)
    write_parquet(ph, PHASE_REF)
    write_parquet(predict(_first_minutes(ev, 10), min_actuations=1), PRED10_REF)
    print("references frozen")


def _same(a: pd.Series, b: pd.Series, tol: float = 1e-6) -> bool:
    """Equal, NaN == NaN, numbers within tol (dtype-independent)."""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    na, nb = a.isna(), b.isna()
    if not na.equals(nb):
        return False
    a, b = a[~na], b[~nb]
    try:
        return bool(np.allclose(a.astype(float), b.astype(float), atol=tol, rtol=0))
    except (TypeError, ValueError):
        return bool((a.astype(str) == b.astype(str)).all())


def check_new_outputs(out: pd.DataFrame, phases: pd.DataFrame) -> None:
    """Internal consistency of the lane / setback / health columns."""
    assert not set(ALLOWED_EVENTS) & {83, 84, 85, 86, 87, 88}, "fault events must not be read"
    assert out.health_status.notna().all(), "every detector gets a health status"
    assert out.health_status.isin(HEALTH).all(), "unknown health status"
    hs = out.health_score.dropna()
    assert ((hs >= 0) & (hs <= 1)).all(), "health score outside [0, 1]"
    assert out.loc[out.health_status.eq("ok"), "health_bad_periods"].fillna("").eq("").all(), \
        "an ok detector lists bad periods"
    ans = out[out.function_pred.notna()]
    sb = ans[ans.function_pred.isin(STOPBAR)]
    assert (sb.distance_ft == 0).all(), "a stop-bar zone must be at 0 ft"
    assert ans[ans.function_pred.isin(["Other", "Bike"])].distance_ft.isna().all(), \
        "Other / Bike must have no setback distance"
    adv = ans[ans.function_pred.isin(["Advance", "Mid"])]
    assert (adv.distance_ft.dropna() >= 0).all() and adv.distance_ft.notna().any()
    assert out.setback_confidence.dropna().isin(["high", "medium", "low"]).all()
    assert out[out.function_pred.isna()].distance_ft.isna().all(), "a withheld detector has a setback"
    assert ans[ans.function_pred.eq("Bike")].lanes.isna().all(), "a Bike detector got a lane"
    ln = ans[ans.lanes.notna()]
    assert len(ln), "no detector got a lane"
    mx = ln.lanes.map(lambda x: max(int(v) for v in str(x).split(",")))
    assert (mx <= ln.phase_n_lanes).all(), "a lane number above n_lanes of its phase"
    assert ((ln.lane_conf >= 0) & (ln.lane_conf <= 1)).all()
    assert list(phases.columns) == PHASE_COLS and len(phases)
    assert (phases.n_lanes >= 1).all() and (phases.n_lanes <= 4).all()
    pt = phases.set_index(["DeviceId", "phase"]).n_lanes
    for (dev, p), g in ln.groupby(["DeviceId", "phase_pred"]):
        top = max(max(int(v) for v in str(x).split(",")) for x in g.lanes)
        assert pt.loc[(dev, p)] == top, f"phase {p}: n_lanes {pt.loc[(dev, p)]} but highest lane {top}"


def check_reproduces_reference() -> str:
    cfg = gb.config(DEFAULT_MODEL_DIR)
    assert cfg is not None, f"no network in {DEFAULT_MODEL_DIR}"
    assert cfg["runtime"] == "onnx" and Path(cfg["weights_path"]).exists()
    ev = read_parquet(SAMPLE)
    minutes = (ev.Timestamp.max() - ev.Timestamp.min()).total_seconds() / 60.0
    assert gb.piece_plan(cfg, minutes) == {"max_chunks": 48, "grid": 0}, \
        "the bundled sample must take the all-pieces (short-sample) path"
    # the long-sample piece placement must be the one the model was evaluated with: a 32-piece grid, then
    # linspace(0, n - 1, 4).round() -- e.g. 24 h = 48 pieces -> pieces 0, 15, 32, 47
    lp = gb.piece_plan(cfg, 1440)
    got_idx = [a // st.CHUNK_MS for a, _ in st.split_range(0, 1440 * 60_000, **lp)]
    assert got_idx == [0, 15, 32, 47], f"long-sample piece placement changed: {got_idx}"

    # -- the network on its own (phase head on every pair; function head)
    got, netp = _net_pass(ev)
    ref = read_parquet(NET_PHASE_REF).sort_values(KEY).reset_index(drop=True)
    assert len(got) == len(ref) and (got[KEY].to_numpy() == ref[KEY].to_numpy()).all(), "network output shape changed"
    d = np.abs(got.p_gru.to_numpy() - ref.p_gru.to_numpy()).max()
    assert d < 1e-4, f"network phase probabilities drifted by {d:.2e}"
    assert (got.groupby(["DeviceId", "Detector"]).p_gru.idxmax().to_numpy() ==
            ref.groupby(["DeviceId", "Detector"]).p_gru.idxmax().to_numpy()).all(), "the winning candidate changed"
    nref = read_parquet(NET_REF)
    assert (netp[["DeviceId", "Detector"]].to_numpy() == nref[["DeviceId", "Detector"]].to_numpy()).all(), \
        "function network output shape changed"
    pc = [c for c in nref.columns if c.startswith("P_")]
    dn = np.abs(netp[pc].to_numpy() - nref[pc].to_numpy()).max()
    assert dn < 1e-4, f"function network probabilities drifted by {dn:.2e}"
    d = max(d, dn)

    # -- the whole pipeline
    out, phases = predict(SAMPLE, min_actuations=1, return_phases=True)
    assert list(out.columns) == OUT_COLS + EXTRA_COLS, "output columns changed"
    pref = read_parquet(PRED_REF)
    assert list(pref.columns) == OUT_COLS + EXTRA_COLS, "reference columns changed"
    m = out.merge(pref, on=["DeviceId", "Detector"], suffixes=("", "_ref"))
    assert len(m) == len(pref), "a detector appeared or disappeared"
    assert _same(m.phase_pred, m.phase_pred_ref, 0), "phase predictions changed"
    dp = np.abs(m.phase_prob.astype(float).fillna(-1) - m.phase_prob_ref.astype(float).fillna(-1)).max()
    assert dp < 1e-6, f"phase probabilities drifted by {dp:.2e}"
    assert m.function_pred.fillna("").equals(m.function_pred_ref.fillna("")), "function predictions changed"
    df_ = np.abs(m.function_prob.astype(float).fillna(-1) - m.function_prob_ref.astype(float).fillna(-1)).max()
    assert df_ < 1e-6, f"function probabilities drifted by {df_:.2e}"
    for c in INVARIANT:
        assert _same(m[c], m[c + "_ref"], 1e-6 if c != "distance_ft" else 0.5), f"{c} changed"
    ph_ref = read_parquet(PHASE_REF)
    assert all(_same(phases[c], ph_ref[c]) for c in PHASE_COLS), "the per-phase lane table changed"
    check_new_outputs(out, phases)

    # -- a short sample (first 10 minutes): no lanes, Count / Yellow_Red twin decode
    short = predict(_first_minutes(ev, 10), min_actuations=1)
    sref = read_parquet(PRED10_REF)
    ms = short.merge(sref, on=["DeviceId", "Detector"], suffixes=("", "_ref"))
    assert len(ms) == len(sref) == len(short), "short sample: a detector appeared or disappeared"
    assert _same(ms.phase_pred, ms.phase_pred_ref, 0), "short sample: phase changed"
    assert ms.function_pred.fillna("").equals(ms.function_pred_ref.fillna("")), "short sample: function changed"
    ds = np.abs(ms.function_prob.astype(float).fillna(-1) - ms.function_prob_ref.astype(float).fillna(-1)).max()
    assert ds < 1e-6, f"short sample: function probabilities drifted by {ds:.2e}"
    assert short.lanes.isna().all(), "a sample under 30 minutes must not get lanes"

    # -- and the refusal rules still behave
    ans = out[out.function_pred.notna()]
    assert ans.function_pred.isin(CLASSES).all()
    assert (ans[PROB_COLS].sum(axis=1) - 1).abs().max() < 1e-6
    strict = predict(SAMPLE, min_actuations=1, min_prob=0.9)
    kept = strict[strict.phase_pred.notna()]
    assert (kept.phase_prob >= 0.9).all() and len(kept) <= out.phase_pred.notna().sum()
    assert strict.phase_guess.notna().sum() == out.phase_pred.notna().sum(), \
        "a refused detector must still carry the model's opinion in phase_guess"
    return (f"{len(out)} detectors, {int(out.phase_pred.notna().sum())} answered, "
            f"{int(out.lanes.notna().sum())} with a lane, {int(out.distance_ft.notna().sum())} "
            f"with a setback, {int(out.health_status.notna().sum())} with a health status; "
            f"max probability drift {max(d, dp, df_):.1e}")


def _predict_with_candidates(ev):
    """predict() plus the per-candidate phase probability table (every (detector, candidate) pair, captured from the
    scoring stage), so the tests cover every probability, not only the winners."""
    cap = {}
    orig = P.score

    def score(*a, **k):
        r = orig(*a, **k)
        cap["cand"] = r[["DeviceId", "Detector", "cand_phase", "p0", "prob"]].copy()
        return r
    P.score = score
    try:
        out, ph = P.predict(ev, min_actuations=1, return_phases=True)
    finally:
        P.score = orig
    return out, ph, cap["cand"]


def _cand_diff(c0, c1, what):
    mc = c0.merge(c1, on=["DeviceId", "Detector", "cand_phase"], suffixes=("", "_s"))
    assert len(mc) == len(c0) == len(c1), f"the candidate set changed under {what}"
    dmax = 0.0
    for c in ("p0", "prob"):
        d = float(np.nanmax(np.abs(mc[c].to_numpy(float) - mc[c + "_s"].to_numpy(float))))
        assert d < PROB_TOL, f"candidate {c} moved by {d:.2e} under {what}"
        dmax = max(dmax, d)
    return len(mc), dmax


def _map_detector_text(out1, inv, pinv=None):
    """text that names a detector ("relative to d12, which it tracks", "det 12: P2 Presence") must name the same
    detector and phase: map d<ch> / det <ch> back through the channel map and P<phase> through the phase map"""
    pinv = pinv or {}

    def f(t):
        if not isinstance(t, str):
            return t
        t = re.sub(r"\bd(\d+)\b", lambda mm: f"d{inv.get(int(mm.group(1)), mm.group(1))}", t)
        t = re.sub(r"\bdet (\d+)\b", lambda mm: f"det {inv.get(int(mm.group(1)), mm.group(1))}", t)
        return re.sub(r"\bP(\d+)\b", lambda mm: f"P{pinv.get(int(mm.group(1)), mm.group(1))}", t)
    for c in HEALTH_TEXT:
        if c in out1:
            out1[c] = out1[c].map(f)
    return out1


def _invariance_one(ev: pd.DataFrame, seed: int, base=None) -> tuple[int, float]:
    if base is None:
        base = _predict_with_candidates(ev)
    out0, ph0, c0 = base
    is_ph = ev.EventId.isin(PHASE_EVENTS + [150])
    phases = sorted(ev.loc[ev.EventId.isin(PHASE_EVENTS), "Parameter"].unique())
    perm = {int(a): int(b) for a, b in zip(phases, np.random.default_rng(seed).permutation(phases))}
    ev2 = ev.copy()
    ev2.loc[is_ph, "Parameter"] = ev2.loc[is_ph, "Parameter"].map(lambda v: perm.get(v, v))
    out1, ph1, c1 = _predict_with_candidates(ev2)
    out1 = _map_detector_text(out1, {}, {v: k for k, v in perm.items()})
    c0 = c0.assign(cand_phase=c0.cand_phase.astype(np.int64).map(lambda v: int(perm[v])).astype(np.int64),
                   Detector=c0.Detector.astype(np.int64))
    c1 = c1.assign(cand_phase=c1.cand_phase.astype(np.int64), Detector=c1.Detector.astype(np.int64))
    n, dmax = _cand_diff(c0, c1, f"renumbering (seed {seed})")
    m = out0.merge(out1, on=["DeviceId", "Detector"], suffixes=("", "_s"))
    assert len(m) == len(out0) == len(out1)
    for c in INVARIANT:
        assert _same(m[c], m[c + "_s"]), f"{c} changed under renumbering"
    for c in OUT_PROBS:
        assert _same(m[c], m[c + "_s"], PROB_TOL), f"{c} changed under renumbering (seed {seed})"
        d = np.abs(m[c].astype(float) - m[c + "_s"].astype(float))
        dmax = max(dmax, float(np.nanmax(d)) if d.notna().any() else 0.0)
    for c in ("phase_pred", "phase_2nd", "phase_guess"):
        a = m[c].map(lambda v: perm.get(int(v)) if pd.notna(v) else -1)
        assert (a.to_numpy() == m[c + "_s"].fillna(-1).astype(int).to_numpy()).all(), f"{c} did not follow"
    assert m.function_pred.fillna("").equals(m.function_pred_s.fillna("")), "function changed under renumbering"
    if len(ph0):
        a = ph0.assign(phase=ph0.phase.map(perm)).sort_values(["DeviceId", "phase"])
        b = ph1.sort_values(["DeviceId", "phase"])
        for c in PHASE_COLS:
            assert _same(a[c], b[c]), f"per-phase {c} changed under renumbering"
    return n, dmax


def check_phase_number_invariance() -> str:
    ev = read_parquet(SAMPLE)
    base = _predict_with_candidates(ev)
    n, dmax = 0, 0.0
    for s in PERM_SEEDS:
        k, d = _invariance_one(ev, s, base)
        n, dmax = n + k, max(dmax, d)
    k, d = _invariance_one(_first_minutes(ev, 10), 7)
    n, dmax = n + k, max(dmax, d)
    return (f"{len(PERM_SEEDS)} renumberings (30 min) + 1 (10 min): {n} candidate probabilities and every output "
            f"probability identical (max |diff| {dmax:.1e} < {PROB_TOL:g}); answers followed, lanes / setback / "
            f"health unchanged")


CHAN_MODES = ("reverse", "shift", "reverse+shift")
CHAN_COLS = [c for c in OUT_COLS + EXTRA_COLS if c not in ("DeviceId", "Detector")]
PHASE_VALUED = ("phase_pred", "phase_2nd", "phase_guess")   # output columns that ARE phase numbers (mapped back)
JOINT_SEEDS = (11, 12, 13)


def _chan_map(ev: pd.DataFrame, mode: str) -> dict:
    is_d = ev.EventId.isin([81, 82]) & (ev.Parameter <= 64)
    chs = sorted(int(x) for x in ev.loc[is_d, "Parameter"].unique())
    tgt = [max(chs) + min(chs) - c for c in chs] if "reverse" in mode else list(chs)
    k = 0
    if "shift" in mode:
        k = 64 - max(tgt) if max(tgt) < 64 else -(min(tgt) - 1)
        assert k != 0, "no room to shift the channels"
    return dict(zip(chs, [c + k for c in tgt]))


def _relabelled_one(ev: pd.DataFrame, cmap: dict, perm: dict, what: str, base=None) -> tuple[int, float]:
    """Apply a channel map and a phase permutation to the raw log; every output must follow both exactly."""
    if base is None:
        base = _predict_with_candidates(ev)
    out0, ph0, c0 = base
    inv = {v: k for k, v in cmap.items()}
    is_d = ev.EventId.isin([81, 82]) & (ev.Parameter <= 64)
    is_ph = ev.EventId.isin(PHASE_EVENTS + [150])
    ev2 = ev.copy()
    ev2.loc[is_d, "Parameter"] = ev2.loc[is_d, "Parameter"].map(lambda v: cmap.get(int(v), int(v)))
    ev2.loc[is_ph, "Parameter"] = ev2.loc[is_ph, "Parameter"].map(lambda v: perm.get(int(v), int(v)))
    out1, ph1, c1 = _predict_with_candidates(ev2)
    back = lambda d: d.astype(np.int64).map(lambda v: int(inv.get(int(v), int(v)))).astype(np.int64)  # noqa: E731
    c0 = c0.assign(Detector=c0.Detector.astype(np.int64),
                   cand_phase=c0.cand_phase.astype(np.int64).map(lambda v: int(perm.get(v, v))).astype(np.int64))
    c1 = c1.assign(Detector=back(c1.Detector), cand_phase=c1.cand_phase.astype(np.int64))
    n, dmax = _cand_diff(c0, c1, what)
    out1 = _map_detector_text(out1.assign(Detector=back(out1.Detector)), inv, {v: k for k, v in perm.items()})
    m = out0.assign(Detector=out0.Detector.astype(np.int64)).merge(out1, on=["DeviceId", "Detector"],
                                                                   suffixes=("", "_s"))
    assert len(m) == len(out0) == len(out1), f"a detector appeared or disappeared under {what}"
    for c in PHASE_VALUED:
        a = m[c].map(lambda v: perm.get(int(v), int(v)) if pd.notna(v) else -1)
        assert (a.to_numpy() == m[c + "_s"].fillna(-1).astype(int).to_numpy()).all(), f"{c} did not follow {what}"
    for c in CHAN_COLS:
        if c in PHASE_VALUED:
            continue
        assert _same(m[c], m[c + "_s"], PROB_TOL), f"{c} changed under {what}"
        if c in OUT_PROBS or c in ("lane_conf", "phase_n_lanes_conf"):
            d = np.abs(m[c].astype(float) - m[c + "_s"].astype(float))
            dmax = max(dmax, float(np.nanmax(d)) if d.notna().any() else 0.0)
    if len(ph0):
        a = ph0.assign(phase=ph0.phase.map(lambda v: perm.get(int(v), int(v)))).sort_values(["DeviceId", "phase"])
        b = ph1.sort_values(["DeviceId", "phase"])
        for c in PHASE_COLS:
            assert _same(a[c], b[c]), f"per-phase {c} changed under {what}"
    return n, dmax


def _chan_invariance_one(ev, mode, base=None):
    return _relabelled_one(ev, _chan_map(ev, mode), {}, f"channel {mode}", base)


def check_channel_invariance() -> str:
    ev = read_parquet(SAMPLE)
    base = _predict_with_candidates(ev)
    n, dmax = 0, 0.0
    for mode in CHAN_MODES:
        k, d = _chan_invariance_one(ev, mode, base)
        n, dmax = n + k, max(dmax, d)
    k, d = _chan_invariance_one(_first_minutes(ev, 10), "reverse")
    n, dmax = n + k, max(dmax, d)
    return (f"{len(CHAN_MODES)} channel renumberings (30 min: {', '.join(CHAN_MODES)}) + reversal (10 min): {n} "
            f"candidate probabilities and every output column identical (max |diff| {dmax:.1e} < {PROB_TOL:g})")


def _joint_one(ev: pd.DataFrame, seed: int, base=None) -> tuple[int, float]:
    """one random bijection of the detector channels onto 1..64 AND one random renumbering of the phases."""
    rng = np.random.default_rng(seed)
    is_d = ev.EventId.isin([81, 82]) & (ev.Parameter <= 64)
    chs = sorted(int(x) for x in ev.loc[is_d, "Parameter"].unique())
    cmap = {c: int(t) for c, t in zip(chs, rng.choice(np.arange(1, 65), size=len(chs), replace=False))}
    phases = sorted(int(x) for x in ev.loc[ev.EventId.isin(PHASE_EVENTS), "Parameter"].unique())
    perm = {a: int(b) for a, b in zip(phases, rng.permutation(phases))}
    return _relabelled_one(ev, cmap, perm, f"joint permutation (seed {seed})", base)


def check_joint_permutation() -> str:
    ev = read_parquet(SAMPLE)
    base = _predict_with_candidates(ev)
    n, dmax = 0, 0.0
    for s in JOINT_SEEDS:
        k, d = _joint_one(ev, s, base)
        n, dmax = n + k, max(dmax, d)
    k, d = _joint_one(_first_minutes(ev, 10), JOINT_SEEDS[0])
    n, dmax = n + k, max(dmax, d)
    return (f"{len(JOINT_SEEDS)} joint channel-bijection + phase renumberings (30 min) + 1 (10 min): {n} candidate "
            f"probabilities and every output identical (max |diff| {dmax:.1e} < {PROB_TOL:g})")


BLOCKED = ("torch", "lightgbm", "scipy", "sklearn", "pyarrow")


def check_no_hidden_dependencies() -> str:
    code = (
        "import sys\n"
        f"BLOCK = {BLOCKED!r}\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCK:\n"
        "            raise ImportError('blocked by check: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.path.insert(0, {str(HERE.parent)!r})\n"
        "from atspm_detector import predict\n"
        f"out = predict({str(SAMPLE)!r}, min_actuations=1)\n"
        "assert len(out) and out.phase_pred.notna().any()\n"
        "assert out.lanes.notna().any() and out.distance_ft.notna().any()\n"
        "assert out.health_status.notna().all()\n"
        "assert not any(m.split('.')[0] in BLOCK for m in sys.modules)\n"
        "print('ok', len(out))\n"
    )
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]
    return "ran with " + ", ".join(BLOCKED) + " blocked"


def check_profile_ignored() -> str:
    """A `profile` argument is accepted and ignored -- same answers as a plain call, one FutureWarning."""
    with warnings.catch_warnings(record=True) as w0:
        warnings.simplefilter("always")
        base = predict(SAMPLE, min_actuations=1)
    mine = [x for x in w0 if "is ignored: atspm_detector" in str(x.message)]
    assert not mine, "a plain predict() gave the profile warning"
    outs = []
    for how in ("argument", "env"):
        old = os.environ.get("DC_FAST_PROFILE")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                if how == "env":
                    os.environ["DC_FAST_PROFILE"] = "le2h"
                    outs.append(predict(SAMPLE, min_actuations=1))
                else:
                    outs.append(predict(SAMPLE, min_actuations=1, profile="le2h"))
            finally:
                if old is None:
                    os.environ.pop("DC_FAST_PROFILE", None)
                else:
                    os.environ["DC_FAST_PROFILE"] = old
        fw = [x for x in w if issubclass(x.category, FutureWarning)
              and "is ignored: atspm_detector" in str(x.message)]
        assert len(fw) == 1, f"profile via {how}: expected one FutureWarning, got {len(fw)}"
    for o in outs:
        assert list(o.columns) == list(base.columns) and len(o) == len(base)
        for c in base.columns:
            assert _same(o[c], base[c], 0.0), f"profile argument changed {c}"
    return "profile='le2h' and DC_FAST_PROFILE=le2h: answers identical to a plain call, one FutureWarning each"


CAT_RE = re.compile(r"^[A-Z][^();]* \((?:bad|suspect|watch)\)(?:; [A-Z][^();]* \((?:bad|suspect|watch)\))*$")


def check_health_v4() -> str:
    """Health smoke test -- the shipped references load; every detector gets a status and a reason
    that starts with its label; categories / config notes parse; a detector held ON for 12 minutes is called 'Stuck
    on' with the period listed, and nothing else in the sample turns bad."""
    from . import health_v4 as hv
    refs = hv._load_refs()
    assert str(refs.d.get("version", "")).startswith("health v4d"), "health references missing or wrong"
    for v in ("v4", "v4c"):
        assert set(refs.cell[v]) == {"chat_frac", "stuck_x", "n3_exc", "n_ep"}, f"limit tables {v} incomplete"
    assert refs.tod_own["groups"] and refs.band_own["groups"], "time-of-day references missing"
    ev = read_parquet(SAMPLE)
    out = predict(ev, min_actuations=1)
    assert out.health_status.isin(HEALTH).all()
    assert all(isinstance(t, str) and re.match(r"^det \d+: ", t) for t in out.health_reason), "reason without label"
    assert all(c == "" or CAT_RE.match(c) for c in out.health_categories.fillna("")), "unreadable categories"
    # inject a detector held ON for 12 minutes (minutes 16-28: the sample has a comms gap at 10.5-15 min, and an ON
    # over a gap is never 'stuck'): the busiest answered Advance zone
    t = pd.to_datetime(ev.Timestamp)
    adv = out[out.function_pred.eq("Advance")].sort_values("n_actuations", ascending=False)
    d = int(adv.Detector.iloc[0])
    t0 = t.min() + _td(minutes=16)
    t1 = t0 + _td(minutes=12)
    ut = np.unique(t[(t >= t0) & (t <= t1)].to_numpy())
    assert len(ut) > 1 and (np.diff(ut) <= np.timedelta64(120, "s")).all(), "the injection window has a comms gap"
    det = ev.EventId.isin([81, 82]) & (ev.Parameter == d)
    ev2 = ev[~(det & (t >= t0) & (t <= t1))]
    add = pd.DataFrame({"DeviceId": ev.DeviceId.iloc[0], "Timestamp": [t0, t1], "EventId": [82, 81], "Parameter": [d, d]})
    ev2 = pd.concat([ev2, add.astype({c: ev.dtypes[c] for c in add.columns if c in ev.dtypes})], ignore_index=True)
    out2 = predict(ev2, min_actuations=1)
    r = out2[out2.Detector == d].iloc[0]
    assert r.health_status in ("suspect", "bad"), f"held ON 12 min: status {r.health_status}"
    assert "Stuck on" in r.health_categories and "stuck on" in r.health_bad_periods, "stuck ON not reported"
    others = out2[out2.Detector != d].merge(out[["DeviceId", "Detector", "health_status"]], on=["DeviceId", "Detector"],
                                            suffixes=("", "_0"))
    worse = others[others.health_status.eq("bad") & others.health_status_0.ne("bad")]
    assert worse.empty, f"injected stuck ON turned others bad: {worse.Detector.tolist()}"
    n_f = int(out.health_status.isin(["suspect", "bad"]).sum())
    return (f"references loaded; {len(out)} detectors ({n_f} flagged on the sample); det {d} held ON 12 min -> "
            f"{r.health_status} ({r.health_categories})")


CHECKS = [("reproduces the stored answers", check_reproduces_reference),
          ("phase-number invariance", check_phase_number_invariance),
          ("channel-order invariance", check_channel_invariance),
          ("joint channel + phase permutation", check_joint_permutation),
          ("no hidden dependencies", check_no_hidden_dependencies),
          ("profile argument ignored", check_profile_ignored),
          ("health smoke test", check_health_v4)]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--freeze" in argv:
        freeze()
        return 0
    if not SAMPLE.exists():
        print(f"FAIL: missing bundled sample {SAMPLE}")
        return 1
    bad = 0
    for name, fn in CHECKS:
        try:
            print(f"PASS  {name}: {fn()}", flush=True)
        except Exception as exc:                                   # noqa: BLE001
            bad += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
    print("\nall checks passed" if not bad else f"\n{bad} check(s) FAILED")
    return 1 if bad else 0


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
