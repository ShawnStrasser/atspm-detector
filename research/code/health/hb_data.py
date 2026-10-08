"""Bins, windows, weak labels and synthetic faults for note 38 (research only)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import health_core as hc  # noqa: E402

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
REPO = Path(__file__).resolve().parents[3]
HB = DCW / "health"
BINS = HB / "bins"
BPH = 12                                   # 5-min bins per hour

# ---- fixed evaluation windows (start, hours); Sept 2026 starts Fri 16:15, Dec 2024 Mon 00:00
EVAL_WIN = {
    "stg": {"2h_a": ("2026-09-21 07:00", 2), "2h_b": ("2026-09-19 13:00", 2),
            "2h_c": ("2026-09-20 17:00", 2), "6h_a": ("2026-09-19 10:00", 6),
            "6h_b": ("2026-09-21 04:00", 6), "24h_a": ("2026-09-19 00:00", 24),
            "24h_b": ("2026-09-20 00:00", 24), "full": ("2026-09-18 16:15", 66.2)},
}


def load_B(per: str, dev: str) -> dict | None:
    f = BINS / per / f"{dev}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    B = {k: z[k] for k in z.files}
    B["start"] = pd.Timestamp(str(B["start"]))
    B["bin_s"] = int(B["bin_s"])
    B["listed"] = None if B["listed"].shape == () else B["listed"]
    return B


def slice_B(B: dict, a: int, b: int, keep=None) -> dict:
    """Bins [a, b) of B; `keep` = boolean mask of detectors to keep."""
    keep = np.ones(len(B["dets"]), bool) if keep is None else keep
    S = {k: B[k][keep][:, a:b].copy() for k in ("n_on", "occ", "n_chat", "dmax")}   # n_flt unused (no faults)
    S["dur_max"] = S["dmax"].max(1) if b > a else np.zeros(keep.sum())
    S["dets"] = B["dets"][keep]
    S["cov"] = B["cov"][a:b].copy()
    S["hour"] = B["hour"][a:b]
    S["start"] = B["start"] + pd.Timedelta(seconds=a * B["bin_s"])
    S["bin_s"] = B["bin_s"]
    S["listed"] = None if B["listed"] is None else B["listed"][keep]
    return S


def win_bins(B: dict, start: str, hours: float) -> tuple[int, int]:
    a = int(round((pd.Timestamp(start) - B["start"]).total_seconds() / B["bin_s"]))
    b = min(a + int(round(hours * BPH)), B["n_on"].shape[1])
    return max(a, 0), b


def stats_of(S: dict) -> pd.DataFrame:
    st = hc.det_stats(S)
    rl = hc.rules(st, S)
    return rl.merge(st, on="detector")


# ============================================================================ weak labels
def weak_labels() -> pd.DataFrame:
    """One row per (DeviceId, detector) with the known-problem flags (cleansing sources)."""
    v3 = pd.read_parquet(REPO / "research" / "labels" / "function_labels_v3.parquet",
                         columns=["DeviceId", "DeviceName", "detector", "function", "source",
                                  "technology", "dead", "dq_flags", "dq_score", "validated",
                                  "failed_checks", "print_confidence"])
    v3["DeviceId"] = v3.DeviceId.str.lower()
    v3["detector"] = pd.to_numeric(v3.detector, errors="coerce")
    name2id = dict(zip(v3.DeviceName, v3.DeviceId))
    card = pd.read_parquet(DCW / "cabinet" / "card_channels.parquet")
    card["DeviceId"] = card.DeviceId.str.lower()
    name2id.update(dict(zip(card.DeviceName, card.DeviceId)))
    dead = pd.read_csv(REPO / "review" / "dead_detectors_from_prints.csv", dtype=str)
    dead["DeviceId"] = dead.signal.map(name2id)
    dead["detector"] = pd.to_numeric(dead.detector)
    lc = pd.read_parquet(DCW / "cabinet" / "label_check_health.parquet")
    lc["DeviceId"] = lc.DeviceId.str.lower()
    keys = pd.concat([v3[["DeviceId", "detector"]], card[["DeviceId", "detector"]],
                      dead[["DeviceId", "detector"]], lc[["DeviceId", "detector"]]]).dropna()
    keys = keys.drop_duplicates()
    W = keys.merge(v3.drop(columns="DeviceName"), how="left", on=["DeviceId", "detector"])
    d = dead.dropna(subset=["DeviceId"]).assign(
        wl_dead_print=True,
        wl_dead_since_dec=lambda x: x["last window with actuations"].eq("Dec 2024"))
    W = W.merge(d[["DeviceId", "detector", "wl_dead_print", "wl_dead_since_dec"]].drop_duplicates(
        ["DeviceId", "detector"]), how="left", on=["DeviceId", "detector"])
    c = card[card.card_suspect.fillna(False).astype(bool)]
    # note 88: a card is erratic only by actuations (stuck-on / chatter), never by detector fault events 83-88
    c = c.assign(wl_card_dead=c.dead.fillna(False).astype(bool),
                 wl_card_erratic=~c.dead.fillna(False).astype(bool)
                 & c.reasons.fillna("").str.contains("stuck-on|chatter"))
    W = W.merge(c[["DeviceId", "detector", "wl_card_dead", "wl_card_erratic"]].drop_duplicates(
        ["DeviceId", "detector"]), how="left", on=["DeviceId", "detector"])
    # note 88: dq health by ACTUATIONS only (stuck-on / chatter in the reasons) -- s_health also carried the detector
    # fault-event part (83-88, user ban 2026-09-28), which kept fault-flagged detectors out of `presumed_healthy`
    act = lc.dq_reasons.fillna("").str.extract(r"health: (.*)", expand=False).fillna("").str.contains("stuck|chatter")
    lcb = lc.assign(wl_dq_health=act | (lc.s_sat < 1))
    W = W.merge(lcb[["DeviceId", "detector", "wl_dq_health"]].drop_duplicates(
        ["DeviceId", "detector"]), how="left", on=["DeviceId", "detector"])
    W["wl_lc_health"] = W.failed_checks.fillna("").str.contains("health")
    for k in ("wl_dead_print", "wl_dead_since_dec", "wl_card_dead", "wl_card_erratic", "wl_dq_health"):
        W[k] = W[k].fillna(False).astype(bool)
    W["wl_any"] = W[[c for c in W if c.startswith("wl_")]].any(axis=1)
    # note 88: dq_score without its fault-event health factor (0.2 when the only health reason was fault events)
    dq = pd.read_parquet(DCW / "cabinet" / "dq_print.parquet", columns=["DeviceId", "detector", "dq_reasons"])
    dq["DeviceId"] = dq.DeviceId.str.lower()
    h = dq.dq_reasons.fillna("").str.extract(r"health: (.*)", expand=False).fillna("")
    dq["fo"] = h.str.contains("faults") & ~h.str.contains("stuck|chatter")
    W = W.merge(dq[["DeviceId", "detector", "fo"]].drop_duplicates(["DeviceId", "detector"]), how="left",
                on=["DeviceId", "detector"])
    W["dq_score"] = np.where(W.fo.eq(True), np.minimum(1.0, (W.dq_score * 5).round(3)), W.dq_score)
    W = W.drop(columns="fo")
    W["presumed_healthy"] = (~W.wl_any & W.validated.eq("pass") & W.dq_score.eq(1.0)
                             & ~W.dead.fillna(False).astype(bool))
    return W


# ========================================================================= synthetic faults
FAULTS = ("dropout", "intermittent", "stuck", "chatter", "random", "erratic", "undercount",
          "overcount", "corr_break", "dead")


def inject(S: dict, i: int, kind: str, rng: np.random.Generator, B_full: dict | None = None,
           a_full: int = 0) -> np.ndarray:
    """Inject fault `kind` into detector row i of window S (in place). Returns the bool mask of
    affected bins."""
    nb = S["n_on"].shape[1]
    m = np.zeros(nb, bool)
    x, o = S["n_on"][i], S["occ"][i]
    if kind in ("dropout", "undercount", "overcount"):
        a = int(rng.integers(int(nb * .1), int(nb * .75) + 1))
        m[a:] = True
    elif kind in ("intermittent", "stuck", "chatter", "random", "erratic"):
        L = int(rng.integers(max(2, nb // 8), max(3, nb // 2) + 1))
        a = int(rng.integers(0, nb - L + 1))
        m[a:a + L] = True
    elif kind in ("corr_break", "dead"):
        m[:] = True
    if kind in ("dropout", "intermittent", "dead"):
        for k in ("n_on", "occ", "n_chat", "dmax"):
            S[k][i][m] = 0
    elif kind == "stuck":
        idx = np.where(m)[0]
        S["n_on"][i][idx[1:]] = 0
        S["n_chat"][i][idx[1:]] = 0
        S["occ"][i][idx] = S["bin_s"]
        S["dmax"][i][idx] = 0
        S["dmax"][i][idx[0]] = len(idx) * S["bin_s"]
    elif kind == "chatter":
        k = rng.uniform(1.5, 5)
        add = rng.poisson(x[m] * (k - 1))
        S["n_on"][i][m] = x[m] + add
        S["n_chat"][i][m] += add
    elif kind == "random":                   # ONs at random, no traffic shape
        lam = max(x.mean(), 0.5) * rng.uniform(0.5, 2)
        new = rng.poisson(lam, m.sum()).astype(np.float32)
        mo = o[m].sum() / max(x[m].sum(), 1)
        S["n_on"][i][m] = new
        S["occ"][i][m] = np.minimum(new * max(mo, 0.3), S["bin_s"])
    elif kind == "erratic":
        f = rng.lognormal(0, rng.uniform(0.6, 1.2), m.sum())
        S["n_on"][i][m] = rng.poisson(x[m] * f)
        S["occ"][i][m] = np.minimum(o[m] * f, S["bin_s"])
    elif kind in ("undercount", "overcount"):
        f = rng.uniform(0.05, 0.4) if kind == "undercount" else rng.uniform(2.5, 5)
        S["n_on"][i][m] = rng.binomial(x[m].astype(int), min(f, 1)) if f < 1 else rng.poisson(x[m] * f)
        S["occ"][i][m] = np.minimum(o[m] * f, S["bin_s"])
    elif kind == "corr_break":               # the detector's own counts from another time of day
        nbf = B_full["n_on"].shape[1]
        j = list(B_full["dets"]).index(S["dets"][i])
        for _ in range(10):
            off = int(rng.integers(6 * BPH, 18 * BPH)) * (1 if rng.random() < .5 else -1)
            a2 = a_full + off
            if 0 <= a2 and a2 + nb <= nbf:
                break
        else:
            a2 = (a_full + nbf // 2) % max(nbf - nb, 1)
        xs = B_full["n_on"][j][a2:a2 + nb].astype(float)
        sc = x.sum() / max(xs.sum(), 1)
        S["n_on"][i] = rng.poisson(xs * sc)
        S["occ"][i] = np.minimum(B_full["occ"][j][a2:a2 + nb] * sc, S["bin_s"])
    S["dur_max"][i] = S["dmax"][i].max()
    return m


def rescore(st: pd.DataFrame, start="2026-01-01", bin_s: int = 300) -> pd.DataFrame:
    """Re-apply the current rule limits (hc.LIM) to stored statistics."""
    stat_cols = [c for c in st.columns if not (c.startswith("s_") or c in
                 ("health_score", "status", "reason"))]
    Bm = {"start": pd.Timestamp(start), "bin_s": bin_s}
    rl = hc.rules(st[stat_cols].reset_index(drop=True), Bm)
    out = st[stat_cols].reset_index(drop=True).copy()
    for c in rl.columns:
        if c != "detector":
            out[c] = rl[c].to_numpy()
    return out
