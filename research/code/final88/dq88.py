"""Note 88: remove detector fault events 83-88 from the label-cleansing chain (user ban 2026-09-28).

Before: dq_core's `health` part flagged a detector on fault events 84-88 (FAULT_N / FAULT_FRAC) and counted 83-88 as
comms-coverage evidence; dq_suspect (cab_final step 5, print location) therefore dropped 233 otherwise-clean labelled
rows from training (v3_retrain --clean; note 87 finding 1), and card_suspect (vd_audit.card) called a card "erratic" on
fault events, which blocked not_checkable print-high rows from --allow-not-checkable-high.  dq_core is now
actuation-only (ALLOWED without 83-88, no fault health); this script recomputes everything that depended on it.

    python dq88.py dq        dq_core (fault-free) on the cabinet_v4l print rows, exactly cab_final.dq_print's input
                             -> f88/dq_print_nf.parquet (+ comparison with the cached cabinet_v4l/dq_print.parquet)
    python dq88.py lch       dq_core (fault-free) on every label_check_health row: stuck-on / chatter verdicts of
                             label_check._health old vs new (label_check's own health is actuation-only, but its
                             dur_max / chatter_frac came from a dq_core run that counted fault events as coverage)
    python dq88.py card      card_suspect with erratic = stuck-on / chatter only -> f88/card_channels_nf.parquet
    python dq88.py labels    research/labels/function_labels_v4m.parquet = v4l with dq_score / dq_flags / dq_suspect
                             from f88/dq_print_nf (nothing else changes) + change list f88/dq_changes.csv
CPU, DuckDB 4 threads / 10 GB.  locked_v2 asserted absent everywhere.  Writes nothing into the cabinet stores.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: E402
import dq_core  # noqa: E402

assert not set(dq_core.ALLOWED) & {83, 84, 85, 86, 87, 88}, "dq_core still reads fault events"
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
CAB = DCW / "cabinet_v4l"
OUT = DCW / "final_v3_work" / "f88"
REPO = rpath.REPO
V4L = REPO / "research" / "labels" / "function_labels_v4l.parquet"
V4M = REPO / "research" / "labels" / "function_labels_v4m.parquet"
LOC = {"Presence": "stopbar", "Count": "stopbar", "Yellow_Red": "stopbar", "Advance": "advance", "Mid": "mid",
       "Bike": "bike"}     # = cab_final.LOC
THREADS = 4


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set[str]:
    return set(pd.read_csv(DCW / "official" / "locked_v2.csv").DeviceId.astype(str).str.lower())


def _con(threads: int = THREADS):
    con = dq_core.duckdb.connect()
    con.execute("SET memory_limit='10GB'")
    con.execute(f"SET threads={THREADS}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    return con


dq_core.connect = _con


def hpart(r) -> str:
    m = re.search(r"health: (.*)", r if isinstance(r, str) else "")
    return m.group(1) if m else ""


def stage_dq():
    pl = pd.read_parquet(CAB / "print_labels.parquet")
    pr = pl[pl.source == "print"].copy()
    assert not pr.DeviceId.str.lower().isin(locked()).any()
    d = pd.DataFrame(dict(
        DeviceId=pr.DeviceId, detector=pr.detector.astype(int),
        phase=pr.phase_timing.fillna(pr.phase_diagram.astype("string")).fillna("?").astype(str),
        location=pr.function.map(LOC).fillna("other"), lane_index=pr.lane_index.astype("float"),
        lanes_spanned=pr.lanes_spanned.astype("float").fillna(1), technology=pr.technology, function=pr.function))
    t0 = time.time()
    dq = dq_core.run(d, period="auto", threads=THREADS)
    dq.attrs.pop("pairs", None)
    log(f"dq: {len(dq)} rows, {dq.DeviceId.nunique()} signals, {time.time() - t0:.0f}s")
    full = dq.copy()
    dq = dq[["DeviceId", "detector", "source", "dq_score", "flags", "reasons", "suspect_config_or_health"]]
    dq.columns = ["DeviceId", "detector", "dq_source", "dq_score", "dq_flags", "dq_reasons", "dq_suspect"]
    OUT.mkdir(parents=True, exist_ok=True)
    dq.to_parquet(OUT / "dq_print_nf.parquet", index=False)
    full[["DeviceId", "detector", "n_on", "dur_max", "chatter_frac", "cov_h", "s_health", "reasons"]].to_parquet(
        OUT / "dq_print_nf_stats.parquet", index=False)
    old = pd.read_parquet(CAB / "dq_print.parquet")
    m = old.merge(dq, on=["DeviceId", "detector"], suffixes=("_o", "_n"))
    assert len(m) == len(old) == len(dq)
    ho = m.dq_reasons_o.map(hpart)
    fault = ho.str.contains("faults")
    same = np.isclose(m.dq_score_o.fillna(-1), m.dq_score_n.fillna(-1))
    r = {"rows": len(m), "suspect_old": int(m.dq_suspect_o.sum()), "suspect_new": int(m.dq_suspect_n.sum()),
         "suspect_true_to_false": int((m.dq_suspect_o & ~m.dq_suspect_n).sum()),
         "suspect_false_to_true": int((~m.dq_suspect_o & m.dq_suspect_n).sum()),
         "rows_with_fault_health_old": int(fault.sum()),
         "score_changed_rows": int((~same).sum()),
         "score_changed_rows_without_fault_reason": int((~same & ~fault).sum()),
         "flips_true_to_false_with_fault_reason": int((m.dq_suspect_o & ~m.dq_suspect_n & fault).sum())}
    json.dump(r, open(OUT / "dq_compare.json", "w"), indent=1)
    log(f"compare: {r}")
    x = m[~same & ~fault]
    if len(x):
        log("score changes without a fault reason (coverage effect of dropping 83-88 from ALLOWED):\n"
            + x[["DeviceId", "detector", "dq_score_o", "dq_score_n", "dq_reasons_o", "dq_reasons_n"]].head(30).to_string())


def stage_lch():
    h = pd.read_parquet(CAB / "label_check_health.parquet")
    assert not h.DeviceId.str.lower().isin(locked()).any()
    d = pd.DataFrame(dict(DeviceId=h.DeviceId, detector=h.detector.astype(int), phase="?", location="other",
                          lane_index=np.nan, lanes_spanned=1.0))
    t0 = time.time()
    n = dq_core.run(d, period="auto", threads=THREADS)
    n.attrs.pop("pairs", None)
    n["DeviceId"] = n.DeviceId.str.lower()
    log(f"lch: {len(n)} rows, {time.time() - t0:.0f}s")
    n = n[["DeviceId", "detector", "n_on", "dur_max", "chatter_frac"]]
    n.to_parquet(OUT / "label_check_health_nf.parquet", index=False)
    m = h.merge(n, on=["DeviceId", "detector"], suffixes=("_o", "_n"), how="left")
    assert len(m) == len(h)

    def verdict(dur, chat, non):
        dur, chat, non = (pd.to_numeric(x, errors="coerce").fillna(0) for x in (dur, chat, non))
        return (dur > 900.0) | ((chat > 0.30) & (non >= 20))
    vo = verdict(m.dur_max_o, m.chatter_frac_o, m.dq_n_on)
    vn = verdict(m.dur_max_n, m.chatter_frac_n, m.n_on)
    r = {"rows": len(m), "unhealthy_old": int(vo.sum()), "unhealthy_new": int(vn.sum()),
         "old_to_ok": int((vo & ~vn).sum()), "ok_to_unhealthy": int((~vo & vn).sum()),
         "n_on_changed": int((m.dq_n_on.fillna(0) != m.n_on.fillna(0)).sum()),
         "dur_max_changed": int((~np.isclose(m.dur_max_o.fillna(-1), m.dur_max_n.fillna(-1))).sum())}
    json.dump(r, open(OUT / "lch_compare.json", "w"), indent=1)
    log(f"label-check health old vs new: {r}")
    x = m[vo != vn]
    if len(x):
        x[["DeviceId", "detector", "dq_n_on", "n_on", "dur_max_o", "dur_max_n", "chatter_frac_o", "chatter_frac_n"]] \
            .to_csv(OUT / "lch_flips.csv", index=False)


def stage_card():
    """card_suspect without fault events: a two-output slot is suspect when both in-use outputs are dead, or both are
    erratic BY ACTUATIONS (stuck-on / chatter in dq_core's reasons). Same rule label_check.card_faults already applies."""
    c = pd.read_parquet(CAB / "card_health.parquet")
    act = lambda s: s.fillna("").map(hpart).str.contains("stuck-on|chatter")  # noqa: E731
    c["both_erratic_nf"] = c.both_in_use & act(c.reasons_upper) & act(c.reasons_lower)
    c["card_suspect_nf"] = c.card_suspect & (c.both_dead | c.both_erratic_nf)   # keeps the all-dead-signal exemption
    ch = pd.read_parquet(CAB / "card_channels.parquet")
    bad = c[c.card_suspect_nf]
    keys = set(zip(bad.DeviceName, bad.det_upper)) | set(zip(bad.DeviceName, bad.det_lower))
    ch["card_suspect_nf"] = [(a, int(b)) in keys for a, b in zip(ch.DeviceName, ch.detector)]
    ch.to_parquet(OUT / "card_channels_nf.parquet", index=False)
    r = {"slots_suspect_old": int(c.card_suspect.sum()), "slots_suspect_new": int(c.card_suspect_nf.sum()),
         "channels_suspect_old": int(ch.card_suspect.sum()), "channels_suspect_new": int(ch.card_suspect_nf.sum()),
         "slots_dropped": c[c.card_suspect & ~c.card_suspect_nf][["DeviceName", "slot"]].astype(str).agg(" ".join, 1).tolist()}
    json.dump(r, open(OUT / "card_compare.json", "w"), indent=1)
    log(f"card: {r}")


def stage_labels():
    v = pd.read_parquet(V4L)
    assert not v.DeviceId.str.lower().isin(locked()).any()
    n = pd.read_parquet(OUT / "dq_print_nf.parquet")
    n["k"] = list(zip(n.DeviceId.str.lower(), n.detector.astype(int)))
    v["k"] = list(zip(v.DeviceId.str.lower(), v.detector.astype(int)))
    nm = n.set_index("k")
    has = v.k.isin(nm.index)
    # v4l rows with a dq value are exactly the print rows the cache scored
    assert (v.dq_score.notna() <= has).all(), "v4l dq row without a print dq row"
    new = v.copy()
    for c in ("dq_score", "dq_flags", "dq_suspect"):
        val = v.k.map(nm[c])
        new[c] = val.where(v.dq_score.notna(), v[c]).astype(v[c].dtype)
    new["label_version"] = "v4m"
    ch = (v.dq_suspect.astype("boolean").fillna(False) != new.dq_suspect.astype("boolean").fillna(False))
    lst = v.loc[ch, ["DeviceId", "DeviceName", "detector", "function", "source", "validated", "train_use_validated",
                     "dq_suspect", "dq_flags"]].assign(dq_suspect_new=new.loc[ch, "dq_suspect"].to_numpy(),
                                                      dq_flags_new=new.loc[ch, "dq_flags"].to_numpy(),
                                                      dq_reasons_old=v.loc[ch, "k"].map(
                                                          pd.read_parquet(CAB / "dq_print.parquet").assign(
                                                              k=lambda x: list(zip(x.DeviceId.str.lower(),
                                                                                   x.detector.astype(int))))
                                                          .set_index("k").dq_reasons).to_numpy())
    lst.to_csv(OUT / "dq_changes.csv", index=False)
    new.drop(columns="k").to_parquet(V4M, index=False)
    tv = v.train_use_validated.eq(True)
    r = {"rows": len(v), "dq_suspect_old": int(v.dq_suspect.eq(True).sum()), "dq_suspect_new": int(new.dq_suspect.eq(True).sum()),
         "changed": int(ch.sum()), "changed_and_train_use_validated": int((ch & tv).sum()),
         "changed_train_by_function": v[ch & tv].function.value_counts().to_dict()}
    json.dump(r, open(OUT / "labels_compare.json", "w"), indent=1)
    log(f"labels v4m: {r}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for st in sys.argv[1:]:
        {"dq": stage_dq, "lch": stage_lch, "card": stage_card, "labels": stage_labels}[st]()
    log(f"== {' '.join(sys.argv[1:])} done ({time.time() - t0:.0f}s)")
