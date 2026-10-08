"""Note 79: the best available health EVALUATION set, one row per (DeviceId, detector), from what already exists.
Classes: dead / stuck / chatter / intermittent / degraded (undercount) / healthy.  Every row carries its source and a
tier that says how independent it is of the hi-res health rules:
    A  independent of the Sept 2026 log rules: alive in Dec 2024 and silent in Sept 2026 (print lists it), share of
       the signal fell > 5x Dec 2024 -> Sept 2026 (history), the user's own answers
    B  partly independent: print-dead never seen alive (may be removed, not broken), card-dead pairs, print-lane advance
       loop counting < ~0.45 of its same-lane stop-bar loop (print lanes + a count ratio)
    C  CIRCULAR: derived by hi-res rules (stuck-on minutes, chatter share, saturation, near-dead) on the stg 66-h log
Negatives = note 38's presumed healthy (print-labelled, all label checks pass, dq score 1) - not proven healthy.
Fault events 83-88 are not used anywhere (the dq 'faults' reasons are ignored).  Locked signals absent.

    python h79_evalset.py -> %DC_WORK%/health79/evalset.parquet
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_calib as C  # noqa: E402
import hb_data as H  # noqa: E402

OUT = H.DCW / "health79"
# the user's answers (review/spotcheck_health.xlsx, all sheets, later answer wins) -> class; 'ok' = he judged healthy
USER = {("03026", 22): "dead",
        ("03033", 8): "stuck", ("07027", 9): "stuck", ("10090", 22): "stuck", ("2B402", 16): "stuck",
        ("2B044", 14): "stuck", ("10045", 4): "stuck", ("07027", 8): "stuck",
        ("01009", 23): "chatter", ("07035", 19): "chatter",
        ("01026", 24): "intermittent", ("14003", 25): "intermittent", ("01026", 25): "intermittent",
        ("2B146", 16): "intermittent", ("05032", 52): "intermittent", ("03024", 4): "intermittent",
        ("10018", 2): "degraded", ("2B525", 7): "degraded", ("2B422", 8): "degraded", ("11042", 22): "degraded",
        ("04055", 24): "ok", ("2B439", 25): "ok", ("2B048", 6): "ok", ("03090", 25): "ok", ("2B067", 42): "ok"}


def name_map():
    v3 = pd.read_parquet(H.REPO / "research" / "labels" / "function_labels_v3.parquet", columns=["DeviceId", "DeviceName"])
    card = pd.read_parquet(H.DCW / "cabinet" / "card_channels.parquet", columns=["DeviceId", "DeviceName"])
    m = dict(zip(card.DeviceName.astype(str), card.DeviceId.str.lower()))
    m.update(dict(zip(v3.DeviceName.astype(str), v3.DeviceId.str.lower())))
    return m


def main():
    W = pd.read_parquet(H.HB / "weak_labels.parquet")
    nm = name_map()
    rows = []

    def add(dev, det, cls, src, tier):
        if dev is not None and pd.notna(det):
            rows.append(dict(DeviceId=dev, detector=int(det), cls=cls, source=src, tier=tier))
    # dead
    for r in W[W.wl_dead_print].itertuples():
        add(r.DeviceId, r.detector, "dead", "print dead, alive Dec 2024" if r.wl_dead_since_dec else "print dead, never seen",
            "A" if r.wl_dead_since_dec else "B")
    for r in W[W.wl_card_dead & ~W.wl_dead_print].itertuples():
        add(r.DeviceId, r.detector, "dead", "card both outputs dead", "B")
    # degraded (undercount)
    deg, _ = C.degraded(pd.read_parquet(H.HB / "real_stats.parquet"))
    for dev, det in deg:
        add(dev, det, "degraded", "share fell > 5x since Dec 2024", "A")
    fi = pd.read_excel(H.REPO / "review" / "field_issues_for_staff.xlsx", dtype=str)
    adv = fi[fi.issue.str.contains("misses vehicles", na=False)]
    for r in adv.itertuples():
        for d in re.findall(r"\d+", str(r._2)):
            add(nm.get(str(r.signal)), d, "degraded", "advance loop < .45 of same-lane stop-bar loop (print lanes)", "B")
    # circular hi-res-rule sets (no fault reasons)
    lc = pd.read_parquet(H.DCW / "cabinet" / "label_check_health.parquet")
    lc["DeviceId"] = lc.DeviceId.str.lower()
    rs = lc.dq_reasons.fillna("")
    for r in lc[rs.str.contains("stuck-on")].itertuples():
        add(r.DeviceId, r.detector, "stuck", "dq: stuck-on (66 h rule)", "C")
    for r in lc[rs.str.contains("chatter|sat:")].itertuples():
        add(r.DeviceId, r.detector, "chatter", "dq: chatter / saturation (66 h rule)", "C")
    for r in lc[rs.str.contains("near-dead")].itertuples():
        add(r.DeviceId, r.detector, "degraded", "dq: near-dead vs phase median", "C")
    card = pd.read_parquet(H.DCW / "cabinet" / "card_channels.parquet")
    card["DeviceId"] = card.DeviceId.str.lower()
    cr = card.reasons.fillna("")
    for r in card[card.card_suspect.fillna(False).astype(bool) & cr.str.contains("stuck-on")].itertuples():
        add(r.DeviceId, r.detector, "stuck", "card erratic: stuck-on", "C")
    for r in card[card.card_suspect.fillna(False).astype(bool) & cr.str.contains("chatter")].itertuples():
        add(r.DeviceId, r.detector, "chatter", "card erratic: chatter", "C")
    # the user's answers win over everything
    for (sig, det), cls in USER.items():
        add(nm.get(sig), det, cls if cls != "ok" else "healthy", "user answer", "U")
    E = pd.DataFrame(rows)
    pri = {"U": 0, "A": 1, "B": 2, "C": 3}
    E["p"] = E.tier.map(pri)
    E = E.sort_values(["DeviceId", "detector", "p"])
    allsrc = E.groupby(["DeviceId", "detector"]).source.apply(lambda s: "; ".join(dict.fromkeys(s)))
    E = E.drop_duplicates(["DeviceId", "detector"]).drop(columns="p").set_index(["DeviceId", "detector"])
    E["all_sources"] = allsrc
    E = E.reset_index()
    neg = W[W.presumed_healthy & ~W.set_index(["DeviceId", "detector"]).index.isin(E.set_index(["DeviceId", "detector"]).index)]
    neg = neg[["DeviceId", "detector"]].assign(cls="healthy", source="presumed healthy (checks pass)", tier="N",
                                               all_sources="presumed healthy")
    E = pd.concat([E, neg], ignore_index=True)
    E = E.merge(W[["DeviceId", "detector", "technology", "function"]].drop_duplicates(["DeviceId", "detector"]),
                on=["DeviceId", "detector"], how="left")
    locked = set(pd.read_csv(H.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    folds = pd.read_csv(H.DCW / "folds_v4.csv")
    train = {d.lower() for d in folds.DeviceId} - locked
    E = E[E.DeviceId.isin(train)]
    assert not E.DeviceId.isin(locked).any()
    OUT.mkdir(parents=True, exist_ok=True)
    E.to_parquet(OUT / "evalset.parquet", index=False)
    print(E.groupby(["cls", "tier"]).size().unstack(fill_value=0))
    print(E.groupby(["cls", "source"]).size().to_string())
    print("signals with any positive:", E[E.cls != "healthy"].DeviceId.nunique())


if __name__ == "__main__":
    main()
