"""Note 38 addendum (2026-09-28): the health scorer without detector fault events (83-88).

The user ruled that fault events must not be used (configured per cabinet, guarantee nothing).
Re-scores the stored Sept-2026 window statistics (act_eval.parquet: note 38 stats + note 40
`rapid`) with the current health_core.rules (no fault rule) and re-defines the real-problem
sets that were partly fault-defined:
  dq health fail / label-check health fail / card erratic = fault events OR stuck-on OR
  chatter (dq: OR saturation).  Non-fault version keeps a detector only if its dq_core
  reason names stuck-on, chatter (or, for dq, saturation); fault-only members are dropped
  (not moved to the healthy set, which stays unchanged so the false-alarm rate is comparable).
Rows that existed only because a channel logged fault events (no ONs, not in the channel
list) disappear from the new scorer's output and are dropped.

    python hb_nofault_eval.py   -> prints before / after tables; writes nofault_eval.parquet
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_calib as C  # noqa: E402
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
ORDER = ["2h", "6h", "24h", "full"]


def main():
    m = pd.read_parquet(H.HB / "act_eval.parquet")
    m["old"] = m.flag2.astype(bool)                       # note 40 scorer (with the fault rule)
    fault_only_row = ~m.listed.astype(bool) & m.n_on.eq(0) & m.occ_frac.eq(0)
    print("rows created only by fault events (dropped):", int(fault_only_row.sum()))
    keep = [c for c in m.columns if c not in ("old", "flag1", "flag2", "wlen", "grp", "H", "pos", "mr")]
    s = pd.concat([H.rescore(g[keep], H.EVAL_WIN["stg"][w][0]).assign(window=w)
                   for w, g in m[~fault_only_row].groupby("window")], ignore_index=True)
    s = s.merge(m.loc[~fault_only_row, ["DeviceId", "window", "detector", "old"]],
                on=["DeviceId", "window", "detector"])
    s["new"] = s.status.isin(["bad", "suspect"])
    s["wlen"] = s.window.str.split("_").str[0]
    deg, _ = C.degraded(pd.read_parquet(H.HB / "real_stats.parquet"))
    s["wl_degraded"] = [(d, k) in deg for d, k in zip(s.DeviceId, s.detector)]
    # ---- non-fault versions of the partly fault-defined sets
    lc = pd.read_parquet(H.DCW / "cabinet" / "label_check_health.parquet")
    lc["DeviceId"] = lc.DeviceId.str.lower()
    r = lc.dq_reasons.fillna("")
    lc["nf_health"] = r.str.contains("stuck-on|chatter")
    lc["nf_dq"] = lc.nf_health | r.str.contains("sat:")
    card = pd.read_parquet(H.DCW / "cabinet" / "card_channels.parquet", columns=["DeviceId", "detector", "reasons"])
    card["DeviceId"] = card.DeviceId.str.lower()
    card["nf_card"] = card.reasons.fillna("").str.contains("stuck-on|chatter")
    s = s.merge(lc[["DeviceId", "detector", "nf_health", "nf_dq"]].drop_duplicates(["DeviceId", "detector"]),
                on=["DeviceId", "detector"], how="left")
    s = s.merge(card[["DeviceId", "detector", "nf_card"]].drop_duplicates(["DeviceId", "detector"]),
                on=["DeviceId", "detector"], how="left")
    for c in ("nf_health", "nf_dq", "nf_card"):
        s[c] = s[c].fillna(False).astype(bool)
    T = lambda c: s[c].eq(True)  # noqa: E731
    groups = {
        "presumed healthy = false alarm": T("presumed_healthy"),
        "dead on print, listed": T("wl_dead_print"),
        "live Dec 2024, dead Sept 2026": T("wl_dead_since_dec"),
        "card both dead": T("wl_card_dead"),
        "share fell >5x since Dec 2024": T("wl_degraded"),
        "dq health fail [old def]": T("wl_dq_health"),
        "label-check health fail [old def]": T("wl_lc_health"),
        "card erratic [old def]": T("wl_card_erratic"),
        "dq health fail [no-fault def]": T("wl_dq_health") & s.nf_dq,
        "label-check health fail [no-fault def]": T("wl_lc_health") & s.nf_health,
        "card erratic [no-fault def]": T("wl_card_erratic") & s.nf_card,
        "fault-only members (dropped)": (T("wl_dq_health") | T("wl_lc_health") | T("wl_card_erratic"))
        & ~(s.nf_dq | s.nf_health | s.nf_card),
    }
    rows = []
    for g, msk in groups.items():
        x = s[msk]
        n = x.drop_duplicates(["DeviceId", "detector"]).shape[0]
        for w, y in x.groupby("wlen"):
            rows.append(dict(group=g, n=n, wlen=w, before=100 * y.old.mean(), after=100 * y.new.mean()))
    t = pd.DataFrame(rows)
    t["cell"] = t.before.map("{:.1f}".format) + " -> " + t.after.map("{:.1f}".format)
    out = t.pivot_table(index=["group", "n"], columns="wlen", values="cell", aggfunc="first")[ORDER]
    print(out.reindex(list(groups), level=0).to_string())
    s.to_parquet(H.HB / "nofault_eval.parquet")


if __name__ == "__main__":
    main()
