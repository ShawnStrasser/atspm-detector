"""Note 43 step 5: tables for the note from h3_final.py's output (note 38's harness).

    python h3_final_eval.py -> prints; %DC_WORK%/health3/final_tables.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
OUT = H.DCW / "health3"
ORDER = ["2h", "6h", "24h", "full"]
# the user's 15 answers (review/spotcheck_health.xlsx, 2026-09-28): what he judged, ? = unsure
USER = [("05032", 52, "flag"), ("01026", 24, "flag"), ("03024", 4, "flag"), ("03026", 22, "flag"),
        ("2B044", 14, "ok"), ("04055", 24, "ok"), ("2B439", 25, "?"), ("01009", 23, "flag"),
        ("14003", 25, "flag"), ("07035", 19, "flag"), ("03033", 8, "flag"), ("2B048", 6, "ok"),
        ("2B525", 7, "flag"), ("03090", 25, "?"), ("07027", 9, "flag")]
# detectors he remarked on in his comments (same file + spotcheck_label_checks.xlsx)
EXTRA = [("01026", 25, "flag"), ("07027", 8, "flag"), ("2B422", 8, "flag"), ("11042", 22, "flag"),
         ("2B422", 10, "?"), ("2B422", 11, "?"), ("04055", 22, "?"), ("07035", 18, "?")]


def groups(s):
    T = lambda c: s[c].eq(True)  # noqa: E731
    return {"presumed healthy = false alarm": T("presumed_healthy"),
            "dead on print": T("wl_dead_print"), "live Dec 2024, dead Sept 2026": T("wl_dead_since_dec"),
            "card both dead": T("wl_card_dead"),
            "dq health fail [no-fault]": T("wl_dq_health") & s.nf_dq,
            "label-check health [no-fault]": T("wl_lc_health") & s.nf_health,
            "card erratic [no-fault]": T("wl_card_erratic") & s.nf_card,
            "share fell >5x since Dec 2024": T("wl_degraded")}


def main():
    f = pd.read_parquet(OUT / "final_eval.parquet")
    f = f[f.window != "dec_full"]
    s = pd.read_parquet(H.HB / "nofault_eval.parquet")
    keys = ["DeviceId", "window", "detector"]
    g = s[keys + ["status", "presumed_healthy", "wl_dead_print", "wl_dead_since_dec", "wl_card_dead", "wl_dq_health",
                  "wl_lc_health", "wl_card_erratic", "nf_dq", "nf_health", "nf_card", "wl_degraded", "wlen"]]
    v2 = g.assign(mode="v2")
    m = pd.concat([v2, f.merge(g.drop(columns="status"), on=keys, how="inner")], ignore_index=True)
    m["flag"] = m.status.isin(["bad", "suspect"])
    m["bad"] = m.status.eq("bad")
    print("rows per mode:", m.groupby("mode").size().to_dict())
    G = groups(m)
    rows = []
    for gname, msk in G.items():
        for (mode, w), x in m[msk].groupby(["mode", "wlen"]):
            rows.append(dict(group=gname, mode=mode, wlen=w, flagged=100 * x.flag.mean(), bad=100 * x.bad.mean(),
                             n=len(x)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "final_tables.csv", index=False)
    for what in ("flagged", "bad"):
        print(f"\n% {what}")
        p = t.pivot_table(index=["group", "mode"], columns="wlen", values=what)[ORDER]
        print(p.round(1).reindex(list(G), level=0).to_string())
    # recovery: detectors whose episode rule was capped (bad -> suspect) by a clean recovery
    x = m[(m["mode"] == "v3_pf")]
    rec = (x.ep_rec.between(.5, 2) & (x.ep_dur >= 900)) | (x.drop_rec.between(.5, 2) & (x.drop_lam >= 30))
    print("\nclean recoveries (v3_pf), rows by window:", x[rec].groupby("wlen").size().to_dict(),
          "| of them suspect:", x[rec].status.eq("suspect").groupby(x[rec].wlen).mean().round(2).to_dict())
    cs = x[(x.ep_co >= 3) & (x.ep_dur >= 900)]
    print("stuck with >= 3 co-stuck (exonerated):", cs.groupby("wlen").size().to_dict())
    # user mini test (66-h window)
    v3 = pd.read_parquet(H.REPO / "research" / "labels" / "function_labels_v3.parquet", columns=["DeviceId", "DeviceName"])
    nm = dict(zip(v3.DeviceName, v3.DeviceId.str.lower()))
    full = m[m.wlen == "full"]
    for mode, x in full.groupby("mode"):
        res = {}
        for name, L in (("answers", USER), ("remarks", EXTRA)):
            ok = n = 0
            miss = []
            for a, d, tgt in L:
                k = x[(x.DeviceId == nm.get(a)) & (x.detector == d)]
                st = k.status.iat[0] if len(k) else "missing"
                fl = st in ("bad", "suspect")
                if tgt != "?":
                    n += 1
                    ok += fl == (tgt == "flag")
                    if fl != (tgt == "flag"):
                        miss.append(f"{a} d{d} {st}")
                else:
                    miss.append(f"({a} d{d} {st})")
            res[name] = f"{ok}/{n}"
            res[name + "_miss"] = ", ".join(miss)
        print(f"{mode:9s} answers {res['answers']} remarks {res['remarks']} | {res['answers_miss']} | {res['remarks_miss']}")
    # FA composition by predicted function (v2 vs v3_pf, 2 h and 66 h)
    for mode in ("v3_blind", "v3_pf"):
        x = m[(m["mode"] == mode) & G["presumed healthy = false alarm"]]
        print(mode, "FA % by predicted function:",
              (100 * x.groupby(["wlen", x.pred_function.fillna("none")]).flag.mean()).unstack().round(1).loc[ORDER].to_string())


if __name__ == "__main__":
    main()
