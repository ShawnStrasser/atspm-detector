"""Note 46: tables from h4_final.py's output (note 38's harness): % flagged / % bad per real-problem group,
mode and window length, the user's answered spot-check rows as a mini-test, and what changed v3 -> v4.

    python h4_final_eval.py [--file final_eval.parquet] -> prints; %DC_WORK%/health4/final_tables.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
OUT = H.DCW / "health4"
ORDER = ["2h", "6h", "24h", "full"]
# the user's answers (review/spotcheck_health.xlsx): 2026-09-28 sheet, then the 2026-09-29 sheet (later answer wins).
# target: flag = bad or suspect, suspect = flagged but not bad, ok = not flagged, ? = unsure (unscored)
USER = {("05032", 52): "flag", ("01026", 24): "flag", ("03024", 4): "flag", ("03026", 22): "flag",
        ("04055", 24): "ok", ("2B439", 25): "ok", ("01009", 23): "flag", ("14003", 25): "flag",
        ("07035", 19): "flag", ("2B048", 6): "ok", ("2B525", 7): "flag", ("03090", 25): "ok", ("07027", 9): "flag",
        # 2026-09-29 sheet
        ("2B146", 16): "flag", ("01026", 25): "flag", ("2B402", 16): "suspect", ("03033", 8): "suspect",
        ("10045", 4): "suspect", ("2B044", 14): "suspect"}
REMARKS = {("07027", 8): "flag", ("2B422", 8): "flag", ("11042", 22): "flag", ("10045", 5): "suspect",
           ("10045", 7): "?", ("10045", 6): "ok"}


def groups(s):
    T = lambda c: s[c].eq(True)  # noqa: E731
    return {"presumed healthy = false alarm": T("presumed_healthy"),
            "dead on print": T("wl_dead_print"), "live Dec 2024, dead Sept 2026": T("wl_dead_since_dec"),
            "card both dead": T("wl_card_dead"),
            "dq health fail [no-fault]": T("wl_dq_health") & s.nf_dq,
            "label-check health [no-fault]": T("wl_lc_health") & s.nf_health,
            "card erratic [no-fault]": T("wl_card_erratic") & s.nf_card,
            "share fell >5x since Dec 2024": T("wl_degraded")}


def hit(status, tgt):
    fl = status in ("bad", "suspect")
    return {"flag": fl, "ok": not fl, "suspect": status == "suspect"}[tgt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="final_eval.parquet")
    a = ap.parse_args()
    f = pd.read_parquet(OUT / a.file)
    f = f[f.window != "dec_full"]
    s = pd.read_parquet(H.HB / "nofault_eval.parquet")
    keys = ["DeviceId", "window", "detector"]
    g = s[keys + ["presumed_healthy", "wl_dead_print", "wl_dead_since_dec", "wl_card_dead", "wl_dq_health",
                  "wl_lc_health", "wl_card_erratic", "nf_dq", "nf_health", "nf_card", "wl_degraded", "wlen"]]
    m = f.merge(g, on=keys, how="inner")
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
        print(p.round(2).reindex(list(G), level=0).to_string())
    print("\nn per group (full):", {k: int((v & m.wlen.eq("full") & m["mode"].eq("v4")).sum()) for k, v in G.items()})
    # user mini test, 66-h window
    v3 = pd.read_parquet(H.REPO / "research" / "labels" / "function_labels_v3.parquet", columns=["DeviceId", "DeviceName"])
    nm = dict(zip(v3.DeviceName, v3.DeviceId.str.lower()))
    full = f[f.window == "full"]
    for mode, x in full.groupby("mode"):
        res = []
        for name, L in (("answers", USER), ("remarks", REMARKS)):
            ok = n = 0
            miss = []
            for (sig, d), tgt in L.items():
                k = x[(x.DeviceId == nm.get(sig)) & (x.detector == d)]
                st = k.status.iat[0] if len(k) else "missing"
                if tgt == "?":
                    continue
                n += 1
                if hit(st, tgt):
                    ok += 1
                else:
                    miss.append(f"{sig} d{d} {st} (want {tgt})")
            res.append(f"{name} {ok}/{n}: " + ", ".join(miss))
        print(f"{mode:14s} " + " | ".join(res))
    # what v4 changed on the healthy set and the positives (66 h)
    for w in ORDER:
        a3 = m[(m["mode"] == "v3") & (m.wlen == w)].set_index(keys)
        a4 = m[(m["mode"] == "v4") & (m.wlen == w)].set_index(keys)
        j = a3[["status", "presumed_healthy"]].join(a4[["status"]], rsuffix="_4", how="inner")
        ch = j[j.status != j.status_4]
        print(f"\n{w}: status changes v3 -> v4 {len(ch)}:", ch.groupby(["status", "status_4"]).size().to_dict(),
              "| on healthy:", ch[ch.presumed_healthy.eq(True)].groupby(["status", "status_4"]).size().to_dict())
    x = m[(m["mode"] == "v4") & (m.wlen == "full")]
    for c in ("s_rapid_hour", "s_choppy", "s_rapid"):
        if c in x:
            print(c, "> 0 on healthy:", round(100 * (x[G["presumed healthy = false alarm"].loc[x.index]][c] > 0).mean(), 2), "%")


if __name__ == "__main__":
    main()
