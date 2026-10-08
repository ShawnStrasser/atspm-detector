"""Note 47: tables from h5_final.py (note 38's harness) against v4 (health4/final_eval_final.parquet): % flagged /
% bad per real-problem group, mode and window; what each new check adds (catches vs false alarms, vs the
shuffled control); the user's answered spot-check rows (all sheets, later answer wins) as a mini-test; and every
detector v4 flagged through an "ON with no OFF" episode, re-checked.

    python h5_final_eval.py -> prints; %DC_WORK%/health5/final_tables.csv, nooff_recheck.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h4_final_eval as E  # noqa: E402
import hb_data as H  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
OUT = H.DCW / "health5"
FINAL = "v5-min_on"               # the shipped defaults: short-ON check as a note only (OPTS min_on=False)
ORDER = E.ORDER
# answers of 2026-09-28 / 29 (h4_final_eval.USER) updated with the v4 sheet (2026-09-29, later answer wins)
USER = dict(E.USER)
USER.update({("2B146", 16): "suspect", ("07035", 19): "flag", ("10018", 2): "bad", ("10090", 22): "bad",
             ("2B067", 42): "ok"})


def hit(status, tgt):
    fl = status in ("bad", "suspect")
    return {"flag": fl, "ok": not fl, "suspect": status == "suspect", "bad": status == "bad"}[tgt]


def main():
    f = pd.read_parquet(OUT / "final_eval.parquet")
    v4 = pd.read_parquet(H.DCW / "health4" / "final_eval_final.parquet")
    v4 = v4[v4.window != "dec_full"].assign(mode="v4")
    f = pd.concat([v4, f], ignore_index=True)
    s = pd.read_parquet(H.HB / "nofault_eval.parquet")
    keys = ["DeviceId", "window", "detector"]
    g = s[keys + ["presumed_healthy", "wl_dead_print", "wl_dead_since_dec", "wl_card_dead", "wl_dq_health",
                  "wl_lc_health", "wl_card_erratic", "nf_dq", "nf_health", "nf_card", "wl_degraded", "wlen"]]
    m = f.merge(g, on=keys, how="inner")
    m["flag"] = m.status.isin(["bad", "suspect"])
    m["bad"] = m.status.eq("bad")
    G = E.groups(m)
    rows = []
    for gname, msk in G.items():
        for (mode, w), x in m[msk].groupby(["mode", "wlen"]):
            rows.append(dict(group=gname, mode=mode, wlen=w, flagged=100 * x.flag.mean(), bad=100 * x.bad.mean(), n=len(x)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "final_tables.csv", index=False)
    for what in ("flagged", "bad"):
        print(f"\n% {what}")
        p = t.pivot_table(index=["group", "mode"], columns="wlen", values=what)[ORDER]
        print(p.round(2).reindex(list(G), level=0).to_string())
    # what each new check adds: rows flagged in v5 but not in v5_a (both checks off), healthy vs positives
    pos = (G["dq health fail [no-fault]"] | G["label-check health [no-fault]"] | G["card erratic [no-fault]"]
           | G["share fell >5x since Dec 2024"])
    m["pos"] = pos
    for c in ("s_short_on", "s_night_drop"):
        for mode in ("v5", "v5_shuf"):
            x = m[m["mode"] == mode]
            print(f"\n{c} > 0 [{mode}] by window: healthy %", (x[x.presumed_healthy.eq(True)].groupby("wlen")[c]
                                                             .apply(lambda v: round(100 * (v > 0).mean(), 2)).to_dict()),
                  "| positives n", x[x.pos].groupby("wlen")[c].apply(lambda v: int((v > 0).sum())).to_dict())
    a = m[m["mode"] == "v5_a"].set_index(keys)
    for mode in ("v5", "v5_shuf", "v5-min_on", "v5-night"):
        b = m[m["mode"] == mode].set_index(keys)
        j = a[["flag", "presumed_healthy", "pos", "wlen"]].join(b[["flag"]], rsuffix="_b", how="inner")
        new = j[~j.flag & j.flag_b]
        print(f"{mode:10s} vs v5_a: newly flagged healthy {new.presumed_healthy.eq(True).groupby(new.wlen).sum().to_dict()}"
              f" positives {new.pos.groupby(new.wlen).sum().to_dict()}")
    # user mini test (66 h)
    v3 = pd.read_parquet(H.REPO / "research" / "labels" / "function_labels_v3.parquet", columns=["DeviceId", "DeviceName"])
    nm = dict(zip(v3.DeviceName, v3.DeviceId.str.lower()))
    full = f[f.window == "full"]
    print()
    for mode, x in full.groupby("mode"):
        res = []
        for name, L in (("answers", USER), ("remarks", E.REMARKS)):
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
        print(f"{mode:10s} " + " | ".join(res))
    # every detector v4 flagged through an ON-with-no-OFF episode, re-checked (all windows)
    no = v4[v4.reason.fillna("").str.contains("no OFF was logged") | v4.bad_periods.fillna("").str.contains("no OFF")]
    v5 = f[f["mode"] == FINAL][keys + ["status", "reason", "s_stuck"]]
    r = no[keys + ["status", "reason"]].merge(v5, on=keys, suffixes=("_v4", "_v5"))
    r = r.merge(g[keys + ["presumed_healthy"]], on=keys, how="left")
    r.to_csv(OUT / "nooff_recheck.csv", index=False)
    print(f"\nv4 rows with a no-OFF episode: {len(no)} (full window {int((no.window == 'full').sum())}); "
          f"no-OFF in the reason (drove the call): {int(no.reason.fillna('').str.contains('no OFF was logged').sum())}")
    rr = r[r.reason_v4.fillna("").str.contains("no OFF was logged")]
    print("status v4 -> v5 where the no-OFF episode was in the reason:", rr.groupby(["status_v4", "status_v5"]).size().to_dict())
    print("  of them presumed healthy:", rr[rr.presumed_healthy.eq(True)].groupby(["status_v4", "status_v5"]).size().to_dict())
    print("  v5 reasons still quoting 'no OFF':", int(f.reason.fillna("").str.contains("no OFF").sum() - v4.reason.fillna("").str.contains("no OFF").sum()))
    for w in ("full",):
        b4 = m[(m["mode"] == "v4") & (m.wlen == w)].set_index(keys)
        b5 = m[(m["mode"] == FINAL) & (m.wlen == w)].set_index(keys)
        j = b4[["status", "presumed_healthy"]].join(b5[["status"]], rsuffix="_5", how="inner")
        ch = j[j.status != j.status_5]
        print(f"\n{w}: status changes v4 -> v5 {len(ch)}:", ch.groupby(["status", "status_5"]).size().to_dict(),
              "| on healthy:", ch[ch.presumed_healthy.eq(True)].groupby(["status", "status_5"]).size().to_dict())


if __name__ == "__main__":
    main()
