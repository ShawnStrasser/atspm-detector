"""Note 40: evaluate actuation-level health statistics with note 38's harness.

Needs hb_real.py / hb_calib.py outputs (rules_real.parquet, real_stats.parquet) and
hb_act.py for both periods (act_stats.parquet, act_stats_dec.parquet).

    python hb_act_eval.py   -> prints the tables of note 40; writes act_eval.parquet

1. Screening: each candidate family, limit = p99.8 of presumed-healthy within the detector's
   behaviour group (pulse / short / mid / long / vlong ON, from the log), OR-ed with the
   existing rules.  Compared with LOOSENING the existing rules to the same false-alarm rate
   (the fair baseline: any rule set buys catches with false alarms).
2. The packaged `rapid` rule (health_core.rapid_ratio), rescored through health_core.rules.
3. Dec 2024 -> Sept 2026: detectors the rules call ok in Dec 2024; does a Dec flag predict
   the detector being dead in Sept 2026 (non-circular: other period, other outcome)?
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_calib as C  # noqa: E402
import hb_data as H  # noqa: E402
import health_core as hc  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

FEATS = {"ioi_lt05": 1, "ioi_lt1": 1, "burst_frac": 1, "gap_lt03": 1, "long_time": 1,
         "lock_ratio": 1, "fo_corr": -1, "fo_hi": 1, "fo_lo": 1, "sticky_bins": 1, "fog_dev": 1,
         "nn_js_dur": 1, "nn_js_ioi": 1}
FAMILIES = {"fast: ON->ON < 0.5 / 1 s": ["ioi_lt05", "ioi_lt1"], "bursts (5+ ONs, gaps < 1 s)": ["burst_frac"],
            "re-trigger < 0.3 s by group": ["gap_lt03"], "long ONs 2-15 min": ["long_time"],
            "flow-occupancy (5 stats)": ["fo_corr", "fo_hi", "fo_lo", "sticky_bins", "fog_dev"],
            "nearest-sibling histogram": ["nn_js_dur", "nn_js_ioi"], "ON locked to color": ["lock_ratio"],
            "all 13": list(FEATS)}
POS = ("wl_dq_health", "wl_lc_health", "wl_card_erratic")


def max_ratio(m: pd.DataFrame) -> pd.Series:
    """The existing rules as one continuous margin: max(statistic / suspect limit); >= 1 = flagged."""
    L = hc.LIM
    dead = (m.n_on == 0) & (m.occ_frac == 0)
    z = pd.DataFrame(index=m.index)
    z["dead"] = np.where(dead, m.dead_lam / L["dead_lam"][0], 0)
    z["drop"] = np.where(~dead & (m.n_on > 0), m.drop_lam / L["drop_lam"][0], 0)
    z["stuck"] = m.dur_max / L["stuck_s"][0]
    z["chat"] = np.where(m.n_on >= 50, m.chat_frac / L["chat"][0], 0)
    z["vol"] = m.max5 / L["max5"][0]
    z["nd"] = np.where(m.night_day > 1, m.night_day / m.sig_night_day.clip(lower=.15) / L["nightday"][0], 0)
    z["lvl"] = np.where(m.level_llr > 50, np.log(1 / m.level_ratio) / np.log(1 / L["level"][0]), 0)
    z["err"] = np.where(m.disp_z > 2, m.disp / L["disp"][0], 0)
    z["corr"] = np.where(m.cov_h >= 12, m.corr_gap / L["corr_gap"][0], 0)
    return z.fillna(0).max(axis=1)


def load():
    s = pd.read_parquet(H.HB / "rules_real.parquet")
    a = pd.read_parquet(H.HB / "act_stats.parquet").drop(columns=["period"], errors="ignore")
    m = s.merge(a, on=["DeviceId", "window", "detector"], how="left")
    m["wlen"] = m.window.str.split("_").str[0]
    m["grp"] = hc.behaviour_group(m.pulse_frac, m.med_dur)
    m["H"] = m.presumed_healthy.eq(True)
    m["pos"] = m[list(POS)].eq(True).any(axis=1)
    m["mr"] = max_ratio(m)
    return m


def limits(m, fs, q=.998):
    h = m[m.H & (m.a_n_on >= hc.RAPID_MIN_ON)]
    return {f: (h[f] * FEATS[f]).groupby(h.grp).quantile(q) for f in fs}


def flag(df, TH):
    out = pd.Series(False, index=df.index)
    for f, th in TH.items():
        out |= (df[f] * FEATS[f] > df.grp.map(th)) & (df.a_n_on >= hc.RAPID_MIN_ON)
    return out


def vs_loose(m, new, rows_label):
    """catch on the positives: old, old OR new, and old loosened to the same false-alarm rate."""
    both = new | m.flag
    r = {"candidate": rows_label}
    for wl in ("2h", "6h", "24h", "full"):
        W = m.wlen.eq(wl)
        fa = both[W & m.H].mean()
        loose = (m.mr >= np.quantile(m.mr[W & m.H], 1 - fa)) | m.flag
        P = W & m.pos
        r[f"{wl} FA"] = f"{100 * m.flag[W & m.H].mean():.1f}->{100 * fa:.1f}"
        r[f"{wl} catch new/loose"] = f"{100 * both[P].mean():.1f}/{100 * loose[P].mean():.1f}"
        d = pd.DataFrame({"d": both[P].astype(int) - loose[P].astype(int), "g": m.DeviceId[P]})
        g = d.groupby("g").d.agg(["sum", "size"])
        rng = np.random.default_rng(0)
        bs = [g.iloc[rng.integers(0, len(g), len(g))].pipe(lambda x: x["sum"].sum() / x["size"].sum())
              for _ in range(300)]
        r[f"{wl} diff 90%CI"] = "{:+.1f}..{:+.1f}".format(*(100 * np.quantile(bs, [.05, .95])))
    return r


def dec_to_sept(m, TH_by_name):
    R = pd.read_parquet(H.HB / "real_stats.parquet")
    D = R[R.period.eq("dec") & R.window.eq("full")]
    ad = pd.read_parquet(H.HB / "act_stats_dec.parquet").drop(columns=["period", "window"])
    D = D.merge(ad, on=["DeviceId", "detector"])
    D["grp"] = hc.behaviour_group(D.pulse_frac, D.med_dur)
    D["mr"] = max_ratio(D)
    S = m[m.window.eq("full")][["DeviceId", "detector", "n_on", "wl_dead_print"]]
    D = D.merge(S.rename(columns={"n_on": "n_on_sept"}), on=["DeviceId", "detector"])
    D["dead_sept"] = D.n_on_sept.eq(0) | D.wl_dead_print.eq(True)
    base = D[(D.a_n_on >= hc.RAPID_MIN_ON) & ~D.status.isin(["bad", "suspect"])].copy()
    rows = []
    base["rapid"] = hc.rapid_ratio(base)
    for name, TH in list(TH_by_name.items()) + [("rapid (packaged)", None)]:
        f = flag(base, TH) if TH is not None else base.rapid.ge(1)
        k = np.quantile(base.mr, 1 - f.mean())
        near = base.mr > k                    # the existing rules' near-misses, same count
        rows.append(dict(candidate=name, n_flag_dec=int(f.sum()), dead_sept_flagged=base.dead_sept[f].mean(),
                         dead_sept_rest=base.dead_sept[~f].mean(), old_nearmiss=base.dead_sept[near].mean()))
    return len(base), base.dead_sept.mean(), pd.DataFrame(rows)


def main():
    m = load()
    print("rows", len(m), "with actuation stats", m.a_n_on.notna().mean().round(3))
    # ---- 1. screening
    T = {k: limits(m, fs) for k, fs in FAMILIES.items()}
    print("== 1. screening: OR with existing rules vs existing rules loosened to the same FA "
          "(positives = dq / label-check health fail / card erratic)")
    print(pd.DataFrame([vs_loose(m, flag(m, T[k]), k) for k in FAMILIES]).to_string(index=False))
    # ---- 2. the packaged rule, through health_core.rules
    m["rapid"] = hc.rapid_ratio(m)
    new = m.rapid >= 1
    print("== 2. packaged rapid rule (health_core.RAPID_LIM)")
    print(pd.DataFrame([vs_loose(m, new, "rapid")]).to_string(index=False))
    s = pd.concat([H.rescore(g, H.EVAL_WIN["stg"][w][0]).assign(window=w)
                   for w, g in m.drop(columns=["wl_degraded"], errors="ignore").groupby("window")],
                  ignore_index=True)
    deg, _ = C.degraded(pd.read_parquet(H.HB / "real_stats.parquet"))
    s["wl_degraded"] = [(d, k) in deg for d, k in zip(s.DeviceId, s.detector)]
    s["flag1"] = s["flag"].astype(bool)          # note 38 rules, as stored
    s["flag2"] = s.status.isin(["bad", "suspect"])   # + rapid
    s["wlen"] = s.window.str.split("_").str[0]
    grp = {"presumed healthy (FA)": "presumed_healthy", "dq health fail": "wl_dq_health",
           "label-check health fail": "wl_lc_health", "card erratic": "wl_card_erratic",
           "share fell >5x since Dec": "wl_degraded", "dead on print": "wl_dead_print"}
    rows = []
    for g, c in grp.items():
        x = s[s[c].eq(True)]
        for wl, y in x.groupby("wlen"):
            rows.append(dict(group=g, wlen=wl, before=100 * y.flag1.mean(), after=100 * y.flag2.mean()))
    t = pd.DataFrame(rows).pivot_table(index="group", columns="wlen", values=["before", "after"])
    print(t.round(1).reindex(columns=["2h", "6h", "24h", "full"], level=1).to_string())
    # circularity: new catches on positives that the 66-h chatter statistic (their label source) already had
    full_chat = set(zip(m.DeviceId[m.window.eq("full") & (m.chat_frac > .3)],
                        m.detector[m.window.eq("full") & (m.chat_frac > .3)]))
    nn = s[s.flag2 & ~s.flag1.astype(bool) & s[list(POS)].eq(True).any(axis=1)]
    print("new-only catches on positives by wlen", nn.groupby("wlen").size().to_dict(),
          "of which 66-h chatter > 30 %:", sum((d, k) in full_chat for d, k in zip(nn.DeviceId, nn.detector)))
    # ---- 3. Dec 2024 -> Sept 2026
    n, br, t3 = dec_to_sept(m, T)
    print(f"== 3. Dec 2024 ok ({n} detectors, >= 50 ONs) -> dead Sept 2026; base rate {br:.3f}")
    print(t3.round(3).to_string(index=False))
    s.to_parquet(H.HB / "act_eval.parquet")


if __name__ == "__main__":
    main()
