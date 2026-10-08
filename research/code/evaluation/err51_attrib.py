"""Note 51 step 4: attribute every function_v3e OOF error (rows from err51_build.py) to label noise / ambiguous by
definition / model-fixable (hierarchical rules, first match wins), on both scoring sets.  Rules are informed by the
example review (err51_examples.py); they are an estimate, not a relabelling.

    python err51_attrib.py > %DC_WORK%/trackA/err51/attrib.txt
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

D = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), "trackA/err51/")
ORDER = ["full", "h24", "h6", "h3", "h1", "m30", "m10", "m5"]
# print subtypes of "Other" (and Mid) whose log behaviour is by construction that of a PM class
OTHER_LIKE = {"superseded_by_radar", "advance_presence", "long_zone", "presence_20_75", "presence_setback_75",
              "presence_setback", "setback_zone", "dump_loop", "on_ramp_count", "departure", "queue", "extension",
              "rail_presence", "adaptive", "eta", "truck", "track", "ped_camera", "explained"}
RADAR_LONG_ADV = {"radar_advance", "radar_far_advance", "radar_advance_trucks", "radar_advance_cars",
                  "radar_sidefire_advance", "radar_side_fire_advance", "radar_advance_count", "radar_count_upstream"}
PM = {"Advance", "Presence", "Count", "Yellow_Red"}


def yr_twin(df):
    """log count ratio of each Yellow_Red / Count row to its closest opposite-class sibling (same labelled phase,
    same window); NaN when the phase has none."""
    k = ["DeviceId", "period", "win", "phase_target"]
    out = pd.Series(np.nan, index=df.index)
    for a, b in (("Yellow_Red", "Count"), ("Count", "Yellow_Red")):
        x = df[df.truth == a][k + ["Detector", "det_n_on"]].reset_index()
        y = df[df.truth == b][k + ["det_n_on"]].rename(columns={"det_n_on": "n2"})
        m = x.merge(y, on=k, how="left")
        m["lr"] = np.abs(np.log((m.det_n_on + 1) / (m.n2 + 1)))
        out.loc[m.groupby("index").lr.min().index] = m.groupby("index").lr.min().values
    return out


def attribute(df):
    df = df.copy()
    df["wo"] = df.wgroup.map({g: i for i, g in enumerate(ORDER)})
    lo = df.groupby(["DeviceId", "Detector"]).wo.transform("min")
    longest_ok = df[df.wo == lo].groupby(["DeviceId", "Detector"]).ok.mean()
    df["long_ok"] = df.set_index(["DeviceId", "Detector"]).index.map(longest_ok).to_numpy() > .5
    df["det_acc"] = df.groupby(["DeviceId", "Detector"]).ok.transform("mean")
    df["twin"] = yr_twin(df)
    # Dec-2024 rows of detectors wrong in Dec (< .2) but right in Sept 2026 (> .8): the channel was re-assigned
    # between the two pulls (labels = the current print / config), so the Dec label is stale
    pa = df.groupby(["DeviceId", "Detector", "period"]).ok.mean().unstack()
    drift = pa[(pa.get("dec") < .2) & (pa.get("stg") > .8)].index
    df["drift"] = df.set_index(["DeviceId", "Detector"]).index.isin(drift) & df.period.eq("dec").to_numpy()
    t, p, st = df.truth, df.pred, df.print_subtype.fillna("none")
    yrc = ((t == "Yellow_Red") & (p == "Count")) | ((t == "Count") & (p == "Yellow_Red"))
    rules = [
        ("A0 Dec-2024 label stale (right in Sept 2026, wrong in Dec)", df.drift),
        ("A1 label-check fail / misconfigured", df.validated.isin(["fail", "misconfigured"])),
        ("A2 unhealthy / no_data / health bad", df.validated.isin(["unhealthy", "no_data"]) | df.health.eq("bad")),
        ("A3 config-only label, model confident (p>=.8)", df.truth_src.eq("config") & (df.p_max >= .8)),
        ("B1 Other subtype that behaves like a PM class", t.eq("Other") & st.isin(OTHER_LIKE) & p.isin(PM | {"Mid"})),
        ("B2 radar long 'Advance' zone called Other", t.eq("Advance") & st.isin(RADAR_LONG_ADV) & p.eq("Other")),
        ("B3 Yellow_Red/Count twin (count within 5 %) or no partner", yrc & ((df.twin < .05) | df.twin.isna())),
        ("B4 Mid / series loop vs Advance / Presence", (t.eq("Mid") & p.isin(["Advance", "Presence"]))
         | (p.eq("Mid") & t.isin(["Advance", "Presence"]))),
        ("A4 config-only label, model unsure", df.truth_src.eq("config")),
        ("C1 phase predicted wrong", df.phase_ok.eq(False)),
        ("C2 short-sample miss (right at the detector's longest window)", df.long_ok & (df.wo > 0)),
        ("C3 systematic, other/unsure Other rows", t.eq("Other") | p.eq("Other")),
        ("C4 systematic, among PM classes", pd.Series(True, index=df.index)),
    ]
    e = df[~df.ok].copy()
    e["cat"] = None
    for name, m in rules:
        sel = e.cat.isna() & m.loc[e.index]
        e.loc[sel, "cat"] = name
    return df, e


def main():
    fr = pd.read_parquet(D + "rows.parquet")
    for sn, m in (("EVERYTHING", fr.setA), ("REALISTIC", fr.setR)):
        df, e = attribute(fr[m])
        n = len(df)
        print(f"\n===== {sn}: rows {n:,}, errors {len(e):,} ({len(e) / n:.4f})")
        g = e.groupby("cat").agg(err=("ok", "size"), p_med=("p_max", "median"), det_acc=("det_acc", "mean"))
        g["share_err"] = (g.err / len(e)).round(3)
        g["pt_of_acc"] = (100 * g.err / n).round(2)
        print(g.round(3).to_string())
        for grp in "ABC":
            s = e.cat.str.startswith(grp).sum()
            print(f"  {grp}: {s:,} errors = {s / len(e):.3f} of errors = {100 * s / n:.2f} pt")
        # excess-rate estimate of label noise (independent of the rules): errors above the print_high rate
        base = 1 - df[df.truth_src.eq("print_high") & df.validated.eq("pass")].ok.mean()
        for col, vals in (("truth_src", ["config", "user_ruling"]),
                          ("validated", ["fail", "misconfigured", "unhealthy", "no_data"])):
            for v in vals:
                x = df[df[col] == v]
                if len(x):
                    print(f"  excess errors {col}={v}: {len(x):,} rows, err {1 - x.ok.mean():.3f} vs {base:.3f} -> "
                          f"{(1 - x.ok.mean() - base) * len(x):,.0f} excess")
        print("\n  pairs inside C2 (short-sample):",
              e[e.cat.str.startswith("C2")].groupby(["truth", "pred"]).size().sort_values(ascending=False).head(6).to_dict())
        print("  C2 by window:", e[e.cat.str.startswith("C2")].wgroup.value_counts().to_dict())
        for c in ("C3", "C4"):
            x = e[e.cat.str.startswith(c)]
            print(f"  pairs inside {c}:", x.groupby(["truth", "pred"]).size().sort_values(ascending=False).head(8).to_dict())
            print(f"  {c} tech:", (x.technology.fillna("none").value_counts() / len(x)).round(2).to_dict(),
                  " det always-wrong share", round((x.det_acc == 0).mean(), 2), " p_max median", round(x.p_max.median(), 2))
        e[["DeviceId", "DeviceName", "Detector", "period", "win", "wgroup", "truth", "pred", "p_max", "cat",
           "print_subtype", "technology", "truth_src"]].to_parquet(D + f"errors_{sn.lower()}.parquet", index=False)


if __name__ == "__main__":
    main()
