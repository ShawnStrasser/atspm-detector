"""Note 70: note 51's error decomposition redone on the CURRENT candidate (cand64 function_rows: trees + fj blend at tree
weight 0.6, lanes D decode, stack pick, ATSPM-only stack-aware score; truth_v3s; Dec-role filter).  Same hierarchical
rules as note 51 (`err51_attrib.attribute`, first match wins) so the buckets are comparable, plus B5 for the stack_extra
error type that did not exist in note 51.  Headline pool >= 30 min, both sets.  Three columns:
  old    function_v3e argmax, note-51 truth / sets, ATSPM-only score (no stack credit)            = note 51 restated
  bridge function_v3e argmax, CURRENT truth / sets / stack credit (labels + scoring fixes only)
  new    the candidate (trees + fj, decoded)                                                     = today
Analysis only; nothing trained.  locked_v2 asserted absent.

    python err70.py > %DC_WORK%/err70/attrib70.txt
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
import err51_attrib as E51  # noqa: E402

OUT = DC_WORK / "err70"
GE30 = ["m30", "h1", "h3", "h6", "h24", "full"]
ATS = {"Advance", "Presence", "Count", "Yellow_Red"}
RUN51 = "run_ad6ea6959f_exclude_min5_clean_valnc_h3"
KEY = ["DeviceId", "Detector", "period", "win"]


def locked() -> set:
    return set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())


def meta() -> pd.DataFrame:
    """per-detector attributes from the CURRENT label table, named as in note 51's rows."""
    import v3_retrain as V
    v = pd.read_parquet(rpath.LABELS_CURRENT)  # note 81: v4l
    v["DeviceId"] = v.DeviceId.str.lower()
    src = v.source.astype("string").fillna("none")
    tsrc = np.select([src.eq("user_ruling"), src.eq("print_high"), src.eq("config"),
                      src.eq("whole_intersection_other")], ["user_ruling", "print_high", "config", "whole_int_other"],
                     "print_med_low")
    return pd.DataFrame({"DeviceId": v.DeviceId, "Detector": v.detector, "truth_src": tsrc,
                         "print_subtype": v.print_subtype, "technology": v.technology, "phase_target": v.phase_target,
                         "stack_relabel": v.stack_relabel.eq(True), "description": v.description,
                         "DeviceName": v.DeviceName, "config_function": v.config_function,
                         "print_function": v.print_function, "dead": v.dead.eq(True),
                         "dq_suspect": v.dq_suspect.eq(True)})


def side(df: pd.DataFrame, r51: pd.DataFrame) -> pd.DataFrame:
    """phase_ok and health per row, exactly as note 51 built them (copied from err51 rows by detector / period)."""
    p = r51[["DeviceId", "Detector", "period", "win", "phase_ok"]].drop_duplicates(KEY)
    h = r51[["DeviceId", "Detector", "period", "health"]].drop_duplicates(["DeviceId", "Detector", "period"])
    df = df.merge(p, on=KEY, how="left").merge(h, on=["DeviceId", "Detector", "period"], how="left")
    df["health"] = df.health.fillna("no_row")
    return df


def attribute(df: pd.DataFrame):
    """note 51's rules + B5 (stack_extra) inserted after A3; df.ok = ATSPM ok."""
    df = df.copy()
    df["wo"] = df.wgroup.map({g: i for i, g in enumerate(E51.ORDER)})
    lo = df.groupby(["DeviceId", "Detector"]).wo.transform("min")
    longest_ok = df[df.wo == lo].groupby(["DeviceId", "Detector"]).ok.mean()
    df["long_ok"] = df.set_index(["DeviceId", "Detector"]).index.map(longest_ok).to_numpy() > .5
    df["det_acc"] = df.groupby(["DeviceId", "Detector"]).ok.transform("mean")
    df["twin"] = E51.yr_twin(df)
    pa = df.groupby(["DeviceId", "Detector", "period"]).ok.mean().unstack()
    drift = pa[(pa.get("dec") < .2) & (pa.get("stg") > .8)].index if {"dec", "stg"} <= set(pa.columns) else []
    df["drift"] = df.set_index(["DeviceId", "Detector"]).index.isin(drift) & df.period.eq("dec").to_numpy()
    t, p, st = df.truth, df.pred, df.print_subtype.fillna("none")
    yrc = ((t == "Yellow_Red") & (p == "Count")) | ((t == "Count") & (p == "Yellow_Red"))
    ex = df.err.eq("stack_extra") if "err" in df else pd.Series(False, index=df.index)
    rules = [
        ("A0 Dec-2024 label stale", df.drift),
        ("A1 label-check fail / misconfigured", df.validated.isin(["fail", "misconfigured"])),
        ("A2 unhealthy / no_data / health bad", df.validated.isin(["unhealthy", "no_data"]) | df.health.eq("bad")),
        ("A3 config-only label, model confident", df.truth_src.eq("config") & (df.p_max >= .8)),
        ("B5 stack extra lane (same role twice)", ex),
        ("B1 Other subtype behaving like a PM class", t.eq("Other") & st.isin(E51.OTHER_LIKE) & p.isin(E51.PM | {"Mid"})),
        ("B2 radar long Advance called Other", t.eq("Advance") & st.isin(E51.RADAR_LONG_ADV) & p.eq("Other")),
        ("B3 YR/Count twin (count within 5 %) or no partner", yrc & ((df.twin < .05) | df.twin.isna())),
        ("B4 Mid / series vs Advance / Presence", (t.eq("Mid") & p.isin(["Advance", "Presence"]))
         | (p.eq("Mid") & t.isin(["Advance", "Presence"]))),
        ("A4 config-only label, model unsure", df.truth_src.eq("config")),
        ("C1 phase predicted wrong", df.phase_ok.eq(False)),
        ("C2 short-sample miss", df.long_ok & (df.wo > 0)),
        ("C3 systematic, ->/from Other", t.eq("Other") | p.eq("Other")),
        ("C4 systematic, among PM classes", t.isin(ATS) & p.isin(ATS)),
        ("D  other / unknown (Mid / Bike <-> PM, rest)", pd.Series(True, index=df.index)),
    ]
    e = df[~df.ok.astype(bool)].copy()
    e["cat"] = None
    for name, m in rules:
        sel = e.cat.isna() & m.loc[e.index]
        e.loc[sel, "cat"] = name
    return df, e


def ge30(df):
    return df[df.wgroup.isin(GE30)]


def old_rows(r51: pd.DataFrame) -> dict:
    """note 51 restated: v3e argmax, old truth / sets, ATSPM-only ok (no stack credit)."""
    r = r51.copy()
    ta, pa = r.truth.isin(ATS), r.pred.isin(ATS)
    r["ok"] = np.where(ta, r.pred == r.truth, ~pa)
    r["err"] = ""
    return {"everything": r[r.setA], "realistic": r[r.setR]}


def bridge_rows(m: pd.DataFrame, r51: pd.DataFrame) -> dict:
    """v3e argmax on the CURRENT truth / sets with stack credit (atspm_score.credit, as cand64)."""
    import atspm_score as S
    import atspm_pick55 as AP
    fr = S.load(RUN51)
    rows = AP.score_rows(fr)
    pred = fr[[f"P_{c}" for c in S.V.C7]].to_numpy().argmax(1)
    out = {}
    for s, ix in rows.items():
        d = S.credit(fr, pred, "truth_v3s", ix, True)
        x = fr.iloc[ix][KEY + ["wgroup", "det_n_on", "validated"]].reset_index(drop=True)
        x["truth"], x["pred"], x["ok"], x["err"] = d.t.to_numpy(), d.p.to_numpy(), d.ok_a.to_numpy(bool), d.err.to_numpy()
        x["p_max"] = fr[[f"P_{c}" for c in S.V.C7]].to_numpy()[ix].max(1)
        x = x.merge(m, on=["DeviceId", "Detector"], how="left")
        out[s] = side(x, r51)
    return out


def new_rows(m: pd.DataFrame, r51: pd.DataFrame) -> dict:
    f = pd.read_parquet(DC_WORK / "cand64" / "function_rows.parquet")
    out = {}
    for s, k in (("everything", "E"), ("realistic", "R")):
        x = f[f[f"ok_{k}_fj"].notna()][KEY + ["wgroup", "det_n_on", "validated", "truth_v3s", "pred_fj", "p_pred_fj",
                                             f"ok_{k}_fj", f"err_{k}_fj", "has_fj", "health_status", "lanes_D"]].copy()
        x = x.rename(columns={"truth_v3s": "truth", "pred_fj": "pred", "p_pred_fj": "p_max", f"ok_{k}_fj": "ok",
                              f"err_{k}_fj": "err"})
        x["ok"] = x.ok.astype(bool)
        x["validated"] = x.validated.astype(str)
        x = x.merge(m, on=["DeviceId", "Detector"], how="left")
        out[s] = side(x, r51)
    return out


def report(tag: str, sets: dict, store: dict):
    for sn, df0 in sets.items():
        df, e = attribute(df0)           # rules see every window (long_ok / drift need them) ...
        df, e = ge30(df), ge30(e)        # ... headline pool >= 30 min
        n = len(df)
        print(f"\n===== {tag} / {sn}: rows {n:,} ({df.DeviceId.nunique()} sig.), ATSPM errors {len(e):,} "
              f"= {100 * len(e) / n:.2f} pt (acc {1 - len(e) / n:.4f})")
        g = e.groupby("cat").agg(err=("ok", "size"), p_med=("p_max", "median"), det_acc=("det_acc", "mean"))
        g["share"] = (g.err / len(e)).round(3)
        g["pt"] = (100 * g.err / n).round(2)
        print(g.round(3).to_string())
        for grp in "ABCD":
            s = e.cat.str.startswith(grp).sum()
            store[(tag, sn, grp)] = (100 * s / n, s / len(e))
            print(f"  {grp}: {s:,} = {s / len(e):.3f} of errors = {100 * s / n:.2f} pt")
        for c in sorted(e.cat.unique()):
            store[(tag, sn, c)] = 100 * (e.cat == c).sum() / n
        store[(tag, sn, "total")] = 100 * len(e) / n
        if tag == "new":
            print("  error type:", e.err.value_counts().to_dict())
            for c in sorted(e.cat.unique()):
                x = e[e.cat == c]
                print(f"  {c[:3]} pairs", x.groupby(["truth", "pred"]).size().sort_values(ascending=False).head(5).to_dict(),
                      "| tech", (x.technology.fillna("none").value_counts(normalize=True).round(2).head(4)).to_dict(),
                      "| subtype", x.print_subtype.fillna("none").value_counts().head(4).to_dict(),
                      "| src", x.truth_src.value_counts().head(3).to_dict())
            OUT.mkdir(parents=True, exist_ok=True)
            e.drop(columns=["wo"]).to_parquet(OUT / f"errors_{sn}.parquet", index=False)


def main():
    lk = locked()
    r51 = pd.read_parquet(DC_WORK / "trackA" / "err51" / "rows.parquet")
    assert not r51.DeviceId.isin(lk).any()
    m = meta()
    assert not m.DeviceId.isin(lk).any()
    store = {}
    report("old", old_rows(r51), store)
    report("bridge", bridge_rows(m, r51), store)
    report("new", new_rows(m, r51), store)
    print("\n===== pt of rows by bucket (>= 30 min): old -> bridge -> new")
    for sn in ("everything", "realistic"):
        for k in ["total", "A", "B", "C", "D"]:
            v = [store.get((t, sn, k), (0, 0)) for t in ("old", "bridge", "new")]
            v = [x[0] if isinstance(x, tuple) else x for x in v]
            print(f"  {sn[:4]} {k:5s} " + " -> ".join(f"{x:.2f}" for x in v))
        cats = sorted({k[2] for k in store if k[1] == sn and len(k[2]) > 5})
        for c in cats:
            v = [store.get((t, sn, c), 0.0) for t in ("old", "bridge", "new")]
            print(f"     {c[:44]:44s} " + " -> ".join(f"{x:.2f}" for x in v))


if __name__ == "__main__":
    main()
