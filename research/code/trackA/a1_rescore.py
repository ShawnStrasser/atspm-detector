"""A1 (part 2) -- the single documented re-score of the locked signals.

Nothing is trained or tuned here.  The predictions are the ones `score_final_v2.py`
already wrote (`dc_work/preds/final_test_DO_NOT_USE/v2`); this only changes the TRUTH
they are compared against:

  * function: the config label -> the user's corrected label (`function_labels_v2`)
  * phase:    a prediction equal to the timing's `switch_phase` or one of its
              `additional_call_phases` now counts as correct.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

# repository root: DC_REPO, else three levels up when run from research/code/trackA
REPO = Path(os.environ.get("DC_REPO") or Path(__file__).resolve().parents[3])
DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
WORK = DCW / "trackA"
PRED = DCW / "preds" / "final_test_DO_NOT_USE" / "v2"
CLASSES5 = ["Advance", "Presence", "Count", "Yellow_Red", "Other"]
CLASSES3 = ["Advance", "Presence", "Count"]
MIN_ACT = 5


def parse_extra(s) -> list[int]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    t = str(s).strip()
    if not t or t.lower() in ("none", "nan"):
        return []
    out = []
    for p in t.replace(";", ",").split(","):
        p = p.strip()
        if p.isdigit() and 1 <= int(p) <= 16:
            out.append(int(p))
    return out


def accept_map() -> pd.DataFrame:
    """(DeviceId, Detector) -> the set of phases the timing itself says this detector
    legitimately serves: call_phase, switch_phase and additional_call_phases, unioned
    over every timing plan."""
    dp = pd.read_parquet(REPO / "data" / "detector_plans.parquet")
    dp["DeviceId"] = dp.DeviceId.str.lower()
    dp["Detector"] = dp.Detector.astype(int)
    rows = {}
    for did, det, cp, sw, add in zip(dp.DeviceId, dp.Detector, dp.call_phase,
                                     dp.switch_phase, dp.additional_call_phases):
        k = (did, det)
        s = rows.setdefault(k, set())
        if 1 <= int(cp) <= 16:
            s.add(int(cp))
        if 1 <= int(sw) <= 16:
            s.add(int(sw))
        s.update(parse_extra(add))
    return rows


WINS = {"full": (None, None), "h6_a0": ("2026-09-18 17:00:00", 360),
        "h6_a3": ("2026-09-21 04:23:00", 360),
        "m30_a0": ("2026-09-18 17:00:00", 30), "m30_a1": ("2026-09-19 12:00:00", 30),
        "m30_a2": ("2026-09-20 10:00:00", 30), "m30_a3": ("2026-09-21 07:30:00", 30)}


def candidates() -> dict:
    """{(DeviceId, win)} -> set of phases with a Begin Green inside the window.
    Same rule the published scoring used (`score_final_v2.candidates`)."""
    import duckdb
    ids = sorted(pd.read_csv(DCW / "official" / "newtest_signals.csv")
                 .DeviceId.str.lower().unique())
    glob = (DCW / "official" / "stg" / "cache" / "events" / "**" / "*.parquet").as_posix()
    con = duckdb.connect()
    con.execute("SET memory_limit='12GB'"); con.execute("SET threads=12")
    con.execute(f"SET temp_directory='{(DCW / 'tmp').as_posix()}'")
    idl = ",".join("'" + d + "'" for d in ids)
    out = {}
    for w, (start, mins) in WINS.items():
        cl = ["EventId = 1", "Parameter BETWEEN 1 AND 16",
              f"lower(CAST(DeviceId AS VARCHAR)) IN ({idl})"]
        if start is not None:
            end = str(pd.Timestamp(start) + pd.Timedelta(minutes=mins))
            cl += [f"Timestamp >= TIMESTAMP '{start}'", f"Timestamp < TIMESTAMP '{end}'"]
        d = con.sql(f"""SELECT DISTINCT lower(CAST(DeviceId AS VARCHAR)) AS DeviceId,
                               CAST(Parameter AS INT) AS p
                        FROM read_parquet('{glob}', union_by_name=true)
                        WHERE {' AND '.join(cl)}""").df()
        for did, g in d.groupby("DeviceId"):
            out[(did, w)] = set(g.p.astype(int))
    con.close()
    return out


def function_block(pred: pd.DataFrame, lab: pd.DataFrame, col: str) -> dict:
    d = pred.merge(lab, on=["DeviceId", "Detector"], how="inner")
    d = d[d.function_pred.notna() & (d.n_actuations >= MIN_ACT) & d[col].notna()]
    if not len(d):
        return {"n": 0}
    y, p = d[col].to_numpy(), d.function_pred.to_numpy()
    out = {"n": int(len(d)), "n_signals": int(d.DeviceId.nunique()),
           "acc5": round(float((y == p).mean()), 4)}
    apc = np.isin(y, CLASSES3)
    out["n_apc"] = int(apc.sum())
    out["acc_apc"] = round(float((y[apc] == p[apc]).mean()), 4) if apc.any() else None
    per = {}
    for c in CLASSES5:
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        per[c] = {"n": tp + fn,
                  "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                  "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    out["per_class"] = per
    for th in (0.7, 0.8, 0.9):
        k = (d.function_prob >= th).to_numpy()
        out[f"cov{th}"] = round(float(k.mean()), 4)
        out[f"acc_at_cov{th}"] = round(float((y[k] == p[k]).mean()), 4) if k.any() else None
    return out


def main() -> None:
    lab = pd.read_parquet(REPO / "research" / "labels" / "function_labels_v2.parquet")
    lab = lab[["DeviceId", "Detector", "func5_config", "func5", "locked"]]

    off = pd.read_parquet(DCW / "official" / "labels_official.parquet")
    off = off[off.target_type == "phase"].copy()
    off["DeviceId"] = off.DeviceId.str.lower()
    off["Detector"] = off.Detector.astype(int)
    plab = off[["DeviceId", "Detector", "target_num"]].rename(
        columns={"target_num": "Phase"})
    acc = accept_map()
    cand = candidates()

    res = {}
    for setname, wins in (("NEWTEST", ["full", "h6_a0", "h6_a3",
                                       "m30_a0", "m30_a1", "m30_a2", "m30_a3"]),):
        for w in wins:
            f = PRED / f"{setname}_v2_{w}.parquet"
            if not f.exists():
                continue
            pr = pd.read_parquet(f)
            pr["DeviceId"] = pr.DeviceId.str.lower()
            pr["Detector"] = pr.Detector.astype(int)
            key = f"{setname}/{w}"
            res[key] = {
                "function_config_labels": function_block(pr, lab, "func5_config"),
                "function_corrected_labels": function_block(pr, lab, "func5"),
            }
            # ---- phase: strict vs "switch_phase / additional_call_phases also OK"
            d = pr.merge(plab, on=["DeviceId", "Detector"], how="inner")
            d = d[d.phase_pred.notna() & (d.n_actuations >= MIN_ACT)]
            # headline rule: the labelled phase must turn green inside the window
            has_cand = np.array([int(ph) in cand.get((di, w), set())
                                 for di, ph in zip(d.DeviceId, d.Phase)])
            d = d[has_cand]
            strict = (d.phase_pred.astype(float) == d.Phase.astype(float)).to_numpy()
            lenient = np.array([
                bool(s) or (int(pp) in acc.get((di, de), set()))
                for s, pp, di, de in zip(strict, d.phase_pred, d.DeviceId, d.Detector)])
            err = d[~strict]
            rec = d[~strict & lenient]
            res[key]["phase"] = {
                "n": int(len(d)),
                "acc_strict": round(float(strict.mean()), 5),
                "acc_lenient": round(float(lenient.mean()), 5),
                "n_errors_strict": int((~strict).sum()),
                "n_errors_lenient": int((~lenient).sum()),
                "n_recovered": int(len(rec)),
                "recovered_pairs": ({f"{int(a)}->{int(b)}": int(v) for (a, b), v in
                                     rec.groupby([rec.Phase.astype(int),
                                                  rec.phase_pred.astype(int)])
                                     .size().sort_values(ascending=False)
                                     .head(12).items()} if len(rec) else {}),
                "n_detectors_with_extra_targets": int(sum(
                    len(acc.get((di, de), set())) > 1
                    for di, de in zip(d.DeviceId, d.Detector))),
            }
    # means over the anchor families
    for fam in ("m30", "h6"):
        keys = [k for k in res if f"/{fam}_" in k]
        if not keys:
            continue
        res[f"NEWTEST/MEAN_{fam}"] = {
            sec: {k: (round(float(np.mean([res[q][sec][k] for q in keys])), 5)
                      if isinstance(res[keys[0]][sec][k], (int, float)) else None)
                  for k in res[keys[0]][sec]}
            for sec in ("function_config_labels", "function_corrected_labels", "phase")}
    json.dump(res, open(WORK / "a1_rescore.json", "w"), indent=1, default=str)

    print("== function, locked 143 (final_v2 predictions, truth changed) ==")
    for k in ("NEWTEST/MEAN_m30", "NEWTEST/MEAN_h6", "NEWTEST/full"):
        if k not in res:
            continue
        a = res[k]["function_config_labels"]; b = res[k]["function_corrected_labels"]
        print(f"{k:20s} n {a['n']:4} -> {b['n']:4}   acc5 {a['acc5']:.4f} -> "
              f"{b['acc5']:.4f}   A/P/C {a['acc_apc']:.4f} -> {b['acc_apc']:.4f}")
    print("\n== phase scoring fix ==")
    for k in ("NEWTEST/MEAN_m30", "NEWTEST/MEAN_h6", "NEWTEST/full"):
        if k not in res:
            continue
        p = res[k]["phase"]
        print(f"{k:20s} n {p['n']:.0f} strict {p['acc_strict']:.5f} -> lenient "
              f"{p['acc_lenient']:.5f}  errors {p['n_errors_strict']:.1f} -> "
              f"{p['n_errors_lenient']:.1f} (recovered {p['n_recovered']:.1f})")
    print("\nfull-window recovered (true_phase, predicted):",
          res["NEWTEST/full"]["phase"]["recovered_pairs"])
    print("\nper-class, full window, corrected labels:")
    print(json.dumps(res["NEWTEST/full"]["function_corrected_labels"]["per_class"],
                     indent=1))


if __name__ == "__main__":
    main()
