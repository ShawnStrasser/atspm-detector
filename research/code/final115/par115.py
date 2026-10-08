"""Note 115 parity runner: one package x one profile over every parity case, in ONE process (warm after the first case).

    python par115.py run <name> <flat-package-dir | src:<dir with detector_classifier>> <profile> [--only s00,s01]
-> s115/parity/out/<name>_<profile>_{det,cand,ph}.parquet + _time.json.  locked_v2 asserted absent."""
from __future__ import annotations
import json, os, sys, time
import os
from pathlib import Path
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
import pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "4")
PD = W / "s115" / "parity"
B71 = W / "bench71"
CASES = []
sig = pd.read_csv(PD / "signals115.csv")
for c in sig.case:
    f = str(PD / "ev" / f"{c}.parquet")
    CASES += [(f"{c}_m30", f, "2026-09-19 07:30:00", "2026-09-19 08:00:00"),
              (f"{c}_h3", f, "2026-09-19 15:00:00", "2026-09-19 18:00:00"),
              (f"{c}_h24", f, None, None)]
for b in ("typical_r8", "typical_r11", "busiest_ev", "busiest_ch"):
    for L in ("m30", "h3", "h24"):
        CASES.append((f"b_{b}_{L}", str(B71 / f"ev_{b}_{L}.parquet"), None, None))


def main():
    _, cmd, name, pkg, profile, *rest = sys.argv
    only = None
    if rest and rest[0] == "--only":
        only = set(rest[1].split(","))
    if pkg.startswith("src:"):
        sys.path.insert(0, pkg[4:])
        from detector_classifier import pipeline as P
    else:
        sys.path.insert(0, pkg)
        import predict as P
    lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not set(sig.DeviceId.str.lower()) & lk
    cap = {}
    orig = P.score

    def score(*a, **k):
        r = orig(*a, **k)
        cap["cand"] = r[["DeviceId", "Detector", "cand_phase", "p0", "prob"]].copy()
        return r
    P.score = score
    dets, cands, phs, tim = [], [], [], {}
    for case, f, s, e in CASES:
        if only and not any(case.startswith(o) for o in only):
            continue
        cap.clear()
        t0 = time.perf_counter()
        out, ph = P.predict(f, start=s, end=e, min_actuations=1, return_phases=True, profile=profile)
        tim[case] = time.perf_counter() - t0
        dets.append(out.assign(case=case))
        phs.append(ph.assign(case=case))
        if "cand" in cap:
            cands.append(cap["cand"].assign(case=case))
        print(f"{case} {tim[case]:.2f}s {len(out)} det", flush=True)
    od = PD / "out"
    od.mkdir(exist_ok=True)
    tag = f"{name}_{profile}"
    pd.concat(dets, ignore_index=True).to_parquet(od / f"{tag}_det.parquet", index=False)
    pd.concat(cands, ignore_index=True).to_parquet(od / f"{tag}_cand.parquet", index=False)
    pd.concat(phs, ignore_index=True).to_parquet(od / f"{tag}_ph.parquet", index=False)
    json.dump(tim, open(od / f"{tag}_time.json", "w"), indent=1)


if __name__ == "__main__":
    main()
