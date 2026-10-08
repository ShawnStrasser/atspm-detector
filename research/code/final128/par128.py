"""Note 128 parity runner (copy of final124/par124.py, outputs s128/par): the note-115 parity set (40 signals x 30 min / 3 h / 24 h + bench71 4 x 3 = 132 cases),
one package per run.  python par124.py <name> <dir containing detector_classifier>
-> s124/par/<name>_{det,cand,ph}.parquet, <name>_time.json"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import pandas as pd
PD = W / "s115" / "parity"
B71 = W / "bench71"
OUT = W / "s128" / "par"
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
    name, pkg = sys.argv[1], sys.argv[2]
    sys.path.insert(0, pkg)
    import warnings
    warnings.simplefilter("ignore")
    from detector_classifier import pipeline as P
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
        cap.clear()
        t0 = time.perf_counter()
        out, ph = P.predict(f, start=s, end=e, min_actuations=1, return_phases=True)
        tim[case] = time.perf_counter() - t0
        dets.append(out.assign(case=case))
        phs.append(ph.assign(case=case))
        if "cand" in cap:
            cands.append(cap["cand"].assign(case=case))
        print(f"{case} {tim[case]:.2f}s {len(out)} det", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(dets, ignore_index=True).to_parquet(OUT / f"{name}_det.parquet", index=False)
    pd.concat(cands, ignore_index=True).to_parquet(OUT / f"{name}_cand.parquet", index=False)
    pd.concat(phs, ignore_index=True).to_parquet(OUT / f"{name}_ph.parquet", index=False)
    (OUT / f"{name}_time.json").write_text(json.dumps(tim))


if __name__ == "__main__":
    main()
