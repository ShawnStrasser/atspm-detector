"""Note 120 parity runner (note-115 parity set, 132 cases) + probe of which models the run loads.

    python par120.py <name> <dir containing detector_classifier> [profile|-]
-> s120/out/<name>_{det,cand,ph}.parquet, _time.json, _probe.json (per case: decoder files, stacker variants,
network used).  profile '-' = do not pass the argument (the post-note-120 API)."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import pandas as pd
PD = W / "s115" / "parity"
B71 = W / "bench71"
OUT = W / "s120" / "out"
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
    profile = sys.argv[3] if len(sys.argv) > 3 else "-"
    only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
    sys.path.insert(0, pkg)
    from detector_classifier import pipeline as P, trees_onnx, stacker, function_stage as fs
    lk = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not set(sig.DeviceId.str.lower()) & lk
    probe = {"onnx": set(), "variant": set(), "net": 0}
    o_load, o_bag, o_net = trees_onnx.load, stacker.Stacker.bag, fs.net_model

    def load(path, *a, **k):
        probe["onnx"].add(Path(path).name)
        return o_load(path, *a, **k)

    def bag(self, variant=None):
        probe["variant"].add(variant or getattr(self, "variant", "mean3"))
        return o_bag(self, variant) if variant is not None else o_bag(self)

    def net_model(*a, **k):
        probe["net"] += 1
        return o_net(*a, **k)
    trees_onnx.load, stacker.Stacker.bag, fs.net_model = load, bag, net_model
    P.trees_onnx.load = load
    P.fs.net_model = net_model
    cap = {}
    orig = P.score

    def score(*a, **k):
        r = orig(*a, **k)
        cap["cand"] = r[["DeviceId", "Detector", "cand_phase", "p0", "prob"]].copy()
        return r
    P.score = score
    dets, cands, phs, tim, pr = [], [], [], {}, {}
    kw = {} if profile == "-" else {"profile": profile}
    for case, f, s, e in CASES:
        if only and not any(case.startswith(o) for o in only):
            continue
        cap.clear()
        probe["onnx"], probe["variant"], probe["net"] = set(), set(), 0
        t0 = time.perf_counter()
        out, ph = P.predict(f, start=s, end=e, min_actuations=1, return_phases=True, **kw)
        tim[case] = time.perf_counter() - t0
        pr[case] = {"onnx": sorted(x for x in probe["onnx"] if "decode" in x), "variant": sorted(probe["variant"]),
                    "net": probe["net"]}
        dets.append(out.assign(case=case))
        phs.append(ph.assign(case=case))
        if "cand" in cap:
            cands.append(cap["cand"].assign(case=case))
        print(f"{case} {tim[case]:.2f}s {len(out)} det {pr[case]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(dets, ignore_index=True).to_parquet(OUT / f"{name}_det.parquet", index=False)
    pd.concat(cands, ignore_index=True).to_parquet(OUT / f"{name}_cand.parquet", index=False)
    pd.concat(phs, ignore_index=True).to_parquet(OUT / f"{name}_ph.parquet", index=False)
    json.dump(tim, open(OUT / f"{name}_time.json", "w"), indent=1)
    json.dump(pr, open(OUT / f"{name}_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
