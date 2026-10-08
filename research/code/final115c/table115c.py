"""Note 115c: table from bench115c runs (2 passes): warm s / peak MB per package/profile/length (mean over the 4 signals
and both passes; pass spread shown), import + model-load (cold - warm), machine load; ratios v7 vs v6b and vs beta.
    python table115c.py   -> %DC_WORK%/s115c/table115c.json + print"""
import json, os
from pathlib import Path
import numpy as np
W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
B = W / "s115c" / "bench"
CFG = ["beta", "v6bfull", "v6ble2h", "v7full", "v7le2h", "v7le2h1t"]
rows, out = {}, {}
for c in CFG:
    for p in (1, 2):
        for f in (B / f"{c}_p{p}").glob("r_*.json"):
            r = json.load(open(f)); rows.setdefault((c, r["L"]), []).append((p, r))
for (c, L), rs in sorted(rows.items()):
    g = lambda k: [r[k] for _, r in rs]
    pw = [np.mean([r["warm_s"] for p, r in rs if p == q]) for q in (1, 2)]
    out[f"{c}|{L}"] = dict(n=len(rs), warm_s=float(np.mean(g("warm_s"))), warm_pass=[round(x, 3) for x in pw],
                         warm_by_sig={r["sig"]: round(r["warm_s"], 3) for p, r in rs if p == 1},
                         peak_mb=float(np.mean(g("peak_all_mb"))), peak_max_mb=float(np.max(g("peak_all_mb"))),
                         import_s=float(np.median(g("import_s"))),
                         load_s=float(np.median([r["cold_s"] - r["warm_s"] for _, r in rs])),
                         cpu_s=float(np.mean(g("warm_cpu_s"))),
                         machine_cpu=float(np.mean(g("machine_cpu_pct_mean"))),
                         others_max=float(max(sum((r.get("others_cpu_pct_mean") or {}).values()) for _, r in rs)))
for k, v in out.items():
    print(f"{k:18s} warm {v['warm_s']:.3f}s (p1/p2 {v['warm_pass']}) peak {v['peak_mb']:.0f} (max {v['peak_max_mb']:.0f}) MB "
          f"import {v['import_s']:.2f} load {v['load_s']:.2f} cpu-s {v['cpu_s']:.2f} mach {v['machine_cpu']:.0f}% oth {v['others_max']:.0f}%")
rat = {}
for L in ("m30", "h3", "h24"):
    for a, b in [("v7full", "v6bfull"), ("v7le2h", "v6ble2h"), ("v7full", "beta"), ("v7le2h", "beta")]:
        if f"{a}|{L}" in out and f"{b}|{L}" in out:
            A, Bb = out[f"{a}|{L}"], out[f"{b}|{L}"]
            # paired per signal ratio, geometric mean
            t = np.exp(np.mean([np.log(A["warm_by_sig"][s] / Bb["warm_by_sig"][s]) for s in A["warm_by_sig"]]))
            rat[f"{a}/{b}|{L}"] = dict(time=round(float(t), 3), peak=round(A["peak_mb"] / Bb["peak_mb"], 3))
            print(f"{a}/{b} {L}: time x{t:.2f}  peak x{A['peak_mb']/Bb['peak_mb']:.2f}")
json.dump(dict(cases=out, ratios=rat), open(W / "s115c" / "table115c.json", "w"), indent=1)
