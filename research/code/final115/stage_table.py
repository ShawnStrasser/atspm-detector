import json, re, sys
import numpy as np
import os
from pathlib import Path
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
tag = sys.argv[1]
T = Path(str(W / r"s115\v7\t"))
R = {}
for f in T.glob(f"*_{tag}.json"):
    m = re.match(rf"(ref|v7)_(.+)_(m30|h3|h24)_(full|le2h)_{tag}\.json", f.name)
    r = json.loads(f.read_text())
    warm = r["calls"][1:]
    R[(m.group(1), m.group(2), m.group(3), m.group(4))] = ({k: float(np.median([c.get(k, 0.0) for c in warm])) for k in warm[0]},
                                                        r["calls"][0]["total"], r["peak_ws_mb"])
stages = ["total", "load", "tables", "streams", "features", "f.lead", "network", "phase_score", "function", "fn.expert",
          "fn.lanes", "post", "post.setback", "post.health"]
out = {}
for L in ("m30", "h3", "h24"):
    for pr in ("full", "le2h"):
        sigs = sorted({k[1] for k in R if k[2] == L and k[3] == pr})
        row = {}
        for s in stages:
            a = [R[("ref", g, L, pr)][0].get(s, 0) for g in sigs]
            b = [R[("v7", g, L, pr)][0].get(s, 0) for g in sigs]
            row[s] = [round(float(np.mean(a)), 3), round(float(np.mean(b)), 3)]
        row["total_ratio_range"] = [round(min(R[("v7", g, L, pr)][0]["total"] / R[("ref", g, L, pr)][0]["total"] for g in sigs), 2),
                                    round(max(R[("v7", g, L, pr)][0]["total"] / R[("ref", g, L, pr)][0]["total"] for g in sigs), 2)]
        row["cold_mean"] = [round(float(np.mean([R[("ref", g, L, pr)][1] for g in sigs])), 2),
                            round(float(np.mean([R[("v7", g, L, pr)][1] for g in sigs])), 2)]
        row["peak_mb_range"] = [[round(min(R[("ref", g, L, pr)][2] for g in sigs)), round(max(R[("ref", g, L, pr)][2] for g in sigs))],
                                [round(min(R[("v7", g, L, pr)][2] for g in sigs)), round(max(R[("v7", g, L, pr)][2] for g in sigs))]]
        out[f"{L}_{pr}"] = row
        print(f"{L} {pr}: " + "  ".join(f"{s} {row[s][0]:.3f}->{row[s][1]:.3f}" for s in stages if max(row[s]) > 0),
              "| ratio", row["total_ratio_range"], "cold", row["cold_mean"], "peak", row["peak_mb_range"])
Path(str(W / rf"s115\v7\stages115_{tag}.json")).write_text(json.dumps(out, indent=1))
