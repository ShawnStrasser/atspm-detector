"""Note 53 (function part): score the TCN function head on fold 0 with the note-59 harness.

Rows = note-55 step-4 scoring rows (ATSPM-only stack-aware score, everything / realistic) restricted
to folds_v4 fold 0; decode = stack-scoped pick on D lanes (note 59 baseline set-up, pick inputs D229).
Arms: trees (note-57 229-feature arm, 3 seeds = the .8914 / .9042 reference), net alone, 0.5/0.5 blend
of the 7-class probabilities.  A detector-window the net did not score keeps the trees' probabilities
(coverage reported).  Signal-bootstrap CI of each arm minus trees, overall and by window.

    python tcn53_feval.py --cands fj,ff
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
import s59_step6 as S59  # noqa: E402
import atspm_score as S  # noqa: E402

ROOT = DC_WORK / "tcn53"
C7 = S59.C7
PC = [f"P_{c}" for c in C7]
WG = S59.WG


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(fr, rows, P):
    for i, c in enumerate(C7):
        fr[f"P_{c}"] = P[:, i]
    pred = S.decode(fr, "lanes5g", "greedy", "strict", pick=True)
    f0 = fr.fold.to_numpy() == 0
    return {s: S.credit(fr, pred, "truth_v3s", r[f0[r]], True) for s, r in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--out", default="feval_f0")
    a = ap.parse_args()
    fr, rows = S59.scoring_frame(S59.PICK_TAG)
    Pt, _ = S59.spec_probs("base", fr)
    res = {}
    base = run(fr, rows, Pt)
    res["trees"] = {s: {"atspm": S.summ(d)["atspm"], "n": int(len(d)),
                        "by_window": {g: S.summ(d[d.wgroup == g]).get("atspm") for g in WG}}
                    for s, d in base.items()}
    log(f"trees fold 0: {json.dumps({s: v['atspm'] for s, v in res['trees'].items()})}")
    for nm in a.cands.split(","):
        f = ROOT / "fpreds" / f"{nm}_f0.parquet"
        if not f.exists():
            log(f"{nm}: missing"); continue
        n = pd.read_parquet(f)
        n = n.astype({"Detector": fr.Detector.dtype})
        x = fr[S59.KEY].merge(n, on=S59.KEY, how="left")
        Pn = x[PC].to_numpy(float)
        have = ~np.isnan(Pn).any(1)
        cov = float(have[fr.fold.to_numpy() == 0].mean())
        for arm, P in (("net", np.where(have[:, None], Pn, Pt)),
                       ("blend", np.where(have[:, None], 0.5 * Pt + 0.5 * np.nan_to_num(Pn), Pt))):
            D = run(fr, rows, P)
            r = {"coverage_fold0_rows": round(cov, 4)}
            for s, d in D.items():
                b0 = base[s]
                bw = {}
                for g in WG:
                    d0, d1 = b0[b0.wgroup == g], d[d.wgroup == g]
                    bw[g] = [round(float(d1.ok_a.mean()), 4),
                             round(100 * (d1.ok_a.mean() - d0.ok_a.mean()), 2)] + S.boot(d0, d1)
                r[s] = {"atspm": S.summ(d)["atspm"],
                        "d_pt": round(100 * (d.ok_a.mean() - b0.ok_a.mean()), 3),
                        "ci": S.boot(b0, d), "by_window_acc_d_ci": bw}
            res[f"{nm}:{arm}"] = r
            log(f"{nm}:{arm} cov {cov:.3f} E {r['everything']['atspm']:.4f} d {r['everything']['d_pt']:+.2f} "
                f"{r['everything']['ci']} | R {r['realistic']['atspm']:.4f} d {r['realistic']['d_pt']:+.2f}")
            log("   by window E: " + " ".join(f"{g} {v[0]:.3f}({v[1]:+.2f})"
                                            for g, v in r["everything"]["by_window_acc_d_ci"].items()))
            json.dump(res, open(ROOT / f"{a.out}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
