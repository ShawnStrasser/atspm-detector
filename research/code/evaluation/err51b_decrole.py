"""Note 51b: re-score function_v3e's six-fold OOF (err51 rows, 3-seed mean) with the Dec-2024 role-drift filter
(`cab_final.py --dec-role` -> cabinet/dec_role_changed.parquet; `v3_retrain.dec_role_rows`). Re-score only, no refit.

    python err51b_decrole.py > %DC_WORK%/trackA/err51/decrole.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trackA"))
import v3_retrain as V  # noqa: E402

D = V.DCW / "trackA" / "err51"
ORDER = ["m5", "m10", "m30", "h1", "h3", "h6", "h24", "full"]


def boot(ok, keep, sig, n=1000, seed=0):
    """signal-cluster bootstrap CI of acc(kept rows) - acc(all rows)."""
    s = pd.DataFrame({"sig": sig, "ok": ok, "k": keep, "ok_k": ok & keep})
    g = s.groupby("sig").agg(n=("ok", "size"), ok=("ok", "sum"), nk=("k", "sum"), okk=("ok_k", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    a = g.to_numpy(float)
    t = a[idx].sum(1)
    d = t[:, 3] / t[:, 2] - t[:, 1] / t[:, 0]
    return np.percentile(d, [2.5, 97.5]) * 100


def main():
    r = pd.read_parquet(D / "rows.parquet")
    lk = set(pd.read_csv(V.DCW / "official" / "locked_v2.csv").DeviceId.str.lower())
    assert not r.DeviceId.isin(lk).any()
    r["drop"] = V.dec_role_rows(r, "truth")
    dr = pd.read_parquet(V.DEC_ROLE_FILE)
    dr["DeviceId"] = dr.DeviceId.str.lower()
    r = r.merge(dr[["DeviceId", "detector", "dec_status_truth"]].rename(columns={"detector": "Detector"})
                .astype({"Detector": r.Detector.dtype}), on=["DeviceId", "Detector"], how="left")
    dec = r.period.eq("dec")
    print("Dec rows by dec_status_truth (everything):", r[dec].dec_status_truth.fillna("no_row").value_counts().to_dict())
    for name, m in (("everything", r.setA), ("realistic", r.setR)):
        x = r[m]
        keep = ~x["drop"]
        ch = x[x["drop"]].drop_duplicates(["DeviceId", "Detector"])
        a0, a1 = x.ok.mean(), x.ok[keep].mean()
        lo, hi = boot(x.ok.to_numpy(), keep.to_numpy(), x.DeviceId.to_numpy())
        print(f"\n{name}: {len(x):,} rows / {x.DeviceId.nunique()} sig. acc7 {a0:.4f} -> {a1:.4f} "
              f"({(a1 - a0) * 100:+.2f} pt [{lo:+.2f}, {hi:+.2f}]); dropped {int(x['drop'].sum()):,} Dec rows "
              f"= {x['drop'].mean():.3f} of rows, {len(ch)} channels / {ch.DeviceId.nunique()} sig.; "
              f"errors {int((~x.ok).sum()):,} -> {int((~x.ok[keep]).sum()):,}; acc of dropped rows {x.ok[x['drop']].mean():.3f}")
        print("  dropped by status (rows, acc):", {k: (len(g), round(g.ok.mean(), 3)) for k, g in
                                                   x[x['drop']].groupby("dec_status_truth")})
        xd = x[x.period.eq("dec")]
        print(f"  Dec period acc {xd.ok.mean():.4f} -> {xd.ok[~xd['drop']].mean():.4f}; Sept {x.ok[x.period.eq('stg')].mean():.4f}")
        print("  by window:", {g: f"{x.ok[x.wgroup.eq(g)].mean():.4f}->{x.ok[x.wgroup.eq(g) & keep].mean():.4f}"
                               for g in ORDER})
        # note 51's drift set (Dec acc < .2, Sept acc > .8; model right/wrong pattern - used ONLY to measure overlap)
        pa = x.groupby(["DeviceId", "Detector", "period"]).ok.mean().unstack()
        drift = pa[(pa.get("dec") < .2) & (pa.get("stg") > .8)].index
        fl = set(zip(ch.DeviceId, ch.Detector))
        caught = sum(k in fl for k in drift)
        st = x.drop_duplicates(["DeviceId", "Detector"]).set_index(["DeviceId", "Detector"]).dec_status_truth
        print(f"  note-51 drift channels {len(drift)}; caught {caught} ({caught / max(len(drift), 1):.2f}); "
              f"status of drift channels {st.reindex(drift).fillna('no_row').value_counts().to_dict()}")
        dd = x[x.period.eq('dec') & x.set_index(['DeviceId', 'Detector']).index.isin(drift)]
        print(f"  drift-channel Dec rows {len(dd):,}, of them dropped {int(dd['drop'].sum()):,}")
    # training rows the filter removes (function_v3e recipe: first.all.wi, min5, clean, valnc, h3; frame v6e)
    V.MIN_ON, V.CLEAN, V.REQ_VAL, V.NC_HIGH, V.HEALTH3 = 5, True, True, True, True
    V.set_frame("v6e")
    fr = pd.read_parquet(V.FEATS, columns=V.KEY + ["det_n_on", "health_flag"])
    fr["DeviceId"] = fr.DeviceId.str.lower()
    V.DEC_ROLE = True
    lab = V.load_labels(fr, "exclude")
    V.DEC_ROLE = False
    _, ok0 = V.variant_target(lab, fr, "first.all.wi")
    trn = ok0 & lab.decchg_train.to_numpy()
    print(f"\ntraining (v6e first.all.wi h3): {int(ok0.sum()):,} rows -> {int((ok0 & ~trn).sum()):,}; dropped "
          f"{int(trn.sum()):,} Dec rows / {fr[trn].drop_duplicates(['DeviceId', 'Detector']).shape[0]} channels / "
          f"{fr[trn].DeviceId.nunique()} sig.")


if __name__ == "__main__":
    main()
