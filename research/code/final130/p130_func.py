"""Note 130 function side: hand (config export) function labels vs the model, both against the cabinet print (high
confidence), same non-locked detectors, Sept-2026 windows >= 30 min, >= 5 actuations. Model = v7 recipe OOF (note 114
mean3 'nou' stacker, seeds 0-2, argmax before the lane / stack decode) and the note-77 champion (decoded, rev81)."""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

W = Path(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")))
LAB = Path(__file__).resolve().parents[2] / "labels" / "function_labels_v4q.parquet"
C7 = np.array(["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"], object)
ATSPM = {"Advance", "Presence", "Count", "Yellow_Red"}
at = lambda s: s.map(lambda x: x if x in ATSPM else ("nonATSPM" if isinstance(x, str) else None))  # noqa: E731
locked = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())

fr = pd.read_parquet(W / "rev81" / "function_rows.parquet",
                     columns=["DeviceId", "Detector", "period", "win", "wgroup", "det_n_on", "pred_champ"])
P = np.mean([np.load(W / "s114" / "oof" / f"P_mean3_nou_s{s}.npy").astype(float) for s in range(3)], 0)
assert len(P) == len(fr)
fr["pred_v7"] = C7[P.argmax(1)]
assert not fr.DeviceId.str.lower().isin(locked).any()
out = {"align_v7_vs_champ_agree_all_rows": round(float((fr.pred_v7 == fr.pred_champ).mean()), 4),
       "wgroups": sorted(fr.wgroup.dropna().unique().tolist())}
GE30 = [g for g in out["wgroups"] if not (str(g).startswith("m5") or str(g).startswith("m10"))]
out["ge30_groups"] = GE30
m = (fr.period == "stg") & fr.wgroup.isin(GE30) & (fr.det_n_on >= 5)
r = fr[m].copy()
r["dev"] = r.DeviceId.str.lower()

L = pd.read_parquet(LAB)
L["dev"] = L.DeviceId.str.lower()
L = L.rename(columns={"detector": "Detector"})
L["Detector"] = L.Detector.astype(r.Detector.dtype)
cols = ["dev", "Detector", "func7_v2", "v2_source", "print_source", "print_function", "print_confidence", "dead",
        "unusual_layout", "source", "user_review", "user_review_v1", "truth_v3s", "n1_model_decided"]
r = r.merge(L[cols].drop_duplicates(["dev", "Detector"]), on=["dev", "Detector"], how="inner")


def boot(v, sig, n=1000, seed=0):
    g = pd.DataFrame({"v": v, "s": sig}).groupby("s").v.agg(["sum", "count"])
    rng = np.random.default_rng(seed); k = len(g); S, C = g["sum"].to_numpy(), g["count"].to_numpy()
    b = [S[i].sum() / C[i].sum() for i in (rng.integers(0, k, k) for _ in range(n))]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def compare(s, truth_col, name):
    s = s.copy()
    s["TR"] = at(s[truth_col]); s["H"] = at(s.func7_v2)
    res = {"rows": len(s), "detectors": int(s[["dev", "Detector"]].drop_duplicates().shape[0]), "signals": int(s.dev.nunique())}
    for mn in ("pred_v7", "pred_champ"):
        s["M"] = at(s[mn])
        hok = (s.H == s.TR).astype(float).to_numpy(); mok = (s.M == s.TR).astype(float).to_numpy()
        # detector level: majority model answer over its windows
        d = s.groupby(["dev", "Detector"]).agg(TR=("TR", "first"), H=("H", "first"),
                                                M=("M", lambda x: x.mode().iloc[0]))
        s["okm"] = mok
        d["mok_share"] = s.groupby(["dev", "Detector"]).okm.mean()
        dis = d[d.M != d.H]
        res[mn] = {"hand_acc_rows": round(float(hok.mean()), 4), "model_acc_rows": round(float(mok.mean()), 4),
                   "diff_ci": boot(mok - hok, s.dev.to_numpy()),
                   "det_hand_wrong": int((d.H != d.TR).sum()), "det_model_major_wrong": int((d.M != d.TR).sum()),
                   "det_model_mean_err": round(float(1 - d.mok_share.mean()), 4),
                   "disagree": {"n": len(dis), "model_right": int((dis.M == dis.TR).sum()),
                                "hand_right": int((dis.H == dis.TR).sum()),
                                "neither": int(((dis.M != dis.TR) & (dis.H != dis.TR)).sum())}}
    out[name] = res


base = r[r.func7_v2.notna() & ~r.dead.fillna(False).astype(bool)]
ph = base[(base.print_source == "print") & (base.print_confidence == "high") & base.print_function.isin(C7)]
compare(ph, "print_function", "vs_print_high")
compare(ph[~ph.unusual_layout.fillna(False).astype(bool)], "print_function", "vs_print_high_no_unusual")
pm = base[(base.print_source == "print") & base.print_confidence.isin(["high", "medium"]) & base.print_function.isin(C7)]
compare(pm, "print_function", "vs_print_high_medium")
# the user's own rulings as truth (source user_ruling / user_review answered)
u = base[(base.source == "user_ruling") & base.truth_v3s.isin(C7)]
compare(u, "truth_v3s", "vs_user_ruling")
out["v2_source_counts_in_print_high"] = ph.drop_duplicates(["dev", "Detector"]).v2_source.value_counts(dropna=False).to_dict()
print(json.dumps(out, indent=1, default=str))
(W / "s130").mkdir(exist_ok=True)
json.dump(out, open(W / "s130" / (Path(__file__).stem + ".json"), "w"), indent=1, default=str)
