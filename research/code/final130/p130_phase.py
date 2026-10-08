"""Note 130 phase side: hand phase labels vs the model, both against controller timing, same detectors.
Non-locked only (OOF). (1) v7 phase recipe OOF (s113 q113 p2_v6_lagt seeds 0-2) on the 2026 pool, scored rows of note 113.
(2) 2025-style model trained on HAND labels (note 10 variant A OOF, Dec-2024 full window)."""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

W = Path(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")))
KEY4 = ["DeviceId", "Detector", "win", "cand_phase"]
DET = ["DeviceId", "Detector", "win"]
locked = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
old_locked = set(pd.read_csv(W / "official" / "newtest_signals.csv").DeviceId.str.lower()) if (W / "official" / "newtest_signals.csv").exists() else set()
plain = lambda s: s.astype(str).str.replace("@stg", "", regex=False).str.lower()  # noqa: E731

hand = pd.read_parquet(W / "labels_dev.parquet")
hand["dev_plain"] = hand.DeviceId.str.lower()
hand = hand.rename(columns={"Phase": "hand"})[["dev_plain", "Detector", "hand", "Function"]]
assert not hand.dev_plain.isin(locked).any()
hand = hand.drop_duplicates(["dev_plain", "Detector"])
out = {"hand_rows": len(hand), "hand_signals": hand.dev_plain.nunique()}

# ---------- (1) v7 recipe OOF, 2026 pool
comb = pd.read_parquet(W / "s95" / "phase" / "pool.parquet", columns=KEY4)
q = pd.read_parquet(W / "s113" / "phase" / "q113.parquet", columns=KEY4 + [f"p2_v6_lagt_s{s}" for s in range(3)])
for c in KEY4:
    assert (q[c].to_numpy() == comb[c].to_numpy()).all()
rows = pd.read_parquet(W / "s95" / "phase" / "score_rows.parquet")
rows["alt"] = rows.alt_s.map(lambda s: {int(x) for x in s.split(",") if x})
assert not rows.dev_plain.isin(locked).any()
preds = {}
for s in range(3):
    col = f"p2_v6_lagt_s{s}"
    d = q[KEY4 + [col]].dropna().sort_values(DET + [col, "cand_phase"], ascending=[True, True, True, False, True])
    t = d.groupby(DET, sort=False).first().reset_index()
    preds[s] = rows[DET].merge(t[DET + ["cand_phase"]], on=DET, how="left").cand_phase.to_numpy()
ge30 = rows.fam.isin(["m30", "h1", "h3", "h6", "h24", "full"]).to_numpy()
okE = np.mean([(preds[s] == rows.Phase.to_numpy()) for s in range(3)], 0)
okR = np.mean([np.maximum(preds[s] == rows.Phase.to_numpy(), [p in a for p, a in zip(preds[s], rows.alt)]) for s in range(3)], 0)
rows["okE_m"], rows["okR_m"] = okE, okR
rows["pred0"] = preds[0]
m_all = rows.everything.to_numpy() & ge30
out["v7_oof_all_ge30"] = {"n": int(m_all.sum()), "E": round(float(okE[m_all].mean()), 4), "R": round(float(okR[m_all].mean()), 4)}

r = rows[m_all].merge(hand, on=["dev_plain", "Detector"], how="inner")
r["handE"] = (r.hand == r.Phase).astype(float)
r["handR"] = np.maximum(r.handE, [h in a for h, a in zip(r.hand, r.alt)])
sig = r.dev_plain
def boot(v, sig, n=1000, seed=0):
    g = pd.DataFrame({"v": v, "s": sig.values}).groupby("s").v.agg(["sum", "count"])
    rng = np.random.default_rng(seed); k = len(g); S, C = g["sum"].to_numpy(), g["count"].to_numpy()
    b = [S[i].sum() / C[i].sum() for i in (rng.integers(0, k, k) for _ in range(n))]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
out["v7_vs_hand_same_rows_ge30"] = {
    "n_det_windows": len(r), "n_detectors": int(r[["dev_plain", "Detector"]].drop_duplicates().shape[0]),
    "signals": int(r.dev_plain.nunique()),
    "hand_E": round(float(r.handE.mean()), 4), "hand_R": round(float(r.handR.mean()), 4),
    "model_E": round(float(r.okE_m.mean()), 4), "model_R": round(float(r.okR_m.mean()), 4),
    "diff_E_ci": boot(r.okE_m.to_numpy() - r.handE.to_numpy(), sig),
    "diff_R_ci": boot(r.okR_m.to_numpy() - r.handR.to_numpy(), sig)}
# by window family
out["v7_vs_hand_by_fam"] = {f: [int(len(g)), round(float(g.handE.mean()), 4), round(float(g.okE_m.mean()), 4)]
                            for f, g in r.groupby("fam")}
# detector level: model majority over its >= 30-min windows (seed 0 + seeds mean of ok), hand per detector
dd = r.groupby(["dev_plain", "Detector"]).agg(Phase=("Phase", "first"), hand=("hand", "first"), handE=("handE", "first"),
                                               handR=("handR", "first"), mE=("okE_m", "mean"), mR=("okR_m", "mean"),
                                               pmaj=("pred0", lambda x: x.mode().iloc[0]), alt=("alt", "first"))
dd["model_major_ok"] = (dd.pmaj == dd.Phase) | np.array([p in a for p, a in zip(dd.pmaj, dd.alt)])
out["detector_level"] = {"n": len(dd), "hand_wrong_E": int((dd.handE == 0).sum()), "hand_wrong_R": int((dd.handR == 0).sum()),
                         "model_mean_err_E": round(float(1 - dd.mE.mean()), 4), "model_mean_err_R": round(float(1 - dd.mR.mean()), 4),
                         "model_major_wrong_R": int((~dd.model_major_ok).sum())}
# disagreements: model majority (seed 0) != hand -> who matches timing (R)
dis = dd[dd.pmaj != dd.hand]
handok = dis.handR == 1
modok = dis.model_major_ok
out["disagree_detector_level"] = {"n": len(dis), "model_right_only": int((modok & ~handok).sum()),
                                  "hand_right_only": int((~modok & handok).sum()), "both_right_(alt)": int((modok & handok).sum()),
                                  "neither": int((~modok & ~handok).sum())}

# ---------- (2) model trained on HAND labels (note 10 variant A), Dec-2024 full window, scored vs official
off = pd.read_parquet(W / "official" / "labels_official.parquet")
off["dev_plain"] = off.DeviceId.str.lower()
off = off[(off.target_type == "phase")]
off["alt"] = [({int(s)} if s > 0 else set()) | ({int(x) for x in a} if a is not None else set())
              for s, a in zip(off.switch_phase, off.additional_call_phases)]
A = pd.read_parquet(W / "preds" / "phase_oof_official_A.parquet")
A["dev_plain"] = A.DeviceId.str.lower()
assert not A.dev_plain.isin(locked).any()
cands = A.groupby(["dev_plain", "Detector"]).cand_phase.apply(set).rename("cands")
topA = A.sort_values(["dev_plain", "Detector", "prob"], ascending=[True, True, False]).groupby(["dev_plain", "Detector"]).first()
topA = topA[["cand_phase"]].rename(columns={"cand_phase": "predA"}).join(cands)
a = topA.reset_index().merge(off[["dev_plain", "Detector", "call_phase", "alt", "n_on_dec2024"]], on=["dev_plain", "Detector"]) \
        .merge(hand, on=["dev_plain", "Detector"])
a = a[(a.n_on_dec2024 >= 5) & [c in s for c, s in zip(a.call_phase, a.cands)]]
a["hE"] = a.hand == a.call_phase
a["mE"] = a.predA == a.call_phase
a["hR"] = a.hE | [h in s for h, s in zip(a.hand, a.alt)]
a["mR"] = a.mE | [h in s for h, s in zip(a.predA, a.alt)]
out["varA_trained_on_hand_full_dec2024"] = {
    "n": len(a), "signals": int(a.dev_plain.nunique()),
    "hand_E": round(float(a.hE.mean()), 4), "model_E": round(float(a.mE.mean()), 4),
    "hand_R": round(float(a.hR.mean()), 4), "model_R": round(float(a.mR.mean()), 4),
    "hand_wrong_R": int((~a.hR).sum()), "model_wrong_R": int((~a.mR).sum()),
    "diff_R_ci": boot(a.mR.astype(float).to_numpy() - a.hR.astype(float).to_numpy(), a.dev_plain)}
d2 = a[a.predA != a.hand]
out["varA_disagree"] = {"n": len(d2), "model_right_only": int((d2.mR & ~d2.hR).sum()), "hand_right_only": int((~d2.mR & d2.hR).sum()),
                        "both": int((d2.mR & d2.hR).sum()), "neither": int((~d2.mR & ~d2.hR).sum())}
print(json.dumps(out, indent=1, default=str))
(W / "s130").mkdir(exist_ok=True)
json.dump(out, open(W / "s130" / (Path(__file__).stem + ".json"), "w"), indent=1, default=str)
