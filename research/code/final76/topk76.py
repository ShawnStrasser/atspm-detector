"""Note 76: how often does the top-K cut of the lag table (12) / similarity graph (8) fall inside a tie (so the old
row_number / sort order decided by channel number)?  Package tables built from the bench extracts, old vs new rule.

    python topk76.py   (2 threads)
"""
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd

W = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
NEW, OLD = W / "final_v3_candidate_v3", W / "final_v3_candidate_v3_pre76"
sys.path.insert(0, str(NEW))
import predict as P  # noqa: E402
import features as F1  # noqa: E402
import features_yellowred as F3  # noqa: E402
import similarity as SIM  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


F3o = load(OLD / "features_yellowred.py", "f3_old")
SIMo = load(OLD / "similarity.py", "sim_old")
rows = []
for sig in ("typical_r8", "typical_r11", "typical_r5", "typical_r15", "busiest_ch", "busiest_ev"):
    for L in ("m30", "h3", "h24"):
        con = P._connect(2, "4GB")
        w0, w1, _ = P.load_events(con, str(W / "bench71" / f"ev_{sig}_{L}.parquet"))
        P.build_chunk_tables(con)
        F1.apply_window(con, w0, w1)
        dm = con.sql("SELECT * FROM devmap").df()
        a = F3o.build(con, "w", dm, F3o.SQL_LAG)
        b = F3.build(con, "w", dm, F3.SQL_LAG)
        sa, sb = SIMo.build_window(con, "w", w1 - w0), SIM.build_window(con, "w", w1 - w0)
        k = ["DeviceId", "det"]
        ja = set(zip(a.det, a.oth))
        jb = set(zip(b.det, b.oth))
        dets = a.det.nunique()
        lag_dets_extra = b.groupby("det").size().gt(a.groupby("det").size().reindex(b.det.unique()).fillna(0)).sum()
        ka = set(zip(sa.Detector, sa.other))
        kb = set(zip(sb.Detector, sb.other))
        sim_dets_extra = sb.groupby("Detector").size().gt(
            sa.groupby("Detector").size().reindex(sb.Detector.unique()).fillna(0)).sum()
        rows.append(dict(sig=sig, L=L, lag_dets=dets, lag_dets_with_tie_at_cut=int(lag_dets_extra),
                         lag_rows_old=len(a), lag_rows_new=len(b), lag_old_not_in_new=len(ja - jb),
                         sim_dets=int(sa.Detector.nunique()), sim_dets_with_tie_at_cut=int(sim_dets_extra),
                         sim_rows_old=len(sa), sim_rows_new=len(sb), sim_old_not_in_new=len(ka - kb)))
        print(rows[-1], flush=True)
        con.close()
d = pd.DataFrame(rows)
print(d.sum(numeric_only=True))
d.to_csv(W / "final_v3_work" / "f76" / "topk76.csv", index=False)
