"""Note 95: the 2026-only PHASE POOL (Sept-2026 staging period only; Dec-2024 dropped everywhere).

Every non-locked signal of folds_v4 with Sept-2026 pair features (761): the note-57 STG (NEWTRAIN) and REL (released
NEWTEST) signals plus the Dec-2024 pool signals' own Sept-2026 rows (src STG2). Unlike the note-57 pool, UNLABELLED
detectors are KEPT as rows (Phase NaN, never trained or scored) in every signal, so the decoder sees the full sibling
context, as in production (orchestrator 2026-10-05). Recipe = t57_phase.stage_pool / run_train.build_stg (T.load_stg
features, official timing labels on DeviceId@stg, add_scorable, folds_v4) + the note-76 order-free patch of
final76/f76_pool.py (excl_partner_diff from f76/excl_stg{,_fx}.parquet, float32 companions, partner ties from
f76/ptie_stg{,_fx}.parquet). Built in signal chunks (memory).

    python pool95.py build     -> %DC_WORK%/s95/phase/pool.parquet (+ pool_meta.json, pool_sim.parquet)
    python pool95.py check     parity: labelled STG / REL rows vs f76/phase/pool.parquet (must be identical)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pds

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402
import train_official as T  # noqa: E402
import f76_pool as F76P  # noqa: E402
import features as F  # noqa: E402

OUT = DC_WORK / "s95" / "phase"
F76PH = DC_WORK / "final_v3_work" / "f76" / "phase"
KEY4 = ["DeviceId", "Detector", "win", "cand_phase"]
DETK = ["DeviceId", "win", "Detector"]
CHUNK = 90


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def plain(s):
    return s.astype(str).str.replace("@stg", "", regex=False).str.lower()


def signals() -> pd.DataFrame:
    f4 = pd.read_csv(DC_WORK / "folds_v4.csv")
    f4["dev"] = f4.DeviceId.str.lower()
    lk = set(pd.read_csv(DC_WORK / "official/locked_v2.csv").DeviceId.str.lower())
    ids = pds.dataset(T.SFEAT / "pair_features_stg.parquet").to_table(columns=["DeviceId"]).column(0).unique().to_pylist()
    have = {d.lower(): d for d in ids}
    old = pd.read_parquet(F76PH / "pool.parquet", columns=["DeviceId", "src"]).drop_duplicates()
    old["dev"] = plain(old.DeviceId)
    srcmap = dict(zip(old.dev, old.src))
    s = f4[f4.dev.isin(have) & ~f4.dev.isin(lk)].copy()
    s["raw"] = s.dev.map(have)
    s["src95"] = s.dev.map(lambda d: {"STG": "STG", "REL": "REL", "DEC": "STG2"}.get(srcmap.get(d), "STG3"))
    assert not s.dev.isin(lk).any()
    return s.reset_index(drop=True)


def load_chunk(raw: list[str]):
    filt = pds.field("DeviceId").isin(raw)
    df = pds.dataset(T.SFEAT / "pair_features_stg.parquet").to_table(filter=filt).to_pandas()
    ex = pds.dataset(T.SFEAT / "pair_features_v2_stg.parquet").to_table(filter=filt).to_pandas()
    df = df.merge(ex, on=T.PAIR_KEY, how="left")
    del ex
    df = T.add_partner_diffs(df, T.PDIFF_FEATS + ["on_lift_green", "occ_lift_green", "f_on_green", "excl_diff_min",
                                                  "release_frac_long", "call43_fwd_lift"])
    sim = pds.dataset(T.SFEAT / "det_similarity_stg.parquet").to_table(filter=filt).to_pandas()
    df["DeviceId"] = df.DeviceId + "@stg"
    sim["DeviceId"] = sim.DeviceId + "@stg"
    return df, sim


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(F76PH / "pool_meta.json"))
    fc = meta["features"]
    s = signals()
    log(f"{len(s)} signals: {s.src95.value_counts().to_dict()}")
    off = T._official_phase().copy()
    off["DeviceId"] = off.DeviceId + "@stg"
    fold = dict(zip(s.dev, s.fold))
    src = dict(zip(s.dev, s.src95))
    parts, sims = [], []
    for i in range(0, len(s), CHUNK):
        raw = s.raw.iloc[i:i + CHUNK].tolist()
        df, sim = load_chunk(raw)
        df = T.attach_labels(df, off)
        df = T.add_scorable(df)
        df["fold"] = plain(df.DeviceId).map(fold).astype(int)
        df["fold_p37"] = df.fold
        df["src"] = plain(df.DeviceId).map(src)
        miss = [c for c in fc if c not in df.columns]
        assert not miss, miss[:5]
        keep = KEY4 + ["src", "fold", "fold_p37", "Phase", "y", "scorable", "det_n_on"] + [c for c in fc if c != "det_n_on"]
        df = df[keep].copy()
        df[fc] = df[fc].astype(np.float32)
        parts.append(df)
        sims.append(sim)
        log(f"chunk {i // CHUNK + 1}: {len(df):,} rows, labelled {int(df.Phase.notna().sum()):,}")
    pool = pd.concat(parts, ignore_index=True)
    del parts
    simc = pd.concat(sims, ignore_index=True)
    pool["scorable"] = pool.scorable.fillna(False).astype(bool)
    pool = pool.sort_values(["win", "DeviceId", "Detector", "cand_phase"]).reset_index(drop=True)
    pool, res = patch76(pool, fc)
    lk = set(pd.read_csv(DC_WORK / "official/locked_v2.csv").DeviceId.str.lower())
    assert not plain(pool.DeviceId).isin(lk).any() and not plain(simc.DeviceId).isin(lk).any()
    pool.to_parquet(OUT / "pool.parquet", index=False)
    simc.to_parquet(OUT / "pool_sim.parquet", index=False)
    meta2 = dict(meta, rows=len(pool), signals=int(pool.DeviceId.nunique()),
                 labelled_rows=int(pool.Phase.notna().sum()), note="note 95: 2026-only pool, unlabelled rows kept",
                 by_src=pool.groupby("src").DeviceId.nunique().to_dict(), patch76=res)
    json.dump(meta2, open(OUT / "pool_meta.json", "w"), indent=1, default=str)
    log(f"pool {pool.shape}; {meta2['signals']} signals; labelled rows {meta2['labelled_rows']:,}; {meta2['by_src']}")


def patch76(pool: pd.DataFrame, fc: list[str]):
    """final76/f76_pool.main, pool part, on this pool (stg excl / ptie scans only)."""
    F76 = F76P.F76
    ex = pd.concat([pd.read_parquet(f) for f in sorted(F76.glob("excl_stg*.parquet"))], ignore_index=True)
    pt = pd.concat([pd.read_parquet(f) for f in sorted(F76.glob("ptie_stg*.parquet"))], ignore_index=True)
    for d in (ex, pt):
        d["key_id"] = F76P.norm_id(d.DeviceId)
    ex = ex.drop_duplicates(["key_id", "Detector", "cand_phase", "win"])
    res = {}
    pool["key_id"] = F76P.norm_id(pool.DeviceId)
    e = pool[["key_id", "Detector", "cand_phase", "win"]].merge(
        ex.astype({"Detector": pool.Detector.dtype, "cand_phase": pool.cand_phase.dtype})[
            ["key_id", "Detector", "cand_phase", "win", "epd_new", "n_tied"]],
        on=["key_id", "Detector", "cand_phase", "win"], how="left")
    assert len(e) == len(pool)
    miss = (e.n_tied.isna() & e.epd_new.isna()).to_numpy()
    res["rows_not_rescanned"] = int(miss.sum())
    new = np.where(miss, pool.excl_partner_diff.to_numpy(float), e.epd_new.to_numpy(float))
    res["epd_changed"] = int(F76P.changed(pool.excl_partner_diff.to_numpy(), new).sum())
    pool["excl_partner_diff"] = new.astype(np.float32)
    del e
    for f in F.RANK_FEATS:
        if f not in pool.columns or f + "__rank" not in fc:
            continue
        cn = F76P.companions(pool, f)
        for c in cn.columns:
            pool[c] = cn[c].to_numpy()
    pt = pt.rename(columns={"det": "Detector", "p": "cand_phase"})
    pt = pt.astype({"Detector": pool.Detector.dtype, "cand_phase": pool.cand_phase.dtype})
    pt = pt.drop_duplicates(["key_id", "Detector", "cand_phase", "win"])
    j = pool[["key_id", "Detector", "cand_phase", "win"]].reset_index().merge(
        pt[["key_id", "Detector", "cand_phase", "win", "partner_set"] + F76P.PMETA + F76P.PEX],
        on=["key_id", "Detector", "cand_phase", "win"], how="inner")
    idx = j["index"].to_numpy()
    res["partner_tie_rows"] = int(len(idx))
    for c in F76P.PMETA + F76P.PEX:
        if c in pool.columns:
            pool.loc[idx, c] = j[c].to_numpy().astype(np.float32)
    lft = j[["index", "key_id", "Detector", "win", "partner_set"]].copy()
    lft["q"] = lft.partner_set.str.split(",")
    lft = lft.explode("q")
    lft["q"] = lft.q.astype(int).astype(pool.cand_phase.dtype)
    right = pool[["key_id", "Detector", "win", "cand_phase"] + F76P.PDIFF].rename(columns={"cand_phase": "q"})
    mm = lft.merge(right, on=["key_id", "Detector", "win", "q"], how="left").groupby("index")[F76P.PDIFF].mean()
    ii = mm.index.to_numpy()
    for f in F76P.PDIFF:
        c = f + "__pdiff"
        if c in pool.columns:
            pool.loc[ii, c] = pool.loc[ii, f].to_numpy(np.float32) - mm[f].to_numpy(np.float32)
    pool = pool.drop(columns=["key_id"])
    for c in fc:
        if pool[c].dtype != np.float32:
            pool[c] = pool[c].astype(np.float32)
    log(f"patch76: {res}")
    return pool, res


def check():
    """The labelled STG / REL rows must equal the f76 pool's rows (same features, same order-free patch)."""
    meta = json.load(open(F76PH / "pool_meta.json"))
    fc = meta["features"]
    a = pd.read_parquet(F76PH / "pool.parquet", filters=[("src", "in", ["STG", "REL"])])
    b = pd.read_parquet(OUT / "pool.parquet", filters=[("src", "in", ["STG", "REL"])])
    b = b[b.Phase.notna()]
    a = a.sort_values(KEY4).reset_index(drop=True)
    b = b.sort_values(KEY4).reset_index(drop=True)
    print("rows", len(a), len(b))
    m = a[KEY4].merge(b[KEY4].assign(_b=1), on=KEY4, how="outer", indicator=True)._merge.value_counts().to_dict()
    print("key match", m)
    if len(a) == len(b) and (a[KEY4].values == b[KEY4].values).all():
        bad = {}
        for c in fc + ["y", "scorable", "fold", "Phase"]:
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            d = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= 1e-5 * np.maximum(1, np.abs(x))))
            if d.any():
                bad[c] = int(d.sum())
        print("columns differing:", len(bad), dict(list(sorted(bad.items(), key=lambda t: -t[1]))[:25]))
        json.dump(dict(rows=len(a), differing=bad), open(OUT / "parity_vs_f76.json", "w"), indent=1)


if __name__ == "__main__":
    {"build": build, "check": check}[sys.argv[1]]()
