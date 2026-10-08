"""Note 80: build the separate training-store copy %DC_WORK%/cabinet_v4l/ for label pass v4l (resumable, idempotent).

    python v4l_store.py setup            copy the training store (records, batches, tables, stacked/) once, then bring in
                                         ONLY the 71 released NEWTEST signals' records from the locked store
    python v4l_store.py stage <DN> ...   copy render / crops of signals to be re-read into the new store (paths rewritten)
    python v4l_store.py check            assert no locked_v2 signal anywhere in the store

The old stores (cabinet/, cabinet_locked/) are only READ. The share is never touched (PDFs were copied earlier).
A released signal's record, extract, PDF copy, renders, crops and activity rows are copied; nothing of the 72 still-locked
NEWTEST or 43 TEST signals is read beyond the DeviceId lists used to exclude them (asserted).
Released records go to batches/batch_36.txt so cab_final treats them as an ordinary finished batch.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

DC_WORK = Path(__import__("os").environ.get("DC_WORK", Path.home() / "dc_work"))
OLD, LK, NEW = DC_WORK / "cabinet", DC_WORK / "cabinet_locked", DC_WORK / "cabinet_v4l"
TOP = ["activity.parquet", "card_channels.parquet", "card_health.parquet", "dq_print.parquet", "dq_print_pairs.parquet",
       "final_rulings.csv", "label_check_extra.parquet", "label_check_health.parquet", "label_check_pairs.parquet",
       "label_check_pplt.parquet", "label_check_stats.parquet", "label_check_thresholds.json", "print_labels.parquet",
       "print_tiers.csv", "scripted_status.csv", "share_listing.csv", "share_path.txt", "sweep_changes.csv",
       "xlsm_zones.parquet", "dec_role_changed.parquet", "dec_role_changed_v3s.parquet", "vd_audit.csv"]
RELEASED_BATCH = 36


def holdout() -> tuple[set[str], set[str]]:
    """(locked_v2 DeviceNames, released DeviceNames); disjoint."""
    lk = pd.read_csv(DC_WORK / "official/locked_v2.csv", dtype=str)
    rel = pd.read_csv(DC_WORK / "official/newtest_released.csv", dtype=str)
    a, b = set(lk.DeviceName), set(rel.DeviceName)
    assert not a & b and len(a) == 115 and len(b) == 71
    return a, b


def _rewrite(obj, old: str, new: str):
    if isinstance(obj, str):
        return obj.replace(old, new)
    if isinstance(obj, list):
        return [_rewrite(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite(v, old, new) for k, v in obj.items()}
    return obj


def setup():
    locked, rel = holdout()
    NEW.mkdir(exist_ok=True)
    for d in ("signals", "batches", "stacked", "render", "crops", "pdf", "lists", "logs"):
        (NEW / d).mkdir(exist_ok=True)
    marker = NEW / "setup.done"
    if not marker.exists():
        for f in TOP:
            if (OLD / f).exists():
                shutil.copy2(OLD / f, NEW / f)
        for p in (OLD / "signals").glob("*.json"):
            shutil.copy2(p, NEW / "signals" / p.name)
        for p in (OLD / "batches").glob("*"):
            if p.is_file():
                shutil.copy2(p, NEW / "batches" / p.name)
        for p in (OLD / "stacked").glob("*.parquet"):
            shutil.copy2(p, NEW / "stacked" / p.name)
        # released NEWTEST: records + extracts + pdf + renders + crops + activity (locked store read for these 71 only)
        done = []
        for dn in sorted(rel):
            src = LK / "signals" / f"{dn}.json"
            if not src.exists():
                print(f"released {dn}: no record in the locked store (no print)")
                continue
            for name in (f"{dn}.json", f"{dn}.extract.json"):
                s = LK / "signals" / name
                if s.exists():
                    r = json.loads(s.read_text(encoding="utf-8"))
                    r = _rewrite(r, str(LK), str(NEW))
                    (NEW / "signals" / name).write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
            if (LK / "render" / dn).exists():
                shutil.copytree(LK / "render" / dn, NEW / "render" / dn, dirs_exist_ok=True)
            for p in (LK / "crops").glob(f"{dn}_*"):
                shutil.copy2(p, NEW / "crops" / p.name)
            ex = json.loads((LK / "signals" / f"{dn}.extract.json").read_text(encoding="utf-8")) \
                if (LK / "signals" / f"{dn}.extract.json").exists() else {}
            for k in ("pdf", "xlsm"):
                p = Path(str(ex.get(k) or ""))
                if p.name and (LK / k / p.name).exists():
                    (NEW / k).mkdir(exist_ok=True)
                    shutil.copy2(LK / k / p.name, NEW / k / p.name)
            st = json.loads(src.read_text(encoding="utf-8")).get("status")
            if st == "visual_done":
                done.append(dn)
            else:
                print(f"released {dn}: status {st} (not in the batch)")
        (NEW / "batches" / f"batch_{RELEASED_BATCH}.txt").write_text("\n".join(done) + "\n", encoding="utf-8")
        a = pd.read_parquet(NEW / "activity.parquet")
        b = pd.read_parquet(LK / "activity.parquet")
        b = b[b.DeviceName.astype(str).isin(rel)]
        a = pd.concat([a[~a.DeviceName.astype(str).isin(rel)], b], ignore_index=True)
        a.to_parquet(NEW / "activity.parquet", index=False)
        R = pd.read_csv(LK / "final_rulings.csv", dtype=str, keep_default_na=False)
        R = R[R.DeviceName.isin(rel)]
        if len(R):
            R.to_csv(NEW / "final_rulings.csv", mode="a", header=False, index=False)
        print(f"setup: {len(done)} released records in batch_{RELEASED_BATCH}; activity rows +{len(b)}; rulings +{len(R)}")
        marker.write_text("ok", encoding="utf-8")
    check()


def stage(names: list[str]):
    """Renders + crops of signals to be re-read: copied into the new store, record paths rewritten (old store untouched)."""
    locked, _ = holdout()
    for dn in names:
        assert dn not in locked, f"{dn} is locked"
        if (OLD / "render" / dn).exists() and not (NEW / "render" / dn).exists():
            shutil.copytree(OLD / "render" / dn, NEW / "render" / dn)
        for p in (OLD / "crops").glob(f"{dn}_*"):
            if not (NEW / "crops" / p.name).exists():
                shutil.copy2(p, NEW / "crops" / p.name)
        for name in (f"{dn}.json", f"{dn}.extract.json"):
            f = NEW / "signals" / name
            if f.exists():
                r = json.loads(f.read_text(encoding="utf-8"))
                r2 = _rewrite(r, str(OLD / "render"), str(NEW / "render"))
                if r2 != r:
                    f.write_text(json.dumps(r2, indent=1, default=str), encoding="utf-8")
        print(f"staged {dn}")


def check():
    locked, rel = holdout()
    names = {p.name.split(".")[0] for p in (NEW / "signals").glob("*.json")}
    bad = names & locked
    assert not bad, f"locked_v2 signals in the v4l store: {sorted(bad)}"
    for d in ("render", "crops"):
        bad = {p.name.split("_")[0] for p in (NEW / d).glob("*")} & locked
        assert not bad, f"locked_v2 signals in {d}: {sorted(bad)}"
    a = pd.read_parquet(NEW / "activity.parquet")
    assert not a.DeviceName.astype(str).isin(locked).any(), "locked_v2 activity in the v4l store"
    print(f"check ok: {len(names)} records, {len(names & rel)} released, 0 locked_v2")


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"setup": lambda: setup(), "stage": lambda: stage(sys.argv[2:]), "check": lambda: check()}[cmd]()
