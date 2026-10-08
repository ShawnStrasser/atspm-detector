"""Note 128 part 1: a start far before the data / an end far after it.  Sliced samples (3 non-locked note-115 signals x
30 min / 3 h / 24 h) are called with start 1 h / 7 days / 365 days early and 2000-01-01, end 2030-01-01, and both;
the answers must equal the old package called with the out-of-range bound left out (= clipped to the data).
    python clip128.py ref <old src>    -> s128/clip/ref.pkl   (old package, bounds clipped)
    python clip128.py new <new src>    -> s128/clip/new.pkl   (new package, far bounds) + times
    python clip128.py cmp"""
from __future__ import annotations
import os, sys, time, pickle, warnings
from pathlib import Path
import pandas as pd
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
OUT = W / "s128" / "clip"
SIG = ["s05", "s20", "s33"]
WINS = {"m30": ("2026-09-19 07:30:00", "2026-09-19 08:00:00"), "h3": ("2026-09-19 15:00:00", "2026-09-19 18:00:00"),
        "h24": (None, None)}
EARLY = {"1h": pd.Timedelta(hours=1), "7d": pd.Timedelta(days=7), "365d": pd.Timedelta(days=365), "y2000": None}
FAR_END = "2030-01-01 00:00:00"


def sample(sig, win):
    f = OUT / f"{sig}_{win}.parquet"
    if not f.exists():
        e = pd.read_parquet(W / "s115" / "parity" / "ev" / f"{sig}.parquet")
        a, b = WINS[win]
        if a:
            e = e[(e.Timestamp >= a) & (e.Timestamp < b)]
        e.to_parquet(f, index=False)
    return str(f)


def cases():
    out = []
    for sig in SIG:
        for win, (a, b) in WINS.items():
            f = sample(sig, win)
            e = pd.read_parquet(f, columns=["Timestamp"])
            a0 = pd.Timestamp(a) if a else e.Timestamp.min().floor("min")
            b0 = pd.Timestamp(b) if b else e.Timestamp.max().ceil("min")
            for k, dt in EARLY.items():
                s = "2000-01-01 00:00:00" if dt is None else str(a0 - dt)
                out.append((f"{sig}_{win}_start-{k}", f, s, str(b0), None, str(b0)))
            out.append((f"{sig}_{win}_end-2030", f, str(a0), FAR_END, str(a0), None))
            out.append((f"{sig}_{win}_both", f, "2000-01-01 00:00:00", FAR_END, None, None))
            out.append((f"{sig}_{win}_exact", f, str(a0), str(b0), str(a0), str(b0)))
    return out


def main():
    mode = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    locked = set(pd.read_csv(W / "official" / "locked_v2.csv").DeviceId.str.lower())
    sig = pd.read_csv(W / "s115" / "parity" / "signals115.csv")
    assert not set(sig[sig.case.isin(SIG)].DeviceId.str.lower()) & locked
    if mode == "cmp":
        R, N = pickle.load(open(OUT / "ref.pkl", "rb")), pickle.load(open(OUT / "new.pkl", "rb"))
        bad = 0
        for k, (out, ph, t) in N.items():
            ro, rp, rt = R[k]
            same = out.equals(ro) and ph.equals(rp)
            if not same:
                bad += 1
                cols = [c for c in out.columns if not out[c].equals(ro[c])]
                print("DIFF", k, cols)
            print(f"{k:28s} new {t:6.2f}s  old(clipped) {rt:6.2f}s  {'identical' if same else 'DIFFERENT'}")
        print("cases", len(N), "different", bad)
        return
    sys.path.insert(0, sys.argv[2])
    warnings.simplefilter("ignore")
    from detector_classifier import predict
    res = {}
    for name, f, s, e, rs, re_ in cases():
        if mode == "ref":
            s, e = rs, re_
        t = time.perf_counter()
        out, ph = predict(f, start=s, end=e, min_actuations=1, return_phases=True)
        res[name] = (out, ph, time.perf_counter() - t)
        print(name, f"{res[name][2]:.2f}s", flush=True)
    pickle.dump(res, open(OUT / f"{mode}.pkl", "wb"))


if __name__ == "__main__":
    main()
