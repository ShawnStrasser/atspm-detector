"""Note 152: cProfile of one assess_group call over a captured set (or each signal alone with --alone).
    python profg152.py <set> [--n N] [--alone] [--top K] [--sort tottime|cumtime]"""
import argparse
import cProfile
import os
import pickle
import pstats
import sys
import time
from pathlib import Path

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.environ.get("P152_SRC") or str(REPO / "src"))
from atspm_detector import health_core as hc, health_v4 as hv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--alone", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--sort", default="cumtime")
    ap.add_argument("--noprof", action="store_true")
    ap.add_argument("--chunk", type=int, default=0)
    a = ap.parse_args()
    recs = pickle.load(open(DCW / "s152" / "cap" / f"{a.set}.pkl", "rb"))
    if a.n:
        recs = recs[:a.n]
    its = [dict(phase=r["phase"], fn_label=r["flab"], fn_prob=r["func"], lanes=r["span"], phase_conf=r["pconf"])
           for r in recs]
    preps = [hc.prep_arrays(r["t"], r["eid"], r["par"]) for r in recs]
    hv._load_refs()
    hv._load_limits()

    def run():
        if a.alone:
            for P, r, it in zip(preps, recs, its):
                P.cache = {}
                hv.assess_group(P, r["start"], r["end"], [it])
        else:
            ch = a.chunk or len(preps)
            for i in range(0, len(preps), ch):
                for P in preps[i:i + ch]:
                    P.cache = {}
                G = hc.prep_group(preps[i:i + ch], names=list(range(len(preps[i:i + ch]))))
                hv.assess_group(G, recs[i]["start"], recs[i]["end"], its[i:i + ch])
    t0 = time.perf_counter()
    if a.noprof:
        run()
        print(f"{len(recs)} signals {time.perf_counter() - t0:.2f}s")
        return
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    print(f"{len(recs)} signals {time.perf_counter() - t0:.2f}s (profiled)")
    pstats.Stats(pr).sort_stats(a.sort).print_stats(a.top)


if __name__ == "__main__":
    main()
