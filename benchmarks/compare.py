"""Flag what got slower in benchmarks/results.jsonl: python benchmarks/compare.py [--file F] [--threshold 0.20]

Only runs of the same kind are compared: same machine, DuckDB version, threads, memory limit and input; a whole-run peak
only with runs of the same implementations and jobs.  Every run's metric is set against the median of the 5 runs before it.  The newest run being more than the threshold
worse is "watch" (one run can be noise); it and the run before it both being worse is "SLOWER" and exit code 1.  The batch
package / oracle ratio has its own limit (1.05, on 2 runs in a row too), and outputs that differed always fail."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BASE_RUNS = 5
RATIO_LIMIT = 1.05


def group_of(line: dict) -> tuple:
    return (line.get("machine"), line.get("duckdb"), line.get("threads"), line.get("memory"),
            json.dumps(line.get("input"), sort_keys=True))


def metrics(line: dict) -> dict:
    """metric -> value, all of them lower is better."""
    out = {f"batch {impl} {job} s": v for impl, t in line.get("seconds", {}).items() for job, v in t.items()}
    u = line.get("updates")
    if u:
        out.update({f"update {k} s": u["update_s"][k] for k in ("median", "p95", "max")})
        out["judge s"], out["store MB"] = u["judge_s"], u["store_mb"]
        out.update({f"{n} s": u[f"{k}_s"] for n, k in (("judge ped", "judge_ped"), ("judge ped model", "judge_ped_model")) if f"{k}_s" in u})
        out.update({f"peak MB after {k}": v for k, v in u.get("peak_mb", {}).items() if v})
    if line.get("peak_mb"):
        jobs = sorted({j for t in line.get("seconds", {}).values() for j in t} | ({"update"} if u else set()))
        out[f"peak MB {line.get('impl')} {'+'.join(jobs)}"] = line["peak_mb"]
    return out


def _worse(lines: list, i: int, metric: str, threshold: float) -> tuple | None:
    """(value, baseline) when run i's metric is worse than the median of the runs before it."""
    if metric not in metrics(lines[i]):
        return None
    before = [m[metric] for m in map(metrics, lines[:i]) if metric in m][-BASE_RUNS:]
    if not before:
        return None
    base, value = statistics.median(before), metrics(lines[i])[metric]
    return (value, base) if value > base * (1 + threshold) else None


def compare(lines: list, threshold: float = 0.20) -> list:
    """one dict per finding: level (SLOWER / watch / DIFFERENT), group, metric, value, baseline."""
    groups = {}
    for line in lines:
        groups.setdefault(group_of(line), []).append(line)
    found = []
    for g, runs in groups.items():
        last, n = runs[-1], len(runs) - 1
        name = f"{g[0]} duckdb {g[1]} {g[4]}"
        bad = {k: v for k, v in last.get("equal_outputs", {}).items() if v == "DIFFERENT"}
        if bad or False in (last.get("updates", {}).get(k) for k in ("equal_to_health_clock", "equal_to_ped_health")):
            found.append({"level": "DIFFERENT", "group": name, "metric": "outputs", "value": str(bad or "updates"), "baseline": "same"})
        for m in metrics(last):
            hit = _worse(runs, n, m, threshold)
            if hit:
                found.append({"level": "SLOWER" if n and _worse(runs, n - 1, m, threshold) else "watch", "group": name,
                              "metric": m, "value": hit[0], "baseline": hit[1]})
        ratios = {}
        for run in runs:
            for j, r in run.get("ratio_package_oracle", {}).items():
                ratios.setdefault(j, []).append(r)
        for j, rs in ratios.items():
            if rs[-1] > RATIO_LIMIT:
                found.append({"level": "SLOWER" if len(rs) > 1 and rs[-2] > RATIO_LIMIT else "watch", "group": name,
                              "metric": f"ratio package / oracle {j}", "value": rs[-1], "baseline": RATIO_LIMIT})
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(Path(__file__).with_name("results.jsonl")))
    ap.add_argument("--threshold", type=float, default=0.20)
    a = ap.parse_args(argv)
    lines = [json.loads(x) for x in Path(a.file).read_text(encoding="utf-8").splitlines() if x.strip()]
    found = compare(lines, a.threshold)
    for f in found:
        print(f"{f['level']:9} {f['metric']}: {f['value']} against {f['baseline']}  [{f['group']}]")
    print(f"{len(lines)} runs; {sum(f['level'] != 'watch' for f in found)} failing, {sum(f['level'] == 'watch' for f in found)} to watch")
    return 1 if any(f["level"] != "watch" for f in found) else 0


if __name__ == "__main__":
    raise SystemExit(main())
