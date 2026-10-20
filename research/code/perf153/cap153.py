"""Note 153 timing: capture the classifier's answers that predict() hands to the health step (`_health(con, W, d,
notes)` of the SQL-health pipeline, commit 90b979a) for the 755 h24_b signals (one day), saved for the health-alone
timing harness tim153.py.   python cap153.py --src <package parent with the SQL pipeline>"""
import argparse
import os
import sys
from pathlib import Path

DCW = Path(os.environ.get("DC_WORK") or (Path.home() / "dc_work"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.src)
    import pandas as pd
    from atspm_detector import pipeline as PP
    keep = []
    orig = PP._health

    def spy(con, W, d, notes):
        keep.append(d.copy())
        return orig(con, W, d, notes)
    PP._health = spy
    ids = pd.read_csv(DCW / "s151" / "h24b_ids.csv").DeviceId.tolist()
    PP.predict(str(DCW / "s151" / "h24b_events.parquet"), device_ids=ids, start="2026-09-28 00:00",
               end="2026-09-29 00:00", threads=4)
    pd.concat(keep, ignore_index=True).to_parquet(DCW / "s153" / "tim" / "answers_755.parquet")
    print("saved", sum(len(k) for k in keep))


if __name__ == "__main__":
    main()
