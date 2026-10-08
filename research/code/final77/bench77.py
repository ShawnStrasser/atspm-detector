"""Note 77: bench75 speed / RAM at 3 h (typical r8 / r11, busiest by channels / events; cold + warm median of 3, fresh
process, 4 threads) on the package after notes 76 + 77.   python bench77.py  -> %DC_WORK%/final_v3_work/f77/bench/"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final75"))
import bench75 as B  # noqa: E402

B.OUT = B.W / "final_v3_work" / "f77" / "bench"
B.LENS = ["h3"]
if __name__ == "__main__":
    import argparse
    B.cmd_run(argparse.Namespace(force=True))
    B.cmd_summary(None)
