"""Note 88: paired quick speed check, v4b then v4d, 3 h only, same machine load (bench84 recipe)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "final84"))
import bench84 as B4  # noqa: E402
B4.B.LENS = ["h3"]
for tag in ("v4b", "v4d"):
    B4.V4B = B4.B.W / f"final_v3_candidate_{tag}"
    B4.out_dir = lambda cfg, t=tag: B4.B.W / "final_v3_work" / "f88" / f"bench_{t}"
    print("==", tag, flush=True)
    B4.run("mean3")
