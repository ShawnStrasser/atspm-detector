"""Note 100c: package stackers for v5b = the note-95 recipes (pkg95 pkgsingle / pkgstacker) on the plain width-32 siba OOF
(x100_w32{,_s1,_s2}, 2026-only, filtered) instead of s95_siba. Output folder `final_v3_work/v3fit100` = a copy of v3fit95
(v3fit95 untouched); only stacker/ changes.

    set F76_ARM=c & python pkg100.py pkgsingle      (run BEFORE pkgstacker, as in pkg95)
    set F76_ARM=c & python pkg100.py pkgstacker
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "final95"))
import pkg95 as P  # noqa: E402

NET = "x100_w32"
FIT100 = P.DCW / "final_v3_work" / "v3fit100"


def main():
    if not FIT100.exists():
        shutil.copytree(P.DCW / "final_v3_work" / "v3fit95", FIT100)
    P.O88.FIT = FIT100
    P.FN.NET95 = NET
    {"pkgsingle": P.stage_pkgsingle, "pkgstacker": P.stage_pkgstacker}[sys.argv[1]](None)
    print(f"{sys.argv[1]} on {NET} -> {FIT100 / 'stacker'}")


if __name__ == "__main__":
    main()
