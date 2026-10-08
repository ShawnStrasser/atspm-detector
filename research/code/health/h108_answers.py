"""Note 108: how the user's earlier answers come out under v108 (SANITY CHECK ONLY - nothing was fitted to them).

v1 (health_review_v1, rows 1-55 + 60): 'is this detector really faulty in the way the check says?' Y / N / ?.
v2 (health_review_v2, 19 rows): his verdict on the note-96 result ('new result is good' = agree; 'No!' = disagree).
v3 (health_review_v3, row 1): '?' with the comment that 2B334 d13 follows the phase neither in counts nor in time ON.

    python h108_answers.py 995 999
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h108_base as B  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)
DCW, OUT = B.DCW, B.OUT
FL = ("suspect", "bad")


def main(tags):
    a1 = pd.read_parquet(DCW / "health96" / "answers.parquet")[["row", "check", "answer", "DeviceId", "window",
                                                                "detector"]]
    a1 = a1[a1.answer.ne("")].assign(src="v1")
    v2 = pd.read_csv(DCW / "health96" / "review_rows.csv")[["n", "rule", "dev", "window", "detector"]]
    an = pd.read_csv(DCW / "health104" / "answers_v2_104.csv")[["n", "ans", "new_status_96"]]
    v2 = v2.merge(an, on="n").rename(columns={"n": "row", "rule": "check", "ans": "answer", "dev": "DeviceId"})
    v2 = v2.assign(src="v2")
    v3 = pd.read_csv(DCW / "health104" / "review_rows_v3b.csv").iloc[:1][["n", "dev", "window", "detector"]]
    v3 = v3.rename(columns={"n": "row", "dev": "DeviceId"}).assign(src="v3", check="erratic (R2)", answer="? not ok")
    A = pd.concat([a1, v2, v3], ignore_index=True)
    cols = ["DeviceId", "window", "detector", "fn", "span", "status", "new_status", "st8", "rules8", "left8", "watch8",
            "dq8"]
    for t in tags:
        R = pd.read_parquet(OUT / f"resolved108_q{t}.parquet", columns=cols)
        A = A.merge(R.rename(columns={"st8": f"q{t}", "rules8": f"rules{t}", "left8": f"left{t}", "watch8": f"watch{t}",
                                      "dq8": f"dq{t}"}).drop(columns=[c for c in ("fn", "span", "status", "new_status")
                                                                       if c in A.columns]),
                    on=["DeviceId", "window", "detector"], how="left")
    show = ["src", "row", "check", "answer", "fn", "span", "status", "new_status"] + \
        [c for t in tags for c in (f"q{t}", f"left{t}", f"watch{t}")]
    print(A[show].to_string())
    # v1 summary: flagged (suspect / bad) and flagged-or-watch by answer
    v = A[A.src == "v1"]
    for c in ["status", "new_status"] + [f"q{t}" for t in tags]:
        print(c, "v1 flagged by answer:", v.groupby("answer")[c].apply(lambda s: f"{s.isin(FL).sum()}/{len(s)}").to_dict(),
              "| flagged or watch:", v.groupby("answer")[c].apply(lambda s: f"{s.isin(FL + ('watch',)).sum()}/{len(s)}")
              .to_dict())
    w = A[A.src == "v2"]
    good = w.answer.str.contains("good|fine", case=False)
    for c in ["new_status"] + [f"q{t}" for t in tags]:
        print(c, "v2 rows he agreed with that keep the agreed status:", int((w[good][c] == w[good].new_status_96).sum()),
              "/", int(good.sum()), "| row 3 ('No!'):", w[w.row == 3][c].iloc[0])
    A.to_csv(OUT / "answers108.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["995", "999"])
