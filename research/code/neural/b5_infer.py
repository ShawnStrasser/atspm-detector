"""Track B, B5: held-out predictions for a sibling-context checkpoint.

`trackb_infer.py` unchanged (same 22 windows, 30-min pieces pooled by mean log-prob, same
output contract) with the loader and the batch forward swapped for the B5 ones.  Every
labelled detector of a signal-window is scored in one forward, so each sees all its
siblings.

    python research/code/neural/b5_infer.py --ckpt tb_b5a_f0 --fold 0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from neural import b5_net as N  # noqa: E402
from neural import infer2 as I2  # noqa: E402
from neural import trackb_infer as TI  # noqa: E402


def load_model(ckpt: str, device: str):
    p = Path(ckpt)
    if not p.exists():
        p = TI.MODELDIR / f"{ckpt}.pt"
    side = json.loads((p.parent / f"{p.stem}.b5.json").read_text())
    d = torch.load(p, map_location=device, weights_only=False)
    m = N.SibNet(d["arch"], side["variant"]).to(device).eval()
    m.load_state_dict(d["state"])
    I2.log(f"{p.name}: {side['variant']}, epoch {d.get('epoch')}, inner-val {d.get('es_score')}")
    return m


def main() -> None:
    I2.load_model = load_model
    I2.forward_batch = N.forward_batch
    TI.main()


if __name__ == "__main__":
    main()
