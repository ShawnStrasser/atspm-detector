"""Track B, B5: train the sibling-context TCN (`b5_net.SibNet`) with the Track B loop.

`trackb_train.run` unchanged -- same data, folds, inner validation, early stopping,
checkpoints, resumability -- with the model, the batch forward and the loss swapped for
the B5 ones.  Extra flag `--variant b5a|b5b`; everything else is `trackb_train.py`'s.
The variant is written next to the checkpoint (`models/<tag>.b5.json`) for inference.

    python research/code/neural/b5_train.py --tag tb_b5a_f0 --fold 0 --arch tcn --variant b5a
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "code")))
import rpath  # noqa: F401,E402
from neural import b5_net as N  # noqa: E402
from neural import train2 as T2  # noqa: E402
from neural import trackb_train as TB  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    variant = "b5a"
    if "--variant" in argv:
        i = argv.index("--variant")
        variant = argv[i + 1]
        del argv[i:i + 2]
    assert variant in ("b5a", "b5b"), variant
    tag = argv[argv.index("--tag") + 1]
    TB.MODELDIR.mkdir(parents=True, exist_ok=True)
    (TB.MODELDIR / f"{tag}.b5.json").write_text(json.dumps({"variant": variant}))

    def factory(arch, dropout=0.0):
        m = N.SibNet(arch, variant, dropout)
        N._CURRENT["model"] = m
        return m

    TB.PairNetB = factory
    TB.forward_batch = N.forward_batch
    T2.forward_batch = N.forward_batch          # inner validation (`T2.evaluate`)
    T2.phase_loss = N.phase_loss                # + auxiliary first-pass loss for b5b
    sys.argv = [sys.argv[0]] + argv
    TB.main()


if __name__ == "__main__":
    main()
