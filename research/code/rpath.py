"""Import path helper for the research scripts.

The research code is a flat namespace of loose scripts, exactly as it was while the
study ran.  Importing this module puts every folder under `research/code/` and the
repository's `src/` (the shipped package `atspm_detector`) on `sys.path`, so a research
script can say `from common import DC_WORK` without knowing where the file sits.

Many scripts were written against the earlier model package, removed from the tree when
the pip package shipped and kept in git tag `beta-final_v2`.  Without it, `decode` and `health`
resolve to `atspm_detector.decode` / `atspm_detector.health` (same names, the final
model's definitions, with a warning); `predict`, `lgbm_numpy`, `gru_blend`, `gru_input`
and `gru_onnx` do not exist.  To run such a script as its note describes (DC_BETA_MODEL):

    git worktree add ../dc-beta beta-final_v2
    set DC_BETA_MODEL=..\\dc-beta\\model

and the earlier model's folder goes back on the path (aliases off).

Every research script starts with the same three lines::

    import sys; from pathlib import Path
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                                if p.name == "code")))
    import rpath  # noqa: F401
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import warnings
from pathlib import Path

CODE = Path(__file__).resolve().parent
REPO = CODE.parents[1]
SRC = REPO / "src"                                   # the shipped package: import atspm_detector
_beta = os.environ.get("DC_BETA_MODEL")              # model folder of a checkout of git tag beta-final_v2
MODEL = Path(_beta).resolve() if _beta else None     # the earlier model package, only when checked out

# note 76: `phasefree/` holds the phase-order-free versions of the earlier model's features.py and
# features_partner.py (ties broken by behaviour, never by phase number).  It sits ahead of the other folders so
# every research script builds features with the fixed definitions.
PHASEFREE = CODE / "phasefree"
_dirs = [CODE, PHASEFREE] + ([MODEL] if MODEL else []) + [SRC] + sorted(
    p for p in CODE.rglob("*") if p.is_dir() and p.name not in ("__pycache__", "phasefree"))
for _d in reversed(_dirs):          # so CODE, phasefree, MODEL and SRC end up first
    _s = str(_d)
    # re-insert rather than skip: the caller has already put CODE on the path itself
    # (that is how this module was found), and merely skipping it would leave CODE
    # *behind* the others, so `import common` could find the wrong module.
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)

# Module names the earlier model package shares with atspm_detector: aliased unless that package is on the path.
FINAL_ALIASES = ("decode", "health")


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self.target = target

    def create_module(self, spec):  # default module object
        return None

    def exec_module(self, module) -> None:
        real = importlib.import_module(self.target)
        module.__dict__.update({k: v for k, v in vars(real).items()
                                if not (k.startswith("__") and k.endswith("__"))})
        module.__doc__, module.__file__ = real.__doc__, real.__file__
        warnings.warn(f"research import {module.__name__!r} uses {self.target} (the final model's "
                      "definitions); set DC_BETA_MODEL for the earlier model's", stacklevel=2)


class _FinalAlias(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if path is not None or name not in FINAL_ALIASES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name)
        if spec is not None and spec.origin is not None:   # a real module of that name wins
            return None
        return importlib.util.spec_from_loader(name, _AliasLoader(f"atspm_detector.{name}"))


if MODEL is None and not any(type(f).__name__ == "_FinalAlias" for f in sys.meta_path):
    sys.meta_path.insert(0, _FinalAlias())

# Function label table used by every research scorer and trainer (orchestrator 2026-10-03, note 81): v4l is the
# scoring truth AND the training labels from now on (v3s schema: `truth_v3s` = this table's truth). The old v3s table
# stays on disk unchanged as LABELS_V3S for before / after comparisons only.
import os as _os
# note 89 (orchestrator 2026-10-04): v4o = v4m (note 88, dq without fault events) + the eight note-87 exclusion groups
# (= v4n) + not_checkable rows, all restored to TRAINING; plain / truth columns equal v4l (scoring sets unchanged),
# training values in <col>_train (read by v3_retrain.load_labels). v4l / v4m / v4n stay on disk for comparisons.
# note 95 (2026-10-05): v4q = v4p (note 94: user review v1, 05999 print, N1 print-vs-hand by model agreement, N2) + the
# user's rulings of 2026-10-05 16:20 (loop 'advance presence' = Advance, channel text vs Function by model agreement,
# 05999 d23 out) + three prints filed under other names. v4o / v4p stay on disk for comparisons.
LABELS_CURRENT = REPO / "research" / "labels" / "function_labels_v4q.parquet"
LABELS_V3S = REPO / "research" / "labels" / "function_labels_v3s.parquet"
DEC_ROLE_CURRENT = Path(_os.environ.get("DC_WORK", Path.home() / "dc_work")) / "lab95" / "dec_role_changed_v4q.parquet"
