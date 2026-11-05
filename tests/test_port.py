"""Stage 1 of the incremental-health design (T1.1, T1.3), as it stands after stages 2-3: the vehicle judge keeps today's
text -- every step of the frozen builder (tests/legacy) that the new plan has under the same name, token for token, but
for the adapters and the few judge steps that now read one -- for the default options, all off, each option switched
alone (today's raw-event steps became the extractor and the adapters; their outputs are tested in test_extract and
test_judge); ped health renders the queries of the frozen oracle v3 (the rules of the 2026-11-02 review), but for the steps that read what a
store keeps of the raw log.  Whitespace between
tokens is free (hq.plan.tokens)."""
import re

import pytest

from atspm_detector import health_duckdb as H
from atspm_detector import ped_health_duckdb as P
from atspm_detector.hq import plan, render
from legacy import health_duckdb_v2 as H0
from legacy import ped_health_duckdb_v3 as P0


def _flip(k, v):
    return {"lane_pairs": [None, "any"], "clean_yardsticks": [False, True], "occ_strict": []}.get(k, [not v])


VEH = [("defaults", {}), ("all off", dict(H.OPTIONS_OFF))]
VEH += [(f"{k}={x}", {k: x}) for k, v in H.OPTIONS.items() for x in _flip(k, v)]
KNOWN_FAIL = {"silent_merge=False"}   # today's builder fails there too (phase_drop's text needs silent_merge's times)
ADAPTERS = set(re.findall(r"-- @step (\w+)", (render.SQL_DIR / "adapters.sql").read_text(encoding="utf-8")))
# judge steps that read an adapter in place of a raw-event step (design 5)
READ_ADAPTER = {"fe", "pmode", "sgb", "pdg", "lcsw", "pkept", "g5"}
# changed on purpose after the freeze: a logging hole no longer ends a run of signal silence (2026-11-03)
VEH_CHANGED = {"sgr"}
# step names today's SQL also uses as a column (flat: xx.flat; tod: lce's CTE column), renamed consistently everywhere;
# lint reports any other
ALIAS = ("flat", "tod")
# changed on purpose after the freeze: the wording of a rise against almost-silent phase mates (2026-11-03)
RISE_NOW = ("CASE WHEN x.mr_r >= 100 THEN 'counts rose far above its phase mates, which gave almost nothing at that time of day on the other days: ' ELSE "
            "format('counts rose to {:.1f} x what its phase mates gave at that time of day on the other days: ', x.mr_r) END")
RISE_THEN = "format('counts rose to {:.1f} x what its phase mates gave at that time of day on the other days: ', x.mr_r)"


@pytest.mark.parametrize("name,opt", VEH, ids=[n for n, _ in VEH])
@pytest.mark.parametrize("periods", [False, True])
def test_vehicle_judge_verbatim(name, opt, periods):
    kw = dict(options=opt, periods=periods)
    if name in KNOWN_FAIL:
        with pytest.raises(Exception):
            H0.build("events", "detectors", **kw)
        with pytest.raises(Exception):
            H.build("events", "detectors", **kw)
        return
    a, b = H.build("events", "detectors", **kw), H0.build("events", "detectors", **kw)
    old = {n: s for n, s, _ in b.ctes}
    kept = [(n, s) for n, s, _ in a.ctes if n in old and n not in ADAPTERS | READ_ADAPTER | VEH_CHANGED]
    assert len(kept) > 150
    for n, s in kept:
        assert plan.tokens(s.replace(RISE_NOW, RISE_THEN)) == plan.tokens(old[n]), n
    assert plan.tokens(H.FINAL) == plan.tokens(H0.FINAL)
    assert plan.tokens(H.PERIODS_FINAL) == plan.tokens(H0.PERIODS_FINAL)
    assert a.lint(allow_alias=ALIAS) == []


PED = [("defaults", {}), ("all off", {k: False for k in P.OPT})]
# the steps over the raw log that became sil + logq (the store keeps those, design 6), and the ones that read them
PED_READ = {"b10", "mins", "hours", "alive", "gap", "hole", "press", "stopl"}
# after the freeze (review of 2026-11-03): the checks walks_no_press and sustained_press, and a presence ON over a log hole in f_stuck
PED_NEW = {"uniq", "pbin", "f_nop", "f_press"}
PED_CHANGED = {"f_stuck"}
PED += [(f"{k}={not v}", {k: not v}) for k, v in P.OPT.items()]


@pytest.mark.parametrize("name,opt", PED, ids=[n for n, _ in PED])
def test_ped_steps(name, opt):
    a, b = dict(P._steps("my_events", opt)), dict(P0._steps("my_events", opt))
    assert set(a) - set(b) - PED_NEW == {"sil", "logq"} and set(b) - set(a) == {"mins", "hours", "alive"}
    assert len(set(a) & set(b) - PED_READ) >= 25
    for n in set(a) & set(b) - PED_READ - PED_CHANGED:
        assert plan.tokens(a[n]) == plan.tokens(b[n]), n


