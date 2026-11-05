"""Stage 2 of the incremental-health design: the extractor writes the store tables (temp tables in a batch run), finalize
settles their rows at the window end, and the adapters rebuild today's steps from them.

T2.1 / T2.2  every adapter -- and the judge steps that read one -- equals the frozen oracle's step of the same name, on
             the bundled sample, a synthetic set and the edge patterns of synth.edges, over each signal's whole log, an
             odd window and 30-minute samples at odd hours (predict's windows_table): integers, texts and flags exact,
             floats within 1e-9 relative (integer ns against today's double seconds), the median ON (dur_p50, pmed)
             exact below 0.5 s (the on_us cap, design 4.1)
T2.1b        the same on small logs built to reach the rare branches (each with a check that it does reach its branch)
T2.3         the adapters' column names and types equal the oracle's
T2.4         every statement of the script runs on empty typed inputs
T2.5         lint: no lambdas / views / macros, every step defined before use; the extractor (extract/*.sql) reads only the
             events, the windows and the detectors table and its own steps, never a store table; ownership (design 3.2): every
             emitted fact comes from its owner, an event of this chunk, recomputed here from the raw events without the
             extractor's steps; no time is written past the signal's last allowed event; labels name only what is open there
plus: the steps marked @emit have the layout of store.sql and are written as temp tables; no rule compares the capped
medians with a limit at or above the cap."""
import re

import duckdb
import numpy as np
import pandas as pd
import pytest

import make_golden as G
import synth
from atspm_detector import health_duckdb as H
from atspm_detector.hq import render
from legacy import health_duckdb_v2 as H0

# adapters, and judge steps reading an adapter whose text changed (fe, gonb's hib, sgb, pdg, lcsw, pmode, periods)
STEPS = ["win", "cov5", "dets", "n5", "occ5", "unk5", "g5", "hold", "act", "tf", "phs3", "sgyb", "q5b", "q5gyb", "fe",
         "gonb", "hib", "sgb", "pdg", "lcsw", "pmode", "pall", "xf", "finV"]
CAPPED = ("dur_p50", "pmed")
STORE = ("health_signal_5min", "health_detector_5min", "health_phase_5min", "health_phase_15min", "health_tick_counts",
         "health_intervals")
_SETS = {}


def _set(name):
    if name not in _SETS:
        ev, det = synth.edges(*synth.make(3, 2, seed=5)) if name == "edge" else G.inputs(name)
        con = G.connect(ev, det)
        con.execute("SET memory_limit = '2GB'; SET max_temp_directory_size = '4GB'")
        _SETS[name] = con
    return _SETS[name]


def _windows(con, kind):
    """the windows table wtab, one window per signal: its whole log, an odd window, or a 30-min sample at an odd hour
    (a different one per signal, as predict's samples)."""
    if kind == "whole":
        sql = "epoch_ns(min(Timestamp)) AS s_ns, epoch_ns(max(Timestamp)) + 1000000000 AS e_ns"
    elif kind == "odd":
        sql = "epoch_ns(min(Timestamp)) + 4033100000000 AS s_ns, epoch_ns(max(Timestamp)) - 3611700000000 AS e_ns"
    else:
        s0 = "epoch_ns(min(Timestamp)) + 1234500000000 + (hash(DeviceId) % 20) * 4321000000000"
        sql = f"{s0} AS s_ns, {s0} + 1800000000000 AS e_ns"
    con.execute(f"CREATE OR REPLACE TABLE wtab AS SELECT DeviceId, {sql} FROM events GROUP BY DeviceId")


def _same(name, a, b):
    assert a[1] == b[1], f"{name}: columns / types {a[1]} != {b[1]}"
    x, y = a[0].copy(), b[0].copy()
    assert len(x) == len(y), f"{name}: rows {len(x)} != {len(y)}"
    if not len(x):
        return
    for c in CAPPED:
        if c in x.columns:
            x[c], y[c] = x[c].clip(upper=0.5), y[c].clip(upper=0.5)
    keys = [c for c in x.columns if not pd.api.types.is_float_dtype(x[c])
            and not x[c].map(lambda v: isinstance(v, (list, np.ndarray))).any()]
    x = x.sort_values(keys, kind="stable").reset_index(drop=True)
    y = y.sort_values(keys, kind="stable").reset_index(drop=True)
    for c in x.columns:
        if pd.api.types.is_float_dtype(x[c]):
            u, v = x[c].to_numpy(float), y[c].to_numpy(float)
            assert np.array_equal(np.isnan(u), np.isnan(v)), f"{name}.{c}: NULLs differ"
            ok = ~np.isnan(u)
            rel = np.abs(u[ok] - v[ok]) / np.maximum(np.abs(u[ok]), 1.0)
            assert (rel <= 1e-9).all(), f"{name}.{c}: max difference {rel.max():.2e}"
        else:
            assert x[c].astype(str).tolist() == y[c].astype(str).tolist(), f"{name}.{c} differs"




def test_bind_on_empty_inputs():
    con = duckdb.connect()
    con.execute("CREATE TABLE events (DeviceId VARCHAR, Timestamp TIMESTAMP, EventId INTEGER, Parameter INTEGER)")
    con.execute("CREATE TABLE detectors (DeviceId VARCHAR, Detector BIGINT, phase DOUBLE, function VARCHAR, "
                "lanes DOUBLE, configured_function VARCHAR, "
                + ", ".join(f"{c} DOUBLE" for c in H.PROB.values()) + ")")
    con.execute("CREATE TABLE hq_wpart AS SELECT 'S1' AS DeviceId, 0::BIGINT AS s_ns, 3600000000000::BIGINT AS e_ns")
    for job in ((True, True, tuple(sorted(H._options(None).items())), True, True, "window"),
                (False, False, tuple(sorted(H.OPTIONS_OFF.items())), False, False, "clock")):
        _, stmts, drops = H._script("events", "detectors", *job)
        for i, st in enumerate(stmts):
            con.execute(st)
            for t in drops[i]:
                con.execute(f"DROP TABLE IF EXISTS {t}")


ALIAS = ("flat", "tod")          # today's step names also used as a column, renamed consistently (stage 1)
SOURCES = {"win", "dcl", "xovd"}  # what the extractor may read besides the events, the carry and its own steps


@pytest.mark.parametrize("opt", [{}, dict(H.OPTIONS_OFF)], ids=["defaults", "all off"])
@pytest.mark.parametrize("origin", ["window", "clock"])
def test_lint(opt, origin):
    q = H.build("events", "detectors", periods=True, options=opt, bin_origin=origin)
    assert q.lint(allow_alias=ALIAS) == []
    names = [n for n, _, _ in q.ctes]
    ext = names[names.index("xev"):names.index("xhold") + 1]      # the extractor's steps (extract/*.sql)
    reads = {}
    for n, sql, _ in q.ctes:
        if n in ext:
            read = {o for o in names if o != n and re.search(rf"\b{o}\b", sql)}
            assert read <= SOURCES | set(ext), (n, read)
            reads[n] = read
    assert "xseg" in reads["xpc"] and "win" in reads["xev"] and "dcl" in reads["xsa"]     # the search does find reads


def test_store_tables_have_the_layout_of_store_sql():
    q = H.build("events", "detectors", windows="wtab", periods=True)
    assert set(q.emits) == set(STORE) and all(k == v for k, v in q.emits.items())
    stmts, _ = q.script(H.FINAL)
    for t in STORE:     # written as temp tables, whatever the number of steps reading them
        assert any(st.startswith(f"CREATE OR REPLACE TEMP TABLE {H.PREFIX}{t} AS") for st in stmts), t
    con = _set("edge")
    _windows(con, "whole")
    got = G.steps(H, con, STORE)
    ddl = duckdb.connect()
    ddl.execute(render.render("store.sql", device_type="VARCHAR"))
    assert {r[0] for r in ddl.execute("SHOW TABLES").fetchall()} == set(STORE) | {"health_active", "health_bins_released", "health_carry", "health_detector_phase", "health_ped_events", "health_log_15min"}
    for t in STORE:
        assert got[t][1] == [(r[0], r[1]) for r in ddl.execute(f"DESCRIBE {t}").fetchall()], t
        assert len(got[t][0]), t


def _raw(con):
    """the window's allowed events, de-duplicated, from the caller's tables only: DeviceId, dn (ns from the window start), eid,
    par, and each signal's window start and last allowed event."""
    good = f"e.EventId IN {tuple(H.ALLOWED)} AND (e.EventId NOT IN (81, 82) OR e.Parameter BETWEEN 1 AND 64)"
    ev = con.execute(f"""SELECT DISTINCT e.DeviceId, epoch_ns(e.Timestamp) - w.s_ns AS dn, e.EventId::INT AS eid,
                         e.Parameter::INT AS par, w.s_ns
                         FROM events e JOIN wtab w USING (DeviceId)
                         WHERE {good} AND epoch_ns(e.Timestamp) >= w.s_ns AND epoch_ns(e.Timestamp) < w.e_ns""").df()
    ev["lz"] = ev.groupby("DeviceId").dn.transform("max")
    return ev


def _chains(ev):
    """the detector events in today's order with their neighbours, and the chains: a run of ONs from its first ON to the
    event after its last ON (an OFF; none = still open)."""
    d = ev[ev.eid.isin([81, 82])].copy()
    d["o"] = (d.eid == 81).astype(int)      # ON before OFF at one instant
    d = d.sort_values(["DeviceId", "par", "dn", "o"], kind="stable").reset_index(drop=True)
    g = d.groupby(["DeviceId", "par"], sort=False)
    d["pe"], d["ne"], d["nd"] = g.eid.shift(1), g.eid.shift(-1), g.dn.shift(-1)
    on = d[d.eid == 82].copy()
    on["start"] = on.pe != 82
    on["cid"] = on.start.cumsum()
    c = on.groupby("cid").agg(DeviceId=("DeviceId", "first"), det=("par", "first"), s=("dn", "first"), n=("dn", "size"),
                              nxe=("ne", "last"), nd=("nd", "last"), lz=("lz", "first")).reset_index(drop=True)
    c["closed"] = c.nxe == 81
    return d, on, c


def _gaps(ev):
    """comms gaps: consecutive allowed events more than 120 s apart (as today's doubles)."""
    t = ev[["DeviceId", "dn"]].drop_duplicates().sort_values(["DeviceId", "dn"])
    t["nt"] = t.groupby("DeviceId").dn.shift(-1)
    return t[t.nt / 1e9 - t.dn / 1e9 > 120.0].rename(columns={"dn": "g0", "nt": "g1"})


def _ns(col):
    """a TIMESTAMP_NS column of a DataFrame as epoch ns."""
    return col.astype("datetime64[ns]").astype("int64")




def check_owners(ev, rows, open_labels=True):
    """the store rows (any number of deltas per key) against the facts recomputed from the raw events ev (_raw).
    open_labels: labels name only what is open at the last event (one chunk; rows of earlier chunks may name what closed
    since)."""
    s_ns = ev.groupby("DeviceId").s_ns.first()
    lz = ev.groupby("DeviceId").lz.first()
    D, P5, S5 = rows["health_detector_5min"], rows["health_phase_5min"], rows["health_signal_5min"]
    GH, TK, I = rows["health_phase_15min"], rows["health_tick_counts"], rows["health_intervals"]
    org = lambda t: t.DeviceId.map(s_ns).astype("int64")
    for t in (D, P5, S5):
        t["k"] = (_ns(t.TimeStamp) - org(t)) // H.NS5
    d, on, c = _chains(ev)
    d["k"] = d.dn // H.NS5

    def equal(got, want, what):
        got = got[got > 0].astype("int64").sort_index()
        want = want[want > 0].astype("int64").sort_index()
        assert got.index.tolist() == want.index.tolist(), what
        assert got.tolist() == want.tolist(), what

    # owner = the event itself: starts, re-logged ONs, the OFF closing a chain, stray OFFs, in the event's bin
    d["pon"] = d.pe == 82
    for col, sel in (("Starts", (d.eid == 82) & ~d.pon), ("Relogged_Ons", (d.eid == 82) & d.pon),
                     ("Closing_Offs", (d.eid == 81) & d.pon), ("Stray_Offs", (d.eid == 81) & ~d.pon)):
        equal(D.groupby(["DeviceId", "Detector", "k"])[col].sum(), d[sel].groupby(["DeviceId", "par", "k"]).size(), col)
    # greens: the green event, in its 5-min bin and its clock hour
    gr = ev[ev.eid == 1].assign(k=lambda x: x.dn // H.NS5, h=lambda x: (x.dn + x.s_ns) // H.NSQ)
    equal(P5.groupby(["DeviceId", "Phase", "k"]).Greens.sum(), gr.groupby(["DeviceId", "par", "k"]).size(), "Greens")
    GH["h"] = _ns(GH.TimeStamp) // H.NSQ
    equal(GH.groupby(["DeviceId", "Phase", "h"]).Greens.sum(), gr.groupby(["DeviceId", "par", "h"]).size(), "Greens per 15 min")
    # the signal's bins: every allowed event, first and last inside the bin
    b = ev.assign(k=ev.dn // H.NS5).groupby(["DeviceId", "k"]).dn.agg(["size", "min", "max"])
    sb = S5.assign(a=_ns(S5.First_Event) - org(S5), z=_ns(S5.Last_Event) - org(S5)).groupby(["DeviceId", "k"]).agg(
        Events=("Events", "sum"), a=("a", "min"), z=("z", "max")).sort_index()
    assert sb.index.tolist() == b.index.tolist()
    assert sb.Events.tolist() == b["size"].tolist() and sb.a.tolist() == b["min"].tolist() and sb.z.tolist() == b["max"].tolist()
    # intervals: a comms gap is owned by the event ending it, a plan by its event, a hold by the OFF closing its chain
    I["ss"], I["ff"] = _ns(I.StartTime) - org(I), _ns(I.EndTime) - org(I)
    rowset = lambda x, cols: sorted(map(tuple, x[cols].astype(object).where(x[cols].notna(), None).values.tolist()), key=repr)
    g = _gaps(ev)
    assert rowset(I[I.Kind == "comms_gap"], ["DeviceId", "ss", "ff"]) == rowset(g.astype({"g1": "int64"}), ["DeviceId", "g0", "g1"])
    assert rowset(I[I.Kind == "plan"], ["DeviceId", "ss", "Value"]) == rowset(ev[ev.eid == 131], ["DeviceId", "dn", "par"])
    cc = c[c.closed].copy()
    cc["f"] = cc.nd.astype("int64")
    cc["dur"] = cc.f / 1e9 - cc.s / 1e9
    gs = g.groupby("DeviceId").g0.apply(lambda x: np.sort(x.to_numpy())).to_dict()

    def over(r):
        a = gs.get(r.DeviceId)
        if r.dur <= 60 or a is None:
            return False
        i = np.searchsorted(a, r.s)
        return bool(i < len(a) and a[i] < r.f)
    cc["gap"] = [over(r) for r in cc.itertuples()]
    h = cc[(cc.dur >= 300) & ~cc.gap].assign(multi=lambda x: x.n > 1)
    assert rowset(I[I.Kind == "long_on"].astype({"Relogged": bool}), ["DeviceId", "Channel", "ss", "ff", "Relogged"]) == rowset(
        h, ["DeviceId", "det", "s", "f", "multi"])
    # tick counts: a closed single chain's length by the closing OFF, a too-fast pair by the next ON
    TK["q"] = (_ns(TK.TimeStamp) - org(TK)) // H.NSQ
    single = cc[(cc.n == 1) & ~cc.gap]
    equal(TK[TK.Kind == "on_us"].groupby(["DeviceId", "Detector"])["Count"].sum(), single.groupby(["DeviceId", "det"]).size(), "on_us")
    o = on.sort_values(["DeviceId", "par", "dn"], kind="stable")
    og = o.groupby(["DeviceId", "par"], sort=False).dn
    o["ta"], o["nx"] = np.round((o.dn - og.shift(1)) / 1e8), og.shift(-1)
    pair = o[(o.ta < 8) & o.nx.notna()].assign(q=lambda x: x.dn // H.NSQ)
    equal(TK[TK.Kind == "gap_pair"].groupby(["DeviceId", "Detector", "q"])["Count"].sum(),
          pair.groupby(["DeviceId", "par", "q"]).size(), "gap_pair")
    # time: every chain's time up to the signal's last allowed event, nothing past it
    op = c[~c.closed]
    want = pd.concat([cc.assign(t=cc.f - cc.s)[["DeviceId", "det", "t"]], op.assign(t=op.lz - op.s)[["DeviceId", "det", "t"]]])
    got = D.assign(t=D.On_Time_ns.fillna(0) + D.Unknown_Time_ns.fillna(0)).groupby(["DeviceId", "Detector"]).t.sum()
    equal(got, want.groupby(["DeviceId", "det"]).t.sum(), "time ON + unknown")
    for t in (D[D.On_Time_ns.notna() | D.Unknown_Time_ns.notna()], P5[P5.Green_Yellow_ns.notna() | P5.Green5_ns.notna()]):
        assert (t.k <= t.DeviceId.map(lz) // H.NS5).all()
    if not open_labels:
        return
    # labels: only what is open at the last event
    lab = D[D.Chain_Tag.notna()]
    assert set(zip(lab.DeviceId, lab.Detector, _ns(lab.Chain_Tag) - org(lab))) <= set(zip(op.DeviceId, op.det, op.s))
    col = ev[ev.eid.isin([1, 8, 9, 10])].copy()
    col["st"] = col.eid.map({1: 0, 8: 1, 9: 2, 10: 2})
    col = col.sort_values(["DeviceId", "par", "dn", "eid"], kind="stable")
    col = col[col.st != col.groupby(["DeviceId", "par"], sort=False).st.shift(1)]
    last = {(a, int(p)): v for (a, p), v in col.groupby(["DeviceId", "par"]).dn.last().items()}
    for t, ph in ((P5, "Phase"), (D, "Piece_Phase")):
        lab = t[t.Piece_Tag.notna()]
        assert all(last[(a, int(p))] == v for a, p, v in zip(lab.DeviceId, lab[ph], _ns(lab.Piece_Tag) - org(lab)))


# Rare branches the sets above do not reach: each scene is a small log, its check that the log reaches the branch, then the
# adapters and judge steps and both output tables against the oracle (T2.1, T3.1).
_BASE = []
COLOUR = [1, 7, 8, 9, 10, 11]


def _scene_base():
    if not _BASE:
        _BASE.extend(synth.make(2, 1, seed=3))
    return _BASE


def _scene_con(rows, drop, end=None):
    """the base log less the rows drop(ev) selects, plus rows; each signal's window: its log, or up to end for all."""
    ev, det = _scene_base()
    add = pd.DataFrame(rows, columns=ev.columns).astype({"EventId": "int32", "Parameter": "int32"})
    con = G.connect(pd.concat([ev[~drop(ev)], add], ignore_index=True), det)
    con.execute("SET memory_limit = '2GB'; SET max_temp_directory_size = '4GB'")
    e_ns = f"{pd.Timestamp(end).value}" if end is not None else "epoch_ns(max(Timestamp)) + 1000000000"
    con.execute(f"CREATE OR REPLACE TABLE wtab AS SELECT DeviceId, epoch_ns(min(Timestamp)) AS s_ns, {e_ns} AS e_ns "
                "FROM events GROUP BY DeviceId")
    return con


def _at(sec):
    return _scene_base()[0].Timestamp.min() + pd.Timedelta(seconds=sec)


def _dets(ev, chans, a, b):
    return (ev.DeviceId == "S00") & ev.EventId.isin([81, 82]) & ev.Parameter.isin(chans) & (ev.Timestamp >= a) & (ev.Timestamp < b)


def _dev(x):
    return x["win"][0].set_index("DeviceId").dev["S00"]


def _relogged_two_offs():
    """ON a, ON b (re-logged), OFF, OFF, ON c: the start before c is a (2.2 s back, not fast), not b (0.7 s back)."""
    chans, rows = (1, 2, 3, 4, 5, 6), []
    for d in chans:
        for k in range(60):
            t = 7200 + k * 40 + d
            rows += [("S00", _at(t + x), e, d) for x, e in ((0, 82), (1.5, 82), (1.8, 81), (1.9, 81), (2.2, 82), (2.4, 81))]
    con = _scene_con(rows, lambda ev: _dets(ev, chans, _at(7200), _at(9700)))

    def reach():
        x = G.steps(H, con, ["xrare"])["xrare"][0]
        assert (x.pst.notna() & (x.pst != x.prv)).sum() >= 300
    return con, reach


def _open_red(gap_after):
    """the signal's log ends in a red of phase 2 with fast starts in its first 2 s and a comms gap starting after them or
    inside them: only the first 2 s decide the piece's validity (up to the window end it would always be invalid)."""
    T = 30000.0
    rows = [("S00", _at(T - 1), 81, 1), ("S00", _at(T), 8, 2), ("S00", _at(T + 4), 9, 2)]
    rows += [("S00", _at(T + 4 + x), e, 1) for x, e in ((0.1, 82), (0.2, 81), (0.7, 82), (0.8, 81), (0.9, 82), (1.0, 81))]
    rows += [("S00", _at(T + 4 + gap_after), 131, 1), ("S00", _at(T + 4 + gap_after + 300), 131, 2)]
    con = _scene_con(rows, lambda ev: (ev.DeviceId == "S00") & (ev.Timestamp >= _at(T)), _at(T + 4 + gap_after + 350))

    def reach():
        x = G.steps(H, con, ["fpc", "xd5l", "win"])
        f, lab = x["fpc"][0], x["xd5l"][0]
        assert f[(f.dev == _dev(x)) & (f.ph == 2) & (f.st == 2)].valid.tolist() == [gap_after >= 2]
        lab = lab[(lab.dev == _dev(x)) & (lab.det == 1) & (lab.pk == 2)]
        assert lab.nfst_r2.sum() == 1 and lab.nchat_r2.sum() == 1
    return con, reach


def _open_green_gap():
    """the signal's log ends in a green of phase 2 with starts in it and a comms gap after them: the piece is invalid, so
    its starts do not count as starts in green."""
    T = 30000.0
    rows = [("S00", _at(T - 1), 81, 1), ("S00", _at(T), 1, 2)]
    rows += [("S00", _at(T + x), e, 1) for x, e in ((1.0, 82), (1.2, 81), (3.0, 82), (3.2, 81))]
    rows += [("S00", _at(T + 4), 131, 1), ("S00", _at(T + 304), 131, 2)]
    con = _scene_con(rows, lambda ev: (ev.DeviceId == "S00") & (ev.Timestamp >= _at(T)), _at(T + 350))

    def reach():
        x = G.steps(H, con, ["fpc", "xd5l", "win"])
        f, lab = x["fpc"][0], x["xd5l"][0]
        assert f[(f.dev == _dev(x)) & (f.ph == 2) & (f.st == 0)].valid.tolist() == [False]
        assert lab[(lab.dev == _dev(x)) & (lab.det == 1) & (lab.pk == 0)].nst_gy.sum() == 2
    return con, reach


def _open_green_chain(tail, green=200.0):
    """Count zone 17 ON from 01:30 with no OFF, its phase's green open from green seconds before the signal's last event, then
    tail seconds of silence to the window end: the overlap runs on to the window end, unless the chain counts as lost
    (> 120 s).  With green under 5 s its 5th second comes after the last event, and the time after 5 s starts there."""
    e = 9300.0
    last = e - tail
    rows = [("S00", _at(5400), 82, 17), ("S00", _at(last - green), 1, 2), ("S00", _at(last), 131, 3)]

    def drop(ev):
        s00 = ev.DeviceId == "S00"
        return (s00 & (ev.Timestamp >= _at(last))) | _dets(ev, [17], _at(5300), _at(e + 100)) | (
            s00 & ev.EventId.isin(COLOUR) & (ev.Parameter == 2) & (ev.Timestamp >= _at(last - green)))
    con = _scene_con(rows, drop, _at(e))

    def reach():
        x = G.steps(H, con, ["fov", "gonb", "win"])
        g, f = x["gonb"][0], x["fov"][0]
        assert (g[(g.dev == _dev(x)) & (g.det == 17)].gon > 0).any() == (tail <= 120)
        assert ((f.dev == _dev(x)) & (f.det == 17)).any() == (tail <= 120)
    return con, reach


def _lost_chain_after_held():
    """Count zone 17 held ON 01:30-02:30 (a held-ON zone), then a chain still open when the signal goes silent for 400 s: lost,
    so its time inside the greens it overlapped does not count (gon stays empty there)."""
    last, e = 11000.0, 11400.0
    rows = [("S00", _at(5400), 82, 17), ("S00", _at(9000), 81, 17), ("S00", _at(10000), 82, 17), ("S00", _at(last), 131, 3)]
    con = _scene_con(rows, lambda ev: ((ev.DeviceId == "S00") & (ev.Timestamp >= _at(last))) | _dets(ev, [17], _at(5300), _at(e + 100)), _at(e))

    def reach():
        x = G.steps(H, con, ["xd5l", "gonb", "win"])
        lab, g = x["xd5l"][0], x["gonb"][0]
        lab = lab[(lab.dev == _dev(x)) & (lab.det == 17) & lab.ctag.notna()]
        assert lab.ong5_ns.sum() > 0
        g = g[(g.dev == _dev(x)) & (g.det == 17)]
        assert g.k.max() >= 10000 // 900 and g[g.k >= 10100 // 900].gon.isna().all()
    return con, reach


def _open_chain_at_gap():
    """a chain whose ON is the signal's last event before a comms gap, still open at the end of the log: the gap starts at its
    start, so it counts as over the gap (unknown time)."""
    T = 30000.0

    def drop(ev):
        s00 = ev.DeviceId == "S00"
        return (s00 & (ev.Timestamp >= _at(T)) & (ev.Timestamp < _at(T + 300))) | _dets(ev, [1], _at(T - 100), _at(86400))
    con = _scene_con([("S00", _at(T - 101), 81, 1), ("S00", _at(T), 82, 1)], drop)

    def reach():
        x = G.steps(H, con, ["xoc", "win"])
        o = x["xoc"][0]
        o = o[(o.dev == _dev(x)) & (o.det == 1)]
        assert len(o) == 1 and o.mk.iloc[0] == o.s.iloc[0]
    return con, reach


def _phase_with_two_greens():
    """phase 8 has only 2 greens, with starts in a green and fast and chatter starts in a red's first 2 s of detector 6: phases
    under 3 greens have no colour timeline, so none of them counts as such."""
    rows = []
    for t in (3600, 7200):
        rows += [("S00", _at(t), 1, 8), ("S00", _at(t + 20), 8, 8), ("S00", _at(t + 24), 9, 8)]
    rows += [("S00", _at(3605 + x), e, 6) for x, e in ((0, 82), (1, 81))]
    rows += [("S00", _at(3624 + x), e, 6) for x, e in ((0.1, 82), (0.2, 81), (0.7, 82), (0.8, 81), (0.9, 82), (1.0, 81))]

    def drop(ev):
        return ((ev.DeviceId == "S00") & ev.EventId.isin(COLOUR) & (ev.Parameter == 8)) | _dets(ev, [6], _at(3590), _at(3640))
    con = _scene_con(rows, drop)

    def reach():
        x = G.steps(H, con, ["d5", "phs3", "win"])
        p, d = x["phs3"][0], x["d5"][0]
        assert not ((p.dev == _dev(x)) & (p.ph == 8)).any()
        d = d[(d.dev == _dev(x)) & (d.det == 6)]
        assert d.nst_gy.sum() >= 1 and d.nfst_r2.sum() >= 1 and d.nchat_r2.sum() >= 1
    return con, reach


def _short_green_in_a_held_zone():
    """a held-ON Count zone's phase with, in one 15-min bin, only a green under 5 s: green and yellow time but none after the
    first 5 s, so that bin has no row in gonb."""
    rows = [("S00", _at(5400), 82, 17), ("S00", _at(9000), 81, 17)]
    rows += [("S00", _at(14300), 8, 2), ("S00", _at(14300.5), 9, 2)]
    rows += [("S00", _at(14500 + x), e, 2) for x, e in ((0, 1), (3, 8), (7, 9))]

    def drop(ev):
        return _dets(ev, [17], _at(5300), _at(86400)) | ((ev.DeviceId == "S00") & ev.EventId.isin(COLOUR) & (ev.Parameter == 2)
                                                         & (ev.Timestamp >= _at(14300)) & (ev.Timestamp < _at(15400)))
    con = _scene_con(rows, drop)

    def reach():
        x = G.steps(H0, con, ["gonb", "win"])
        y = G.steps(H, con, ["p5", "win"])["p5"][0]
        g = x["gonb"][0]
        assert not ((g.dev == _dev(x)) & (g.det == 17) & (g.k == 16)).any() and ((g.dev == _dev(x)) & (g.det == 17) & (g.k == 15)).any()
        y = y[(y.dev == _dev(x)) & (y.ph == 2) & (y.b // 3 == 16)]
        assert y.gy_ns.notna().any() and y.g5_ns.isna().all()
    return con, reach


def _chain_in_an_invalid_open_green():
    """a held-ON Count zone with a closed chain inside the open green the log ends in, a comms gap after it: the piece is
    invalid, so that time does not count (gon stays empty there)."""
    T = 12000.0
    rows = [("S00", _at(5400), 82, 17), ("S00", _at(9000), 81, 17), ("S00", _at(T), 1, 2), ("S00", _at(T + 10), 82, 17),
            ("S00", _at(T + 30), 81, 17), ("S00", _at(T + 40), 131, 1), ("S00", _at(T + 340), 131, 2)]
    con = _scene_con(rows, lambda ev: ((ev.DeviceId == "S00") & (ev.Timestamp >= _at(T))) | _dets(ev, [17], _at(5300), _at(T)), _at(T + 350))

    def reach():
        x = G.steps(H, con, ["fpc", "xd5l", "gonb", "win"])
        f, lab, g = x["fpc"][0], x["xd5l"][0], x["gonb"][0]
        assert f[(f.dev == _dev(x)) & (f.ph == 2) & (f.st == 0)].valid.tolist() == [False]
        assert lab[(lab.dev == _dev(x)) & (lab.det == 17) & lab.ptag.notna()].ong5_ns.sum() == 20 * 10 ** 9
        assert g[(g.dev == _dev(x)) & (g.det == 17) & (g.k == T // 900)].gon.isna().tolist() == [True]
    return con, reach


def _fast_ons_only():
    """a detector's only ONs: 3 fast ones in a row, so every fast ON with a next ON has a fast next gap (pair_share 0, not
    empty)."""
    t = 3 * 3600
    rows = [("S00", _at(t + k * 0.6 + x), e, 2) for k in range(3) for x, e in ((0, 82), (0.3, 81))]
    con = _scene_con(rows, lambda ev: _dets(ev, [2], _at(0), _at(86400)))

    def reach():
        x = G.steps(H0, con, ["tf", "win"])
        t = x["tf"][0]
        assert t[(t.dev == _dev(x)) & (t.det == 2)].pair_share.tolist() == [0.0]
    return con, reach


def _no_length_on_in_green():
    """Count zone 17 held ON 01:30-02:30 (a held-ON zone), then later only an ON and an OFF at one instant, inside a green
    after its first 5 s: that bin has time ON 0 (gon 0.0), not none; at the green's 5th second exactly it has none."""
    ev = _scene_base()[0]
    g = ev[(ev.DeviceId == "S00") & (ev.EventId == 1) & (ev.Parameter == 2) & (ev.Timestamp >= _at(4 * 3600))].Timestamp.sort_values()
    inside, at_a5 = g.iloc[0] + pd.Timedelta(seconds=7), g.iloc[40] + pd.Timedelta(seconds=5)
    rows = [("S00", _at(5400), 82, 17), ("S00", _at(9000), 81, 17)]
    rows += [("S00", t, e, 17) for t in (inside, at_a5) for e in (82, 81)]
    con = _scene_con(rows, lambda e: _dets(e, [17], _at(5300), _at(86400)))

    def reach():
        x = G.steps(H0, con, ["gonb", "win"])
        w = x["win"][0].set_index("DeviceId").loc["S00"]
        got = x["gonb"][0]
        got = got[(got.dev == w.dev) & (got.det == 17)].set_index("k").gon
        k = lambda t: int((t.value - int(w.s_ns)) // (int(w.bs) * 10 ** 9))
        assert got[k(inside)] == 0.0 and pd.isna(got[k(at_a5)])
    return con, reach


SCENES = {"re-logged chain, two OFFs, start": _relogged_two_offs,
          "open red, gap after its first 2 s": lambda: _open_red(5.0),
          "open red, gap inside its first 2 s": lambda: _open_red(1.0),
          "open green, gap after its starts": _open_green_gap,
          "open green and chain, tail 40 s": lambda: _open_green_chain(40.0),
          "open green and chain, tail 110 s": lambda: _open_green_chain(110.0),
          "open green and chain, lost": lambda: _open_green_chain(400.0),
          "open green 2 s before the last event": lambda: _open_green_chain(40.0, 2.0),
          "chain lost after a held zone": _lost_chain_after_held,
          "open chain starting at a comms gap": _open_chain_at_gap,
          "phase with two greens": _phase_with_two_greens,
          "short green in a held zone's bin": _short_green_in_a_held_zone,
          "chain in an invalid open green": _chain_in_an_invalid_open_green,
          "only fast ONs": _fast_ons_only,
          "ON of no length in a green": _no_length_on_in_green}




def test_capped_medians_compared_below_the_cap():
    """the on_us tick counts keep lengths exactly up to 1 s, so the median ON is exact up to 0.5 s: every comparison of
    dur_p50 / pmed in the judge must be with a limit below that (design 4.1)."""
    text = "".join(p.read_text(encoding="utf-8") for p in (render.SQL_DIR / "judge").glob("*.sql"))
    uses = re.findall(r"\b(dur_p50|pmed)\b(?:, \d+\))?\s*(<=|<|>=|>)\s*(\{\{[^}]*\}\}|[\d.]+)", text)
    assert len(uses) >= 3, uses
    for _, _, lim in uses:
        assert (H.PULSE["s"] if "PULSE" in lim else float(lim)) < 0.5, lim
