"""The network's input side: the interval streams of one sample and the 1-s raster helpers.

The network reads the same events as the trees, as intervals on a millisecond timeline.  The definitions are exactly the
ones it was TRAINED on (research `build_cache.py` -> `neural/ncache2.py`):

    detector ON   event 82 -> the next 81 on that channel        (= predict's `onev_all`, read from it)
    green         event 1  -> event 8, else event 10, else the next event 1
    yellow        event 8  -> event 10, else event 11
    red clearance event 10 -> event 11
    call          event 43 -> the next 44 on that phase
    coordinated   event 131 with a pattern of 1..253, until the next 131
    candidates    every phase with a Begin Green (event 1) in the sample   (= predict's `cand`)

The colour cycles use events 1 / 8 / 10 / 11 only (no 7 / 9 fall-backs, unlike predict's `cyc_all`); that table
(`cyc5`) is built once per call and shared with the expert function features.  A sample longer than `CHUNK_MS` is cut
by `split_range` into pieces of at most 30 minutes, the window the network was trained on.  numpy, pandas and the
caller's DuckDB connection only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIN_MS = 1000                      # 1 s raster
CHUNK_MS = 30 * 60 * 1000          # a sample is scored in pieces of at most 30 minutes

# detector ON intervals: predict.build_chunk_tables' onev_all (82 -> next 81 on the channel, the training rule)
SQL_DET = """SELECT m.DeviceId, o.det::INT AS ch, o.t_on AS a, o.t_off AS b
FROM onev_all o JOIN devmap m USING (dev)"""
SQL_DET_CH = "SELECT DISTINCT m.DeviceId, o.det::INT AS ch FROM onev_all o JOIN devmap m USING (dev)"
SQL_CAND = "SELECT m.DeviceId, c.p::INT AS p FROM cand c JOIN devmap m USING (dev) ORDER BY m.DeviceId, p"

# colour cycles from events 1 / 8 / 10 / 11 (the network's and the expert features' definition), built once per call
SQL_CYC5_TABLE = """CREATE TEMP TABLE IF NOT EXISTS cyc5 AS
WITH g AS (
  SELECT dev, Parameter::SMALLINT AS p, Timestamp AS t, EventId,
         SUM(CASE WHEN EventId=1 THEN 1 ELSE 0 END) OVER (
             PARTITION BY dev, Parameter
             ORDER BY t, CASE EventId WHEN 1 THEN 0 WHEN 8 THEN 1 WHEN 10 THEN 2 ELSE 3 END
             ROWS UNBOUNDED PRECEDING) AS cyc
  FROM evd WHERE EventId IN (1,8,10,11) AND Parameter BETWEEN 1 AND 16
), c AS (
  SELECT dev, p, cyc, min(t) FILTER (EventId=1) AS green_start, min(t) FILTER (EventId=8) AS yellow_start,
         min(t) FILTER (EventId=10) AS red_start, min(t) FILTER (EventId=11) AS redclr_end
  FROM g WHERE cyc > 0 GROUP BY 1,2,3
)
SELECT *, LEAD(green_start) OVER (PARTITION BY dev, p ORDER BY green_start) AS next_green
FROM c WHERE green_start IS NOT NULL"""

SQL_CYC = """SELECT m.DeviceId, c.p::INT AS p,
       epoch_ms(c.green_start)/1000.0 AS g0,
       epoch_ms(coalesce(c.yellow_start, c.red_start, c.next_green))/1000.0 AS g1,
       epoch_ms(c.yellow_start)/1000.0 AS y0,
       epoch_ms(coalesce(c.red_start, c.redclr_end))/1000.0 AS y1,
       epoch_ms(c.red_start)/1000.0 AS r0,
       epoch_ms(c.redclr_end)/1000.0 AS r1
FROM cyc5 c JOIN devmap m USING (dev)"""

SQL_CALL = """
WITH e AS (SELECT DeviceId, Parameter::INT AS p, Timestamp AS ts, EventId
           FROM ev WHERE EventId IN (43,44) AND Parameter BETWEEN 1 AND 16),
     d AS (SELECT *, LEAD(ts) OVER w AS nts, LEAD(EventId) OVER w AS nev FROM e
           WINDOW w AS (PARTITION BY DeviceId, p
                        ORDER BY ts, CASE WHEN EventId=43 THEN 0 ELSE 1 END))
SELECT DeviceId, p, epoch_ms(ts)/1000.0 AS a, epoch_ms(nts)/1000.0 AS b
FROM d WHERE EventId = 43 AND nev = 44 AND nts IS NOT NULL"""

SQL_COORD = """
SELECT DeviceId, epoch_ms(Timestamp)/1000.0 AS a,
       epoch_ms(LEAD(Timestamp) OVER (PARTITION BY DeviceId ORDER BY Timestamp))/1000.0 AS b,
       (Parameter BETWEEN 1 AND 253) AS is_coord
FROM ev WHERE EventId = 131"""


def ensure_cyc5(con) -> None:
    """The shared 1 / 8 / 10 / 11 cycle table (needs predict's `evd`)."""
    con.execute(SQL_CYC5_TABLE)


def _ptr(keys: np.ndarray, order) -> np.ndarray:
    """Offsets of each channel of `order` in the channel-sorted `keys`."""
    ptr = np.zeros(len(order) + 1, dtype=np.int64)
    if len(keys):
        k = np.sort(np.asarray(keys))
        o = np.asarray(order)
        ptr[1:] = np.cumsum(np.searchsorted(k, o, "right") - np.searchsorted(k, o, "left"))
    return ptr.astype(np.int32)


def _ms(x, t0_ms: int, span: int) -> np.ndarray:
    v = np.rint(np.asarray(x, dtype=np.float64) * 1000.0 - t0_ms)
    return np.clip(v, -1000, span + 1000).astype(np.int32)


# an interval is fetched only if it reaches within this many seconds of a scored piece;
# the ones it drops cover none of the pieces' 1 s bins, so the raster is unchanged
_PIECE_MARGIN_S = 10.0   # > the 8 s minimum raster a very short piece is padded to


def _near(sql: str, windows, a: str, b: str, order: str) -> str:
    """`sql` restricted to rows whose [a, b] interval overlaps one of `windows` (intervals are defined over the whole
    sample first and only then filtered, so every kept interval is exactly the one the full build gives)."""
    if windows is None:
        return f"SELECT * FROM ({sql}) q ORDER BY {order}"
    m = _PIECE_MARGIN_S
    cond = " OR ".join(f"({a} < {w1 / 1000.0 + m!r} AND {b} > {w0 / 1000.0 - m!r})"
                       for w0, w1 in windows)
    return f"SELECT * FROM ({sql}) q WHERE {cond or 'FALSE'} ORDER BY {order}"


def build_streams(con, t0_ms: int, t1_ms: int, windows=None) -> dict:
    """-> {DeviceId: interval bundle} (all int32 milliseconds from `t0_ms`):

        cand    [K]     candidate phases (every phase with a Begin Green)
        det_ch  [D]     detector channels with a completed actuation in the sample
        det_ptr [D+1]   offsets into det_on / det_off;  det_on, det_off: detector ON intervals
        g_* / y_* / r_* / c_*   per candidate phase: green / yellow / red-clearance / call intervals (ptr, on, off)
        co_on, co_off   intervals during which the signal is coordinated

    `windows` (epoch-ms pairs, None = the whole sample) are the pieces the network will read: only intervals reaching
    them are fetched.  Candidates and detector channels still come from the whole sample."""
    span = int(t1_ms - t0_ms)
    ensure_cyc5(con)
    cands = con.sql(SQL_CAND).df()
    chans = con.sql(SQL_DET_CH).df() if windows is not None else None
    dets = con.sql(_near(SQL_DET, windows, "a", "b", "DeviceId, ch, a")).df()
    # a row's green / yellow / red intervals all lie between g0 and its latest end
    cyc = con.sql(_near(SQL_CYC, windows, "g0", "greatest(g1, y1, r1)", "DeviceId, p, g0")).df()
    calls = con.sql(_near(SQL_CALL, windows, "a", "b", "DeviceId, p, a")).df()
    # the last 131 runs to the end of the sample (b is NULL there)
    coord = con.sql(_near(SQL_COORD, windows, "a", "coalesce(b, 1e18)", "DeviceId, a")).df()

    gc = {k: v for k, v in cands.groupby("DeviceId", sort=False)}
    gd = {k: v for k, v in dets.groupby("DeviceId", sort=False)}
    gy = {k: v for k, v in cyc.groupby("DeviceId", sort=False)}
    gl = {k: v for k, v in calls.groupby("DeviceId", sort=False)}
    go = {k: v for k, v in coord.groupby("DeviceId", sort=False)}
    src_ch = dets if chans is None else chans
    gch = {k: np.sort(v.ch.astype(int).unique()) for k, v in src_ch.groupby("DeviceId")}

    out = {}
    for dev in sorted(set(cands.DeviceId) | set(gch)):
        c = gc.get(dev)
        cand = np.sort(c.p.astype(int).unique()) if c is not None else np.zeros(0, dtype=np.int64)
        z = {"cand": cand.astype(np.int16)}
        d, ch = gd.get(dev), gch.get(dev)
        if ch is None or not len(ch):
            z.update(det_ch=np.zeros(0, np.int16), det_ptr=np.zeros(1, np.int32),
                     det_on=np.zeros(0, np.int32), det_off=np.zeros(0, np.int32))
        elif d is None or not len(d):          # channels present, none reaching a piece
            z.update(det_ch=ch.astype(np.int16), det_ptr=np.zeros(len(ch) + 1, np.int32),
                     det_on=np.zeros(0, np.int32), det_off=np.zeros(0, np.int32))
        else:
            z["det_ch"] = ch.astype(np.int16)
            z["det_ptr"] = _ptr(d.ch.astype(int).to_numpy(), list(ch))
            z["det_on"] = _ms(d.a, t0_ms, span)
            z["det_off"] = _ms(d.b, t0_ms, span)
        p, l = gy.get(dev), gl.get(dev)
        for tag, src, c0, c1 in (("g", p, "g0", "g1"), ("y", p, "y0", "y1"),
                                 ("r", p, "r0", "r1"), ("c", l, "a", "b")):
            ons, offs, ptr = [], [], [0]
            if src is not None:
                pv = src.p.astype(int).to_numpy()
                A = pd.to_numeric(src[c0], errors="coerce").to_numpy(dtype=np.float64)
                B = pd.to_numeric(src[c1], errors="coerce").to_numpy(dtype=np.float64)
            for ph in cand:
                if src is None:
                    ptr.append(ptr[-1])
                    continue
                sel = pv == int(ph)
                a, b = A[sel], B[sel]
                ok = np.isfinite(a) & np.isfinite(b) & (b > a)
                a, b = a[ok], b[ok]
                o = np.argsort(a, kind="stable")
                ons.append(a[o])
                offs.append(b[o])
                ptr.append(ptr[-1] + int(len(a)))
            z[tag + "_on"] = _ms(np.concatenate(ons) if ons else np.zeros(0), t0_ms, span)
            z[tag + "_off"] = _ms(np.concatenate(offs) if offs else np.zeros(0), t0_ms, span)
            z[tag + "_ptr"] = np.array(ptr, dtype=np.int32)
        co = go.get(dev)
        if co is not None and len(co):
            k = co[co.is_coord.fillna(False).astype(bool)]
            a = _ms(k.a, t0_ms, span)
            b = _ms(k.b.fillna((t1_ms + 1000.0) / 1000.0), t0_ms, span)
            m = b > a
            z["co_on"], z["co_off"] = a[m], b[m]
        else:
            z["co_on"] = np.zeros(0, np.int32)
            z["co_off"] = np.zeros(0, np.int32)
        out[dev] = z
    return out


# ------------------------------------------------------------- raster pieces
def cover(on: np.ndarray, off: np.ndarray, w0: int, w1: int, T: int,
          bw: int = BIN_MS) -> np.ndarray:
    """Fraction of each 1 s bin covered by the sorted, disjoint intervals [on, off)."""
    out = np.zeros(T, dtype=np.float32)
    if on.size == 0:
        return out
    i0 = int(np.searchsorted(off, w0, "right"))
    i1 = int(np.searchsorted(on, w1, "left"))
    if i1 <= i0:
        return out
    a = (on[i0:i1].astype(np.float64) - w0) / bw
    b = (off[i0:i1].astype(np.float64) - w0) / bw
    np.clip(a, 0.0, T, out=a)
    np.clip(b, 0.0, T, out=b)
    cum = np.concatenate(([0.0], np.cumsum(b - a)))
    x = np.arange(T + 1, dtype=np.float64)
    j = np.searchsorted(a, x, "right")
    G = np.zeros(T + 1)
    m = j >= 1
    jm = j[m] - 1
    G[m] = cum[jm] + np.minimum(x[m], b[jm]) - a[jm]
    out[:] = np.diff(G)
    return np.clip(out, 0.0, 1.0)


def onrate(on: np.ndarray, w0: int, w1: int, T: int, bw: int = BIN_MS) -> np.ndarray:
    """Number of detector ON onsets in each bin."""
    if on.size == 0:
        return np.zeros(T, dtype=np.float32)
    i0 = int(np.searchsorted(on, w0, "left"))
    i1 = int(np.searchsorted(on, w1, "left"))
    if i1 <= i0:
        return np.zeros(T, dtype=np.float32)
    idx = ((on[i0:i1].astype(np.int64) - w0) // bw).astype(np.int64)
    return np.bincount(idx, minlength=T)[:T].astype(np.float32)


def split_range(t0_ms: int, t1_ms: int, chunk_ms: int = CHUNK_MS,
                max_chunks: int = 0, grid: int = 0) -> list[tuple[int, int]]:
    """Cut [t0, t1) into pieces of at most 30 minutes (the network's native window).

    A period shorter than that stays one short piece.  `grid` (0 = off) first thins the pieces to at most `grid` evenly
    spaced ones, then `max_chunks` thins that list the same way -- the two-step placement the long-sample evaluation
    used (note 33: a 32-piece grid, then `linspace(0, n - 1, K).round()`)."""
    total = max(int(t1_ms) - int(t0_ms), 1000)
    if total <= chunk_ms:
        return [(int(t0_ms), int(t1_ms))]
    n = total // chunk_ms
    rem = total - n * chunk_ms
    out = [(int(t0_ms) + i * chunk_ms, int(t0_ms) + (i + 1) * chunk_ms) for i in range(n)]
    if rem >= 60_000:
        out.append((int(t0_ms) + n * chunk_ms, int(t1_ms)))
    if grid and len(out) > grid:
        idx = np.linspace(0, len(out) - 1, grid).round().astype(int)
        out = [out[i] for i in idx]
    if max_chunks and len(out) > max_chunks:
        idx = np.linspace(0, len(out) - 1, max_chunks).round().astype(int)
        out = [out[i] for i in idx]
    return out
