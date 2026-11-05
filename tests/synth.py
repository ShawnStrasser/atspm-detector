"""Seeded synthetic controller logs for the health tests and the benchmark (stage 0 of the incremental-health design).

    ev, det = make(n_signals=3, days=2, seed=0)      # events (DeviceId, Timestamp, EventId, Parameter), detectors table

Each signal: an 8-phase ring-barrier controller (1+5 | 2+6 || 3+7 | 4+8; begin green 1, green termination 7, yellow 8 /
9, red clearance 10 / 11) under time-of-day plans (131 at each change); vehicles arrive as a Poisson process with a daily
profile and actuate presence (stop bar: held while queued), advance and count (pulse) detectors; ped inputs 2 / 4 / 6 /
8 (presses 90 / 89, call 45, walk 21 / 22 / 23 at the next green) and input 9 served by ped overlap 1 (67 / 68 with
phase 2); preemptions (102 / 105 ... 104 / 111).  With faults=True each signal gets its own share of:
  stuck ON (re-logged ONs, then the OFF hours later), a silent advance detector, chatter (ON / OFF within 0.1 s), a lane
  closure (one lane's vehicles move to the other lane), double-counting (too-fast actuations), comms gaps (> 120 s with
  nothing logged), log holes (40 s), an extra plan change, a stuck ped button, a dead ped input, orphan clearances
  (walk 21 lost), dropped ped calls, a preemption with no logged end, duplicate rows and an out-of-order block.
Times are on the controller's 0.1-s ticks.  The same seed always gives the same log.  edges(ev, det) adds edge patterns
of the detector logic to such a log (stage 2-3 tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd

PHASES = (1, 2, 3, 4, 5, 6, 7, 8)
# share of the cycle's green time; barrier A = 1+2 / 5+6, barrier B = 3+4 / 7+8 (both rings alike)
SPLIT = {1: 0.10, 2: 0.42, 3: 0.08, 4: 0.20, 5: 0.10, 6: 0.42, 7: 0.08, 8: 0.20}
Y, RC = 4.0, 2.0                          # yellow, red clearance (s)
# plans by hour: (from hour, plan, cycle s)
PLANS = ((0, 1, 80.0), (6, 2, 120.0), (9, 3, 100.0), (15, 2, 120.0), (19, 3, 100.0), (22, 1, 80.0))
# detectors: channel -> (phase, function, lane); veh/h per lane at the peak by phase
DETS = {1: (2, "Presence", 1), 2: (2, "Presence", 2), 9: (2, "Advance", 0), 17: (2, "Count", 0),
        3: (6, "Presence", 1), 4: (6, "Presence", 2), 10: (6, "Advance", 0), 18: (6, "Count", 0),
        5: (4, "Presence", 1), 6: (8, "Presence", 1), 7: (1, "Presence", 1), 8: (5, "Presence", 1),
        11: (3, "Presence", 1), 12: (7, "Presence", 1)}
PEAK = {1: 120, 2: 650, 3: 90, 4: 260, 5: 120, 6: 650, 7: 90, 8: 260}
LANES = {2: 2, 6: 2}
PROBS = {"Advance": "p_advance", "Presence": "p_presence", "Count": "p_count", "Yellow_Red": "p_yellow_red",
         "Other": "p_other", "Mid": "p_mid", "Bike": "p_bike"}


def _profile(h):
    """the daily traffic profile (0-1) at hour of day h (float)."""
    am, pm = np.exp(-0.5 * ((h - 8.0) / 1.5) ** 2), np.exp(-0.5 * ((h - 17.0) / 2.0) ** 2)
    day = np.clip((h - 5.0) / 3.0, 0, 1) * np.clip((23.0 - h) / 3.0, 0, 1)
    return 0.04 + 0.35 * day + 0.6 * np.maximum(am, pm)


def _poisson(rng, t0, t1, rate_fn):
    """arrival times in [t0, t1) of a Poisson process with rate rate_fn(t) per second (5-min steps)."""
    out = []
    for a in np.arange(t0, t1, 300.0):
        n = rng.poisson(rate_fn(a + 150.0) * 300.0)
        out.append(a + rng.random(n) * 300.0)
    return np.sort(np.concatenate(out)) if out else np.zeros(0)


def _timing(rng, t0, days, night_skip=0.35):
    """the controller's phase intervals: {phase: (green start, green end)} and its events (t, eid, par)."""
    ev, green = [], {p: [] for p in PHASES}
    t, end = t0, t0 + days * 86400.0
    last_plan = None
    while t < end:
        h = ((t - t0) % 86400.0) / 3600.0
        plan, cyc = [(p, c) for f, p, c in PLANS if h >= f][-1]
        if plan != last_plan:
            ev.append((t, 131, plan))
            last_plan = plan
        night = h < 5.0 or h >= 23.0
        for barrier in (((1, 2), (5, 6)), ((3, 4), (7, 8))):
            gs = {}
            for left, thru in barrier:                     # left phases skip at night now and then
                gs[left] = 0.0 if night and rng.random() < night_skip else max(5.0, SPLIT[left] * cyc - Y - RC)
                gs[thru] = max(7.0, SPLIT[thru] * cyc - Y - RC)
            blen = max(sum(gs[p] + Y + RC for p in ring if gs[p] > 0) for ring in barrier)
            for ring in barrier:
                s = t
                for i, p in enumerate(ring):
                    g = gs[p]
                    if g <= 0:
                        continue
                    if i == 1:                              # the through phase ends at the barrier
                        g = t + blen - s - Y - RC
                    green[p].append((s, s + g))
                    ev += [(s, 1, p), (s + g, 7, p), (s + g, 8, p), (s + g + Y, 9, p), (s + g + Y, 10, p),
                           (s + g + Y + RC, 11, p)]
                    s += g + Y + RC
            t += blen
    return {p: np.array(v).reshape(-1, 2) for p, v in green.items()}, ev


def _intervals(on, off):
    """union of [on, off) intervals -> sorted, merged arrays."""
    if len(on) == 0:
        return on, off
    o = np.argsort(on, kind="stable")
    on, off = on[o], off[o]
    new = np.r_[True, on[1:] > np.maximum.accumulate(off)[:-1]]
    return on[new], np.maximum.reduceat(off, np.flatnonzero(new))


def _next_green(g, t):
    """for each time t: (start, end) of the green of phase intervals g in progress at t or the next one (NaN: none)."""
    if len(g) == 0:
        return np.full(len(t), np.nan), np.full(len(t), np.nan)
    i = np.searchsorted(g[:, 1], t, side="right")
    ok = i < len(g)
    s, e = np.full(len(t), np.nan), np.full(len(t), np.nan)
    s[ok], e[ok] = g[i[ok], 0], g[i[ok], 1]
    return s, e


def _signal(rng, t0, days, faults):
    green, ev = _timing(rng, t0, days)
    T1 = t0 + days * 86400.0
    rate = {p: (lambda t, p=p: PEAK[p] / 3600.0 * _profile(((t - t0) % 86400.0) / 3600.0)) for p in PHASES}
    arr = {p: _poisson(rng, t0, T1, rate[p]) for p in PHASES}
    # fault plan of this signal (times as offsets of day 0 / day 1)
    d1 = t0 + (86400.0 if days > 1 else 0.0)
    F = {}
    if faults:
        F = {"closure": (d1 + 10 * 3600.0, d1 + 13 * 3600.0), "silent": (t0 + 11 * 3600.0, t0 + 15 * 3600.0),
             "stuck": (d1 + 7 * 3600.0 + rng.integers(0, 1800), 2.5 * 3600.0),
             "double": (t0 + 16 * 3600.0, t0 + 19 * 3600.0)}
    # per detector: ON / OFF intervals
    det_ev = []
    for ch, (ph, fn, lane) in DETS.items():
        a = arr[ph]
        if lane:                                         # this lane's share of the vehicles
            nl = LANES.get(ph, 1)
            pick = rng.integers(0, nl, len(a)) == (lane - 1) if nl > 1 else np.ones(len(a), bool)
            if "closure" in F and ph == 2:               # lane 1 closed: its vehicles take lane 2
                c0, c1 = F["closure"]
                inside = (a >= c0) & (a < c1)
                pick = np.where(inside, lane == 2, pick)
            a = a[pick]
        if fn == "Advance":
            on = a - 6.0 + rng.uniform(-0.3, 0.3, len(a))
            off = on + rng.uniform(0.25, 0.6, len(a))
            if "double" in F and ch == 9:                # each vehicle counted twice (too-fast actuations)
                d0, d1_ = F["double"]
                m = (on >= d0) & (on < d1_)
                on2 = on[m] + 0.3
                on, off = np.concatenate([on, on2]), np.concatenate([off, on2 + 0.1])
        elif fn == "Count":
            on = a + 1.5 + rng.uniform(0, 0.4, len(a))
            off = on + rng.uniform(0.1, 0.15, len(a))
        else:                                            # presence at the stop bar: held while queued
            gs, ge = _next_green(green[ph], a)
            in_green = (a >= gs) & (a < ge)
            q = ~in_green & ~np.isnan(gs)
            on = a.copy()
            off = np.where(in_green, a + rng.uniform(0.4, 1.0, len(a)), np.nan)
            if q.any():                                  # queue: departs at green start + its place x 2 s
                key = gs[q]
                order = np.argsort(key, kind="stable")
                k = key[order]
                place = np.arange(len(k)) - np.searchsorted(k, k, side="left")
                dep = np.empty(len(k))
                dep[order] = k + 1.5 + 2.0 * place
                off[q] = dep
            off = np.where(np.isnan(off), a + 0.8, off)
        on, off = _intervals(np.round(on, 1), np.round(np.maximum(off, on + 0.1), 1))
        if fn == "Count" and faults and ch == 18:       # chatter: a second short pulse 0.1 s after some OFFs
            m = rng.random(len(off)) < 0.15
            c_on = off[m] + 0.1
            det_ev += [(t, 82, ch) for t in c_on] + [(t, 81, ch) for t in c_on + 0.1]
        if "silent" in F and ch == 10:
            s0, s1 = F["silent"]
            keep = (on < s0) | (on >= s1)
            on, off = on[keep], off[keep]
        if "stuck" in F and ch == 3:                     # stuck ON with re-logged ONs every 10 min, then the OFF
            s0, dur = F["stuck"]
            keep = (off < s0) | (on > s0 + dur)
            on, off = on[keep], off[keep]
            det_ev += [(s0 + k * 600.0, 82, ch) for k in range(int(dur // 600) + 1)] + [(s0 + dur, 81, ch)]
        det_ev += [(t, 82, ch) for t in on] + [(t, 81, ch) for t in off]
    ev += det_ev
    ev += _ped(rng, t0, T1, green, faults)
    return ev


def _ped(rng, t0, T1, green, faults):
    ev = []
    stuck = (t0 + 13 * 3600.0, 2 * 3600.0) if faults else None
    for inp, ph in ((2, 2), (4, 4), (6, 6), (8, 8), (9, 2)):
        day = 8.0 if inp == 4 else 3.0                   # presses an hour by day (07-20 h), 0.4 at night
        rate = (lambda t, day=day: (day if 7 <= ((t - t0) % 86400.0) / 3600.0 < 20 else 0.4) / 3600.0)
        press = _poisson(rng, t0, T1, rate)
        if faults and inp == 4:                          # dead: no presses 08-17 h on day 0
            press = press[(press < t0 + 8 * 3600.0) | (press >= t0 + 17 * 3600.0)]
        if faults and inp == 6:                          # stuck button: ON for hours (no other press meanwhile)
            press = press[(press < stuck[0] - 1.0) | (press > stuck[0] + stuck[1] + 1.0)]
        press = np.round(press, 1)
        rel = np.round(press + rng.uniform(0.1, 0.4, len(press)), 1)
        ev += [(t, 90, inp) for t in press] + [(t, 89, inp) for t in rel]
        if faults and inp == 6:
            ev += [(stuck[0], 90, inp), (stuck[0] + stuck[1], 89, inp)]
            press = np.sort(np.append(press, stuck[0]))
        # served at the next green start of the phase: call 45 (phase inputs), walk 21 / 22 / 23 or overlap walk 67 / 68
        starts = green[ph][:, 0]
        k = np.searchsorted(starts, press + 0.05, side="right")          # the next green start after the call
        gs = np.where(k < len(starts), starts[np.minimum(k, len(starts) - 1)], np.nan)
        served = set()
        for t, g in zip(press, gs):
            if np.isnan(g):
                continue
            if inp != 9:
                ev.append((round(t + 0.05, 1), 45, ph))
            if faults and inp == 8 and rng.random() < 0.08:   # call dropped: no walk for 3 greens, then called again
                k = np.searchsorted(green[ph][:, 0], g)
                g = green[ph][k + 3, 0] if k + 3 < len(green[ph]) else np.nan
                if not np.isnan(g):
                    ev.append((round(g - 20.0, 1), 45, ph))
            if np.isnan(g) or g in served:
                continue
            served.add(g)
            if inp == 9:
                ev += [(g, 67, 1), (g + 7.0, 68, 1)]
            else:
                lost = faults and rng.random() < 0.05          # orphan clearance: the 21 never logged
                ev += ([] if lost else [(g, 21, ph)]) + [(g + 7.0, 22, ph), (g + 17.0, 23, ph)]
    # preemptions: call input on 102, entry 105, input off 104, exit 111 (number 1); one with no logged end
    for k, t in enumerate(np.arange(t0 + 5 * 3600.0, T1, 7.3 * 3600.0)):
        t = round(float(t + rng.integers(0, 600)), 1)
        ev += [(t, 102, 1), (t + 2.0, 105, 1)]
        if not (faults and k == 1):
            ev += [(t + 75.0, 104, 1), (t + 80.0, 111, 1)]
    return ev


def make(n_signals: int = 3, days: int = 2, seed: int = 0, start: str = "2026-10-05", faults: bool = True):
    """(events, detectors) of n_signals synthetic signals over `days` days from `start` (local midnight)."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start).value / 1e9
    parts, dets = [], []
    for s in range(n_signals):
        dev = f"S{s:02d}"
        ev = pd.DataFrame(_signal(rng, t0, days, faults), columns=["t", "EventId", "Parameter"])
        ev = ev[(ev.t >= t0) & (ev.t < t0 + days * 86400.0)]
        if faults:
            for g0 in (t0 + 6.5 * 3600.0 + s * 900.0, t0 + days * 86400.0 - 9 * 3600.0):    # comms gaps of 6 min
                ev = ev[(ev.t < g0) | (ev.t >= g0 + 360.0)]
            for h0 in t0 + rng.uniform(0, days * 86400.0 - 60.0, 4):                       # log holes of 40 s
                ev = ev[(ev.t < h0) | (ev.t >= h0 + 40.0)]
            ev = pd.concat([ev, pd.DataFrame({"t": [t0 + 14.25 * 3600.0], "EventId": [131], "Parameter": [9]})])
        ev = ev.assign(DeviceId=dev)
        parts.append(ev)
        for ch, (ph, fn, lane) in DETS.items():
            row = {"DeviceId": dev, "Detector": ch, "phase": float(ph), "function": fn, "lanes": 1.0}
            row.update({c: (0.9 if f == fn else 0.1 / 6) for f, c in PROBS.items()})
            dets.append(row)
    ev = pd.concat(parts, ignore_index=True)
    ev["t"] = np.round(ev.t.to_numpy(), 1)
    ev = ev.sort_values(["DeviceId", "t", "EventId", "Parameter"], kind="stable").reset_index(drop=True)
    if faults:                                           # exact duplicates, and an out-of-order block of rows
        dup = ev.sample(frac=0.002, random_state=seed)
        ev = pd.concat([ev, dup], ignore_index=True)
        n = len(ev)
        blk = np.arange(n // 3, n // 3 + 500)
        ev.iloc[blk] = ev.iloc[blk[::-1]].to_numpy()
    ts = pd.to_datetime((ev.t.to_numpy() * 10).round().astype("int64") * 100, unit="ms")
    out = pd.DataFrame({"DeviceId": ev.DeviceId.to_numpy(), "Timestamp": ts,
                        "EventId": ev.EventId.astype("int32").to_numpy(),
                        "Parameter": ev.Parameter.astype("int32").to_numpy()})
    return out, pd.DataFrame(dets)


def edges(ev: pd.DataFrame, det: pd.DataFrame):
    """the log of make(3, 2, seed, faults=True) with edge patterns of the detector logic added (stage 2-3 of the
    incremental-health design): a Count zone held ON for 100 min at night (held-ON check, a chain over many greens), OFF
    OFF before an ON and an ON before OFF OFF, a detector with only OFFs, a start right after re-logged ONs, plan changes
    at both ends of a silence (timing-plan switch), a chain still open at the end, a phase with two greens; signal S01
    silent for its last 10 min with a chain open (lost) and a Presence zone pulsing; on S02 a whole phase silent for
    3 h on day 1 while its greens go on."""
    t0 = ev.Timestamp.min().normalize()
    S = lambda h, m=0, s=0.0: t0 + pd.Timedelta(hours=h, minutes=m, seconds=s)
    drop = lambda d, chans, a, b: ~((ev.DeviceId == d) & ev.EventId.isin([81, 82]) & ev.Parameter.isin(chans)
                                    & (ev.Timestamp >= a) & (ev.Timestamp < b))
    rows = []
    d = "S00"
    # det 17 (Count, phase 2) held ON 01:30-03:10 (count zone held ON in light traffic: hicd; a chain over many greens)
    ev = ev[drop(d, [17], S(1, 29), S(3, 11))]
    rows += [(d, S(1, 30), 82, 17), (d, S(3, 10), 81, 17)]
    # det 5: OFF OFF before an ON, an ON before OFF OFF (rare neighbour cases); det 30 only OFFs (a leading OFF)
    ev = ev[drop(d, [5], S(3, 59), S(4, 2))]
    rows += [(d, S(4, 0, 0.0), 82, 5), (d, S(4, 0, 0.5), 81, 5), (d, S(4, 0, 0.9), 81, 5), (d, S(4, 0, 1.3), 82, 5),
             (d, S(4, 0, 1.4), 81, 5), (d, S(4, 0, 1.5), 81, 5), (d, S(4, 0, 1.6), 82, 5), (d, S(4, 0, 1.7), 81, 5)]
    rows += [(d, S(2, 0, 1.0), 81, 30), (d, S(9, 0, 1.0), 81, 30)]
    # re-logged ONs right before a start (pe=81, ppe=82, p3e=82) on det 6
    ev = ev[drop(d, [6], S(4, 29), S(4, 32))]
    rows += [(d, S(4, 30, 0.0), 82, 6), (d, S(4, 30, 0.4), 82, 6), (d, S(4, 30, 0.6), 81, 6), (d, S(4, 30, 0.9), 82, 6),
             (d, S(4, 30, 1.2), 81, 6)]
    # plan changes at both ends of det 10's silence (11-15 h): a timing-plan switch
    rows += [(d, S(11, 1), 131, 4), (d, S(14, 59), 131, 5)]
    # a chain still open at the end (det 11 ON for the last 40 min); a lone phase 16 with two greens
    end = ev[ev.DeviceId == d].Timestamp.max()
    ev = ev[drop(d, [11], end - pd.Timedelta(minutes=41), end + pd.Timedelta(hours=1))]
    rows += [(d, end - pd.Timedelta(minutes=40), 82, 11)]
    rows += [(d, S(6, 0), 1, 16), (d, S(6, 0, 20.0), 8, 16), (d, S(6, 0, 24.0), 9, 16),
             (d, S(7, 0), 1, 16), (d, S(7, 0, 20.0), 8, 16)]
    # S01: silent for its last 10 minutes with det 12 ON (a lost chain at the window end); det 7 (Presence) pulses
    e1 = ev[ev.DeviceId == "S01"].Timestamp.max()
    ev = ev[~((ev.DeviceId == "S01") & (ev.Timestamp > e1 - pd.Timedelta(minutes=10)))]
    ev = ev[drop("S01", [12], e1 - pd.Timedelta(minutes=13), e1)]
    rows += [("S01", e1 - pd.Timedelta(minutes=12), 82, 12)]
    m = (ev.DeviceId == "S01") & (ev.Parameter == 7) & ev.EventId.isin([81, 82])
    on = ev[m & (ev.EventId == 82)].Timestamp.sort_values()
    ev = ev[~m]
    rows += [("S01", t, 82, 7) for t in on] + [("S01", t + pd.Timedelta(milliseconds=100), 81, 7) for t in on]
    # S02: phase 6's detectors silent 10-13 h on day 1 while its greens go on (a whole phase drop)
    ev = ev[drop("S02", [3, 4, 10, 18], S(34), S(37))]
    add = pd.DataFrame(rows, columns=["DeviceId", "Timestamp", "EventId", "Parameter"])
    add["Timestamp"] = add.Timestamp.dt.round("100ms")
    add = add.astype({"EventId": "int32", "Parameter": "int32"})
    ev = pd.concat([ev, add], ignore_index=True).sort_values(["DeviceId", "Timestamp"], kind="stable").reset_index(drop=True)
    return ev, det
