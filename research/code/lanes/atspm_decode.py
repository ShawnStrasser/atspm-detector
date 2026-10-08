"""Per-lane ATSPM decode (note 54; user rule 2026-09-30 "ATSPM classes, one per lane"). numpy only.

Post-processing on the function head's class probabilities and the lane output (lane_output.py, note 42):
within one predicted phase, at most ONE detector per ATSPM class (Advance, Presence, Count, Yellow_Red) per lane.
Other / Mid / Bike (and any future ETA class) may repeat.

    decode_group(P, lanes, mode="greedy", span="strict", pick=None) -> class index per detector

P      (n, 7) probabilities, column order C7.
lanes  list of frozensets of lane numbers per detector (empty = no lane known: < 10 actuations, Bike, or the lane
       output put the detector on another phase) -> the detector is unconstrained (keeps its argmax, blocks nobody).
mode   "greedy"  (detector, class) pairs taken in order of probability; a detector whose ATSPM class is taken in one of
                 its lanes moves on to its next pair (which may be another, still free ATSPM class, e.g. YR after Count);
       "nonatspm" the loser falls straight to its best non-ATSPM class (the user's wording);
       "exact"   maximises the sum of log-probabilities under the constraint (branch and bound; small groups).
span   how a detector listed on several lanes is treated:
       "strict"  it holds its ATSPM class on EVERY lane it spans (one Yellow_Red bar across both lanes excludes a
                 per-lane Yellow_Red; a multi-lane radar advance zone excludes loops of that class underneath);
       "single"  spanning detectors are unconstrained (they neither block nor are blocked) - guards against the lane
                 decode's spanning errors (spanning precision .80-.86, recall .60-.71 in note 42).
pick   (note 55, user rule 2026-09-30) which member of a stacked same-role group gets the ATSPM class; hi-res only
       (lanes/ln6_pick.py), no technology or print fact. dict of per-detector arrays:
         unhealthy  bool    health_core v5 on the sample says bad / suspect -> (a) it loses
         span_lanes list of frozensets or None: the detector behaves like one zone spanning several lanes (its ONs =
                    the union of >= 2 non-co-located detectors' ONs, e.g. a radar advance zone over lane-by-lane loops)
                    -> (b) it is not a lane of its own: it is put on all its covered detectors' lanes and the
                    lane-by-lane detectors win over it (Advance / Presence / Count only; a Yellow_Red bar may span)
         track      float   correlation of its counts with the phase's (lane's) predicted-Count total; NaN = no Count
                    -> (c) among Advance claimants the one that tracks the stop-bar counts best wins
       With pick, every detector's FIRST-choice ATSPM claim is settled before any second choice, in the order
       (healthy, not spanning, track [Advance only], probability); the losers then continue as in "greedy"
       (next free class, user-approved). Optional "stack": per detector the in-group indices it is stacked with
       (hi-res co-location or spanning cover, ln6_pick.py); then the keys act only INSIDE a stack (a cluster of
       same-class first-choice claims), clusters taken by their best probability; without it the keys act on all
       first-choice claims (note 55's first version, costs 0.17 pt). pick={"order_only": True} = the same first-choice-first order by probability
       alone (the control that isolates the three keys).
       "stack_loser" (note 56b; default "next_free" = note 55/56): "nonatspm" = a member that loses its FIRST-choice
       class to a partner in its hi-res stack takes its best non-ATSPM class (user: in a stack "the other is Other");
       "nonatspm_ap" = the same for Advance / Presence claims only.
       Non-stack losers always continue to the next free class.
       Optional "span_key" (note 56): per detector bool used for key (b) instead of bool(span_lanes) - lets one flag
       decide the in-stack order and another the lane extension.
DEFAULT (orchestrator, note 56): the stack-scoped pick is the default decode wherever its hi-res inputs exist - build
       them with stack_pick(); (a) = note 56's stack-relative health flag (lanes/ln7_stackhealth.py: chi >= 10 and
       >= 3 x the best partner's), (b) = note 55's span flag, (c) = note 55's track. Without inputs: greedy (note 54).
"""
from __future__ import annotations

import numpy as np

C7 = ["Advance", "Presence", "Count", "Yellow_Red", "Mid", "Bike", "Other"]
ATSPM = np.array([c in ("Advance", "Presence", "Count", "Yellow_Red") for c in C7])
A_IDX = np.flatnonzero(ATSPM)
N_IDX = np.flatnonzero(~ATSPM)


def _crank(P) -> np.ndarray:
    """note 77: rank of each row by its probability content (lexicographic, highest first) -- the tie-break of every
    greedy step below instead of the row (= channel) order; equal ranks only for identical probability rows."""
    P = np.asarray(P, float)
    o = np.lexsort(tuple(-P[:, c] for c in range(P.shape[1] - 1, -1, -1)), axis=0) if len(P) else np.zeros(0, int)
    r = np.empty(len(P), int)
    r[o] = np.arange(len(P))
    return r


def _pair_order(P, crank=None) -> np.ndarray:
    """(row, class) pairs by probability, ties by content rank then class (was: stable flat argsort = row order)."""
    P = np.asarray(P, float)
    n, k = P.shape
    cr = _crank(P) if crank is None else crank
    rows, cls = np.repeat(np.arange(n), k), np.tile(np.arange(k), n)
    o = np.lexsort((cls, cr[rows], -P.ravel()))
    return np.column_stack([rows[o], cls[o]])


def _keys(lanes, span):
    out = []
    for s in lanes:
        if not s or (span == "single" and len(s) > 1):
            out.append(None)
        else:
            out.append(tuple(sorted(s)))
    return out


def _conflicts(pred, keys) -> bool:
    used = set()
    for c, k in zip(pred, keys):
        if k is None or not ATSPM[c]:
            continue
        for ln in k:
            if (ln, c) in used:
                return True
            used.add((ln, c))
    return False


SPAN_CLS = {0, 1, 2}                 # Advance, Presence, Count: a spanning zone of these is not a lane of its own


DEFAULT_PICK = "stack"


def stack_pick(det, lanes, unhealthy, span, span_peers, coloc_peers, track) -> dict:
    """The default pick dict for one phase group (all arrays per detector, in group order). `span_peers` /
    `coloc_peers`: comma-separated detector numbers (hi-res, ln6_pick.py); `lanes`: list of frozensets."""
    det = [int(d) for d in det]
    pos = {d: k for k, d in enumerate(det)}
    n = len(det)
    sp, stack = [], [set() for _ in range(n)]
    for k in range(n):
        u = frozenset()
        if span[k]:
            for q in str(span_peers[k] or "").split(","):
                if q and int(q) in pos:
                    u |= lanes[pos[int(q)]]
        sp.append(u or None)
        qs = [x for x in str(coloc_peers[k] or "").split(",") if x]
        if span[k]:
            qs += [x for x in str(span_peers[k] or "").split(",") if x]
        for q in qs:
            if int(q) in pos:
                stack[k].add(pos[int(q)])
                stack[pos[int(q)]].add(k)
    return {"stack": stack, "unhealthy": np.asarray(unhealthy, bool), "span_lanes": sp,
            "track": np.asarray(track, float)}


def _pick_lanes(lanes, pred, pick):
    if not pick or pick.get("order_only") or pick.get("span_lanes") is None:
        return list(lanes)
    out = []
    for i, s in enumerate(lanes):
        x = pick["span_lanes"][i]
        out.append(frozenset(s) | x if (x and s and pred[i] in SPAN_CLS) else s)
    return out


def _pick_order(P, pred, pick):
    """(detector, class) pairs: first-choice ATSPM claims by (healthy, not spanning, track, P), then the rest by P.
    Ties: content rank (note 77), never the row order."""
    n = len(P)
    cr = _crank(P)
    first = [i for i in range(n) if ATSPM[pred[i]]]
    if pick.get("order_only"):
        key = lambda i: (P[i, pred[i]], -cr[i])  # noqa: E731
    else:
        unh = np.asarray(pick.get("unhealthy", np.zeros(n, bool)), bool)
        sp = pick.get("span_lanes") or [None] * n
        sk = pick.get("span_key")                        # note 56: optional separate "loses to lane-by-lane" flag
        sk = [bool(x) for x in sp] if sk is None else [bool(x) for x in sk]
        tr = np.asarray(pick.get("track", np.full(n, np.nan)), float)

        def key(i):
            t = tr[i] if (pred[i] == 0 and np.isfinite(tr[i])) else -2.0
            return (not unh[i], not (sk[i] and pred[i] in SPAN_CLS), t, P[i, pred[i]], -cr[i])
    stack = pick.get("stack")
    if stack is None:                                    # keys act globally over all first-choice claims
        first = sorted(first, key=key, reverse=True)
    else:                                                # keys act only inside a hi-res stack of same-class claims
        par = {i: i for i in first}

        def root(i):
            while par[i] != i:
                par[i] = par[par[i]]
                i = par[i]
            return i
        for i in first:
            for j in stack[i] or ():
                if j in par and pred[j] == pred[i]:
                    par[root(i)] = root(j)
        cl = {}
        for i in first:
            cl.setdefault(root(i), []).append(i)
        cl = sorted(cl.values(), key=lambda m: (-max(P[i, pred[i]] for i in m), min(cr[i] for i in m)))
        first = [i for m in cl for i in sorted(m, key=key, reverse=True)]
    rest = _pair_order(P, cr)
    fs = set(first)
    rest = [(i, c) for i, c in rest if not (i in fs and c == pred[i])]
    return [(i, int(pred[i])) for i in first] + rest


def decode_group(P: np.ndarray, lanes, mode: str = "greedy", span: str = "strict", pick: dict | None = None) -> np.ndarray:
    P = np.asarray(P, float)
    pred = P.argmax(1)
    keys = _keys(_pick_lanes(lanes, pred, pick), span)
    if not _conflicts(pred, keys):
        return pred
    if mode == "exact":
        return _exact(P, keys, pred)
    n = len(P)
    out = np.full(n, -1)
    used = set()
    if pick:
        order = _pick_order(P, pred, pick)
    else:
        order = _pair_order(P)
    lost = np.zeros(n, bool)
    slr = (pick or {}).get("stack_loser", "nonatspm_ap")
    sl = bool(pick) and pick.get("stack") is not None and slr in ("nonatspm", "nonatspm_ap")
    sl_cls = {0, 1} if slr == "nonatspm_ap" else {0, 1, 2, 3}
    for i, c in order:
        if out[i] >= 0:
            continue
        if not ATSPM[c]:
            out[i] = c
            continue
        if lost[i] and mode == "nonatspm":
            continue
        k = keys[i]
        if k is None:
            out[i] = c
            continue
        if all((ln, c) not in used for ln in k):
            out[i] = c
            used.update((ln, c) for ln in k)
        else:
            lost[i] = True
            if mode == "nonatspm":
                out[i] = N_IDX[np.argmax(P[i, N_IDX])]
            elif sl and c == pred[i] and c in sl_cls and any(out[j] == c for j in (pick["stack"][i] or ())):
                # note 56b (user, stacks): a stacked member that loses its first-choice class to a stack partner is
                # Other-like (best non-ATSPM class), not the next free ATSPM class
                out[i] = N_IDX[np.argmax(P[i, N_IDX])]
    return out


def _exact(P, keys, pred0, max_nodes: int = 200_000):
    """Max sum log p subject to one detector per (lane, ATSPM class); detectors without a lane key keep argmax."""
    L = np.log(np.clip(P, 1e-9, 1))
    free = [i for i, k in enumerate(keys) if k is not None]
    out = pred0.copy()
    # options per constrained detector: its top-4 classes + its best non-ATSPM class (always feasible)
    opts = {}
    for i in free:
        top = list(np.argsort(-P[i])[:4])
        b = int(N_IDX[np.argmax(P[i, N_IDX])])
        if b not in top:
            top.append(b)
        opts[i] = sorted(top, key=lambda c: -L[i, c])
    cr = _crank(P)
    order = sorted(free, key=lambda i: (-P[i].max(), cr[i]))
    best_rest = np.array([L[i, opts[i][0]] for i in order])
    suffix = np.concatenate([np.cumsum(best_rest[::-1])[::-1], [0.0]])
    best = [-np.inf, None]
    nodes = [0]
    cur = {}

    def rec(j, used, score):
        nodes[0] += 1
        if nodes[0] > max_nodes:
            return
        if score + suffix[j] <= best[0]:
            return
        if j == len(order):
            best[0], best[1] = score, dict(cur)
            return
        i = order[j]
        for c in opts[i]:
            if ATSPM[c]:
                ks = [(ln, c) for ln in keys[i]]
                if any(k in used for k in ks):
                    continue
                cur[i] = c
                rec(j + 1, used | set(ks), score + L[i, c])
            else:
                cur[i] = c
                rec(j + 1, used, score + L[i, c])
        cur.pop(i, None)

    rec(0, frozenset(), 0.0)
    if best[1] is None:                                   # node cap hit before any leaf: fall back to greedy
        return decode_group(P, [frozenset(k) if k else frozenset() for k in keys], "greedy", "strict")
    for i, c in best[1].items():
        out[i] = c
    return out
