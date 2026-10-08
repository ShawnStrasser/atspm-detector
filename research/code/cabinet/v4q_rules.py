"""Note 95: label table v4q = v4p + the user's rulings of 2026-10-05 ~16:20 + the orchestrator's note-93 items + the
three cabinet prints filed under other names (labels only; nothing is trained or scored here).

    python v4q_rules.py build     research/labels/function_labels_v4q.parquet + function_label_changes_v4q.csv
    python v4q_rules.py dec       %DC_WORK%/lab95/dec_role_changed_v4q.parquet (Dec-role table, v4p recipe)
    python v4q_rules.py locked    research/labels/function_labels_locked_v3.parquet (labels only, no model)

Rules, in this order (each row keeps the first v4q rule that touched it in `rule_v4q`):
  Q1 05999 det 23 -> out of training AND scoring (user: exclude; print said Presence, earlier user word Advance).
  Q2 2B061: only the phase-4 channels out (already so in v4p; asserted, nothing else excluded).
  Q3 loop detectors whose hand label (config Function) was "advance presence" (v2 map -> Other): the hand label is
     Advance. The five channels the user named (2B331 d22, 2B417 d8, 04026 d8 / 9 / 21) -> Advance as user rulings.
     Every other such loop: hand label := Advance; no print reading or a print reading that agrees -> Advance; a print
     reading that disagrees -> print-vs-hand rule (model agreement, as N1): hand kept / print used / left out.
  Q4 channel text vs config Function (04034-type, regex parse of the channel description as in note 94 `desc`): where
     the label is the config Function (hand) and the text names another ATSPM class -> keep whichever the model agrees
     with; neither -> out. Rows that N1 left out (print vs Function, model neither) are restored when the model agrees
     with the text. Rows labelled by a print reading or a user ruling are not touched.
  Q5 04016 d23 / 04040 d28 (orchestrator, note 93: look like Count zones labelled Presence) -> model agreement between
     Presence and Count; neither -> out.
  Q6 prints 5CE166 (05166), 04CA156 (04156), 5CE069 (05069), read only where the device match is certain
     (%DC_WORK%/lab95/<name>_print_reading.json): print columns written; print agrees with the hand label (or no hand
     label) -> print (high) / hand; print disagrees -> N1 rule.
"Model" = the note-77 champion out-of-fold prediction already used by N1 (lab93/detpred.parquet): the detector's
majority class over its >= 30-min windows, agreement = share >= .6 and >= 2 windows. Rows decided by agreement carry
`n1_model_decided` = True (report headlines with and without them). locked_v2 asserted absent.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse as up

import numpy as np
import pandas as pd

from cab_common import DC_WORK, REPO

V4P = REPO / "research/labels/function_labels_v4p.parquet"
V4Q = REPO / "research/labels/function_labels_v4q.parquet"
CHG = REPO / "research/labels/function_label_changes_v4q.csv"
LOCKED_V2 = REPO / "research/labels/function_labels_locked_v2.parquet"
LOCKED_V3 = REPO / "research/labels/function_labels_locked_v3.parquet"
CHG_L = REPO / "research/labels/function_label_changes_locked_v3.csv"
LAB = DC_WORK / "lab95"
DETPRED = DC_WORK / "lab93" / "detpred.parquet"
AGREE_SHARE, AGREE_MIN_WIN = 0.6, 2
ATSPM = {"Advance", "Presence", "Count", "Yellow_Red"}
NAMED_Q3 = [("2B331", 22), ("2B417", 8), ("04026", 8), ("04026", 9), ("04026", 21)]
Q5 = [("04016", 23), ("04040", 28)]
PRINTS = {"05166": "5CE166", "04156": "04CA156", "05069": "5CE069"}
# note 94 desc_conflicts regexes, except the bare token "CO" (ambiguous in this agency's texts: "call only" / radar
# "CO" data channels) no longer counts as Count; the description is URL-decoded first (note 94 parsed it encoded)
PAT = [("Yellow_Red", r"\by/?r\b|yellow|yel[- ]?red"), ("Count", r"\bcount|\bcnt\b"),
       ("Presence", r"presence|\bpres\b|stop ?bar"), ("Advance", r"\badv"), ("Bike", r"bike"), ("Mid", r"\bmid\b")]
LBL = ("function", "label_print_first", "label_print_first_train", "truth_v3s")


def locked() -> set[str]:
    return set(pd.read_csv(DC_WORK / "official/locked_v2.csv").DeviceId.astype(str).str.lower())


def desc_class(t) -> str | None:
    if pd.isna(t):
        return None
    t = up.unquote(str(t)).lower()
    hits = [k for k, p in PAT if re.search(p, t)]
    return hits[0] if len(hits) == 1 else None


def is_loop(d: pd.DataFrame) -> pd.Series:
    desc = d.description.astype("string").map(lambda t: up.unquote(t) if pd.notna(t) else "").str.lower()
    tech = d.technology.astype("string")
    return (tech.eq("loop") | (tech.isna() & desc.str.contains(r"\bloops?\b|\bvd ?#", regex=True))).fillna(False)


class Book:
    """Row edits with a change log (first rule per row wins in rule_v4q; every edit is logged)."""

    def __init__(self, d: pd.DataFrame):
        self.d, self.chg = d, []

    def _log(self, i, rule, why):
        d = self.d
        self.chg.append(dict(i=i, rule=rule, why=why, old_train=d.at[i, "label_print_first_train"],
                             old_truth=d.at[i, "truth_v3s"], old_excl=bool(d.at[i, "exclude_train_score_train"])))
        if pd.isna(d.at[i, "rule_v4q"]):
            d.at[i, "rule_v4q"] = rule

    def label(self, i, lab, rule, why, *, truth="same", source=None, restore=False, model=False):
        """Set the training label (and truth: 'same' = new label where the row had truth, 'set' = always, None = clear)."""
        d = self.d
        self._log(i, rule, why)
        had_truth = pd.notna(d.at[i, "truth_v3s"])
        for c in LBL[:3]:
            d.at[i, c] = lab
        if truth == "set" or (truth == "same" and had_truth):
            d.at[i, "truth_v3s"] = lab
        elif truth is None:
            d.at[i, "truth_v3s"] = None
        if source:
            d.at[i, "source"] = source
        if restore:
            for c in ("exclude_train_score", "exclude_train_score_train"):
                d.at[i, c] = False
            ok = not bool(d.at[i, "dead"]) and not bool(d.at[i, "unusual_layout_train"])
            for c in ("train_use", "train_use_train"):
                d.at[i, c] = ok
            if d.at[i, "validated_train"] not in ("pass",):
                for c in ("validated", "validated_train"):
                    d.at[i, c] = "pass" if ok else d.at[i, c]
        if model:
            d.at[i, "n1_model_decided"] = True
        if bool(d.at[i, "exclude_train_score_train"]):
            d.at[i, "truth_v3s"] = None                 # excluded by another rule: never scored

    def exclude(self, i, rule, why):
        d = self.d
        self._log(i, rule, why)
        for c in ("exclude_train_score", "exclude_train_score_train"):
            d.at[i, c] = True
        for c in ("train_use", "train_use_train"):
            d.at[i, c] = False
        d.at[i, "truth_v3s"] = None
        d.at[i, "n1_model_decided"] = False


def model_choice(d, i, options: list[str]):
    """The option the model agrees with (share >= .6, >= 2 windows), else None; plus a short text."""
    mp, sh, nw = d.at[i, "n1_model_pred"], d.at[i, "n1_model_share"], d.at[i, "n1_model_nwin"]
    txt = f"model {mp} {sh:.2f} ({int(nw)} win)" if pd.notna(mp) else "model none (no >= 30-min window)"
    if pd.isna(mp) or not (sh >= AGREE_SHARE and nw >= AGREE_MIN_WIN):
        return None, txt
    return (mp if mp in options else None), txt


def n1_only_excluded(d, o, i) -> bool:
    """Row out of training only because N1 (v4p) left it out: v4o had it in."""
    return (str(d.at[i, "rule_v4p"]) == "N1_left_out") and not bool(o.at[i, "exclude_train_score_train"])


def build():
    d = pd.read_parquet(V4P)
    o = pd.read_parquet(REPO / "research/labels/function_labels_v4o.parquet")
    assert (d.DeviceId.values == o.DeviceId.values).all() and (d.detector.values == o.detector.values).all()
    lk = locked()
    assert not d.DeviceId.str.lower().isin(lk).any(), "locked_v2 in v4p"
    n0 = len(d)
    d["rule_v4q"] = pd.Series(pd.NA, index=d.index, dtype="string")
    d["func7_v2_v4p"] = d.func7_v2
    d["advpres_loop"] = False
    d["desc_class"] = d.description.map(desc_class)
    d["q_flag"] = pd.Series(pd.NA, index=d.index, dtype="string")
    B = Book(d)
    name = d.DeviceName.astype("string")
    key = {(str(a), int(b)): i for i, a, b in zip(d.index, name.fillna(""), d.detector.astype(int))}
    skip_user = (d.source.astype("string").isin(["user_ruling", "review_round1"]).fillna(False)
                 | d.v2_source.astype("string").eq("review_round1").fillna(False)
                 | d.rule_v4l.astype("string").str.startswith("R1").fillna(False)
                 | d.stack_relabel.eq(True) | d.rule_v4p.astype("string").str.startswith("U_").fillna(False))
    # Q1 -- 05999 d23
    B.exclude(key[("05999", 23)], "Q1_user_exclude", "user 2026-10-05: exclude (print Presence vs earlier user Advance)")
    d.at[key[("05999", 23)], "source"] = "user_ruling"
    # Q2 -- 2B061: phase 4 only
    s = name.eq("2B061").fillna(False)
    assert d.loc[s & d.phase_target.eq("P4"), "exclude_train_score_train"].all()
    assert not (s & ~d.phase_target.eq("P4").fillna(False) & d.rule_v4p.astype("string").str.startswith("U_").fillna(False)).any()
    # Q3 -- loops with hand label "advance presence"
    cf = d.config_function.astype("string").str.strip().str.lower()
    q3 = cf.eq("advance presence").fillna(False) & is_loop(d)
    d.loc[q3, "advpres_loop"] = True
    d.loc[q3, "func7_v2"] = "Advance"
    for nm, det in NAMED_Q3:
        i = key[(nm, det)]
        assert q3[i], (nm, det)
        B.label(i, "Advance", "Q3_user_named", "user 2026-10-05: loop 'advance presence' = Advance (named)", truth="set",
                source="user_ruling", restore=True)
        d.at[i, "n1_model_decided"] = False
    named = {key[k] for k in NAMED_Q3}
    for i in np.flatnonzero(q3.to_numpy()):
        i = d.index[i]
        if i in named or skip_user[i]:
            continue
        pf = d.at[i, "print_function"] if d.at[i, "print_source"] == "print" else None
        excl = bool(d.at[i, "exclude_train_score_train"])
        n1x = n1_only_excluded(d, o, i)
        if pd.isna(pf) or pf == "Advance":
            if excl and not n1x:
                d.at[i, "q_flag"] = "Q3_label_but_excluded_by_other_rule"
                for c in LBL[:3]:
                    d.at[i, c] = "Advance"
                continue
            if d.at[i, "label_print_first_train"] == "Advance" and not excl:
                d.at[i, "q_flag"] = "Q3_already_advance"
                if d.at[i, "n1_model_decided"] and pf == "Advance":
                    d.at[i, "n1_model_decided"] = False       # hand and print now agree: no model needed
                continue
            tr = "set" if (pf == "Advance" and d.at[i, "print_confidence"] == "high") else "same"
            B.label(i, "Advance", "Q3_loop_advpres", f"hand 'advance presence' on a loop = Advance; print {pf}",
                    truth=tr, restore=n1x)
            d.at[i, "n1_model_decided"] = False
        else:
            ch, txt = model_choice(d, i, ["Advance", pf])
            why = f"hand Advance (loop 'advance presence') vs print {pf} ({d.at[i, 'print_confidence']}); {txt}"
            if excl and not n1x:
                d.at[i, "q_flag"] = "Q3_conflict_but_excluded_by_other_rule"
                continue
            if ch is None:
                d.at[i, "n1_flag"] = "left_out_model_neither_or_unsure"
                B.exclude(i, "Q3_N1_left_out", why)
            else:
                d.at[i, "n1_flag"] = "hand_kept_model_agrees" if ch == "Advance" else "print_used_model_agrees"
                if d.at[i, "label_print_first_train"] == ch and not excl:
                    d.at[i, "n1_model_decided"] = True
                    d.at[i, "q_flag"] = "Q3_N1_nochange"
                    continue
                B.label(i, ch, "Q3_N1_model_agrees", why, truth="set", restore=True, model=True)
    # Q5 -- 04016 d23, 04040 d28 (before Q4 so the general text rule does not touch them)
    q5 = set()
    for nm, det in Q5:
        i = key[(nm, det)]
        q5.add(i)
        ch, txt = model_choice(d, i, ["Presence", "Count"])
        why = f"note 93: looks like a Count zone labelled Presence; text '{up.unquote(str(d.at[i, 'description']))}'; {txt}"
        if ch is None:
            B.exclude(i, "Q5_left_out", why)
        else:
            B.label(i, ch, "Q5_model_agrees", why, truth="set", restore=True, model=True)
    # Q4 -- channel text vs config Function
    q4 = (d.desc_class.isin(ATSPM) & d.func7_v2.notna() & d.desc_class.ne(d.func7_v2)
          & d.func7_v2.isin(ATSPM | {"Other"})).fillna(False)
    d["desc_conflict"] = q4
    for i in np.flatnonzero(q4.to_numpy()):
        i = d.index[i]
        if i in q5 or i in named or skip_user[i] or d.at[i, "advpres_loop"]:
            d.at[i, "q_flag"] = d.at[i, "q_flag"] if pd.notna(d.at[i, "q_flag"]) else "Q4_skip_user_or_q3"
            continue
        hand, txtc = d.at[i, "func7_v2"], d.at[i, "desc_class"]
        lab, excl = d.at[i, "label_print_first_train"], bool(d.at[i, "exclude_train_score_train"])
        n1x = n1_only_excluded(d, o, i)
        ch, txt = model_choice(d, i, [hand, txtc])
        why = (f"text '{up.unquote(str(d.at[i, 'description']))}' = {txtc} vs config Function {hand}; "
               f"print {d.at[i, 'print_function']}; {txt}")
        has_print = d.at[i, "print_source"] == "print" and pd.notna(d.at[i, "print_function"])
        if not excl and lab == hand and has_print:
            d.at[i, "q_flag"] = "Q4_print_decided"                     # print agrees with the Function (or N1 kept it)
        elif not excl and lab == hand:                                 # case A: the Function alone decides the label
            if ch is None:
                B.exclude(i, "Q4_left_out", why)
            elif ch == hand:
                d.at[i, "n1_model_decided"] = True
                d.at[i, "q_flag"] = "Q4_function_kept_model_agrees"
                if pd.isna(d.at[i, "rule_v4q"]):
                    d.at[i, "rule_v4q"] = "Q4_function_kept"
            else:
                B.label(i, txtc, "Q4_text_model_agrees", why, truth="set", model=True)
        elif excl and n1x:                                             # case B: N1 left it out
            if ch == txtc:
                B.label(i, txtc, "Q4_text_restores_n1", why, truth="set", restore=True, model=True)
            else:
                d.at[i, "q_flag"] = "Q4_n1_left_out_stays"
        elif excl:
            d.at[i, "q_flag"] = "Q4_excluded_by_other_rule"
        else:
            d.at[i, "q_flag"] = "Q4_label_from_print"                  # case C: print (or other) decided
    # Q6 -- prints filed under other names
    p6 = apply_prints(d, B, key, skip_user)
    d["train_use_validated"] = d.train_use.astype(bool) & d.validated.eq("pass").fillna(False).astype(bool)
    d["label_version"] = "v4q"
    assert len(d) == n0 and not d.DeviceId.str.lower().isin(lk).any()
    d.to_parquet(V4Q, index=False)
    changes(d, B, p6)


def apply_prints(d, B, key, skip_user) -> dict:
    out = {}
    for nm, fn in PRINTS.items():
        f = LAB / f"{nm}_print_reading.json"
        if not f.exists():
            out[nm] = "no reading file"
            continue
        r = json.loads(f.read_text())
        dets = r.get("detectors", [])
        if not dets:
            out[nm] = "skipped: " + str(r.get("note", ""))[:200]
            continue
        rows = []
        for x in dets:
            k = (nm, int(x["detector"]))
            if k not in key:
                out.setdefault(nm + "_not_in_table", []).append(int(x["detector"]))
                continue
            i = key[k]
            rows.append(i)
            vals = dict(print_source="print", print_function=x["function"], print_subtype=x.get("subtype"),
                        print_confidence=x["confidence"], confidence_reason=x.get("reason"),
                        phase_diagram=x.get("phase_input_file"), lane_index=x.get("lane_index"),
                        lanes_spanned=x.get("lanes_spanned"), lane_type=x.get("lane_type"),
                        print_flags=",".join(x.get("flags", [])))
            for c, v in vals.items():
                try:
                    d.at[i, c] = v
                except (TypeError, ValueError):
                    d.at[i, c] = None if v is None else int(v)
            if x.get("technology"):
                d.at[i, "technology"] = x["technology"]
            ph = r.get("phases", {}).get(str(x.get("phase_input_file")), {})
            if ph.get("n_lanes") is not None:
                d.at[i, "n_lanes_phase"] = ph["n_lanes"]
            if "dead" in x.get("flags", []):
                d.at[i, "dead"] = True
            pf, hand, conf = x["function"], d.at[i, "func7_v2"], x["confidence"]
            if skip_user[i]:
                d.at[i, "q_flag"] = "Q6_user_ruling_kept"
                continue
            if pd.isna(hand) or pf == hand:
                if conf == "high":
                    B.label(i, pf, "Q6_print_agrees", f"print {fn} {pf} (high) agrees with hand {hand}", truth="set",
                            source="print_high")
                    d.at[i, "n1_model_decided"] = False
                else:
                    d.at[i, "q_flag"] = f"Q6_print_{conf}_agrees"
            else:
                tc = d.at[i, "desc_class"]             # the channel text is a hand label too (Q4)
                opts = [hand, pf] + ([tc] if tc in ATSPM else [])
                ch, txt = model_choice(d, i, opts)
                why = f"print {fn} {pf} ({conf}) vs hand {hand} (text {tc}); {txt}"
                if ch is None:
                    d.at[i, "n1_flag"] = "left_out_model_neither_or_unsure"
                    B.exclude(i, "Q6_N1_left_out", why)
                else:
                    d.at[i, "n1_flag"] = "print_used_model_agrees" if ch == pf else "hand_kept_model_agrees"
                    B.label(i, ch, "Q6_N1_model_agrees", why, truth="set", restore=True, model=True,
                            source=f"print_{conf}" if ch == pf else "config")
        tier = r.get("tier")
        sig = d.DeviceName.astype("string").eq(nm).fillna(False)
        if tier:
            d.loc[sig, "tier"] = tier
            d.loc[sig, "tier_train"] = tier
        out[nm] = f"{len(rows)} detectors applied; tier {tier}; unmatched {len(r.get('unmatched', []))}"
    return out


def changes(d, B, p6):
    ch = pd.DataFrame(B.chg).drop_duplicates("i", keep="first")
    x = d.loc[ch.i, ["DeviceName", "DeviceId", "detector", "phase_target", "technology", "description", "func7_v2_v4p",
                     "func7_v2", "desc_class", "print_function", "print_confidence", "n1_model_pred", "n1_model_share",
                     "label_print_first_train", "truth_v3s", "exclude_train_score_train", "n1_model_decided"]].reset_index(drop=True)
    x["description"] = x.description.map(lambda t: up.unquote(str(t)) if pd.notna(t) else t)
    x.columns = ["DeviceName", "DeviceId", "detector", "phase", "technology", "description", "hand_label_v4p",
                 "hand_label", "text_class", "print_label", "print_confidence", "model_pred", "model_share",
                 "new_train_label", "new_truth", "new_excluded", "n1_model_decided"]
    x.insert(5, "rule", ch.rule.values)
    x.insert(6, "reason", ch.why.values)
    x["old_train_label"], x["old_truth"], x["old_excluded"] = ch.old_train.values, ch.old_truth.values, ch.old_excl.values
    x.loc[x.new_excluded.astype(bool), "new_train_label"] = None
    x.loc[x.old_excluded.astype(bool), "old_train_label"] = None
    f = lambda c: x[c].astype("string").fillna("-")  # noqa: E731
    x = x[(f("old_train_label") != f("new_train_label")) | (f("old_truth") != f("new_truth"))].copy()
    x["atspm_truth_changed"] = (f("old_truth") != f("new_truth")) & (x.old_truth.isin(ATSPM) | x.new_truth.isin(ATSPM))
    pri = {"Q1": 0, "Q3": 1, "Q5": 2, "Q6": 3, "Q4": 4}
    x["_p"] = x.rule.str[:2].map(pri).fillna(9)
    x = x.sort_values(["_p", "atspm_truth_changed", "DeviceName", "detector"],
                      ascending=[True, False, True, True]).drop(columns="_p")
    x.to_csv(CHG, index=False)
    p = pd.read_parquet(V4P)
    s = dict(rows=len(d), changed=len(x), by_rule=x.rule.value_counts().to_dict(),
             atspm_truth_changed=int(x.atspm_truth_changed.sum()),
             q_flags=d.q_flag.value_counts().to_dict(),
             q3_loops=int(d.advpres_loop.sum()), q4_conflicts=int(d.desc_conflict.sum()), prints=p6,
             trainable_labelled_v4p=int((~p.exclude_train_score_train.astype(bool) & p.label_print_first_train.notna()).sum()),
             trainable_labelled_v4q=int((~d.exclude_train_score_train.astype(bool) & d.label_print_first_train.notna()).sum()),
             truth_rows_v4p=int(p.truth_v3s.notna().sum()), truth_rows_v4q=int(d.truth_v3s.notna().sum()),
             atspm_truth_v4p=int(p.truth_v3s.isin(ATSPM).sum()), atspm_truth_v4q=int(d.truth_v3s.isin(ATSPM).sum()),
             n1_model_decided_truth_v4p=int((p.n1_model_decided & p.truth_v3s.notna()).sum()),
             n1_model_decided_truth_v4q=int((d.n1_model_decided & d.truth_v3s.notna()).sum()),
             n1_model_decided_atspm_v4q=int((d.n1_model_decided & d.truth_v3s.isin(ATSPM)).sum()))
    LAB.mkdir(exist_ok=True, parents=True)
    (LAB / "v4q_summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str))


def dec():
    """Dec-role table for v4q (v4p recipe: recompute rows whose label / truth changed)."""
    import cab_final as CF
    p = pd.read_parquet(V4P)
    q = pd.read_parquet(V4Q)
    t = pd.read_parquet(DC_WORK / "lab93" / "dec_role_changed_v4p.parquet")
    assert (t.DeviceId.str.lower().values == q.DeviceId.str.lower().values).all() and (t.detector.values == q.detector.values).all()
    same = ((p.label_print_first_train.astype("string").fillna("-") == q.label_print_first_train.astype("string").fillna("-"))
            & (p.truth_v3s.astype("string").fillna("-") == q.truth_v3s.astype("string").fillna("-"))).to_numpy()
    a = CF.dec_role_changed(q, q.label_print_first_train)
    b = CF.dec_role_changed(q, q.truth_v3s)
    n = t.copy()
    ch = ~same
    n.loc[ch, "label_train"] = a.label.to_numpy()[ch]
    n.loc[ch, "dec_status_train"] = a.dec_status.to_numpy()[ch]
    n.loc[ch, "dec_role_changed_train"] = False
    n.loc[ch, "label_truth"] = b.label.to_numpy()[ch]
    n.loc[ch, "dec_status_truth"] = b.dec_status.to_numpy()[ch]
    n.loc[ch, "dec_role_changed_truth"] = (b.dec_status.isin(["function", "phase"]) & b.label.notna()).to_numpy()[ch]
    assert not n.DeviceId.str.lower().isin(locked()).any()
    out = LAB / "dec_role_changed_v4q.parquet"
    n.to_parquet(out, index=False)
    print(f"{int(ch.sum())} rows recomputed; truth flags {int(t.dec_role_changed_truth.sum())} -> "
          f"{int(n.dec_role_changed_truth.sum())} -> {out}")


def locked_build():
    """Locked key v3 = locked v2 + Q3 (loop 'advance presence' = Advance) + Q4 (text vs Function). LABELS ONLY: where a
    rule needs model agreement the row is flagged `n1_pending_exam` and left out of `truth_v3` until the exam."""
    d = pd.read_parquet(LOCKED_V2)
    assert d.DeviceId.str.lower().isin(locked()).all()
    d["rule_v3"] = pd.Series(pd.NA, index=d.index, dtype="string")
    d["pending_reason"] = pd.Series(pd.NA, index=d.index, dtype="string")
    d.loc[d.n1_pending_exam.astype(bool), "pending_reason"] = "N1 print vs hand"
    d["func7_v2_v2"] = d.func7_v2
    d["truth_v3"] = d.truth_v2
    d["function_v3"] = d.function
    skip = (d.source.astype("string").isin(["user_ruling", "review_round1"]).fillna(False)
            | d.v2_source.astype("string").str.startswith("review").fillna(False)
            | d.rule_v2.astype("string").str.contains("R1|STACK", regex=True).fillna(False))
    cf = d.config_function.astype("string").str.strip().str.lower()
    q3 = cf.eq("advance presence").fillna(False) & is_loop(d)
    d["advpres_loop"] = q3
    d.loc[q3, "func7_v2"] = "Advance"
    log = []
    for i in d.index[q3]:
        if skip[i] or bool(d.at[i, "exclude_score"]):
            continue
        pf = d.at[i, "print_function"] if d.at[i, "print_source"] == "print" else None
        old = (d.at[i, "function_v3"], d.at[i, "truth_v3"], bool(d.at[i, "n1_pending_exam"]))
        if pd.isna(pf) or pf == "Advance":
            d.at[i, "function_v3"] = "Advance"
            d.at[i, "n1_pending_exam"] = False
            d.at[i, "pending_reason"] = None
            if pd.notna(d.at[i, "truth_v2_oldrule"]) or (pf == "Advance" and d.at[i, "print_confidence"] == "high"):
                d.at[i, "truth_v3"] = "Advance"
            d.at[i, "rule_v3"] = "Q3_loop_advpres"
        else:
            d.at[i, "n1_pending_exam"] = True
            d.at[i, "pending_reason"] = "Q3 hand Advance vs print " + str(pf)
            d.at[i, "truth_v3"] = None
            d.at[i, "rule_v3"] = "Q3_pending_exam"
        log.append((i, *old))
    d["desc_class"] = d.description.map(desc_class)
    q4 = (d.desc_class.isin(ATSPM) & d.func7_v2.notna() & d.desc_class.ne(d.func7_v2)
          & d.func7_v2.isin(ATSPM | {"Other"}) & ~q3).fillna(False)
    d["desc_conflict"] = q4
    for i in d.index[q4]:
        if skip[i] or bool(d.at[i, "exclude_score"]) or d.at[i, "function_v3"] != d.at[i, "func7_v2"]:
            continue
        if d.at[i, "print_source"] == "print" and pd.notna(d.at[i, "print_function"]):
            continue                                   # a print reading decides (or N1 already pending), as in training
        old = (d.at[i, "function_v3"], d.at[i, "truth_v3"], bool(d.at[i, "n1_pending_exam"]))
        d.at[i, "n1_pending_exam"] = True
        d.at[i, "pending_reason"] = f"Q4 text {d.at[i, 'desc_class']} vs Function {d.at[i, 'func7_v2']}"
        d.at[i, "truth_v3"] = None
        d.at[i, "rule_v3"] = "Q4_pending_exam"
        log.append((i, *old))
    d["label_use_v3"] = d.function_v3.notna() & ~d.exclude_score & ~d.unusual_layout.astype(bool) & ~d.dead.eq(True)
    d["label_version"] = "locked_v3"
    d.to_parquet(LOCKED_V3, index=False)
    L = pd.DataFrame(log, columns=["i", "old_label", "old_truth", "old_pending"])
    x = d.loc[L.i, ["DeviceName", "set", "detector", "phase_target", "technology", "description", "func7_v2_v2",
                    "desc_class", "print_function", "print_confidence", "rule_v3", "pending_reason", "function_v3",
                    "truth_v3", "n1_pending_exam"]].reset_index(drop=True)
    x["description"] = x.description.map(lambda t: up.unquote(str(t)) if pd.notna(t) else t)
    x["old_label"], x["old_truth"], x["old_pending"] = L.old_label.values, L.old_truth.values, L.old_pending.values
    x.to_csv(CHG_L, index=False)
    s = dict(rows=len(d), touched=len(x), by_rule=x.rule_v3.value_counts().to_dict(),
             pending_v2=int(pd.read_parquet(LOCKED_V2).n1_pending_exam.sum()), pending_v3=int(d.n1_pending_exam.sum()),
             truth_v2=int(d.truth_v2.notna().sum()), truth_v3=int(d.truth_v3.notna().sum()),
             atspm_truth_v3=int(d.truth_v3.isin(ATSPM).sum()))
    print(json.dumps(s, indent=1))
    (LAB / "locked_v3_summary.json").write_text(json.dumps(s, indent=1))


if __name__ == "__main__":
    {"build": build, "dec": dec, "locked": locked_build}[sys.argv[1]]()
