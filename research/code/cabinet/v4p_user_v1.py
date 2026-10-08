"""Note 94: label table v4p = v4o + the user's answers on review sheet v1 (rows 1-17, 36, 37) + the 05999 print found
under another file name + the user's rulings of 2026-10-05 (labels only; nothing is trained or scored here).

    python v4p_user_v1.py corrections   research/labels/function_label_corrections_user_v1.csv (answers verbatim)
    python v4p_user_v1.py build         research/labels/function_labels_v4p.parquet + function_label_changes_v4p.csv
    python v4p_user_v1.py desc          %DC_WORK%/lab93/desc_conflicts.csv (config Function vs channel text, 04034-type)

Rules applied by `build`, in this order (each row keeps the first rule that touched it in `rule_v4p`):
  U  user answers (corrections csv): relabel -> source user_ruling, label + truth = the user's class;
     exclude / exclude_phase / exclude_signal -> out of training AND scoring.
  P  05999 print reading (%DC_WORK%/lab93/05999_print_reading.json; print filed as 5CE055 on the share).
  N1 print vs hand (user 2026-10-05, USER_INPUT q1, "for train and test labels"): where a print reading and the hand
     label (func7_v2 = config Function, round-1 review folded in) disagree: model agrees with the hand label -> hand;
     model agrees with the print -> print; otherwise out of training AND scoring (flag n1_flag). "Model" = the note-77
     champion out-of-fold prediction (rev81 function_rows, >= 30-min windows), agreement = the detector's majority
     class over its windows with share >= AGREE_SHARE and >= AGREE_MIN_WIN windows. Not applied to user rulings,
     review-round-1 labels, R1 (a loop is never Count / Yellow_Red: config Count on a loop is not a valid hand label)
     or stacked-group relabels. Rows decided by model agreement are marked `n1_model_decided` so a scorer can leave
     them out (truth chosen by agreeing with a model is biased towards that model).
  N2 channels 40 / 41 with no timing phase, no config text and no config function (user 2026-10-05 q4: custom dummy
     detectors) -> out of training AND scoring.
Training values: the plain column AND its g89 `<col>_train` twin are both written (v3_retrain copies `_train` over).
locked_v2 asserted absent.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd

from cab_common import DC_WORK, REPO

V4O = REPO / "research/labels/function_labels_v4o.parquet"
V4P = REPO / "research/labels/function_labels_v4p.parquet"
CORR = REPO / "research/labels/function_label_corrections_user_v1.csv"
CHG = REPO / "research/labels/function_label_changes_v4p.csv"
SHEET = REPO / "review/function_label_review_v1.xlsx"
LAB = DC_WORK / "lab93"
READ05999 = LAB / "05999_print_reading.json"
DETPRED = LAB / "detpred.parquet"          # detpred.py: champion OOF per detector, >= 30-min windows
AGREE_SHARE, AGREE_MIN_WIN = 0.6, 2
ATSPM = {"Advance", "Presence", "Count", "Yellow_Red"}
ROWS = list(range(1, 18)) + [36, 37]

# (sheet row, DeviceName, detector, action, new label, note). action: relabel | exclude | exclude_phase:<P> |
# exclude_signal | confirm_phase | signal_note
ANSWERS = [
    (1, "2B091", 21, "exclude_signal", None, "loop cannot be a stop-bar count; no good record -> whole signal out (row 2)"),
    (2, "2B091", 9, "exclude_signal", None, "skeptical of all this signal's labels: throw the whole signal out"),
    (2, "2B091", 27, "exclude_signal", None, "same"),
    (2, "2B091", 28, "exclude_signal", None, "same"),
    (3, "2C009", 19, "confirm_phase", None, "label (phase 7) correct"),
    (4, "2C009", 5, "confirm_phase", None, "label (phase 3) correct"),
    (5, "05999", 15, "relabel", "Advance", "user: advance (print 5CE055 agrees: loop 5 upstream in the LT bay)"),
    (5, "05999", 23, "relabel", "Advance", "user: advance (print 5CE055 shows loops 21,22 at the RT-lane stop bar: CONFLICT, flagged)"),
    (6, "05999", 25, "relabel", "Advance", "user: advance (= config; print agrees: loop 20 upstream)"),
    (7, "2B061", 9, "exclude_phase:P4", None, "field does not match the print, labels unknown: phase-4 detectors out (user: 'or potentially the entire signal')"),
    (8, "2B095", 7, "relabel", "Advance", "user: det 7 is advance"),
    (8, "2B095", 14, "relabel", "Presence", "user: det 14 is presence"),
    (9, "05166", 9, "relabel", "Advance", "user: det 9 is advance"),
    (9, "05166", 10, "relabel", "Presence", "user: det 10 is presence (row 9 and row 10)"),
    (10, "05166", 10, "relabel", "Presence", "user: presence"),
    (11, "10073", 61, "exclude", None, "extends phase 1 but does not call it: unusual, do not train on it"),
    (12, "2B331", 1, "relabel", "Advance", "advance loops; hand 'advance presence' is not valid for loops"),
    (12, "2B331", 2, "relabel", "Advance", "same"),
    (12, "2B331", 16, "relabel", "Advance", "same"),
    (12, "2B331", 17, "relabel", "Advance", "same"),
    (12, "2B331", 23, "relabel", "Advance", "same"),
    (13, "2B054", 9, "exclude", None, "phase-4 labels cannot be confirmed, seem wrong"),
    (13, "2B054", 23, "exclude", None, "same row (det 23 is phase 8)"),
    (14, "2B054", 8, "exclude", None, "loops probably replaced by radar; changepoint last year: throw out"),
    (15, "2B551", 22, "relabel", "Advance", "user: 22 advance"),
    (15, "2B551", 23, "relabel", "Presence", "user: 23 presence"),
    (15, "2B551", 24, "relabel", "Advance", "user: other P8 lane, 24 advance"),
    (15, "2B551", 25, "relabel", "Presence", "user: other P8 lane, 25 presence"),
    (16, "04034", 8, "relabel", "Presence", "user: the label says Presence (= channel text 'Rad B - Presence'; config Function says Advance)"),
    (16, "04034", 22, "relabel", "Presence", "same (channel text 'Rad D - RL Presence')"),
    (16, "04034", 23, "relabel", "Presence", "same (channel text 'Rad D - LL Presence')"),
    (17, "06048", 24, "relabel", "Advance", "user: 24 advance (print_high reading said Presence)"),
    (17, "06048", 25, "relabel", "Presence", "user: 25 presence (print_low reading said Bike)"),
    (36, "05048", 14, "exclude", None, "not set to call or extend, not bike, purpose unknown"),
    (36, "05048", 28, "exclude", None, "same"),
    (37, "01067", 21, "relabel", "Presence", "rail crossing: two stop bars, no advance detectors; model right"),
    (37, "01067", 0, "signal_note", None, "unusual but valid layout (rail crossing, two stop bars): keep in training"),
]


def locked() -> set[str]:
    return set(pd.read_csv(DC_WORK / "official/locked_v2.csv").DeviceId.astype(str).str.lower())


def md5(p) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def corrections():
    """Answers verbatim from the sheet (opened read-only, md5 checked) + the action taken."""
    from openpyxl import load_workbook
    h0 = md5(SHEET)
    wb = load_workbook(SHEET, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    it = ws.iter_rows(values_only=True)
    next(it)
    hdr = list(next(it))
    rows = {}
    for r in it:
        if r[0] is None:
            continue
        rows[int(r[0])] = dict(zip(hdr, r))
    wb.close()
    assert md5(SHEET) == h0, "review sheet changed while reading"
    out = []
    for row, dn, det, act, lab, note in ANSWERS:
        s = rows[row]
        assert str(s["Signal"]) == dn, (row, s["Signal"], dn)
        out.append(dict(DeviceName=dn, detector=det, kind="phase" if act == "confirm_phase" else "function",
                        source_sheet="review/function_label_review_v1.xlsx", sheet_row=row,
                        label_shown=str(s["Label"]).replace("\n", "; "), model_said=str(s["Model says"]).replace("\n", "; "),
                        action=act, new_label=lab, note=note, user_words=str(s["Answer"]).strip()))
    assert {o["sheet_row"] for o in out} == set(ROWS), "every answered row must be covered"
    pd.DataFrame(out).to_csv(CORR, index=False)
    print(f"{len(out)} lines -> {CORR}; sheet md5 {h0} unchanged")


def _set_label(d, m, lab, src, rule, why, chg):
    for i in np.flatnonzero(m.to_numpy()):
        chg.append(dict(i=i, rule=rule, why=why, old_train=d.at[d.index[i], "label_print_first_train"],
                        old_truth=d.at[d.index[i], "truth_v3s"], old_excl=bool(d.at[d.index[i], "exclude_train_score_train"])))
    for c in ("function", "label_print_first", "label_print_first_train", "truth_v3s"):
        d.loc[m, c] = lab
    d.loc[m, "source"] = src
    d.loc[m & d.rule_v4p.isna(), "rule_v4p"] = rule


def _exclude(d, m, rule, why, chg):
    for i in np.flatnonzero(m.to_numpy()):
        chg.append(dict(i=i, rule=rule, why=why, old_train=d.at[d.index[i], "label_print_first_train"],
                        old_truth=d.at[d.index[i], "truth_v3s"], old_excl=bool(d.at[d.index[i], "exclude_train_score_train"])))
    for c in ("exclude_train_score", "exclude_train_score_train"):
        d.loc[m, c] = True
    for c in ("train_use", "train_use_train"):
        d.loc[m, c] = False
    d.loc[m, "truth_v3s"] = None
    d.loc[m & d.rule_v4p.isna(), "rule_v4p"] = rule


def apply_05999(d, key):
    r = json.loads(READ05999.read_text())
    ph = r["phases"]
    rows = []
    for x in r["detectors"]:
        m = key.eq(f"{r['DeviceId']}_{x['detector']}")
        assert m.sum() == 1, x
        i = d.index[m][0]
        p = str(x["phase_input_file"])
        d.loc[i, ["print_source", "print_function", "print_subtype", "print_confidence", "confidence_reason",
                  "technology", "phase_diagram", "lane_index", "lanes_spanned", "lane_type", "n_lanes_phase",
                  "print_flags"]] = ["print", x["function"], x["subtype"], x["confidence"], x["reason"], "loop",
                                     x["phase_input_file"], x["lane_index"], x["lanes_spanned"], x["lane_type"],
                                     ph.get(p, {}).get("n_lanes"), ",".join(x.get("flags", []))]
        if "dead" in x.get("flags", []):
            d.loc[i, "dead"] = True
        rows.append(i)
    allhigh = all(x["confidence"] == "high" for x in r["detectors"] if x["function"] != "Other")
    sig = d.DeviceId.str.lower().eq(r["DeviceId"])
    d.loc[sig, "tier"] = "complete_high" if allhigh else "complete_mixed"
    d.loc[sig, "tier_train"] = d.loc[sig, "tier"]
    # standard rule for the print rows that agree with the hand label or have none (conflicts go to N1 below)
    for i in rows:
        pf, hand, conf = d.at[i, "print_function"], d.at[i, "func7_v2"], d.at[i, "print_confidence"]
        if pd.isna(hand) or pf == hand:
            use = pf if conf == "high" or pd.isna(hand) else hand
            for c in ("function", "label_print_first", "label_print_first_train"):
                d.at[i, c] = use
            d.at[i, "source"] = f"print_{conf}" if pd.isna(hand) or conf == "high" else "config"
            d.at[i, "truth_v3s"] = pf if conf == "high" else None
    return sig


def build():
    d = pd.read_parquet(V4O)
    lk = locked()
    assert not d.DeviceId.str.lower().isin(lk).any(), "locked_v2 in v4o"
    n0 = len(d)
    d["rule_v4p"] = pd.Series(pd.NA, index=d.index, dtype="string")
    d["user_review_v1"] = pd.Series(pd.NA, index=d.index, dtype="string")
    d["signal_note"] = pd.Series(pd.NA, index=d.index, dtype="string")
    d["n1_flag"] = pd.Series(pd.NA, index=d.index, dtype="string")
    d["n1_model_decided"] = False
    key = d.DeviceId.str.lower() + "_" + d.detector.astype(int).astype(str)
    nm = d.dropna(subset=["DeviceName"]).drop_duplicates("DeviceName")
    ids = dict(zip(nm.DeviceName.astype(str), nm.DeviceId.str.lower()))
    chg: list[dict] = []
    c = pd.read_csv(CORR, dtype={"DeviceName": str})
    # P -- 05999 print reading first (so the user's answers win over it)
    sig05999 = apply_05999(d, key)
    d.loc[sig05999 & d.rule_v4p.isna(), "rule_v4p"] = "P_print_5CE055"
    # U -- user answers
    for r in c.itertuples():
        sid = ids[r.DeviceName]
        if r.action == "exclude_signal":
            m = d.DeviceId.str.lower().eq(sid)
            _exclude(d, m, "U_exclude_signal", f"v1 row {r.sheet_row}", chg)
        elif str(r.action).startswith("exclude_phase"):
            m = d.DeviceId.str.lower().eq(sid) & d.phase_target.eq(r.action.split(":")[1])
            _exclude(d, m, "U_exclude_phase", f"v1 row {r.sheet_row}: {r.note}", chg)
        elif r.action == "signal_note":
            d.loc[d.DeviceId.str.lower().eq(sid), "signal_note"] = r.note
            continue
        else:
            m = key.eq(f"{sid}_{int(r.detector)}")
            assert m.sum() == 1, (r.DeviceName, r.detector, int(m.sum()))
            if r.action == "exclude":
                _exclude(d, m, "U_exclude", f"v1 row {r.sheet_row}: {r.note}", chg)
            elif r.action == "relabel":
                _set_label(d, m, r.new_label, "user_ruling", "U_relabel", f"v1 row {r.sheet_row}: {r.note}", chg)
                for cc in ("validated", "validated_train"):
                    d.loc[m & ~d.dead.eq(True), cc] = "pass"
                for cc in ("exclude_train_score", "exclude_train_score_train"):
                    d.loc[m, cc] = False
                for cc in ("train_use", "train_use_train"):
                    d.loc[m, cc] = ~d.loc[m, "dead"].eq(True) & ~d.loc[m, "unusual_layout_train"].eq(True)
            elif r.action == "confirm_phase":
                assert d.loc[m, "phase_user_confirmed"].all(), r
        d.loc[m, "user_review_v1"] = d.loc[m, "user_review_v1"].fillna("") + f"row {r.sheet_row}: {r.action};"
    # N1 -- print vs hand
    p = pd.read_parquet(DETPRED)
    p["key"] = p.DeviceId.str.lower() + "_" + p.detector.astype(int).astype(str)
    d["_key"] = key.values
    d = d.merge(p[["key", "model_pred", "share", "n_win", "p_mean"]].rename(
        columns={"key": "_key", "model_pred": "n1_model_pred", "share": "n1_model_share", "n_win": "n1_model_nwin",
                 "p_mean": "n1_model_p"}), on="_key", how="left").drop(columns=["_key"])
    assert len(d) == n0
    sure = d.n1_model_share.ge(AGREE_SHARE) & d.n1_model_nwin.ge(AGREE_MIN_WIN)
    conflict = (d.print_source.eq("print") & d.print_function.notna() & d.func7_v2.notna()
                & d.print_function.ne(d.func7_v2)).fillna(False)
    skip = (d.source.astype("string").isin(["user_ruling", "review_round1"]).fillna(False)
            | d.v2_source.astype("string").eq("review_round1").fillna(False)
            | d.rule_v4l.astype("string").str.startswith("R1").fillna(False)
            | d.stack_relabel.eq(True) | d.rule_v4p.astype("string").str.startswith("U_").fillna(False))
    n1 = conflict & ~skip
    hand_ok = n1 & sure & d.n1_model_pred.eq(d.func7_v2)
    print_ok = n1 & sure & d.n1_model_pred.eq(d.print_function)
    neither = n1 & ~hand_ok & ~print_ok
    d.loc[n1, "n1_flag"] = np.where(hand_ok[n1], "hand_kept_model_agrees",
                                    np.where(print_ok[n1], "print_used_model_agrees", "left_out_model_neither_or_unsure"))
    d.loc[hand_ok | print_ok, "n1_model_decided"] = True
    for m, rule in ((hand_ok, "N1_hand_model_agrees"), (print_ok, "N1_print_model_agrees")):
        for i in np.flatnonzero(m.to_numpy()):
            lab = d.at[i, "func7_v2"] if rule.startswith("N1_hand") else d.at[i, "print_function"]
            old_t, old_l = d.at[i, "truth_v3s"], d.at[i, "label_print_first_train"]
            if old_t == lab and old_l == lab and not d.at[i, "exclude_train_score_train"]:
                d.at[i, "rule_v4p"] = d.at[i, "rule_v4p"] if pd.notna(d.at[i, "rule_v4p"]) else rule + "_nochange"
                continue
            _set_label(d, pd.Series(d.index == i, index=d.index), lab, d.at[i, "source"], rule,
                       f"print {d.at[i, 'print_function']} ({d.at[i, 'print_confidence']}) vs hand {d.at[i, 'func7_v2']}; "
                       f"model {d.at[i, 'n1_model_pred']} {d.at[i, 'n1_model_share']:.2f}", chg)
            if d.at[i, "exclude_train_score"] is True or d.at[i, "exclude_train_score"] == True:  # noqa: E712
                d.at[i, "truth_v3s"] = old_t      # out of scoring by another rule: scoring truth untouched
    for i in np.flatnonzero(neither.to_numpy()):
        mp = d.at[i, "n1_model_pred"]
        why = (f"print {d.at[i, 'print_function']} ({d.at[i, 'print_confidence']}) vs hand {d.at[i, 'func7_v2']}; model "
               + (f"{mp} {d.at[i, 'n1_model_share']:.2f}" if pd.notna(mp) else "none (no >= 30-min window)"))
        _exclude(d, pd.Series(d.index == i, index=d.index), "N1_left_out", why, chg)
    # N2 -- channels 40 / 41, no phase, no text, no config function
    desc = d.description.astype("string").fillna("").str.strip()
    n2 = (d.detector.isin([40, 41]) & d.phase_target.isna() & desc.eq("") & d.config_function.isna()
          & ~d.exclude_train_score_train.eq(True))
    _exclude(d, n2, "N2_dummy_40_41", "channel 40/41, no timing phase, no config text or function (user q4)", chg)
    n2all = d.detector.isin([40, 41]) & d.phase_target.isna() & desc.eq("") & d.config_function.isna()
    d.loc[n2all & d.rule_v4p.isna(), "rule_v4p"] = "N2_dummy_40_41"
    d["train_use_validated"] = d.train_use.astype(bool) & d.validated.eq("pass").fillna(False).astype(bool)
    d["label_version"] = "v4p"
    assert not d.DeviceId.str.lower().isin(lk).any()
    d.to_parquet(V4P, index=False)
    # change list
    ch = pd.DataFrame(chg)
    ch = ch.drop_duplicates("i", keep="first")
    x = d.loc[ch.i, ["DeviceName", "DeviceId", "detector", "phase_target", "technology", "func7_v2", "print_function",
                     "print_confidence", "n1_model_pred", "n1_model_share", "label_print_first_train", "truth_v3s",
                     "exclude_train_score_train"]].reset_index(drop=True)
    x.columns = ["DeviceName", "DeviceId", "detector", "phase", "technology", "hand_label", "print_label",
                 "print_confidence", "model_pred", "model_share", "new_train_label", "new_truth", "new_excluded"]
    x.insert(5, "rule", ch.rule.values)
    x.insert(6, "reason", ch.why.values)
    x["old_train_label"], x["old_truth"], x["old_excluded"] = ch.old_train.values, ch.old_truth.values, ch.old_excl.values
    x.loc[x.new_excluded.astype(bool), "new_train_label"] = None
    x.loc[x.old_excluded.astype(bool), "old_train_label"] = None
    # rows that had no label before and have none now (unlabelled dummies, unlabelled phase-4 channels) are not changes
    x = x[~(x.old_train_label.isna() & x.old_truth.isna() & x.new_train_label.isna() & x.new_truth.isna())]
    x = x[(x.old_train_label.astype("string").fillna("-") != x.new_train_label.astype("string").fillna("-"))
          | (x.old_truth.astype("string").fillna("-") != x.new_truth.astype("string").fillna("-"))]
    x["atspm_truth_changed"] = (x.old_truth.astype("string").fillna("-") != x.new_truth.astype("string").fillna("-")) & (
        x.old_truth.isin(ATSPM) | x.new_truth.isin(ATSPM))
    pri = {"U_": 0, "P_": 1, "N1": 2, "N2": 3}
    x["_p"] = x.rule.str[:2].map(pri).fillna(9)
    x = x.sort_values(["_p", "atspm_truth_changed", "DeviceName", "detector"], ascending=[True, False, True, True]).drop(columns="_p")
    x.to_csv(CHG, index=False)
    s = dict(rows=len(d), changed=len(x), by_rule=x.rule.value_counts().to_dict(),
             atspm_truth_changed=int(x.atspm_truth_changed.sum()),
             n1_conflicts=int(n1.sum()), n1_flags=d.n1_flag.value_counts().to_dict(),
             n1_by_conf=pd.crosstab(d.loc[n1, "print_confidence"], d.loc[n1, "n1_flag"]).to_dict(),
             n2_rows_all=int(n2all.sum()), n2_newly_excluded=int(n2.sum()),
             excluded_train=int(d.exclude_train_score_train.eq(True).sum()),
             truth_rows=int(d.truth_v3s.notna().sum()), truth_rows_v4o=int(pd.read_parquet(V4O).truth_v3s.notna().sum()))
    (LAB / "v4p_summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str))


def desc_conflicts():
    """04034-type: the config Function column contradicts the technician's channel text on the same row."""
    import re
    d = pd.read_parquet(V4P)
    pat = [("Yellow_Red", r"\by/?r\b|yellow|yel[- ]?red"), ("Count", r"\bcount|\bcnt\b|\bco\b"),
           ("Presence", r"presence|\bpres\b|stop ?bar"), ("Advance", r"\badv"), ("Bike", r"bike"), ("Mid", r"\bmid\b")]
    def cls(t):
        t = str(t).lower()
        hits = [k for k, p in pat if re.search(p, t)]
        return hits[0] if len(hits) == 1 else None
    d["desc_class"] = d.description.map(lambda t: cls(t) if pd.notna(t) else None)
    m = d.desc_class.notna() & d.func7_v2.notna() & d.desc_class.ne(d.func7_v2) & d.desc_class.isin(ATSPM) \
        & d.func7_v2.isin(ATSPM | {"Other"})
    x = d.loc[m, ["DeviceName", "detector", "phase_target", "technology", "description", "config_function", "func7_v2",
                  "desc_class", "print_function", "print_confidence", "label_print_first_train", "truth_v3s",
                  "n1_model_pred", "n1_model_share", "rule_v4p"]]
    x["model_sides_with"] = np.where(x.n1_model_pred.eq(x.desc_class), "channel_text",
                                     np.where(x.n1_model_pred.eq(x.func7_v2), "config_function", "neither"))
    x.to_csv(LAB / "desc_conflicts.csv", index=False)
    print(len(x), x.model_sides_with.value_counts().to_dict())


def dec():
    """Dec-role table for v4p: v4o's rows where label and truth are unchanged; recomputed (function / phase rule) where
    v4p changed them. Training flag stays cleared (note 89: the Dec-role group is trained on)."""
    import cab_final as CF
    o = pd.read_parquet(V4O)
    p = pd.read_parquet(V4P)
    t = pd.read_parquet(DC_WORK / "x89" / "dec_role_changed_v4o.parquet")
    assert (t.DeviceId.str.lower().values == p.DeviceId.str.lower().values).all() and (t.detector.values == p.detector.values).all()
    same = ((o.label_print_first_train.astype("string").fillna("-") == p.label_print_first_train.astype("string").fillna("-"))
            & (o.truth_v3s.astype("string").fillna("-") == p.truth_v3s.astype("string").fillna("-"))).to_numpy()
    a = CF.dec_role_changed(p, p.label_print_first_train)
    b = CF.dec_role_changed(p, p.truth_v3s)
    n = t.copy()
    ch = ~same
    n.loc[ch, "label_train"] = a.label.to_numpy()[ch]
    n.loc[ch, "dec_status_train"] = a.dec_status.to_numpy()[ch]
    n.loc[ch, "dec_role_changed_train"] = False
    n.loc[ch, "label_truth"] = b.label.to_numpy()[ch]
    n.loc[ch, "dec_status_truth"] = b.dec_status.to_numpy()[ch]
    n.loc[ch, "dec_role_changed_truth"] = (b.dec_status.isin(["function", "phase"]) & b.label.notna()).to_numpy()[ch]
    assert not n.DeviceId.str.lower().isin(locked()).any()
    out = DC_WORK / "lab93" / "dec_role_changed_v4p.parquet"
    n.to_parquet(out, index=False)
    print(f"{int(ch.sum())} rows recomputed; truth flags {int(t.dec_role_changed_truth.sum())} -> "
          f"{int(n.dec_role_changed_truth.sum())}; train {int(n.dec_role_changed_train.sum())} -> {out}")


if __name__ == "__main__":
    {"corrections": corrections, "build": build, "desc": desc_conflicts, "dec": dec}[sys.argv[1]]()
