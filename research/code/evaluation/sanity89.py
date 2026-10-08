"""Note 89: 20-row sanity sheet of labels that STAY excluded from training after the group decisions (user asked for
a small random check instead of the 1,001-row sheet). Pool = %DC_WORK%/x89[/<src>]/excluded_<tag>.parquet (written by
g89_groups.py labels); the user's own rules / rulings (R1, YR identical to Count, user_ruling, R2 / R4) are left out.
Random 20 (seed 89). Charts from saved event data (review87's chart cache and drawing code); nothing is re-scored.

    python sanity89.py --pool <path to excluded_<tag>.parquet>    -> review/label_sanity_20.xlsx + review/label_sanity_20_charts/
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import urllib.parse  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

CODE = next(p for p in Path(__file__).resolve().parents if p.name == "code")
for _p in (CODE, CODE / "evaluation", CODE / "trackA"):
    sys.path.insert(0, str(_p))
import rpath  # noqa: E402
import review87 as R87  # noqa: E402

NAME = "label_sanity_20"
QUESTION = ("Should this label be used? Y / N / ?  Each row is a detector label on a training signal that is still "
            "left out of training after our data test (20 picked at random). Y = use it for training, N = keep it out, "
            "? = can't tell.")
USER_RULES = {"user_ruling", "R1_loop_config_count", "yr_identical_to_count"}


def why(r) -> str:
    rs = r.reason
    if rs == "not_checkable_not_readmitted":
        return ("No behaviour check could be run on it (nothing to compare it with), and the label is not a "
                f"high-confidence print reading (source: {r.source}), so it is not used.")
    if rs.startswith("window_level"):
        return ("Label passed its checks, but every sample of it is cut by a per-sample gate: fewer than 5 "
                "actuations, or a bad-health period overlapping the sample.")
    if rs == "clean_extra_unusual_signal":
        return ("Its whole signal is on a hand-kept 'unusual layout' list in the training code (not the print "
                "reader's flag), so none of its labels are used.")
    if rs == "not_train_use":
        return f"The label pipeline marked it not usable for training (check result: {r.validated or 'none'})."
    if rs == "no_validation_data":
        return "Not enough data to check its behaviour against its label."
    if rs == "dead":
        return ("Marked dead by the label pipeline (no or almost no actuations in the newest log: "
                f"{int(r.n_on_new) if pd.notna(r.n_on_new) else 0}), so not used.")
    if rs.startswith("validated_"):
        return R87.why_check(str(r.failed_checks), r.validation_reason)
    return rs


HEAD = ["#", "Signal", "Detector", "Label", "What this row asks", "Why it is left out", "Chart", "Print", "Answer"]
WIDTH = (5, 9, 9, 16, 50, 60, 7, 7, 24)


def read_answers(xl: Path) -> dict:
    """(signal, detector) -> the user's non-empty Answer cell, from either sheet layout."""
    from openpyxl import load_workbook
    wb = load_workbook(xl, read_only=True, data_only=True)
    ws = wb["labels to check"]
    head = list(next(ws.iter_rows(min_row=2, max_row=2, values_only=True)))
    ia, idt = head.index("Answer"), head.index("Detector")
    out = {}
    for r in ws.iter_rows(min_row=3, values_only=True):
        if r[0] is not None and r[ia] is not None and str(r[ia]).strip():
            out[(str(r[1]), int(r[idt]))] = r[ia]
    wb.close()
    return out


def write_sheet(R: pd.DataFrame, links: list, xl: Path, answers: list | None = None):
    """note-91 layout: label as 'P6 Advance' (phase and function together), plus a one-line question per row.
    links: per row (chart target, print target or None). Answers already typed in `xl` are carried to the same
    (signal, detector) row; refuses to drop any."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    import review72 as R72
    if answers is None and xl.exists():
        old = read_answers(xl)
        answers = [old.pop((str(r.DeviceName), int(r.lead)), None) for r in R.itertuples()]
        if old:
            raise RuntimeError(f"{xl}: answers on rows that are not in the new build, not overwriting: {old}")
    wb = Workbook()
    ws = wb.active
    ws.title = "labels to check"
    ws.append([QUESTION])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(HEAD))
    ws.row_dimensions[1].height = 48
    ws.append(HEAD)
    link = Font(color="0563C1", underline="single")
    cc, cp = HEAD.index("Chart") + 1, HEAD.index("Print") + 1
    for i, r in enumerate(R.itertuples(), 1):
        lab = R72.pf_txt(r.phase, r.had)
        ask = (f"Label says det {int(r.lead)} is {lab}, but it is left out of training (reason in the next column). "
               "Should it be used for training?")
        ch, pdf = links[i - 1]
        ans = answers[i - 1] if answers is not None else None
        ws.append([i, r.DeviceName, int(r.lead), lab, ask, r.why_lead, "chart", "print" if pdf else "none",
                   "" if ans is None else ans])
        n = ws.max_row
        c = ws.cell(n, cc)
        c.hyperlink = ch
        c.font = link
        if pdf:
            c = ws.cell(n, cp)
            c.hyperlink = pdf
            c.font = link
    for j, w in enumerate(WIDTH):
        ws.column_dimensions[chr(65 + j)].width = w
    for c in ws[2]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
    for row in ws.iter_rows(min_row=1):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A3"
    wb.save(xl)


def main(a):
    P = pd.read_parquet(a.pool)
    assert not P.DeviceId.str.lower().isin(R87.locked()).any()
    pool = P[~P.reason.isin(USER_RULES)]
    S = pool.sample(20, random_state=89).reset_index(drop=True)
    S["DeviceName"] = S.DeviceName.where(S.DeviceName.notna(), S.DeviceId.str[:8])
    rows = []
    for r in S.itertuples():
        rows.append(dict(DeviceId=r.DeviceId, DeviceName=r.DeviceName, key="sanity", kind="det", dets=[int(r.Detector)],
                         lead=int(r.Detector), n_dets=1, phases=[], phase=str(r.phase_target) if pd.notna(r.phase_target)
                         else "", had=r.label_print_first if isinstance(r.label_print_first, str) else r.function,
                         action="left out of training", why_lead=why(r), also_lead=[], technology=r.technology,
                         period="stg", reason=r.reason))
    R = pd.DataFrame(rows)
    R87.build_rows = lambda: R               # review87's chart-data extraction on these 20 signals only
    R87.stage_data()
    out = rpath.REPO / "review"
    charts = R87.draw(R, out / f"{NAME}_charts")
    links = []
    for r in R.itertuples():
        pdf = R87.pdf_for(r.DeviceName)
        links.append((f"{NAME}_charts/{Path(charts[R87.stem(r)]).name}",
                      "file:///" + urllib.parse.quote(str(pdf).replace("\\", "/"), safe=":/()_-.,'") if pdf else None))
    write_sheet(R, links, out / f"{NAME}.xlsx")
    R.drop(columns=["dets", "phases", "also_lead"]).to_parquet(Path(a.pool).parent / f"{NAME}_rows.parquet", index=False)
    print(json.dumps(dict(pool=len(pool), by_reason=S.reason.value_counts().to_dict(),
                          with_print=int(sum(R87.pdf_for(n) is not None for n in R.DeviceName))), default=str))
    print(R[["DeviceName", "lead", "phase", "had", "reason"]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    main(ap.parse_args())
