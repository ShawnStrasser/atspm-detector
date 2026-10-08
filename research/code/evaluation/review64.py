"""Note 64 part 2: builder for the user's FINAL error-review spreadsheet (what the candidate still gets wrong).

Reads the integrated OOF rows of `cand64.py` (`%DC_WORK%/cand64/phase_rows.parquet`, `function_rows.parquet`). Scope:
Sept-2026 log (period stg; the labels describe today's cabinet), samples >= 30 min (30 min / 1 h / 3 h / 6 h / 24 h / full).
One ITEM = one detector. A detector is listed when the candidate is wrong, after the exclusions below, in at least half of
its >= 30-min samples (a consistent disagreement, not a one-window wobble). Phase errors take precedence over function
errors of the same detector (a wrong phase usually drags the function with it).
EXCLUDED (counted in the dry run, never listed):
  phase     label check fail / misconfigured; high-confidence print phase differs from the timing (known mislabel);
            prediction = the timing's switch phase or an additional call phase; prediction = the phase that runs with an
            overlap the detector calls (call_overlap / additional_call_overlaps; overlap green best-matched to a phase).
  function  label check fail / misconfigured; known field issue on the label row (e.g. presence zone set to pulse);
            Yellow_Red <-> Count swaps between co-actuating stop-bar twins (scoring question already open with the user,
            AGENTS: never ask the same thing twice); detectors whose phase is wrong (listed as a phase item instead).
Ranked: signals by their most confident item, items by confidence (mean probability the model gave its own answer over
the wrong samples). Charts are drawn from SAVED sample data (`%DC_WORK%/cand64/review_data/`), never re-scored.

    python review64.py                  # dry run: counts by error type -> %DC_WORK%/cand64/review_dryrun.json (default)
    python review64.py --preview 4      # + chart data, charts and a sheet for the top 4 items into %DC_WORK%/cand64/review_preview/
    python review64.py --write          # the real thing into review/ (ONLY when the orchestrator says so)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if p.name == "code")))
import rpath  # noqa: F401,E402
from common import DC_WORK  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
C64 = DC_WORK / "cand64"
DATA = C64 / "review_data"
STG = DC_WORK / "official" / "stg" / "cache"
GE30 = ["m30", "h1", "h3", "h6", "h24", "full"]
ATS = {"Advance", "Presence", "Count", "Yellow_Red"}
PDFS = [DC_WORK / "cabinet" / "pdf"]
TWIN_THR = 0.4
MAJ = 0.5
QUESTION = ("These are the detectors where the finished model still disagrees with the label on samples of 30 minutes or "
            "longer (Sept 2026 log). Detectors already known to be mislabelled or misconfigured, and cases explained by a "
            "switch phase, an additional call phase or an overlap call, are left out. For each row: is the label right "
            "(L), the model right (M), or can you not tell (?)")
NAME = {"Advance": "advance", "Presence": "stop-bar presence", "Count": "stop-bar count", "Yellow_Red": "yellow-red",
        "Mid": "mid", "Bike": "bike", "Other": "other"}
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1", "#7a7a7a"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def locked() -> set:
    return set(pd.read_csv(DC_WORK / "official" / "locked_v2.csv").DeviceId.str.lower())


# ------------------------------------------------------------------------------------------------ facts
def label_facts() -> pd.DataFrame:
    v = pd.read_parquet(rpath.LABELS_CURRENT,  # note 81: v4l
                        columns=["DeviceId", "DeviceName", "detector", "phase_target", "technology", "field_issue",
                                 "validated", "print_source", "print_confidence", "phase_diagram", "truth_v3s"])
    v["DeviceId"] = v.DeviceId.str.lower()
    return v.rename(columns={"detector": "Detector"}).drop_duplicates(["DeviceId", "Detector"])


def overlap_phases() -> pd.DataFrame:
    """(DeviceId, Detector) -> set of phases that run with an overlap the detector calls (Sept log)."""
    off = pd.read_parquet(DC_WORK / "official" / "labels_official.parquet",
                          columns=["DeviceId", "Detector", "call_overlap", "additional_call_overlaps"])
    off["DeviceId"] = off.DeviceId.str.lower()
    ov = pd.read_csv(DC_WORK / "official" / "ovl_stg" / "overlap_green_similarity.csv")
    ov["DeviceId"] = ov.DeviceId.str.lower()
    best = {(d, int(o)): int(p) for d, o, p in zip(ov.DeviceId, ov.overlap, ov.best_phase) if pd.notna(p)}
    out = []
    for d, det, co, ad in zip(off.DeviceId, off.Detector, off.call_overlap, off.additional_call_overlaps):
        ovs = ({int(co)} if pd.notna(co) and co > 0 else set()) | \
              {int(x) for x in str(ad if ad is not None else "").replace(";", ",").split(",") if x.strip().isdigit()}
        ph = {best[(d, o)] for o in ovs if (d, o) in best}
        if ph:
            out.append((d, int(det), ph))
    return pd.DataFrame(out, columns=["DeviceId", "Detector", "ovl_phases"])


def twin_set(f: pd.DataFrame) -> set:
    """(DeviceId, Detector, win) of stop-bar zones that co-actuate both ways (>= .4, |lag| <= 1 s; note 62's rule on the
    >= 30-min windows) with another detector of the same predicted phase that the model calls Count or Yellow_Red."""
    p = pd.read_parquet(DC_WORK / "s62" / "pairs_m30_h1_h3_h6_h24_full.parquet")
    p = p[(p.period == "stg") & (np.minimum(p.m_ab, p.m_ba) >= TWIN_THR) & (p.lag.abs() <= 1.0)]
    s = set()
    for d, w, a, b in zip(p.DeviceId, p.win, p.da, p.db):
        s.add((d, int(a), w))
        s.add((d, int(b), w))
    return s


# ------------------------------------------------------------------------------------------------ items
def build_items(arm: str):
    lk = locked()
    L = label_facts()
    # ---------------- phase
    ph = pd.read_parquet(C64 / "phase_rows.parquet")
    ph = ph[ph.DeviceId.str.endswith("@stg") & ph.fam.isin(GE30) & ph.everything].copy()
    ph["DeviceId"] = ph.dev_plain
    assert not ph.DeviceId.isin(lk).any()
    ph = ph.merge(overlap_phases(), on=["DeviceId", "Detector"], how="left")
    ph = ph.merge(L[["DeviceId", "Detector", "field_issue"]], on=["DeviceId", "Detector"], how="left")
    alt = ph.alt_s.fillna("").map(lambda s: {int(x) for x in s.split(",") if x})
    wrong = ph.pred_cand != ph.Phase
    why = np.full(len(ph), "", object)
    ovl = np.array([isinstance(o, set) and p in o for p, o in zip(ph.pred_cand, ph.ovl_phases)])
    altm = np.array([p in a for p, a in zip(ph.pred_cand, alt)])
    for m, r in ((ph.lab_bad.to_numpy(), "label check fail / misconfigured"),
                 (ph.print_phase_disagrees.to_numpy(), "print phase differs from the timing (known mislabel)"),
                 (altm, "switch phase / additional call phase"),
                 (ovl, "phase of a called overlap")):
        why[wrong.to_numpy() & (why == "") & m] = r
    ph["why_excl"] = why
    ph["err_kept"] = wrong & (ph.why_excl == "")
    # ---------------- function
    f = pd.read_parquet(C64 / "function_rows.parquet")
    f = f[(f.period == "stg") & f.wgroup.isin(GE30) & f[f"ok_E_{arm}"].notna()].copy()
    assert not f.DeviceId.isin(lk).any()
    f = f.merge(L[["DeviceId", "Detector", "field_issue"]].astype({"Detector": f.Detector.dtype}),
                on=["DeviceId", "Detector"], how="left")
    pred = f[f"pred_{arm}"].to_numpy(object)
    err = f[f"err_E_{arm}"].to_numpy(object)
    wrongf = f[f"ok_E_{arm}"].to_numpy() == 0
    tw = twin_set(f)
    istw = np.array([(d, int(x), w) in tw for d, x, w in zip(f.DeviceId, f.Detector, f.win)])
    yrc = np.array([{t, p} == {"Count", "Yellow_Red"} for t, p in zip(f.truth_v3s, pred)])
    phw = ph[ph.pred_cand != ph.Phase][["DeviceId", "Detector", "win"]].assign(phase_wrong=True)
    f = f.merge(phw.astype({"Detector": f.Detector.dtype}).drop_duplicates(), on=["DeviceId", "Detector", "win"], how="left")
    whyf = np.full(len(f), "", object)
    for m, r in ((f.validated.isin(["fail", "misconfigured"]).to_numpy(), "label check fail / misconfigured"),
                 (f.field_issue.notna().to_numpy(), "known field issue"),
                 (yrc & istw, "yellow-red / count twin swap (open question)"),
                 (f.phase_wrong.eq(True).to_numpy(), "phase wrong (phase item)")):
        whyf[wrongf & (whyf == "") & m] = r
    f["why_excl"] = whyf
    f["err_kept"] = wrongf & (whyf == "")
    f["etype"] = np.where(err == "stack_extra", "stacked extra lane",
                          [f"{NAME.get(t, t)} called {NAME.get(p, p)}" for t, p in zip(f.truth_v3s, pred)])
    f["ekind"] = err
    f["p_pred"] = f[f"p_pred_{arm}"]
    f["pred"] = pred
    # ---------------- per detector
    gk = ["DeviceId", "Detector"]
    P = ph.groupby(gk).agg(n=("win", "size"), n_err=("err_kept", "sum"), n_wrong=("pred_cand", lambda s: 0)).reset_index()
    P["n_wrong"] = ph.assign(w=wrong).groupby(gk).w.sum().to_numpy()
    lead = ph[ph.err_kept].groupby(gk + ["pred_cand"]).agg(k=("win", "size"), conf=("p_cand", "mean")).reset_index()
    lead = lead.sort_values(["k", "conf"], ascending=False).drop_duplicates(gk)
    P = P.merge(lead, on=gk, how="left").merge(ph.groupby(gk).Phase.first().reset_index(), on=gk)
    P["item"] = P.n_err >= MAJ * P.n
    F = f.groupby(gk).agg(n=("win", "size"), n_err=("err_kept", "sum")).reset_index()
    leadf = f[f.err_kept].groupby(gk + ["pred", "ekind", "etype"]).agg(k=("win", "size"), conf=("p_pred", "mean")).reset_index()
    leadf = leadf.sort_values(["k", "conf"], ascending=False).drop_duplicates(gk)
    F = F.merge(leadf, on=gk, how="left").merge(f.groupby(gk).truth_v3s.first().reset_index(), on=gk)
    F["item"] = F.n_err >= MAJ * F.n
    ph_items = P[P.item].assign(kind="phase", etype="phase wrong")
    fn_items = F[F.item & ~F.set_index(gk).index.isin(ph_items.set_index(gk).index)].assign(kind="function")
    items = pd.concat([ph_items.rename(columns={"pred_cand": "pred_phase_item"}), fn_items], ignore_index=True)
    names = L.set_index(["DeviceId", "Detector"])
    items = items.merge(L[["DeviceId", "Detector", "DeviceName", "phase_target", "technology", "truth_v3s"]].rename(
        columns={"truth_v3s": "label_function"}), on=gk, how="left")
    items["sig_rank"] = items.groupby("DeviceId").conf.transform("max")
    items = items.sort_values(["sig_rank", "DeviceId", "conf"], ascending=[False, True, False]).reset_index(drop=True)
    # function prediction of phase items (for the 'Model says' cell) and phase of function items
    fp = f.groupby(gk).pred.agg(lambda s: s.value_counts().index[0]).rename("model_function")
    items = items.merge(fp.reset_index(), on=gk, how="left")
    pp = ph.groupby(gk).pred_cand.agg(lambda s: s.value_counts().index[0]).rename("model_phase")
    items = items.merge(pp.reset_index(), on=gk, how="left")
    items["model_function"] = np.where(items.kind == "function", items.pred, items.model_function)
    fph = f.groupby(gk).pred_phase.agg(lambda s: s.value_counts().index[0] if s.notna().any() else np.nan)
    items["model_phase"] = items.model_phase.fillna(items.set_index(gk).index.map(fph).to_series(index=items.index))
    del names
    # ---------------- dry-run counts
    dry = {"scope": "Sept 2026 log, samples >= 30 min, detector listed when wrong (after exclusions) in >= half of its "
                    f"samples; function arm = {arm}",
           "detector_windows": {"phase_scored": int(len(ph)), "phase_wrong": int(wrong.sum()),
                                "phase_wrong_kept": int(ph.err_kept.sum()),
                                "function_scored": int(len(f)), "function_wrong": int(wrongf.sum()),
                                "function_wrong_kept": int(f.err_kept.sum())},
           "excluded_detector_windows": {"phase": ph.why_excl[ph.why_excl != ""].value_counts().to_dict(),
                                         "function": f.why_excl[f.why_excl != ""].value_counts().to_dict()},
           "detectors_with_any_kept_error": {"phase": int((P.n_err > 0).sum()), "function": int((F.n_err > 0).sum())},
           "rows_by_error_type": items.etype.value_counts().to_dict(),
           "rows_by_kind": items.kind.value_counts().to_dict(),
           "rows_by_score_kind": items.ekind.fillna("phase").value_counts().to_dict(),
           "rows_total": int(len(items)), "signals": int(items.DeviceId.nunique()),
           "rows_by_min_confidence": {f">= {t:.1f}": [int((items.conf >= t).sum()), int(items[items.conf >= t].DeviceId.nunique())]
                                      for t in (0.5, 0.6, 0.7, 0.8, 0.9)},
           "rows_by_kind_conf_ge_0.8": items[items.conf >= 0.8].kind.value_counts().to_dict(),
           "rows_by_technology": items.technology.fillna("unknown").value_counts().to_dict(),
           "prints_on_file": int(sum(pdf_for(n) is not None for n in items.DeviceName.drop_duplicates()))}
    return items, dry


def pdf_for(name):
    if not isinstance(name, str):
        return None
    for d in PDFS:
        fs = sorted(d.glob(f"{name}_*.pdf"))
        if fs:
            return fs[0]
    return None


# ------------------------------------------------------------------------------------------------ chart data + charts
def stem_of(r) -> str:
    return f"{r.DeviceName or r.DeviceId[:8]}_d{int(r.Detector)}_{r.kind}"


def save_chart_data(items: pd.DataFrame):
    """per item: the detector's ONs and its labelled phase's detectors' ONs over the Sept sample, and the signal's phase
    cycles -> parquet under review_data/ (charts are drawn from these, nothing re-scored)."""
    import duckdb
    DATA.mkdir(parents=True, exist_ok=True)
    cn = duckdb.connect()
    cn.execute("SET memory_limit='10GB'")
    cn.execute("SET threads=6")
    cn.execute(f"SET temp_directory='{(DC_WORK / 'tmp').as_posix()}'")
    L = label_facts()
    for r in items.itertuples():
        st = stem_of(r)
        if (DATA / f"{st}_on.parquet").exists():
            continue
        tgt = r.phase_target if isinstance(r.phase_target, str) else ""
        mates = L[(L.DeviceId == r.DeviceId) & (L.phase_target == tgt)].Detector.astype(int).tolist() if tgt else []
        dets = sorted(set(mates) | {int(r.Detector)})
        iv = cn.sql(f"""SELECT Detector::INT det, t_on, t_off FROM '{(STG / 'det_intervals.parquet').as_posix()}'
                        WHERE lower(DeviceId) = '{r.DeviceId}' AND Detector IN ({','.join(map(str, dets))})
                        ORDER BY t_on""").df().drop_duplicates()
        iv.to_parquet(DATA / f"{st}_on.parquet", index=False)
        cy = cn.sql(f"""SELECT Phase::INT phase, green_start, yellow_start, red_start, next_green
                        FROM '{(STG / 'phase_cycles.parquet').as_posix()}' WHERE lower(DeviceId) = '{r.DeviceId}'
                        ORDER BY green_start""").df()
        cy.to_parquet(DATA / f"{st}_cycles.parquet", index=False)


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def green_share(on: np.ndarray, cy: pd.DataFrame) -> pd.Series:
    """share of ON starts that fall in each phase's green (phases with a green in the sample)."""
    out = {}
    for p, g in cy.groupby("phase"):
        gs = g.green_start.to_numpy("datetime64[ns]").astype("int64")
        ge = g.yellow_start.fillna(g.red_start).to_numpy("datetime64[ns]").astype("int64")
        i = np.searchsorted(gs, on, side="right") - 1
        ok = (i >= 0) & (on < ge[np.clip(i, 0, None)])
        out[int(p)] = float(ok.mean()) if len(on) else np.nan
    return pd.Series(out).sort_index()


def draw(items: pd.DataFrame, out_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for r in items.itertuples():
        st = stem_of(r)
        iv = pd.read_parquet(DATA / f"{st}_on.parquet")
        cy = pd.read_parquet(DATA / f"{st}_cycles.parquet")
        det = int(r.Detector)
        me = iv[iv.det == det]
        on = me.t_on.to_numpy("datetime64[ns]").astype("int64")
        lab_ph = int(str(r.phase_target)[1:]) if isinstance(r.phase_target, str) and r.phase_target[1:].isdigit() else None
        mod_ph = int(r.model_phase) if pd.notna(r.model_phase) else None
        fig = plt.figure(figsize=(10, 6.2), facecolor=SURF)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], width_ratios=[1.6, 1])
        # (1) top: phase item = share of ONs starting in each phase's green; function item = 15-min counts vs phase mates
        ax = fig.add_subplot(gs[0, :])
        _style(ax)
        if r.kind == "phase":
            sh = green_share(on, cy)
            col = [C[1] if p == mod_ph else C[0] if p == lab_ph else "#c9c8c3" for p in sh.index]
            ax.bar([f"P{p}" for p in sh.index], 100 * sh.to_numpy(), color=col)
            ax.set_ylabel("% of its actuations that start in green", color=INK2, fontsize=9)
            ax.set_title(f"blue = labelled phase P{lab_ph}, orange = model's phase P{mod_ph}", fontsize=9, color=INK2,
                         loc="left")
        else:
            iv["t"] = iv.t_on.dt.floor("15min")
            cnt = iv.groupby(["t", "det"]).size().unstack(fill_value=0)
            others = [d for d in cnt.columns if d != det][:4]
            for i, d in enumerate([det] + others):
                if d in cnt:
                    ax.plot(cnt.index, cnt[d], color=C[i], lw=1.9 if d == det else 1.1,
                            label=f"det {d}" + (" (this one)" if d == det else ""))
            ax.set_ylabel("actuations per 15 min", color=INK2, fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
        # (2) bottom left: 10 busy daytime minutes, ONs + green of the labelled (and model) phase
        a2 = fig.add_subplot(gs[1, 0])
        _style(a2)
        c10 = me.set_index("t_on").resample("10min").size()
        c10 = c10[(c10.index.hour >= 7) & (c10.index.hour < 19)]
        if len(c10):
            busy = c10[c10 >= c10.quantile(0.6)]
            t0 = busy.index[len(busy) // 2]
        else:
            t0 = me.t_on.min() if len(me) else cy.green_start.min()
        t0 = pd.Timestamp(t0)
        t1 = t0 + pd.Timedelta(minutes=10)
        rows_y = []
        for p in dict.fromkeys(q for q in (lab_ph, mod_ph) if q is not None):
            g = cy[(cy.phase == p) & (cy.yellow_start >= t0) & (cy.green_start <= t1)]
            y = len(rows_y)
            a2.barh([y] * len(g), (g.yellow_start - g.green_start).dt.total_seconds() / 86400, left=g.green_start,
                    height=0.6, color=C[2], alpha=0.8)
            rows_y.append(f"P{p} green" + (" (label)" if p == lab_ph else " (model)"))
        dets = [det] + ([d for d in sorted(iv.det.unique()) if d != det][:3] if r.kind == "function" else [])
        for d in dets:
            s = iv[(iv.det == d) & (iv.t_off >= t0) & (iv.t_on <= t1)]
            y = len(rows_y)
            a2.barh([y] * len(s), (s.t_off - s.t_on).dt.total_seconds().fillna(0).clip(lower=1.0) / 86400, left=s.t_on,
                    height=0.6, color=C[0] if d == det else C[5])
            rows_y.append(f"det {d}")
        a2.set_yticks(range(len(rows_y)))
        a2.set_yticklabels(rows_y, fontsize=8)
        a2.set_xlim(t0, t1)
        a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        a2.grid(axis="y", visible=False)
        a2.grid(axis="x", color=GRID, lw=0.8)
        a2.set_title("10 busy daytime minutes (ONs under 1 s drawn 1 s wide)", fontsize=9, color=INK2, loc="left")
        # (3) bottom right: how long each ON lasts
        a3 = fig.add_subplot(gs[1, 1])
        _style(a3)
        du = (me.t_off - me.t_on).dt.total_seconds().dropna().clip(0, 30)
        a3.hist(du, bins=np.r_[0, 0.15, 0.5, 1, 2, 3, 5, 8, 12, 20, 30], color=C[0])
        a3.set_xscale("symlog", linthresh=1)
        a3.set_xlabel("seconds ON (one bar = one actuation length band)", color=INK2, fontsize=8)
        a3.set_title(f"{(du <= 0.15).mean():.0%} of ONs are one-tick pulses", fontsize=9, color=INK2, loc="left")
        lab = f"P{lab_ph} {NAME.get(r.label_function, r.label_function)}"
        mod = f"P{mod_ph} {NAME.get(r.model_function, r.model_function)}"
        fig.suptitle(f"{r.DeviceName} det {det} ({r.technology or 'technology unknown'}): label {lab}, model {mod}  -  "
                     f"Sept 2026 log, {len(me):,} actuations", x=0.01, ha="left", fontsize=10, color=INK)
        fig.tight_layout()
        png = out_dir / f"{st}.png"
        fig.savefig(png, dpi=100, facecolor=SURF)
        plt.close(fig)
        res[st] = png
    return res


def why_text(r) -> str:
    k = f"wrong in {int(r.n_err)} of {int(r.n)} samples"
    if r.kind == "phase":
        return f"model puts it on P{int(r.model_phase)}; {k}"
    return f"{r.etype}; {k}"


def write_sheet(items: pd.DataFrame, charts: dict, xl: Path, chart_rel: str):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    ws = wb.active
    ws.title = "still wrong"
    ws.append([QUESTION])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=10)
    ws.row_dimensions[1].height = 62
    cols = ["#", "Signal", "Det", "Label", "Model says", "How sure", "Why listed", "Chart", "Print", "Your answer (L / M / ?)"]
    ws.append(cols)
    link = Font(color="0563C1", underline="single")
    for i, r in enumerate(items.itertuples(), 1):
        lab = f"P{str(r.phase_target)[1:]} {NAME.get(r.label_function, r.label_function)}"
        mod = (f"P{int(r.model_phase)} " if pd.notna(r.model_phase) else "") + NAME.get(r.model_function, r.model_function)
        pdf = pdf_for(r.DeviceName)
        ws.append([i, r.DeviceName, int(r.Detector), lab, mod, f"{r.conf:.0%}", why_text(r), "open chart",
                   "open print" if pdf else "no print on file", ""])
        n = ws.max_row
        c = ws.cell(n, 8)
        c.hyperlink = f"{chart_rel}/{Path(charts[stem_of(r)]).name}"
        c.font = link
        if pdf:
            c = ws.cell(n, 9)
            c.hyperlink = "file:///" + urllib.parse.quote(str(pdf), safe=":\\/()_-.,'")
            c.font = link
    for j, w in enumerate((4, 9, 5, 22, 22, 8, 34, 11, 11, 12)):
        ws.column_dimensions[chr(65 + j)].width = w
    for c in ws[2]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
    for row in ws.iter_rows(min_row=1):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A3"
    xl.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xl)
    log(f"saved {xl} ({len(items)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="trees", choices=["trees", "fj"],
                    help="function arm: trees (six folds) or fj (only once fj exists for all six folds)")
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--min-conf", type=float, default=0.0, dest="min_conf",
                    help="list only items the model is at least this sure of (the dry run counts every threshold)")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.arm == "fj":
        meta = json.load(open(C64 / "function_meta.json"))
        assert len(meta["fj_folds"]) == 6, "fj arm needs all six folds"
    items, dry = build_items(a.arm)
    items.drop(columns=[c for c in items.columns if items[c].map(lambda v: isinstance(v, set)).any()]) \
        .to_parquet(C64 / "review_items.parquet", index=False)
    json.dump(dry, open(C64 / "review_dryrun.json", "w"), indent=1, default=str)
    log(json.dumps(dry, indent=1, default=str))
    items = items[items.conf >= a.min_conf].reset_index(drop=True)
    if a.preview:
        sel = pd.concat([items[items.kind == "phase"].head((a.preview + 1) // 2),
                         items[items.kind == "function"].head(a.preview // 2)])
        save_chart_data(sel)
        d = C64 / "review_preview"
        ch = draw(sel, d / "charts")
        write_sheet(sel, ch, d / "final_error_review_preview.xlsx", "charts")
    if a.write:
        save_chart_data(items)
        ch = draw(items, REPO / "review" / "final_error_review_charts")
        write_sheet(items, ch, REPO / "review" / "final_error_review.xlsx", "final_error_review_charts")


if __name__ == "__main__":
    main()
