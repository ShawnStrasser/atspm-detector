import os
import re
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_v4_stats.py")
s = open(p, encoding="utf-8").read()
a = s.index("# note 128: the six rolling medians (13 five-minute bins)")
b = s.index("def _win_median(R, h=6):")
s = s[:a] + s[b:]                      # the SQL text goes (ported to numpy below)
a = s.index("def colour_stats(CB, lanes, con=None):")
b = s.index("# ============================================================================ time of day")
new = open(os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"s128\patch\fast_np.py"), encoding="utf-8").read() + '''def colour_stats(CB, lanes):
    """too-fast / too-many statistics per detector from the colour bins (h117_study.stats + h118c_fast.fast_stats;
    note 128: the research DuckDB query ported to numpy with the same float32 / float64 arithmetic, so health opens no
    database connection).  lanes: detector -> model lane span.  Rows sorted by detector."""
    cols = ["detector", "q5_gy", "q5_all", "fo_all", "fem_all", "n_spk", "fo_c", "fem_c", "n_spk_c"]
    if CB is None or not len(CB):
        return pd.DataFrame(columns=cols)
    cb = CB.assign(ln=np.fmax(pd.to_numeric(CB.detector.map(lanes), errors="coerce").fillna(1).to_numpy(float), 1),
                   **_colour_medians(CB))
    A = _fast_stats(cb)
    A["zf"] = (A.fo_all - A.fem_all) / np.sqrt(A.fem_all + 1)
    A["zf_c"] = (A.fo_c - A.fem_c) / np.sqrt(A.fem_c + 1)
    return A


'''
s = s[:a] + new + s[b:]
s = s.replace('''  colour_stats                   h117_events.one + h117_study + h118c_fast    (red / green, too fast, too many)''',
              '''  colour_stats                   h117_events.one + h117_study + h118c_fast    (red / green, too fast, too many;
                                                                              the research SQL in numpy since note 128)''')
open(p, "w", encoding="utf-8").write(s)

p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\health_v4.py")
s = open(p, encoding="utf-8").read()
for a_, b_ in [('''           refs: Refs | None = None, stage1=None, keep_all: bool = False, con=None):''',
                '''           refs: Refs | None = None, stage1=None, keep_all: bool = False):'''),
               ('''    `keep_all` returns every internal column (research comparison).  `con`: an open DuckDB connection to reuse
    (predict passes its own; else a private one is opened for the one small query)."""''',
                '''    `keep_all` returns every internal column (research comparison)."""'''),
               ('''    CS = hs.colour_stats(CB, lanes, con)''', '''    CS = hs.colour_stats(CB, lanes)''')]:
    assert s.count(a_) == 1, a_
    s = s.replace(a_, b_)
open(p, "w", encoding="utf-8").write(s)
p = os.path.join(os.environ.get("DC_WORK", os.path.expanduser("~/dc_work")), r"final_v7_next\src\detector_classifier\pipeline.py")
s = open(p, encoding="utf-8").read()
a_ = '''                h, sig_note = hv4.assess(prep, start_ts, end_ts, phase, flab, func, span, pconf or None, con=con)'''
assert s.count(a_) == 1
s = s.replace(a_, '''                h, sig_note = hv4.assess(prep, start_ts, end_ts, phase, flab, func, span, pconf or None)''')
open(p, "w", encoding="utf-8").write(s)
print("patched")
