"""Independent re-scoring of the 119 locked exam (second implementation, DuckDB SQL).

Written from the specification only; does not read score119.py or its outputs.
Output: %DC_WORK%/s119/verify119.json
"""
import json
import os
from pathlib import Path

import duckdb

WORK = Path(os.environ.get("DC_WORK", Path.home() / "dc_work"))
S = WORK / "s119"
REPO = Path(__file__).resolve().parents[3]
KEY = REPO / "research" / "labels" / "function_labels_locked_v3.parquet"

WINDOWS = [
    ("m5_a", "2026-09-21 07:45", 5), ("m5_b", "2026-09-19 12:20", 5),
    ("m5_c", "2026-09-19 22:10", 5), ("m5_d", "2026-09-18 17:05", 5),
    ("m30_a", "2026-09-21 07:30", 30), ("m30_b", "2026-09-19 12:00", 30),
    ("m30_c", "2026-09-19 21:30", 30), ("m30_d", "2026-09-18 17:00", 30),
    ("h1_a", "2026-09-20 17:00", 60), ("h1_b", "2026-09-20 02:00", 60),
    ("h1_c", "2026-09-21 09:00", 60),
    ("h3_a", "2026-09-21 06:00", 180), ("h3_b", "2026-09-19 14:00", 180),
    ("h24_a", "2026-09-19 00:00", 1440), ("h24_b", "2026-09-20 00:00", 1440),
]
FAMS = ["m5", "m30", "h1", "h3", "h24"]
GE30 = ("m30", "h1", "h3", "h24")
ATSPM = ("Advance", "Presence", "Count", "Yellow_Red")


def p(path):
    return str(path).replace("\\", "/")


def main():
    c = duckdb.connect()
    c.execute("SET memory_limit='10GB'; SET threads=4")
    c.execute(f"SET temp_directory='{p(WORK / 'tmp')}'")

    c.execute("CREATE TABLE wins(win VARCHAR, fam VARCHAR, t0 TIMESTAMP, t1 TIMESTAMP)")
    for w, st, m in WINDOWS:
        c.execute("INSERT INTO wins VALUES (?, split_part(?, '_', 1), CAST(? AS TIMESTAMP), "
                  "CAST(? AS TIMESTAMP) + to_minutes(?))", [w, w, st, st, m])

    c.execute(f"""CREATE TABLE locked AS SELECT lower(DeviceId) dev, "set" AS st
                  FROM read_csv_auto('{p(WORK / 'official' / 'locked_v2.csv')}')""")
    assert c.sql("SELECT count(*) FROM locked").fetchone()[0] == 115

    # predictions (both models); files named <DeviceId>.parquet only
    for m in ("v7", "beta"):
        files = sorted(x for x in (S / "pred" / m).glob("*.parquet")
                       if not x.name.endswith((".time.parquet", ".phases.parquet")))
        assert len(files) == 115, (m, len(files))
        flist = ", ".join(f"'{p(f)}'" for f in files)
        c.execute(f"""CREATE TABLE pred_{m} AS SELECT lower(DeviceId) dev, Detector::BIGINT det, win,
              coalesce(phase_pred, phase_guess) ans_ph,
              coalesce(function_pred, function_guess) ans_fn,
              n_actuations n_act, p_advance, p_presence, p_count, p_yellow_red
            FROM read_parquet([{flist}], union_by_name=true)""")
        dup = c.sql(f"SELECT count(*) - count(DISTINCT (dev, det, win)) FROM pred_{m}").fetchone()[0]
        assert dup == 0, (m, dup)

    # candidate phases per signal-window
    evs = ", ".join(f"'{p(S / 'ev' / (d + '.parquet'))}'" for d in
                    [r[0] for r in c.sql("SELECT dev FROM locked").fetchall()]
                    if (S / 'ev' / (d + '.parquet')).exists())
    c.execute(f"""CREATE TABLE cand AS SELECT DISTINCT lower(e.DeviceId) dev, w.win, e.Parameter::BIGINT ph
        FROM read_parquet([{evs}]) e JOIN wins w ON e.Timestamp >= w.t0 AND e.Timestamp < w.t1
        WHERE e.EventId = 1 AND e.Parameter BETWEEN 1 AND 16""")

    # ---------------- phase ----------------
    c.execute(f"""CREATE TABLE ptruth AS SELECT lower(o.DeviceId) dev, o.Detector::BIGINT det,
              o.target_num::BIGINT tph,
              list_distinct(list_concat(
                 CASE WHEN o.switch_phase > 0 THEN [o.switch_phase::BIGINT] ELSE [] END,
                 list_transform(list_filter(regexp_split_to_array(coalesce(o.additional_call_phases, ''), '[^0-9]+'),
                                            x -> x <> ''), x -> x::BIGINT))) alt
        FROM '{p(WORK / 'official' / 'labels_official.parquet')}' o
        JOIN locked l ON lower(o.DeviceId) = l.dev WHERE o.target_type = 'phase'""")
    c.execute(f"""CREATE TABLE pdis AS SELECT lower(DeviceId) dev, detector det FROM '{p(KEY)}'
        WHERE print_source = 'print' AND print_confidence = 'high' AND phase_diagram IS NOT NULL
          AND regexp_matches(phase_target, '^P[0-9]+$')
          AND phase_diagram <> CAST(substr(phase_target, 2) AS INTEGER)""")
    c.execute("""CREATE TABLE prow AS
        SELECT t.dev, t.det, w.win, w.fam, t.tph, t.alt,
               coalesce(a.n_act, 0) n_act,
               EXISTS (SELECT 1 FROM cand k WHERE k.dev = t.dev AND k.win = w.win AND k.ph = t.tph) green,
               EXISTS (SELECT 1 FROM pdis d WHERE d.dev = t.dev AND d.det = t.det) pdis,
               a.ans_ph a7, b.ans_ph ab
        FROM ptruth t CROSS JOIN wins w
        LEFT JOIN pred_v7 a ON a.dev = t.dev AND a.det = t.det AND a.win = w.win
        LEFT JOIN pred_beta b ON b.dev = t.dev AND b.det = t.det AND b.win = w.win""")

    out = {"phase": {}, "function": {}, "lanes": {}, "counts": {}}
    fam_sets = {f: (f,) for f in FAMS}
    fam_sets["ge30"] = GE30

    def fin(fams):
        return "(" + ", ".join(f"'{f}'" for f in fams) + ")"

    for m, col in (("v7", "a7"), ("beta", "ab")):
        for name, fams in fam_sets.items():
            r = c.sql(f"""SELECT count(*) n,
                    avg(CASE WHEN {col} IS NOT NULL AND {col} = tph THEN 1.0 ELSE 0.0 END) acc,
                    count(*) FILTER (WHERE NOT pdis) nr,
                    avg(CASE WHEN {col} IS NOT NULL AND ({col} = tph OR list_contains(alt, {col}))
                             THEN 1.0 ELSE 0.0 END) FILTER (WHERE NOT pdis) accr
                FROM prow WHERE n_act >= 5 AND green AND fam IN {fin(fams)}""").fetchone()
            out["phase"][f"{m}/{name}/E"] = {"acc": r[1], "n": r[0]}
            out["phase"][f"{m}/{name}/R"] = {"acc": r[3], "n": r[2]}
    r = c.sql(f"""SELECT count(*) FILTER (WHERE n_act = 0), count(*) FILTER (WHERE n_act BETWEEN 1 AND 4),
                  count(*) FILTER (WHERE n_act >= 5 AND NOT green)
                  FROM prow WHERE fam IN {fin(GE30)}""").fetchone()
    out["counts"]["phase_ge30"] = {"n_act0": r[0], "n_act1_4": r[1], "never_green": r[2],
                                   "n_pdis_detectors": c.sql("SELECT count(*) FROM pdis").fetchone()[0]}

    # ---------------- function ----------------
    c.execute(f"""CREATE TABLE key AS SELECT lower(DeviceId) dev, detector det, truth_v3, n1_pending_exam n1p,
                  func7_v2, print_function, exclude_score, yr_count_identical, stack_group, stack_role
                  FROM '{p(KEY)}'""")
    # n1 decisions with v7 answers on ge30 windows with n_act >= 5
    c.execute(f"""CREATE TABLE n1v AS
        WITH v AS (SELECT k.dev, k.det, a.ans_fn FROM key k
                   JOIN pred_v7 a ON a.dev = k.dev AND a.det = k.det
                   JOIN wins w ON w.win = a.win
                   WHERE k.n1p AND a.n_act >= 5 AND w.fam IN {fin(GE30)}),
             cnt AS (SELECT dev, det, ans_fn, count(*) c FROM v GROUP BY ALL),
             tot AS (SELECT dev, det, sum(c) n, max(c) mx FROM cnt GROUP BY ALL),
             top AS (SELECT c.dev, c.det, c.ans_fn maj, t.n, t.mx, count(*) OVER (PARTITION BY c.dev, c.det) nties
                     FROM cnt c JOIN tot t USING (dev, det) WHERE c.c = t.mx
                     QUALIFY row_number() OVER (PARTITION BY c.dev, c.det ORDER BY c.ans_fn) = 1)
             -- a tie means share <= 0.5, so it is 'out_share' whichever tied class is kept
        SELECT k.dev, k.det, k.func7_v2, k.print_function, t.maj, coalesce(t.n, 0) n, t.mx, t.nties,
          CASE WHEN coalesce(t.n, 0) < 2 THEN 'out_fewwin'
               WHEN t.mx::DOUBLE / t.n < 0.6 THEN 'out_share'
               WHEN t.maj IS NOT DISTINCT FROM k.func7_v2 THEN 'hand'
               WHEN t.maj IS NOT DISTINCT FROM k.print_function THEN 'print'
               ELSE 'out_neither' END decision
        FROM key k LEFT JOIN top t USING (dev, det) WHERE k.n1p""")
    assert c.sql("SELECT count(*) - count(DISTINCT (dev, det)) FROM n1v").fetchone()[0] == 0
    out["counts"]["n1_decisions"] = dict(c.sql("SELECT decision, count(*) FROM n1v GROUP BY 1").fetchall())
    out["counts"]["n1_by_class"] = {f"{a}/{b}": n for a, b, n in c.sql(
        """SELECT decision, CASE decision WHEN 'hand' THEN func7_v2 WHEN 'print' THEN print_function END, count(*)
           FROM n1v WHERE decision IN ('hand', 'print') GROUP BY ALL ORDER BY ALL""").fetchall()}

    c.execute("""CREATE TABLE ftruth AS
        SELECT k.*, coalesce(k.truth_v3,
                  CASE n.decision WHEN 'hand' THEN n.func7_v2 WHEN 'print' THEN n.print_function END) t,
               (k.truth_v3 IS NULL AND n.decision IN ('hand', 'print')) is_n1
        FROM key k LEFT JOIN n1v n USING (dev, det)""")
    c.execute("""CREATE TABLE fkeep AS SELECT * FROM ftruth
        WHERE t IN ('Advance','Presence','Count','Yellow_Red','Other','Mid','Bike')
          AND exclude_score IS NOT TRUE AND yr_count_identical IS NOT TRUE""")
    # stack groups whose every key member has t == stack_role
    c.execute("""CREATE TABLE okgrp AS SELECT stack_group, any_value(stack_role) AS srole FROM ftruth
        WHERE stack_group IS NOT NULL GROUP BY 1
        HAVING bool_and(t IS NOT DISTINCT FROM stack_role) AND count(DISTINCT stack_role) = 1""")
    out["counts"]["stack_groups_ok"] = c.sql("SELECT count(*) FROM okgrp").fetchone()[0]

    for m in ("v7", "beta"):
        c.execute(f"""CREATE OR REPLACE TABLE fr_{m} AS
            SELECT f.dev, f.det, f.t, f.is_n1, w.win, w.fam, b.ans_fn ans,
                   CASE WHEN g.stack_group IS NOT NULL THEN f.stack_group END sg, g.srole,
                   CASE g.srole WHEN 'Advance' THEN b.p_advance WHEN 'Presence' THEN b.p_presence
                               WHEN 'Count' THEN b.p_count WHEN 'Yellow_Red' THEN b.p_yellow_red END prole,
                   CASE WHEN f.t IN {ATSPM} THEN coalesce(b.ans_fn = f.t, false)
                        ELSE b.ans_fn IS NOT NULL AND b.ans_fn NOT IN {ATSPM} END base
            FROM fkeep f JOIN pred_v7 a ON a.dev = f.dev AND a.det = f.det
            JOIN wins w ON w.win = a.win
            LEFT JOIN pred_{m} b ON b.dev = f.dev AND b.det = f.det AND b.win = a.win
            LEFT JOIN okgrp g ON g.stack_group = f.stack_group
            WHERE a.n_act >= 5""")
        # stack credit
        c.execute(f"""CREATE OR REPLACE TABLE fs_{m} AS
            WITH x AS (SELECT *,
                    count(*) OVER gw ng,
                    count(*) FILTER (WHERE ans = srole) OVER gw n_role,
                    count(*) FILTER (WHERE ans IS NOT NULL AND ans NOT IN {ATSPM}) OVER gw n_non,
                    row_number() OVER (PARTITION BY sg, win, (ans = srole) ORDER BY prole DESC NULLS LAST, det) rk_role,
                    row_number() OVER (PARTITION BY sg, win, (ans IS NOT NULL AND ans NOT IN {ATSPM})
                                       ORDER BY prole DESC NULLS LAST, det) rk_non
                 FROM fr_{m} WINDOW gw AS (PARTITION BY sg, win))
            SELECT *, CASE
                WHEN sg IS NULL OR ng < 2 THEN base
                WHEN n_role > 0 THEN CASE WHEN ans = srole THEN rk_role = 1
                                          WHEN ans IS NOT NULL AND ans NOT IN {ATSPM} THEN true
                                          ELSE base END
                WHEN n_non > 0 THEN CASE WHEN ans IS NOT NULL AND ans NOT IN {ATSPM} THEN rk_non > 1
                                         ELSE base END
                ELSE base END ok
            FROM x""")
        out["counts"][f"stack_rows_{m}"] = c.sql(
            f"SELECT count(*), count(*) FILTER (WHERE ok <> base) FROM fs_{m} WHERE sg IS NOT NULL AND ng >= 2").fetchone()
        for name, fams in fam_sets.items():
            r = c.sql(f"SELECT avg(ok::INT), count(*) FROM fs_{m} WHERE fam IN {fin(fams)}").fetchone()
            out["function"][f"{m}/{name}"] = {"acc": r[0], "n": r[1]}
        r = c.sql(f"SELECT avg(ok::INT), count(*) FROM fs_{m} WHERE fam IN {fin(GE30)} AND NOT is_n1").fetchone()
        out["function"][f"{m}/ge30_no_n1"] = {"acc": r[0], "n": r[1]}
        r = c.sql(f"SELECT count(*) FROM fs_{m} WHERE fam IN {fin(GE30)} AND is_n1").fetchone()
        out["function"][f"{m}/ge30_n1_rows"] = r[0]
        for cls, a, n in c.sql(f"""SELECT CASE WHEN t IN {ATSPM} THEN t ELSE 'nonATSPM' END cls,
                avg(ok::INT), count(*) FROM fs_{m} WHERE fam IN {fin(GE30)} GROUP BY 1 ORDER BY 1""").fetchall():
            out["function"][f"{m}/ge30/{cls}"] = {"acc": a, "n": n}

    # ---------------- lanes (v7) ----------------
    files = sorted((S / "pred" / "v7").glob("*.phases.parquet"))
    flist = ", ".join(f"'{p(f)}'" for f in files)
    c.execute(f"""CREATE TABLE lp AS SELECT lower(DeviceId) dev, phase::BIGINT ph, win, n_lanes
                  FROM read_parquet([{flist}], union_by_name=true)""")
    c.execute(f"""CREATE TABLE lt AS SELECT lower(DeviceId) dev, CAST(substr(phase_target, 2) AS BIGINT) ph,
                  mode(n_lanes_phase) nl FROM '{p(KEY)}'
                  WHERE regexp_matches(phase_target, '^P[0-9]+$') AND n_lanes_phase IS NOT NULL GROUP BY ALL""")
    r = c.sql(f"""SELECT count(*), avg((l.n_lanes = t.nl)::INT) FROM lp l JOIN lt t USING (dev, ph)
                  JOIN wins w ON w.win = l.win WHERE w.fam IN {fin(GE30)} AND l.n_lanes IS NOT NULL""").fetchone()
    out["lanes"]["v7/ge30"] = {"n": r[0], "exact": r[1], "n_files": len(files)}

    (S / "verify119.json").write_text(json.dumps(out, indent=1, default=str))
    for sec in ("phase", "function", "lanes"):
        for k, v in out[sec].items():
            if isinstance(v, dict) and v.get("acc") is not None:
                print(f"{sec:9s} {k:28s} {v['acc']:.4f}  n={v['n']}")
            else:
                print(f"{sec:9s} {k:28s} {v}")
    print(json.dumps(out["counts"], default=str))


if __name__ == "__main__":
    main()
