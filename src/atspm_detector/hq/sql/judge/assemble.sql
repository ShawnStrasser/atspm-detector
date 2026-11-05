{# The scored table X, holds with their flags, limits, the pre-pass and the clean silence yardstick -> xf
   (health_duckdb._assemble, up to the signal-wide checks). #}
{% import "_lib.sql" as lib with context %}
{% import "judge/verdict.sql" as vd with context %}
{% set mh_on = opt['mate_drop'] or opt['count_rise'] %}
-- @step xx0 mat
SELECT x.*, a.rep_frac, a.max_on_s, a.rep_time_s, a.dur_p50, a.n_on_a, a.n_rep, a.n_start, a.n_ge5, a.n_ge60,
       a.long_max_s, s.elhi_cnt, s.elhi_occ, k.c_occ_like, q.q5_gy, q.q5_all, f.fe_n, f.fe_p, f.fe_phi, t.N, t.n_fast,
       t.fast_rate, t.gap_lim, t.pair_share, x.n_on / x.hours AS rate,
       CASE WHEN x.n_on / x.hours < 20 THEN 'low' WHEN x.n_on / x.hours < 100 THEN 'medium' ELSE 'high' END AS band,
       x.fn || ' ' || x.span AS type, greatest(x.ep_dur, a.max_on_s) AS stuck_x,
       coalesce(s.elhi_occ > 0 AND s.elhi_occ > s.elhi_cnt, false) AS queue_pat8,
       CASE WHEN x.ph IS NOT NULL THEN hm.nh - (x.status1 <> 'bad')::INT END AS n_hmates,
       o.hi_min_c, o.hi_occ_max, coalesce(g.geh_ok, false) AS geh_ok, g.geh_night,
       CASE WHEN g.geh_night IS NOT NULL THEN g.geh_night_n0 END AS geh_night_n,
       CASE WHEN g.geh_night IS NOT NULL THEN g.geh_night_exp0 END AS geh_night_exp, g.geh_drop, g.drop_t, g.drop_t2,
       g.occ_after,
       g.own_before, g.own_after, g.exp_after, g.occ_own, g.occ_mates, p.spk_bursts, p.spk_gmax, p.spk_daily, p.spk_lvl,
       m.mocc AS occ{% if mh_on %}, coalesce(mb.mh_ok, false) AS mh_ok, mb.md_h, mb.md_geh, mb.md_r, mb.md_a, mb.md_z, mb.md_own, mb.md_exp, mb.mr_h, mb.mr_geh, mb.mr_r, mb.mr_a, mb.mr_z{% endif %}

FROM xm x LEFT JOIN act a USING (dev, det) LEFT JOIN slp s USING (dev, det) LEFT JOIN lkc k USING (dev, det)
LEFT JOIN q5 q USING (dev, det) LEFT JOIN fe f USING (dev, det) LEFT JOIN tf t USING (dev, det)
LEFT JOIN ohi o USING (dev, det) LEFT JOIN geh g USING (dev, det) LEFT JOIN spk p USING (dev, det)
LEFT JOIN (SELECT DISTINCT dev, det, mocc FROM gb) m USING (dev, det){% if mh_on %}

LEFT JOIN mhbest mb USING (dev, det){% endif %}

LEFT JOIN (SELECT dev, ph, sum((status1 <> 'bad')::INT) AS nh FROM xm GROUP BY dev, ph) hm
  ON hm.dev = x.dev AND hm.ph = x.ph
{% include "judge/limits.sql" %}
{% include "judge/tod.sql" %}
{% if opt['no_daily_pattern'] %}
{# note 154m: day (07-19) and night (01-05) levels of the detector and of its phase mates (no mates: the rest of the
   signal), and how much its hour-of-day profile follows theirs.  No daily pattern = the mates clearly follow the day
   (>= 100 an hour by day, night <= 30 % of it), the detector counts (>= 1.5 an hour by day; less is a nearly dead zone)
   and either keeps >= 80 % of its daytime level at night with a profile unlike theirs, or is a low-volume zone (< 30 an
   hour) keeping >= 30 % (and 2.5 x the mates' share) of it at night; a zone ON >= 25 % of the daytime hours and >= 3 x
   its night share while it counts LESS by day (< 0.8 x night) is a queue zone: vehicles sit on it -- neither noise nor
   busy at night.  own / ref per hour of day are averages of whole counts (exact); their means over
   the hours are taken in hour order, so the text's "~26 an hour" never depends on the order DuckDB's threads add them
   up in (note 154ad) #}
-- @step flatm mat
SELECT *, coalesce(occ_d >= 900 AND occ_d >= 3 * occ_n AND own_d < 0.8 * own_n, false) AS qzone FROM (
WITH t AS (SELECT dev, hid, sum(M) AS S FROM mh GROUP BY dev, hid),
pm AS (SELECT m.dev, z.det, m.hid, sum(m.M) AS P FROM mh m JOIN dz zm USING (dev, det)
       JOIN dz z ON z.dev = m.dev AND z.ph = zm.ph AND z.det <> m.det WHERE z.ph IS NOT NULL GROUP BY m.dev, z.det, m.hid),
r AS (SELECT m.dev, m.det, x.hod, avg(m.M) AS own, avg(m.O) AS occ, avg(coalesce(p.P, t.S - m.M)) AS ref
      FROM mh m JOIN hix x USING (dev, hid) JOIN t USING (dev, hid) LEFT JOIN pm p USING (dev, det, hid)
      WHERE x.ch GROUP BY m.dev, m.det, x.hod)
SELECT dev, det, list_avg(list(own ORDER BY hod) FILTER (hod >= 7 AND hod < 19)) AS own_d,
       list_avg(list(own ORDER BY hod) FILTER (hod >= 1 AND hod < 5)) AS own_n,
       list_avg(list(ref ORDER BY hod) FILTER (hod >= 7 AND hod < 19)) AS rest_d,
       list_avg(list(ref ORDER BY hod) FILTER (hod >= 1 AND hod < 5)) AS rest_n,
       avg(occ) FILTER (hod >= 7 AND hod < 19) AS occ_d, avg(occ) FILTER (hod >= 1 AND hod < 5) AS occ_n,
       corr(own, ref) AS prof_corr, count(*) AS n_hod
FROM r GROUP BY dev, det)
-- @step flat mat
SELECT * FROM flatm
WHERE n_hod >= 20 AND rest_d >= 100 AND rest_n <= 0.3 * rest_d AND own_d >= 1.5 AND NOT qzone
  AND ((own_n >= 0.8 * own_d AND coalesce(prof_corr, 0) < 0.6)
       OR (own_d < 30 AND own_n >= greatest(0.3, 2.5 * rest_n / rest_d) * own_d AND coalesce(prof_corr, 0) < 0.8))
{% endif %}
{# Count zone held ON / extension; the queue excuse is never for a counting zone #}
-- @step xx mat
SELECT x.*, t.tod_z, t.tod_dph, t.tod_n_night, t.tod_r, t.tod_mate_r, t.tod_d0,
       {% if opt['no_daily_pattern'] %}f.det IS NOT NULL AS flat, f.own_d AS flat_d, f.own_n AS flat_n, f.rest_d AS flat_rd, f.rest_n AS flat_rn, coalesce(fm.qzone, false) AS qzone,{% else %}false AS flat, NULL::DOUBLE AS flat_d, NULL::DOUBLE AS flat_n, NULL::DOUBLE AS flat_rd, NULL::DOUBLE AS flat_rn, false AS qzone,{% endif %}

       x.fn = 'Count' AND (len(x.alts) = 0 OR list_has_all(['Count'], x.alts)) AND NOT x.c1 AS count_on_ok,
       x.c1 AS c1_pre, CASE WHEN x.c1 THEN 'C1' WHEN x.c2 THEN 'C2' ELSE '' END AS cfg,
       x.fn IN {{ NO_SHARED|inlist }} AND NOT list_has_any(x.alts, ['Presence', 'Other', 'Mid']) AS noq,
       CASE WHEN x.fn = 'Advance' THEN x.q5_all ELSE x.q5_gy END AS q5_use
FROM (SELECT x.*, coalesce(x.fn = 'Count' AND ((x.rep_time_s >= 600 AND x.rep_time_s >= 0.70 * x.occ * x.hours * 3600)
                           OR (x.n_rep >= 20 AND x.n_rep >= 0.05 * x.n_on)), false) AS c1,
             coalesce(x.fn = 'Count' AND x.cmode = 'pulse' AND (x.n_ge60 >= 1 OR x.n_ge5 >= 3), false) AS c2
      FROM xx0 x) x
LEFT JOIN tod t USING (dev, det)
{{ 'LEFT JOIN flat f USING (dev, det) LEFT JOIN flatm fm USING (dev, det)' if opt['no_daily_pattern'] else '' }}
{# holds with the queue / traffic test (episode_flags) #}
-- @step ef mat
SELECT e.*, e.q_ok AND e.dur < 3600 AND NOT e.common AS q1,
       e.q_ok AND e.dur >= 3600 AND coalesce(e.cover >= 0.75, false) AND NOT e.lifted0 AND NOT e.common AS q1b,
       e.q_ok AND e.lifted0 AS lifted,
       CASE WHEN e.fn IN {{ NO_SHARED|inlist }} THEN 0 ELSE e.co5 END >= 3 AND NOT e.common AS shared
FROM (SELECT e.*, coalesce(e.q_ok0 AND e.dur >= 3600 AND e.cover >= 0.75 AND e.busy_min_e >= 60 AND e.busy_occ_e < 0.5
                           AND NOT e.ref_held_e, false) AS lifted0, coalesce(e.q_ok0, false) AS q_ok
      FROM (SELECT e.*, x.fn, coalesce(CASE WHEN x.noq THEN 0 ELSE e.n_hpeer_e END, 0) > 0 AND e.phx >= 1.5
                   AND e.corr_h_e >= 0.70 AND e.refx_e >= 0.5 AND coalesce(e.light_e, 1) = 0 AND e.trafx_e >= 0.5 AS q_ok0
            FROM (SELECT e.*, e.cover_e AS cover, CASE WHEN coalesce(e.cover_e, 0) >= 0.75
                         THEN greatest(coalesce(e.phx_h_e, 0), 1.5) ELSE e.phx_h_e END AS phx FROM ectx e) e
            LEFT JOIN xx x USING (dev, det)) e) e
-- @step efl mat
SELECT e.*, l.lim_stuck, CASE WHEN e.common THEN 'together' WHEN e.q1 OR e.q1b THEN 'queue'
       WHEN e.shared THEN 'shared' WHEN e.lifted THEN 'not_queue' ELSE 'counted' END AS label,
       NOT e.q1 AND NOT e.q1b AND NOT e.shared AS counted
FROM ef e JOIN limits l USING (dev, det) WHERE e.dur >= l.lim_stuck
{# stuck on: counted holds over the limit; total time and number against the limits; caps (health_v4._stuck) #}
-- @step stk mat
SELECT x.dev, x.det, CASE WHEN coalesce(e.n_common, 0) > 0 THEN 1.0
  WHEN coalesce(e.n_ep, 0) > 0 AND e.n_ep = e.n_lift AND sv > {{ 0.35|dbl }} THEN 0.35
  WHEN coalesce(e.n_over, 0) > 0 AND coalesce(e.n_ep, 0) = 0 AND coalesce(e.n_sh, 0) = 0 AND e.n_q > 0 THEN 0.0
  WHEN coalesce(e.n_ep, 0) = 0 AND coalesce(e.n_sh, 0) > 0 THEN 0.35 ELSE sv END AS s_stuck_v,
  coalesce(e.n_ep, 0) AS st_n_ep, coalesce(e.tot, 0) AS st_tot_s
FROM (SELECT x.dev, x.det, CASE WHEN x.{{ 's_stuck_u' if opt['no_recovery_cap'] else 's_stuck' }} = 0.35
                                    AND coalesce(e.n_ep, 0) <= 1 AND s0 > 0.35 THEN 0.35 ELSE s0 END AS sv
      FROM (SELECT x.*, CASE WHEN st IS NULL THEN coalesce(sn, 0) ELSE greatest(st, coalesce(sn, 0)) END AS s0 FROM (
              SELECT x.dev, x.det, x.s_stuck, {{ 'x.s_stuck_u,' if opt['no_recovery_cap'] else '' }}
                     {{ lib.sc('coalesce(e.tot, 0.0)', 'l.lim_stuck', 'l.bad_stuck') }} AS st,
                     {{ lib.sc('coalesce(e.n_ep, 0)::DOUBLE', 'l.lim_n_ep', 'l.bad_n_ep') }} AS sn
              FROM xx x JOIN limits l USING (dev, det) LEFT JOIN (
                SELECT dev, det, count(*) FILTER (counted) AS n_ep, sum(dur) FILTER (counted) AS tot FROM efl
                GROUP BY dev, det) e USING (dev, det)) x) x
      LEFT JOIN (SELECT dev, det, count(*) FILTER (counted) AS n_ep FROM efl GROUP BY dev, det) e USING (dev, det)) x
LEFT JOIN (SELECT dev, det, count(*) AS n_over, count(*) FILTER (counted) AS n_ep, sum(dur) FILTER (counted) AS tot,
                  count(*) FILTER (q1 OR q1b) AS n_q, count(*) FILTER (shared AND NOT q1 AND NOT q1b) AS n_sh,
                  count(*) FILTER (counted AND lifted) AS n_lift, count(*) FILTER (common) AS n_common
           FROM efl GROUP BY dev, det) e USING (dev, det)
{# pre-pass: phase mates with a finding of their own (not a silence) leave the silence yardstick #}
{{ vd.verdict('pre', 'xx') }}
-- @step flag
SELECT dev, det, f_stuck OR f_count_on OR f_prof OR f_rapid OR f_volume OR f_spiky AS flagged FROM preV
{# goes silent, clean yardstick (h118c_dropout): references by the phase of the scored detectors only #}
-- @step dcph mat
SELECT d.dev, d.det, x.ph FROM dets d LEFT JOIN xbase x USING (dev, det)
-- @step dcrf
SELECT a.dev, a.det, b.det AS r, coalesce(a.ph = b.ph, false) AS same, coalesce(f.flagged, false) AS fl
FROM dcph a JOIN dcph b ON a.dev = b.dev AND a.det <> b.det JOIN dstat sb ON sb.dev = b.dev AND sb.det = b.det AND sb.live
LEFT JOIN twin t ON t.dev = a.dev AND t.i = a.det AND t.j = b.det LEFT JOIN flag f ON f.dev = b.dev AND f.det = b.det
WHERE t.i IS NULL
-- @step dccand mat
SELECT x.dev, x.det, coalesce(k.nsame, 0) >= 2 AS isph, coalesce(k.nkeep, 0) >= 2 AS keep2
FROM xx x LEFT JOIN (SELECT dev, det, count(*) FILTER (same) AS nsame, count(*) FILTER (same AND NOT fl) AS nkeep
                     FROM dcrf GROUP BY dev, det) k USING (dev, det)
WHERE x.drop_lam >= 30 AND x.n_on > 0
{# members of S0 (yardstick as in stage 1) and S1 (without the flagged detectors); dropped = flagged ones left out #}
-- @step dcm
SELECT c.dev, c.det, 0 AS v, f.r FROM dccand c JOIN dcrf f USING (dev, det) WHERE c.isph AND f.same
UNION ALL SELECT c.dev, c.det, 0, d.det FROM dccand c JOIN dets d ON d.dev = c.dev AND d.det <> c.det WHERE NOT c.isph
UNION ALL SELECT c.dev, c.det, 1, f.r FROM dccand c JOIN dcrf f USING (dev, det) WHERE c.isph AND c.keep2 AND f.same
          AND NOT f.fl
UNION ALL SELECT c.dev, c.det, 1, d.det FROM dccand c JOIN dets d ON d.dev = c.dev AND d.det <> c.det
          LEFT JOIN flag f ON f.dev = d.dev AND f.det = d.det WHERE NOT (c.isph AND c.keep2)
          AND NOT coalesce(f.flagged, false)
-- @step dcnd
SELECT c.dev, c.det, CASE WHEN c.isph AND c.keep2 THEN count(*) FILTER (f.same AND f.fl)
                          ELSE count(*) FILTER (fl2) END AS n_dropped
FROM dccand c JOIN (SELECT a.dev, a.det, a.same, a.fl, coalesce(g.flagged, false) AS fl2 FROM
                      (SELECT d.dev, c.det, d.det AS r, coalesce(f.same, false) AS same, coalesce(f.fl, false) AS fl
                       FROM dccand c JOIN dets d ON d.dev = c.dev AND d.det <> c.det
                       LEFT JOIN dcrf f ON f.dev = c.dev AND f.det = c.det AND f.r = d.det) a
                    LEFT JOIN flag g ON g.dev = a.dev AND g.det = a.r) f USING (dev, det)
GROUP BY c.dev, c.det, c.isph, c.keep2
-- @step dcrs0 mat
SELECT u.dev, u.det, m.v, u.a, u.z, m.r, w.nb5 FROM runs u JOIN dcm m ON m.dev = u.dev AND m.det = u.det
JOIN win w ON w.dev = u.dev
{{ lib.at('dcrs', 'dcrs0', 'r', ('a', 'z', 'nb5'), 'cn') }}
-- @step dcrs
SELECT dev, det, v, a, z, sum(cn_z - cn_a) AS s_in, sum(cn_nb5) AS s_tot FROM dcrs_3 GROUP BY dev, det, v, a, z
-- @step dc
SELECT c.dev, c.det, coalesce(r0.lam, 0.0) AS drop_lam_chk, coalesce(r1.lam, 0.0) AS drop_lam_c,
       coalesce(r1.b0, -1) AS drop_b0_c, coalesce(r1.b1, -1) AS drop_b1_c, n.n_dropped
FROM dccand c JOIN dcnd n USING (dev, det)
LEFT JOIN (SELECT dev, det, v, max(lam) AS lam, first(a ORDER BY round(lam, 6) DESC, a) AS b0, first(z ORDER BY round(lam, 6) DESC, a) AS b1
           FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM dcrs r JOIN dstat s USING (dev, det)
                 {{ lib.cap() }})
           WHERE lam > 0 GROUP BY dev, det, v) r0 ON r0.dev = c.dev AND r0.det = c.det AND r0.v = 0
LEFT JOIN (SELECT dev, det, v, max(lam) AS lam, first(a ORDER BY round(lam, 6) DESC, a) AS b0, first(z ORDER BY round(lam, 6) DESC, a) AS b1
           FROM (SELECT r.*, s.n_on / greatest(r.s_tot - r.s_in, 1) * r.s_in AS lam FROM dcrs r JOIN dstat s USING (dev, det)
                 {{ lib.cap() }})
           WHERE lam > 0 GROUP BY dev, det, v) r1 ON r1.dev = c.dev AND r1.det = c.det AND r1.v = 1
{% set chain = opt['signal_silent'] or opt['phase_drop'] %}
{% set own = opt['own_history'] %}
-- @step {{ 'xfa0' if own else 'xfa' if chain else 'xf' }} mat
SELECT x.* REPLACE (
    CASE WHEN m THEN x.drop_lam * c.drop_lam_c / c.drop_lam_chk ELSE x.drop_lam END AS drop_lam,
    CASE WHEN m THEN c.drop_b0_c ELSE x.drop_b0 END AS drop_b0,
    CASE WHEN m THEN c.drop_b1_c ELSE x.drop_b1 END AS drop_b1,
    CASE WHEN m THEN CASE WHEN x.drop_in_stuck THEN 0.0 WHEN x.drop_lam * c.drop_lam_c / c.drop_lam_chk < 30 THEN 0.0
                          ELSE 0.35 + 0.65 * least((x.drop_lam * c.drop_lam_c / c.drop_lam_chk - 30) / 70, 1) END
         ELSE x.s_dropout END AS s_dropout)
FROM (SELECT x.*, coalesce(c.n_dropped > 0 AND c.drop_lam_chk > 0, false) AS m FROM xx x LEFT JOIN dc c USING (dev, det)) x
LEFT JOIN dc c USING (dev, det)
{% if own %}
{# note 154j: the detector's own count at the same clock times on the other days, outside its silences that rate on
   their own (>= 30 expected; a quiet night is a reference, a repeating outage is not) #}
-- @step ownh mat
SELECT * FROM (
WITH r AS (SELECT dev, det, drop_b0, drop_b1 FROM xfa0 WHERE drop_b1 > drop_b0 AND drop_lam > 0),
g AS (SELECT g.dev, g.det, g.b, g.n, g.cov, (w.s_ns // {{ NS5 }} + g.b) % 288 AS bod, r.drop_b0, r.drop_b1
      FROM g5 g JOIN r USING (dev, det) JOIN win w USING (dev)),
ref AS (SELECT dev, det, bod, avg(n) AS mn FROM g
        WHERE cov AND NOT (b >= drop_b0 AND b < drop_b1)
          AND NOT EXISTS (SELECT 1 FROM {{ 'silall' if opt['all_silences'] else 'runs' }} u
                          WHERE u.dev = g.dev AND u.det = g.det {{ 'AND u.lam >= 30' if opt['all_silences'] else '' }}
                          AND g.b >= u.a AND g.b < u.z)
        GROUP BY dev, det, bod)
SELECT g.dev, g.det, sum(f.mn) AS own_typ, count(f.mn) AS nref, count(*) AS nb
FROM g LEFT JOIN ref f USING (dev, det, bod) WHERE g.b >= g.drop_b0 AND g.b < g.drop_b1 GROUP BY g.dev, g.det)
-- @step {{ 'xfa' if chain else 'xf' }} mat
SELECT x.* REPLACE (CASE WHEN o.nref >= 0.8 * o.nb AND x.drop_lam > 3 * o.own_typ + 10 THEN 3 * o.own_typ + 10
                         ELSE x.drop_lam END AS drop_lam)
FROM xfa0 x LEFT JOIN ownh o USING (dev, det)
{% endif %}
