{# The output rows (health_duckdb._outputs_rest / _periods_basic): status, score, reason, categories, config and watch
   notes per detector; the stage-1 rows of detectors without a known function; signal notes. #}
{% import "_lib.sql" as lib with context %}
{% set lc = opt['lane_closure'] %}
{% set pmo = opt['pulse_mode'] %}
{% set nl = 'greatest(round_even(x.lanes, 0)::BIGINT, 1)' %}
-- @step {{ 'outx0' if pmo else 'outx' }}
SELECT v.dev, v.det, {{ "CASE WHEN k.dev IS NOT NULL AND v.health_st IN ('ok', 'watch', 'not_enough_data') THEN CASE WHEN k.all_switch THEN 'timing_plan' ELSE 'lane_closure' END ELSE v.health_st END" if lc else 'v.health_st' }} AS health_status,
       CASE WHEN v.health_st <> 'not_enough_data' AND NOT isnan(v.score) THEN v.score END AS health_score,
       CASE WHEN v.health_st IN ('suspect', 'bad') {{ 'OR k.dev IS NOT NULL' if lc else '' }}
            THEN b.lb || ' - ' || coalesce(c.parts, v.health_st)
            WHEN v.health_st = 'watch' THEN b.lb || ' - watch' || CASE WHEN v.wt <> '' THEN ': ' || v.wt ELSE '' END
            WHEN v.health_st = 'not_enough_data' THEN b.lb || ' - too few actuations to judge'
            ELSE b.lb || ' - nothing unusual' END AS health_reason,
       coalesce(c.health_categories, '') AS health_categories,
       CASE x.cfg WHEN 'C1' THEN format('Extension time on a count zone: {:,} of {:,} ONs logged again without an OFF',
                                        x.n_rep, trunc(x.n_on)::BIGINT)
                  WHEN 'C2' THEN 'Set to pulse but holds ON: ' || CASE WHEN x.long_max_s IS NOT NULL
                       THEN x.n_ge5 || ' ONs over 5 s, longest ' || {{ lib.dur('x.long_max_s') }}
                       ELSE 'holds ON although set to pulse' END ELSE '' END AS health_config,
       v.wt AS health_watch
FROM (SELECT v.*, array_to_string(list_filter([
        CASE WHEN v.w_no_yardstick THEN 'no healthy phase mate to compare with' END,
        CASE {% for k in KEYS %}WHEN v.b_{{ k }} THEN {{ ('borderline ' ~ CATEGORY[k]|lower)|str }}{% if not loop.last %} {% endif %}{% endfor %} END,
        CASE WHEN v.w_dropout_short AND NOT (v.b_dropout OR v.b_level) THEN 'borderline counts fall' END],
        lambda x: x IS NOT NULL), '; ') AS wt FROM finV v) v
JOIN xf x USING (dev, det) LEFT JOIN cats c USING (dev, det)
{{ 'LEFT JOIN lcx k USING (dev, det)' if lc else '' }}
JOIN (SELECT x.dev, x.det, 'det ' || x.det || ': ' || CASE WHEN x.ph IS NOT NULL THEN 'P' || trunc(x.ph)::BIGINT || ' ' ELSE '' END || replace(x.fn, 'Yellow_Red', 'Yellow-red') || CASE WHEN x.lanes IS NOT NULL AND coalesce(c.lane_cat, false) THEN ' (covers '
             || {{ nl }} || ' lane' || CASE WHEN {{ nl }} > 1 THEN 's' ELSE '' END || ')' ELSE '' END AS lb
      FROM xf x LEFT JOIN cats c USING (dev, det)) b USING (dev, det)
{% if pmo %}
{# note 154aa: looks set to pulse mode -- suspect at least (a worse rating stays), the note added #}
{% set txt = "format('Configuration: looks set to pulse mode: {:.0f} % of its {:,} calls last 0.2 s or less; a presence detector should keep its call ON while a vehicle is on it', 100 * p.ps, p.np)" %}
{% set up = "p.dev IS NOT NULL AND o.health_status IN ('ok', 'watch', 'lane_closure', 'timing_plan')" %}
-- @step outx
SELECT o.dev, o.det, CASE WHEN {{ up }} THEN 'suspect' ELSE o.health_status END AS health_status,
       CASE WHEN {{ up }} THEN least(coalesce(o.health_score, 1.0), 0.69) ELSE o.health_score END AS health_score,
       CASE WHEN p.dev IS NULL THEN o.health_reason
            WHEN o.health_status IN ('ok', 'watch') THEN split_part(o.health_reason, ' - ', 1) || ' - ' || {{ txt }}
            ELSE o.health_reason || '; ' || {{ txt }} END AS health_reason,
       CASE WHEN p.dev IS NULL THEN o.health_categories WHEN o.health_categories = '' THEN 'Configuration (suspect)'
            ELSE o.health_categories || '; Configuration (suspect)' END AS health_categories,
       o.health_config, o.health_watch
FROM outx0 o LEFT JOIN pmode p USING (dev, det)
{% endif %}
{# detectors without a known predicted function: the stage-1 checks as they are (pass-2 status) #}
{% set s1 = '(SELECT * REPLACE (status_u AS status, hs_u AS hs, s_dropout_u AS s_dropout, s_stuck_u AS s_stuck) FROM stage1)' if opt['no_recovery_cap'] else 'stage1' %}
-- @step outf
SELECT s.dev, s.det, s.status AS health_status, CASE WHEN s.status <> 'not_enough_data' THEN s.hs END AS health_score,
       'det ' || s.det || ': ' || CASE s.status
         WHEN 'ok' THEN format('{:,.0f} actuations in {:.1f} h, nothing unusual', s.n_on, g.n_cov * 300 / 3600.0)
         WHEN 'not_enough_data' THEN format('only {:.0f} actuations in {:.1f} h - too few to judge', s.n_on,
                                            g.n_cov * 300 / 3600.0)
         ELSE array_to_string(list_filter([CASE WHEN s.s_stuck > 0 THEN 'stayed ON continuously for '
                                                  || round(s.ep_dur / 60)::BIGINT || ' min' END,
                                             CASE WHEN s.s_dropout > 0 THEN 'went silent' END,
                                             CASE WHEN s.s_night_drop > 0 THEN 'misses vehicles at night' END,
                                             CASE WHEN s.s_level > 0 THEN 'its share of the traffic dropped' END],
                                            lambda x: x IS NOT NULL), '; ')
         END AS health_reason, '' AS health_categories, '' AS health_config, '' AS health_watch
FROM {{ s1 }} s
JOIN sig0 g USING (dev) JOIN win w USING (dev) ANTI JOIN xbase x USING (dev, det)
{% if periods %}
{# note 154aa: the periods of the detectors without a known function (outf, suspect / bad): the stage-1 silence and
   longest hold (option basic_periods), every other finding over the window; severity from the finding's stage-1 score
   (0.35 suspect -> 0.5, 0.75 -> 0.8); no type, so no limit -> pbasic (columns as pall) #}
{% set bp = opt['basic_periods'] %}
{% set hold = 's.ep_t0 IS NOT NULL' if bp else 'false' %}
{% set gap = 's.drop_b1 > s.drop_b0' if bp else 'false' %}
-- @step pbasic
WITH s AS (SELECT s.* FROM {{ s1 }} s ANTI JOIN xbase x USING (dev, det) WHERE s.status IN ('suspect', 'bad'))
SELECT s.dev, s.det, 20 AS o, w.s_ns AS a, w.e_ns AS z, 'stuck on' AS what, {{ lib.conf('s.s_stuck', '0.35', '0.75') }} AS sev, NULL::DOUBLE AS lim,
       false AS lc, true AS ww FROM s JOIN win w USING (dev) WHERE s.s_stuck > 0 AND NOT {{ hold }}
UNION ALL SELECT s.dev, s.det, 22, w.s_ns, w.e_ns, 'silent', {{ lib.conf('s.s_dropout', '0.35', '0.75') }}, NULL, false, true
       FROM s JOIN win w USING (dev) WHERE s.s_dropout > 0 AND NOT {{ gap }}
UNION ALL SELECT s.dev, s.det, 23, w.s_ns, w.e_ns, 'misses vehicles at night', {{ lib.conf('s.s_night_drop', '0.35', '0.75') }}, NULL, false, true
       FROM s JOIN win w USING (dev) WHERE s.s_night_drop > 0
UNION ALL SELECT s.dev, s.det, 24, w.s_ns, w.e_ns, 'counts dropped', {{ lib.conf('s.s_level', '0.35', '0.75') }}, NULL, false, true
       FROM s JOIN win w USING (dev) WHERE s.s_level > 0{% if bp %}

UNION ALL SELECT s.dev, s.det, 0, w.s_ns + round(s.ep_t0 * 1e9)::BIGINT, w.s_ns + round(s.ep_t1 * 1e9)::BIGINT, 'stuck on',
       {{ lib.conf('s.s_stuck', '0.35', '0.75') }}, NULL, false, false FROM s JOIN win w USING (dev) WHERE s.s_stuck > 0 AND {{ hold }}
UNION ALL SELECT s.dev, s.det, 2, w.s_ns + s.drop_b0 * {{ NS5 }}, least(w.s_ns + s.drop_b1 * {{ NS5 }}, w.e_ns), 'silent',
       {{ lib.conf('s.s_dropout', '0.35', '0.75') }}, NULL, false, false FROM s JOIN win w USING (dev) WHERE s.s_dropout > 0 AND {{ gap }}{% endif %}

{% endif %}
{# signal notes: many detectors with an unusual night; holds shared by several detectors (common failure) #}
-- @step cgrp
SELECT dev, g, min(s) AS s0, first(dur ORDER BY s, dur) AS dur0, count(DISTINCT round_even(s / 60, 0)) AS k
FROM (SELECT e.dev, e.s, e.dur, list_sort(list_distinct(list_append(coalesce(e.mates, []), e.det))) AS g
      FROM efl e WHERE e.common)
GROUP BY dev, g
-- @step snote
SELECT w.dev, array_to_string(list_filter([
  CASE WHEN t.dev IS NOT NULL THEN t.n_odd || ' of ' || t.n_ok || ' detectors at this signal had an unusual night on '
       || {{ lib.hm('t.d0', '%a %d %b') }} || ': a signal-wide event or an input change, not one bad detector' END,
  c.txt{% if opt['signal_silent'] %}, CASE WHEN sr.dev IS NOT NULL THEN 'no detector logged an actuation ' || {{ lib.hm('w.s_ns + sr.a * ' ~ NS5) }} || '-' || {{ lib.hm('w.s_ns + sr.z * ' ~ NS5) }} || ' while the signal kept serving greens: all detector inputs lost (detector rack / cabinet)' END{% endif %}], lambda x: x IS NOT NULL), '; ') AS health_signal_note
FROM win w LEFT JOIN todnote t USING (dev){{ ' LEFT JOIN sgr sr USING (dev)' if opt['signal_silent'] else '' }}
LEFT JOIN (SELECT g.dev, string_agg(len(g.g) || ' detectors (det ' || array_to_string(g.g, ', ') || ') held ON together '
                  || CASE WHEN g.k = 1 THEN 'from ' || {{ lib.hm('w.s_ns + round(g.s0 * 1e9)::BIGINT') }} || ' for '
                          || {{ lib.dur('g.dur0') }} ELSE g.k || ' times from ' || {{ lib.hm('w.s_ns + round(g.s0 * 1e9)::BIGINT') }} END
                  || ' (same start and end): one common failure, check the cabinet first', '; ' ORDER BY g.s0, g.g) AS txt
           FROM cgrp g JOIN win w USING (dev) GROUP BY g.dev) c USING (dev)
