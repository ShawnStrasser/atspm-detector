{# Per-detector [suspect, bad] limits of every check (health_duckdb._limits: health_v4.limits); the least strict
   plausible function when the model is unsure. #}
-- @step ltab mat
{{ T['ltab']|values('tab, key, lo, hi') }}
-- @step lref mat
{{ T['lref']|values('tab, lvl, fn, span, cmode, wg, band, v') }}
{# the functions whose limits apply: its own, and when the model is unsure (top < .7) every plausible one #}
-- @step lcand mat
SELECT x.dev, x.det, x.fn AS fa, true AS own, x.span, x.cmode, x.wg, x.band,
       CASE WHEN x.fn = 'Count' THEN '|' || CASE WHEN x.cmode = 'normal' THEN 'normal' ELSE 'pulse' END ELSE '' END AS cms,
       x.cmode AS cmr
FROM xx0 x
UNION ALL
SELECT x.dev, x.det, a.fa, false, x.span, x.cmode, x.wg, x.band,
       CASE WHEN a.fa = 'Count' THEN '|' || CASE WHEN x.dur_p50 <= 0.25 THEN 'pulse' ELSE 'normal' END ELSE '' END,
       CASE WHEN a.fa <> 'Count' THEN '' WHEN x.dur_p50 IS NULL THEN 'unknown' WHEN x.dur_p50 <= 0.25 THEN 'pulse'
            ELSE 'normal' END
FROM xx0 x, (SELECT unnest(x2.alts) AS fa, x2.dev, x2.det FROM xx0 x2) a
WHERE a.dev = x.dev AND a.det = x.det AND a.fa <> x.fn
{% macro tab(t, col='lo', split=false) %}(SELECT max(l.{{ col }}) FROM ltab l WHERE l.tab = '{{ t }}' AND l.key = c.fa || '|' || c.span{% if split %} || c.cms{% endif %}){% endmacro %}
{% macro ref(t, cm_fn=false) %}(SELECT first(l.v ORDER BY l.lvl) FROM lref l WHERE l.tab = '{{ t }}' AND (l.fn IS NULL OR l.fn = c.fa) AND (l.span IS NULL OR l.span = c.span) AND (l.cmode IS NULL OR l.cmode = {{ 'c.cmr' if cm_fn else 'c.cmode' }}) AND (l.wg IS NULL OR l.wg = c.wg) AND (l.band IS NULL OR l.band = c.band)){% endmacro %}
-- @step lval
SELECT c.dev, c.det, c.own, {{ ref('n_ep') }} AS n_ep, {{ tab('per_1000') }} AS rapid,
       {{ tab('stuck_s', 'lo', true) }} AS st_lo, {{ tab('stuck_s', 'hi', true) }} AS st_hi,
       {{ tab('silent_exp', 'lo') }} AS si_lo, {{ tab('silent_exp', 'hi') }} AS si_hi,
       {{ ref('stuck_x', true) }} AS stuck_x, {{ ref('q5_gy') }} AS q5gy, {{ ref('q5_all') }} AS q5all
FROM lcand c
{% set tf, cur, sh = LJ['too_fast'], LJ['current'], LJ['short'] %}
{% set a, b, c = sh['stuck_bad'] %}
-- @step limits mat
SELECT x.dev, x.det,
  v.n_ep + 1 AS lim_n_ep, (v.n_ep + 1) * {{ sh['n_holds_bad_x']|dbl }} AS bad_n_ep,
  v.rapid AS lim_rapid, v.rapid * {{ tf['bad_x']|dbl }} AS bad_rapid,
  {{ LJ['count_drops_geh'][0]|dbl }} AS lim_level, {{ LJ['count_drops_geh'][1]|dbl }} AS bad_level,
  {{ LJ['night_geh'][0]|dbl }} AS lim_night_drop, {{ LJ['night_geh'][1]|dbl }} AS bad_night_drop,
  {{ LJ['count_held_on_min'][0]|dbl }} AS lim_count_on, {{ LJ['count_held_on_min'][1]|dbl }} AS bad_count_on,
  CASE WHEN x.wg = 'h24' THEN greatest(v.st_lo, {{ LJ['hold_floor_s']|dbl }})
       ELSE least(greatest(v.sx, 300.0), 900.0) END AS lim_stuck,
  CASE WHEN x.wg = 'h24' THEN greatest(v.st_hi, v.st_lo, {{ LJ['hold_floor_s']|dbl }})
       ELSE greatest({{ a|dbl }} + {{ b|dbl }} * least(greatest(v.sx, 300.0), 900.0), {{ c|dbl }} * least(greatest(v.sx, 300.0), 900.0))
       END AS bad_stuck,
  CASE WHEN x.wg = 'h24' THEN v.si_lo ELSE {{ cur['silent_exp']|dbl }} END AS lim_dropout,
  CASE WHEN x.wg = 'h24' THEN {% if opt['silent_limits'] %}greatest(v.si_hi, v.si_lo * {{ sh['silent_bad_x']|dbl }}){% else %}v.si_hi{% endif %}

       ELSE {{ D['silent_bad']|dbl }} END AS bad_dropout,
  CASE WHEN x.wg = 'h24' THEN (SELECT max(l.lo) FROM ltab l WHERE l.tab = 'busy_z' AND l.key = x.fn || '|' || x.span)
       ELSE {{ cur['busy_z']|dbl }} END AS lim_prof,
  CASE WHEN x.wg = 'h24' THEN {{ LJ['too_many_h24_veh_h_lane'][0]|dbl }}
       ELSE greatest(CASE WHEN x.fn = 'Advance' THEN v.q5all ELSE v.q5gy END, {{ cur['too_many_veh_h_lane']|dbl }}) END
       AS lim_volume,
  CASE WHEN x.wg = 'h24' THEN {{ LJ['too_many_h24_veh_h_lane'][1]|dbl }}
       ELSE greatest(CASE WHEN x.fn = 'Advance' THEN v.q5all ELSE v.q5gy END, {{ cur['too_many_veh_h_lane']|dbl }})
            * {{ sh['too_many_bad_x']|dbl }} END AS bad_volume
FROM xx0 x JOIN (
  SELECT dev, det, max(n_ep) AS n_ep, max(rapid) AS rapid, max(st_lo) AS st_lo, max(st_hi) AS st_hi,
         max(si_lo) AS si_lo, max(si_hi) AS si_hi, max(q5gy) AS q5gy, max(q5all) AS q5all,
         CASE WHEN bool_or(own AND stuck_x IS NOT NULL) THEN max(stuck_x) END AS sx
  FROM lval GROUP BY dev, det) v USING (dev, det)
