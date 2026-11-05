{# The check scores and the resolver / severity (health_duckdb._verdict: health_v4.scores + status) for the rows of src
   -> {p}V; lc: src carries lc_d / lc_l (a silence / count drop explained by a lane closure: not scored). #}
{% import "_lib.sql" as lib with context %}
{% macro real(k) %}{% if k == 'dropout' and opt['no_recovery_cap'] %}(f_dropout AND NOT coalesce(v_dropout = 0.35, false) AND drop_b1 - drop_b0 >= 3 AND drop_lam >= 50){% elif k in ('stuck', 'dropout') %}(f_{{ k }} AND NOT coalesce(v_{{ k }} = 0.35, false)){% else %}f_{{ k }}{% endif %}{% endmacro %}
{% macro count_groups(groups) %}{% for g in groups %}({% for k in g %}{{ real(k) }}{% if not loop.last %} OR {% endif %}{% endfor %})::INT{% if not loop.last %} + {% endif %}{% endfor %}{% endmacro %}
{% macro verdict(p, src, lc=false) %}
{% set P = LJ['busy'] %}
{% set sd = lib.sc('x.drop_lam', 'l.lim_dropout', 'l.bad_dropout') %}
{% set g0, g1 = M['geh'][0]|dbl, M['geh'][1]|dbl %}
{# note 154n: a run of low / high hours against the phase mates (a lane closure explains a drop) #}
{% if opt['mate_drop'] %}{% set md %}CASE WHEN x.fn = 'Bike' THEN NULL {{ 'WHEN x.lc_l OR x.lc_md THEN 0.0' if lc else '' }} WHEN x.md_h IS NOT NULL THEN {{ lib.sc('x.md_geh', g0, g1) }} WHEN x.mh_ok THEN 0.0 END{% endset %}{% else %}{% set md = 'NULL::DOUBLE' %}{% endif %}
{% if opt['count_rise'] %}{% set rise %}CASE WHEN x.fn = 'Bike' THEN NULL {{ 'WHEN x.lc_mr THEN 0.0' if lc else '' }} WHEN x.mr_h IS NOT NULL THEN {{ lib.sc('x.mr_geh', g0, g1) }} WHEN x.mh_ok THEN 0.0 END{% endset %}{% else %}{% set rise = 'NULL::DOUBLE' %}{% endif %}
{% set lvl = lib.sc('x.geh_drop', 'l.lim_level', 'l.bad_level') %}
-- @step {{ p }}S mat
SELECT x.dev, x.det, s.s_stuck_v AS v_stuck,
  CASE WHEN x.count_on_ok THEN {{ lib.sc('coalesce(x.hi_min_c, 0)', 'l.lim_count_on', 'l.bad_count_on') }} END AS v_count_on,
  CASE WHEN x.fn = 'Bike' THEN NULL WHEN x.silent THEN 0.0 {{ 'WHEN x.lc_d THEN 0.0' if lc else '' }}
       {{ 'WHEN x.drop_b1 - x.drop_b0 < 3 THEN least(x.sd, 0.5)' if opt['silent_limits'] else '' }} ELSE x.sd END AS v_dropout,
  CASE WHEN x.fn = 'Bike' OR NOT x.geh_ok THEN NULL {{ 'WHEN x.lc_l THEN 0.0' if lc else '' }}
       {% if opt['long_zone'] %}WHEN x.geh_drop IS NOT NULL AND x.occ_own >= 3.0 AND x.occ_own >= 3.0 * coalesce(x.occ_mates, 1.0) THEN {{ lvl }} WHEN x.occ_own >= 1.0 AND coalesce(x.occ_after, 0) >= 0.25 THEN 0.0{% endif %}

       WHEN x.geh_drop IS NOT NULL AND ((x.occ_mates IS NOT NULL AND x.occ_own / greatest(x.occ_mates, 1e-9) < 0.8)
            {{ 'OR (x.occ_mates IS NULL AND x.occ_own < 0.8) OR x.own_after < 0.1 * x.exp_after' if opt['drop_no_mates'] else '' }})
       THEN {{ lvl }} ELSE 0.0 END AS v_lvl_old,
  {{ md }} AS v_lvl_md,
  CASE WHEN v_lvl_old IS NULL AND v_lvl_md IS NULL THEN NULL
       ELSE greatest(coalesce(v_lvl_old, 0.0), coalesce(v_lvl_md, 0.0)) END AS v_level,
  {{ rise }} AS v_rise,
  CASE WHEN x.fn = 'Bike' THEN NULL {{ 'WHEN x.moved AND x.lc_d THEN 0.0' if lc else '' }}
       {{ 'WHEN x.moved AND coalesce(x.drop_lam, 0) < 50 THEN 0.0' if opt['silent_limits'] else '' }}
       WHEN x.moved AND coalesce(x.drop_lam, 0) < 100 THEN least(x.sd, 0.5)
       WHEN x.moved THEN x.sd ELSE x.nd END AS v_night_drop,
  CASE WHEN x.fn <> 'Bike' AND coalesce(CASE WHEN x.silent THEN 0.0 ELSE x.sd END, 0)
            >= {{ '0.42' if opt['no_recovery_cap'] else '0.35' }}
       THEN CASE WHEN x.pv IS NOT NULL THEN 0.0 END ELSE x.pv END AS v_prof,
  CASE WHEN x.N >= 50 THEN {{ lib.sc('x.xr + 1e-9 * (x.xr > 0)::INT', '1.0', 'l.bad_rapid / l.lim_rapid') }} END AS v_rapid,
  CASE WHEN x.q5_use IS NOT NULL THEN {{ lib.sc('x.q5_use / l.lim_volume', '1.0', 'l.bad_volume / l.lim_volume') }} END
       AS v_volume,
  CASE WHEN x.spk_lvl >= 2 THEN 1.0 WHEN x.spk_lvl >= 1 THEN 0.5 WHEN x.spk_lvl IS NOT NULL THEN 0.0 END AS v_spiky,
  x.silent AS silent_night
FROM (SELECT x.*, coalesce(x.sd0, 0) >= 0.35 AND x.nf >= 0.8 AS silent,
             coalesce(x.sd0, 0) >= 0.35 AND x.nf >= 0.8 AND coalesce(x.sd0, 0) > coalesce(x.nd, 0) + 1e-9 AS moved,
             x.sd0 AS sd
      FROM (SELECT x.*, CASE WHEN x.n_on > 0 THEN CASE WHEN x.drop_in_stuck THEN 0.0
                 {% if not opt['no_recovery_cap'] %}WHEN x.s_dropout = 0.35 AND x.drop_lam > 30.001 AND {{ sd }} > 0.35 THEN 0.35{% endif %} ELSE {{ sd }} END END AS sd0,
                   {{ lib.sc('x.geh_night', 'l.lim_night_drop', 'l.bad_night_drop') }} AS nd,
                   CASE {{ 'WHEN x.flat THEN 0.8 WHEN x.qzone THEN 0.0' if opt['no_daily_pattern'] else '' }} WHEN x.tod_z IS NULL THEN NULL
                        WHEN x.tod_n_night >= {{ P['min_night']|dbl }} AND coalesce(x.tod_dph, -1) >= {{ P['mates_delta']|dbl }}
                             AND x.tod_z >= l.lim_prof * {{ P['z_x']|dbl }} AND x.tod_z >= l.lim_prof
                             {{ 'AND x.tod_z >= 1.5 * l.lim_prof' if opt['busy_ease'] else '' }} THEN 0.8
                        WHEN x.tod_n_night >= {{ P['min_night']|dbl }} AND coalesce(x.tod_dph, -1) >= {{ P['mates_delta']|dbl }}
                             AND x.tod_z >= l.lim_prof * {{ P['z_x']|dbl }} THEN 0.5 ELSE 0.0 END AS pv,
                   CASE WHEN x.n_fast >= 3 THEN x.fast_rate / l.lim_rapid ELSE 0.0 END AS xr,
                   CASE WHEN x.drop_b1 > x.drop_b0 THEN (SELECT count(*) FILTER (b.hr >= 21 OR b.hr < 5) FROM bins5 b
                        WHERE b.dev = x.dev AND b.b >= x.drop_b0 AND b.b < x.drop_b1) / (x.drop_b1 - x.drop_b0)
                        ELSE 0.0 END AS nf
            FROM {{ src }} x JOIN limits l USING (dev, det)) x) x
JOIN limits l USING (dev, det) JOIN stk s USING (dev, det)
-- @step {{ p }}F0
SELECT s.*, {% for k in KEYS %}coalesce(v_{{ k }} >= 0.35, false) AS f_{{ k }}{% if not loop.last %}, {% endif %}{% endfor %}, x.queue_pat8, x.c_occ_like, x.hours, x.status1, x.status AS status2, x.n_hmates, x.n_on,
       x.drop_b0, x.drop_b1, x.fn, x.c1_pre, x.hi_min_c, x.fe_n, x.fe_phi{{ ', x.drop_lam' if opt['no_recovery_cap'] else '' }}
FROM {{ p }}S s JOIN {{ src }} x USING (dev, det)
{# Q2 (queue pattern explains a count drop), R6 (silence in a sample under 1 h: watch), counts fall = one finding #}
-- @step {{ p }}F1
SELECT * REPLACE (
    CASE WHEN f_dropout AND f_level AND NOT (v_dropout >= v_level) THEN false ELSE f_dropout END AS f_dropout,
    CASE WHEN f_dropout AND f_level AND v_dropout >= v_level THEN false ELSE f_level END AS f_level)
FROM (SELECT * REPLACE (f_level AND NOT (queue_pat8 AND coalesce(c_occ_like >= 0.8, false)
                                         AND NOT coalesce(v_lvl_md > v_lvl_old, false)) AS f_level,
                        f_dropout AND NOT (hours < 1) AS f_dropout),
             f_dropout AND hours < 1 AS w_dropout_short,
             status2 <> status1 AND n_hmates = 0 AND status1 IN ('suspect', 'bad') AS w_no_yardstick
      FROM {{ p }}F0)
{% set nf %}{% for k in KEYS %}f_{{ k }}::INT{% if not loop.last %} + {% endif %}{% endfor %}{% endset %}
-- @step {{ p }}F2
SELECT *, ({{ nf }}) = 1 AND ({% for k in STAT %}(f_{{ k }} AND v_{{ k }} < 0.42){% if not loop.last %} OR {% endif %}{% endfor %}) AS r7 FROM {{ p }}F1
-- @step {{ p }}F3
SELECT * REPLACE ({% for k in KEYS %}f_{{ k }} AND NOT r7 AS f_{{ k }}{% if not loop.last %}, {% endif %}{% endfor %}), {% for k in KEYS %}f_{{ k }} AND r7 AS b_{{ k }}{% if not loop.last %}, {% endif %}{% endfor %} FROM {{ p }}F2
{% set prod %}{% for k in KEYS %}(1 - CASE WHEN f_{{ k }} THEN v_{{ k }} ELSE 0 END){% if not loop.last %} * {% endif %}{% endfor %}{% endset %}
{% set anyfin %}{% for k in KEYS %}v_{{ k }} IS NOT NULL{% if not loop.last %} OR {% endif %}{% endfor %}{% endset %}
-- @step {{ p }}F4
SELECT *, CASE WHEN {{ anyfin }} THEN {{ prod }} END AS score, {{ count_groups(EVS) }} AS n_ev, {{ count_groups(FAMS) }} AS n_fam,
       (drop_b1 - drop_b0) * 300 / 3600.0 AS silent_h,
       CASE WHEN {{ prod }} < 0.25 AND ({{ anyfin }}) THEN 'bad' WHEN {{ prod }} < 0.70 AND ({{ anyfin }}) THEN 'suspect'
            WHEN r7 OR w_dropout_short OR w_no_yardstick THEN 'watch' WHEN n_on < 20 THEN 'not_enough_data'
            ELSE 'ok' END AS st0
FROM {{ p }}F3
-- @step {{ p }}V mat
SELECT * EXCLUDE (st0, st1, st2, st3),
       CASE WHEN st3 = 'bad' AND f_rapid AND ({{ nf }}) = 1 AND fe_n >= 8 AND fe_phi <= 2.0
                 AND fn IN ('Presence', 'Other', 'Mid') THEN 'suspect' ELSE st3 END AS health_st,
       CASE WHEN st3 = 'bad' AND f_rapid AND ({{ nf }}) = 1 AND fe_n >= 8 AND fe_phi <= 2.0
                 AND fn IN ('Presence', 'Other', 'Mid') THEN 'F'
            WHEN st3 <> st2 THEN 'C' WHEN st2 <> st1 THEN 'P' WHEN st1 <> st0 AND n_fam < 2 THEN 'A' ELSE '' END AS sev_rule
FROM (SELECT *, CASE WHEN fn = 'Count' AND c1_pre AND hi_min_c >= 30 AND st2 IN ('ok', 'watch') THEN 'suspect'
                     ELSE st2 END AS st3
      FROM (SELECT *, CASE WHEN st1 = 'suspect' AND (f_prof OR (f_dropout AND silent_h >= 4.0)) THEN 'bad' ELSE st1 END
                   AS st2
            FROM (SELECT *, CASE WHEN st0 = 'suspect' AND (n_fam >= 2 OR n_ev >= 2) THEN 'bad' ELSE st0 END AS st1
                  FROM {{ p }}F4)))
{% endmacro %}
