{# note 154aa: the periods table (health(..., periods=True); health_duckdb._periods) -> pall (dev, det, o, a, z ns, what,
   sev, lim, lc, ww): where each finding the rules kept saw the problem (suspect / bad detectors only), with its severity
   on the check's own suspect (lo) -> bad (hi) scale (x = its strength) and lim = the type's bad limit of an event check
   (s), plus every lane closure / timing-plan switch; a finding without a period of its own spans the window (ww). #}
{% import "_lib.sql" as lib with context %}
{% import "adapters.sql" as ad with context %}
{% set lc = opt['lane_closure'] %}
{% set P = LJ['spiky'] %}
{% set g0, g1 = M['geh'][0]|dbl, M['geh'][1]|dbl %}
{% macro sc(k) %}, r.v_{{ k }}, 0.35, 0.75, NULL{% endmacro %}
{% set sil = 'x.drop_lam, l.lim_dropout, l.bad_dropout, NULL FROM r JOIN xf x USING (dev, det) JOIN limits l USING (dev, det) JOIN win w USING (dev)' %}
{% set spl = D['spk_lim']|dbl %}{# spiky: its bad number of 10-min bursts, in seconds #}
{# too-fast ONs per hour (adapter) #}
{{ ad.tfh() }}
-- @step pkept mat
WITH r AS (SELECT * FROM finV WHERE health_st IN ('suspect', 'bad')),
runs_hi AS (SELECT dev, det, min(k) AS a, max(k) + 1 AS z FROM (SELECT dev, det, k, k - row_number() OVER
            (PARTITION BY dev, det ORDER BY k) AS g FROM hib) GROUP BY dev, det, g),
fh AS (SELECT dev, det, h, cf, ca FROM tfh),
runs_fast AS (SELECT dev, det, min(h) AS a, max(h) + 1 AS z FROM (
              SELECT f.dev, f.det, f.h, f.h - row_number() OVER (PARTITION BY f.dev, f.det ORDER BY f.h) AS g
              FROM fh f JOIN limits l USING (dev, det)
              WHERE f.cf >= 3 AND 1000.0 * f.cf / greatest(f.ca, 1) >= l.lim_rapid) GROUP BY dev, det, g),
qv AS (SELECT q.dev, q.det, q.b FROM q5b q JOIN xf x USING (dev, det) JOIN limits l USING (dev, det)
       WHERE x.fn = 'Advance' AND ((q.n::FLOAT / 300::FLOAT) * 3600::FLOAT)::DOUBLE / q.ln1 > l.lim_volume
       UNION ALL SELECT q.dev, q.det, q.b FROM q5gyb q JOIN xf x USING (dev, det) JOIN limits l USING (dev, det)
       WHERE x.fn <> 'Advance' AND q.q5_gy > l.lim_volume),
runs_vol AS (SELECT dev, det, min(b) AS a, max(b) + 1 AS z FROM (SELECT dev, det, b, b - row_number() OVER
             (PARTITION BY dev, det ORDER BY b) AS g FROM qv) GROUP BY dev, det, g)
  SELECT r.dev, r.det, 0 AS o, w.s_ns + round(e.s * 1e9)::BIGINT AS a, w.s_ns + round(e.f * 1e9)::BIGINT AS z,
         'stuck on' AS what, e.dur AS x, l.lim_stuck AS lo, l.bad_stuck AS hi, l.bad_stuck AS lim
         FROM r JOIN efl e USING (dev, det) JOIN win w USING (dev) JOIN limits l USING (dev, det)
         WHERE r.f_stuck AND e.label <> 'queue'
  UNION ALL SELECT r.dev, r.det, 1, w.s_ns + u.a * w.bs * 1000000000, least(w.s_ns + u.z * w.bs * 1000000000, w.e_ns),
         'count zone held ON', (u.z - u.a) * w.bs / 60.0, l.lim_count_on, l.bad_count_on, 60 * l.bad_count_on
         FROM r JOIN runs_hi u USING (dev, det) JOIN win w USING (dev) JOIN limits l USING (dev, det)
         WHERE r.f_count_on OR r.sev_rule = 'C'
  UNION ALL SELECT r.dev, r.det, 2, w.s_ns + r.drop_b0 * {{ NS5 }}, w.s_ns + r.drop_b1 * {{ NS5 }}, 'silent', {{ sil }}
         WHERE r.f_dropout
  UNION ALL SELECT r.dev, r.det, 3, w.s_ns + r.drop_b0 * {{ NS5 }}, w.s_ns + r.drop_b1 * {{ NS5 }}, 'silent at night', {{ sil }}
         WHERE r.f_night_drop AND r.silent_night
  {% if opt['all_silences'] %}UNION ALL SELECT r.dev, r.det, 2, w.s_ns + u.a * {{ NS5 }}, w.s_ns + u.z * {{ NS5 }}, 'silent', u.lam, l.lim_dropout, l.bad_dropout, NULL FROM r JOIN silmore u USING (dev, det) JOIN win w USING (dev) JOIN limits l USING (dev, det) WHERE r.f_dropout OR r.f_night_drop{% endif %}

  {% if opt['drop_end'] %}UNION ALL SELECT r.dev, r.det, 4, (h.hid0 + u.ja) * {{ NSH }}, least((h.hid0 + u.jz) * {{ NSH }}, w.e_ns), 'counts dropped'{{ sc('level') }} FROM r JOIN lvr u USING (dev, det) JOIN win w USING (dev) JOIN (SELECT dev, min(hid) AS hid0 FROM hix GROUP BY dev) h USING (dev) WHERE r.f_level{{ ' AND NOT coalesce(r.v_lvl_md > r.v_lvl_old, false)' if opt['mate_drop'] else '' }}{% else %}UNION ALL SELECT r.dev, r.det, 4, epoch_ns(x.drop_t), coalesce(epoch_ns(x.drop_t2), w.e_ns), 'counts dropped'{{ sc('level') }} FROM r JOIN xf x USING (dev, det) JOIN win w USING (dev) WHERE r.f_level AND x.drop_t IS NOT NULL{% endif %}

  UNION ALL SELECT r.dev, r.det, 5, w.s_ns + u.a * {{ NSH }}, least(w.s_ns + u.z * {{ NSH }}, w.e_ns), 'too-fast actuations'
         {{ sc('rapid') }} FROM r JOIN runs_fast u USING (dev, det) JOIN win w USING (dev) WHERE r.f_rapid
  UNION ALL SELECT r.dev, r.det, 6, w.s_ns + u.a * {{ NS5 }}, least(w.s_ns + u.z * {{ NS5 }}, w.e_ns), 'too many for the traffic'
         {{ sc('volume') }} FROM r JOIN runs_vol u USING (dev, det) JOIN win w USING (dev) WHERE r.f_volume
  UNION ALL SELECT r.dev, r.det, 7, x.tod_d0 + {{ NSH }}, x.tod_d0 + 5 * {{ NSH }}, 'busy at night'{{ sc('prof') }}
         FROM r JOIN xf x USING (dev, det) WHERE r.f_prof AND x.tod_d0 IS NOT NULL AND NOT coalesce(x.flat, false)
  UNION ALL SELECT r.dev, r.det, 7, w.s_ns, w.e_ns, 'no daily pattern'{{ sc('prof') }}
         FROM r JOIN xf x USING (dev, det) JOIN win w USING (dev) WHERE r.f_prof AND coalesce(x.flat, false)
  UNION ALL SELECT r.dev, r.det, 8, w.s_ns + u.a * 600000000000, least(w.s_ns + u.z * 600000000000, w.e_ns),
         'spiky counts',
         (SELECT max(o.g) FROM o10s o WHERE o.dev = u.dev AND o.det = u.det AND o.t >= u.a AND o.t < u.z AND o.sp),
         {{ P['geh']|dbl }}, {{ P['geh_big']|dbl }}, {{ spl }}
         FROM r JOIN spk_runs u USING (dev, det) JOIN win w USING (dev) WHERE r.f_spiky
  {% if opt['mate_drop'] %}UNION ALL SELECT r.dev, r.det, 4, u.a * {{ NSH }}, least(u.z * {{ NSH }}, w.e_ns), 'counts dropped', u.rgeh, {{ g0 }}, {{ g1 }}, NULL FROM r JOIN mhrun u USING (dev, det) JOIN win w USING (dev) WHERE r.f_level AND coalesce(r.v_lvl_md > r.v_lvl_old, false) AND u.kind = 'drop' AND u.hours >= 2 AND u.rgeh >= {{ g0 }}
  {% endif %}{% if opt['count_rise'] %}UNION ALL SELECT r.dev, r.det, 10, u.a * {{ NSH }}, least(u.z * {{ NSH }}, w.e_ns), 'counts too high', u.rgeh, {{ g0 }}, {{ g1 }}, NULL FROM r JOIN mhrun u USING (dev, det) JOIN win w USING (dev) WHERE r.f_rise AND u.kind = 'rise' AND u.hours >= 2 AND u.rgeh >= {{ g0 }}{% endif %}

  {% if lc %}UNION ALL SELECT dev, det, 9, a_ns, z_ns, CASE WHEN switch THEN 'timing plan' ELSE 'lane closure' END, NULL, NULL, NULL, NULL FROM lcres{% endif %}

{# a finding without a period of its own: one row over the window, at the finding's own score (0.35 suspect -> 0.5,
   0.75 bad -> 0.8; >= 0.8 when the rules rate the finding bad) #}
{% set v %}CASE f.key {% for k in KEYS %}WHEN '{{ k }}' THEN v.v_{{ k }}{% if not loop.last %} {% endif %}{% endfor %} END{% endset %}
-- @step pwhole
SELECT f.dev, f.det, 20 + f.ki AS o, w.s_ns AS a, w.e_ns AS z, CASE f.key WHEN 'stuck' THEN 'stuck on' WHEN 'count_on' THEN 'count zone held ON' WHEN 'dropout' THEN 'silent'
         WHEN 'night_drop' THEN CASE WHEN v.silent_night THEN 'silent at night' ELSE 'misses vehicles at night' END
         WHEN 'level' THEN 'counts dropped' WHEN 'rapid' THEN 'too-fast actuations'
         WHEN 'volume' THEN 'too many for the traffic' WHEN 'spiky' THEN 'spiky counts' WHEN 'rise' THEN 'counts too high'
         WHEN 'prof' THEN CASE WHEN coalesce(x.flat, false) THEN 'no daily pattern' ELSE 'busy at night' END END AS what,
       CASE WHEN f.sev = 'bad' THEN greatest({{ lib.conf(v, '0.35', '0.75') }}, 0.8) ELSE {{ lib.conf(v, '0.35', '0.75') }} END AS conf,
       CASE f.key WHEN 'stuck' THEN l.bad_stuck WHEN 'count_on' THEN 60 * l.bad_count_on WHEN 'spiky' THEN {{ spl }} END AS lim
FROM fnd f JOIN finV v USING (dev, det) JOIN xf x USING (dev, det) JOIN win w USING (dev) JOIN limits l USING (dev, det)
WHERE v.health_st IN ('suspect', 'bad')
{# in a lane closure / plan switch of its own, or within 30 min of one (the lane reopening) #}
-- @step pall
SELECT p.dev, p.det, p.o, p.a, p.z, p.what,
       CASE WHEN p.o IN (2, 3) AND p.z - p.a >= 4 * {{ NSH }} AND p.x >= p.lo THEN greatest(p.conf, 0.8) ELSE p.conf END AS sev,
       p.lim, {{ '(SELECT count(*) > 0 FROM lcres c WHERE c.dev = p.dev AND c.det = p.det AND c.a_ns < p.z + 1800000000000 AND c.z_ns > p.a - 1800000000000)' if lc else 'false' }} AS lc, p.what IN {{ WHOLE|inlist }} AS ww
FROM (SELECT p.*, {{ lib.conf('p.x', 'p.lo', 'p.hi') }} AS conf FROM pkept p) p
UNION ALL
SELECT w.dev, w.det, w.o, w.a, w.z, w.what, w.conf, w.lim, false, true FROM pwhole w
WHERE NOT EXISTS (SELECT 1 FROM pkept p WHERE p.dev = w.dev AND p.det = w.det AND p.what = w.what){% if opt['pulse_mode'] %}

UNION ALL
SELECT p.dev, p.det, 30, w.s_ns, w.e_ns, 'pulse mode', 0.5, NULL, false, true FROM pmode p JOIN win w USING (dev){% endif %}

