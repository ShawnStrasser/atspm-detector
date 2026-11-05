{# Categories with severity, the plain reason per finding, lane closures / plan switches as info (health_duckdb._outputs,
   up to the categories). #}
{% import "_lib.sql" as lib with context %}
{% set lc = opt['lane_closure'] %}
{% if opt['pulse_mode'] %}
{# note 154aa: complete calls (one ON to its OFF, nothing lost or re-logged) of the Presence detectors; note 154ac (user:
   count, yellow-red and advance zones may pulse, presence may not): a zone configured as Count / Advance (Advance_Presence
   too) / Yellow_Red, where the detectors table gives the configured function, is never flagged #}
-- @step pmode mat
SELECT * FROM (
  SELECT o.dev, o.det, o.np, o.nps / o.np AS ps, o.med AS pmed
  FROM onmed o JOIN xf x USING (dev, det) JOIN dcl c USING (dev, det)
  WHERE x.fn = 'Presence' AND NOT coalesce(lower(c.cfn) IN ('count', 'yellow_red', 'yellow-red')
                                          OR lower(c.cfn) LIKE 'advance%', false))
WHERE np >= {{ PULSE['n'] }} AND ps >= {{ PULSE['share']|dbl }} AND round(pmed, 3) < {{ PULSE['s']|dbl }}
{% endif %}
{% set vsel %}{% for k in KEYS %}WHEN '{{ k }}' THEN v.v_{{ k }}{% if not loop.last %} {% endif %}{% endfor %}{% endset %}
{# the reported findings: the kept ones, plus the Count zone held ON of rule C #}
-- @step fnd mat
SELECT v.dev, v.det, k.key, k.ki, k.cat, coalesce(CASE k.key {{ vsel }} END, 0.0) AS sc,
       CASE WHEN (v.sev_rule = 'C' AND k.key = 'count_on') OR (v.sev_rule = 'F' AND k.key = 'rapid') THEN 'suspect'
            WHEN v.sev_rule = 'P' AND (k.key = 'prof' OR (k.key = 'dropout' AND v.silent_h >= 4.0)) THEN 'bad'
            WHEN coalesce(CASE k.key {{ vsel }} END, 0.0) >= 0.75 THEN 'bad' ELSE 'suspect' END AS sev
FROM finV v, ({{ T['kv']|values('key, ki, cat') }}) k WHERE CASE k.key {% for k in KEYS %}WHEN '{{ k }}' THEN v.f_{{ k }}{% if not loop.last %} {% endif %}{% endfor %} END OR (k.key = 'count_on' AND v.sev_rule = 'C')
{# stuck-on holds over the detector's limit, labelled (health_v4._stuck_eps) #}
-- @step eps
SELECT e.dev, e.det, count(*) FILTER (e.label <> 'queue') AS n_cnt, sum(e.dur) FILTER (e.label <> 'queue') AS tot,
       min(e.s) FILTER (e.label <> 'queue') AS t0, count(*) FILTER (e.label = 'not_queue') AS n_nq,
       min(e.busy_occ_e) FILTER (e.label = 'not_queue') AS nq_occ, count(*) FILTER (e.label = 'together') AS n_tog,
       sum(e.dur) FILTER (e.label = 'together') AS tog_tot, min(e.s) FILTER (e.label = 'together') AS tog_t0,
       arg_min(e.dur, e.s) FILTER (e.label = 'together') AS tog_dur,
       list_sort(list_distinct(flatten(list(e.mates) FILTER (e.label = 'together')))) AS tog_mates
FROM efl e GROUP BY e.dev, e.det
{# the reason texts #}
{% set why = "CASE WHEN e.n_nq > 0 AND e.n_nq = e.n_cnt THEN format('; not a queue: at the same traffic it was ON {:.0f} % of the time', 100 * e.nq_occ) ELSE '' END" %}
{% set stuck %}CASE WHEN e.n_tog > 0 THEN 'held ON ' || CASE WHEN e.n_tog = 1
        THEN {{ lib.dur('e.tog_dur') }} || ' from ' || {{ lib.hm(lib.at_ns('e.tog_t0')) }}
        ELSE e.n_tog || ' times from ' || {{ lib.hm(lib.at_ns('e.tog_t0')) }} || ', ' || {{ lib.dur('e.tog_tot') }} || ' in total' END
        || ' together with det ' || array_to_string(e.tog_mates, ', ') || ' (same start and end: a common failure)'
      WHEN e.n_cnt > 1 THEN 'held ON ' || e.n_cnt || ' times, ' || {{ lib.dur('e.tot') }} || ' in total' || {{ why }}
      WHEN e.n_cnt = 1 THEN 'held ON ' || {{ lib.dur('e.tot') }} || ' from ' || {{ lib.hm(lib.at_ns('e.t0')) }} || {{ why }}
      ELSE 'held ON up to ' || {{ lib.dur('x.stuck_x') }} END{% endset %}
{% if opt['silent_merge'] %}{# note 154: a window over a day names the day too #}
{% set t0 %}CASE WHEN w.hours > 24.5 THEN {{ lib.hm('w.s_ns + x.drop_b0 * ' ~ NS5) }} ELSE {{ lib.hm('w.s_ns + x.drop_b0 * ' ~ NS5, '%H:%M') }} END{% endset %}
{% set t1 %}CASE WHEN w.hours > 24.5 THEN {{ lib.hm('w.s_ns + x.drop_b1 * ' ~ NS5) }} ELSE {{ lib.hm('w.s_ns + x.drop_b1 * ' ~ NS5, '%H:%M') }} END{% endset %}
{% set silent %}'no actuations ' || {{ t0 }} || '-' || {{ t1 }} || format('; ~{:,.0f} expected', x.drop_lam){% endset %}
{% else %}
{% set silent %}'no actuations ' || {{ lib.hm('w.s_ns + x.drop_b0 * ' ~ NS5, '%H:%M') }} || '-' || {{ lib.hm('w.s_ns + x.drop_b1 * ' ~ NS5, '%H:%M') }} || format('; ~{:,.0f} expected', x.drop_lam){% endset %}
{% endif %}
{% if opt['signal_silent'] %}
{% set silent = silent ~ " || CASE WHEN x.sig_silent THEN '; every detector at the signal was silent while it kept serving greens (all inputs lost: detector rack / cabinet)' ELSE '' END" %}
{% endif %}
{% if opt['phase_drop'] %}{# note 154b: the whole phase fell together (t0 / t1: silent_merge's; without it this fails, as before) #}
{% set silent %}CASE WHEN x.ph_lock THEN 'logs the same count as ' || (x.lk_n - 1) || ' other detectors on P' || trunc(x.ph)::BIGINT || ' every hour ' || {{ t0 }} || '-' || {{ t1 }} || format(' (~{:,.0f} an hour each): the inputs no longer see vehicles and log the same pulses (detector card in fail-safe, or a call placed on them)', x.lk_md) WHEN x.ph_drop THEN 'whole phase dropped ' || {{ t0 }} || '-' || {{ t1 }} || format(': {:,.0f} actuations where ~{:,.0f} expected; ', x.ph_obs, x.ph_exp) || x.ph_k || ' of ' || x.ph_m || ' detectors on P' || trunc(x.ph)::BIGINT || ' fell together while the rest of the signal counted normally (approach closed, or a shared card / cable)' ELSE {{ silent }} END{% endset %}
{% endif %}
{% if opt['drop_end'] %}{# note 154b: plain words #}
{% set level %}CASE WHEN x.own_after < 0.1 * x.exp_after THEN 'nearly stopped counting'
       WHEN x.occ_own >= 3.0 AND x.occ_own >= 3.0 * coalesce(x.occ_mates, 1.0) THEN format('counts fell while its time ON rose {:.0f} x (holds ON longer: stuck, or a changed '
                                         || 'setting)', x.occ_own)
       ELSE 'counts and occupancy both fell' END || CASE WHEN x.drop_t IS NOT NULL THEN ' from ' || strftime(x.drop_t, '%a %H:%M') ELSE '' END
       || CASE WHEN x.drop_t2 IS NOT NULL THEN ' to ' || strftime(x.drop_t2, '%a %H:%M') ELSE '' END
       || format(': {:.0f} per hour where its phase mates'' traffic gives ~{:.0f} (GEH {:.1f})', x.own_after,
                 x.exp_after, x.geh_drop){% endset %}
{% else %}
{% set level %}'counts dropped' || CASE WHEN x.drop_t IS NOT NULL THEN ' from ' || strftime(x.drop_t, '%a %H:%M')
       ELSE '' END || format(': {:.0f} per hour where its phase mates'' traffic gives ~{:.0f} (GEH {:.1f}); '
                             || 'its occupancy fell too', x.own_after, x.exp_after, x.geh_drop){% endset %}
{% endif %}
{% if opt['all_silences'] %}
-- @step silmore mat
SELECT u.dev, u.det, u.a, u.z, u.lam FROM silall u JOIN xf x USING (dev, det) JOIN limits l USING (dev, det)
JOIN win w USING (dev)
WHERE u.lam >= l.lim_dropout AND NOT (u.a < x.drop_b1 AND u.z > x.drop_b0) {% if lc %}AND NOT EXISTS (SELECT 1 FROM lcres c WHERE c.dev = u.dev AND c.det = u.det AND least(c.z_ns, w.s_ns + u.z * {{ NS5 }}) - greatest(c.a_ns, w.s_ns + u.a * {{ NS5 }}) >= 0.5 * (u.z - u.a) * {{ NS5 }}){% endif %}

  AND (u.lam >= 50 OR (SELECT count(*) FILTER (b.hr >= 21 OR b.hr < 5) FROM bins5 b
                       WHERE b.dev = u.dev AND b.b >= u.a AND b.b < u.z) < 0.8 * (u.z - u.a))
{% set silent %}{{ silent }} || coalesce((SELECT '; also ' || string_agg({{ lib.hm('w.s_ns + u.a * ' ~ NS5) }} || '-' || {{ lib.hm('w.s_ns + u.z * ' ~ NS5) }} || format(' (~{:,.0f})', u.lam), ', ' ORDER BY u.a) FILTER (u.rk <= 3) || CASE WHEN count(*) > 3 THEN format(' and {} more', count(*) - 3) ELSE '' END FROM (SELECT *, row_number() OVER (ORDER BY round(lam, 6) DESC, a) AS rk FROM silmore WHERE dev = x.dev AND det = x.det) u), ''){% endset %}
{% endif %}
{% if opt['mate_drop'] %}{# note 154n: the run against the phase mates, when it rates the drop #}
{% set level %}CASE WHEN coalesce(v.v_lvl_md, 0) > coalesce(v.v_lvl_old, 0) THEN format('counts fell to {:.0f} % of what its phase mates gave at that time of day on the other days: ', 100 * x.md_r) || strftime(make_timestamp(x.md_a * 3600000000), '%a %H:%M') || '-' || strftime(make_timestamp(x.md_z * 3600000000), '%H:%M') || format(' ({} h, ~{:.0f} an hour where ~{:.0f} expected; GEH {:.0f})', x.md_h, x.md_own, x.md_exp, x.md_geh) ELSE {{ level }} END{% endset %}
{% endif %}
{% if opt['count_rise'] %}
{# against almost-silent mates the multiple means nothing (mr_r = own / max(mates, 1e-9) reached 2.3e12): plain words from 100 x #}
{% set rise %}CASE WHEN x.mr_r >= 100 THEN 'counts rose far above its phase mates, which gave almost nothing at that time of day on the other days: ' ELSE format('counts rose to {:.1f} x what its phase mates gave at that time of day on the other days: ', x.mr_r) END || strftime(make_timestamp(x.mr_a * 3600000000), '%a %H:%M') || '-' || strftime(make_timestamp(x.mr_z * 3600000000), '%H:%M') || format(' ({} h, GEH {:.0f}): extra calls', x.mr_h, x.mr_geh){% endset %}
{% else %}
{% set rise = "''" %}
{% endif %}
-- @step evid
SELECT f.dev, f.det, f.key, f.ki,
       CASE WHEN f.key = 'prof' AND x.flat THEN 'No daily pattern' ELSE f.cat END AS cat, f.sc, f.sev, CASE f.key
  WHEN 'stuck' THEN {{ stuck }}
  WHEN 'count_on' THEN format('count zone ON up to {:.0f} % in light traffic for {:.0f} min', 100 * x.hi_occ_max,
                              x.hi_min_c)
  WHEN 'dropout' THEN {{ silent }}
  WHEN 'night_drop' THEN CASE WHEN v.silent_night THEN {{ silent }} WHEN x.geh_night_exp IS NOT NULL
       THEN format('night {:,.0f} actuations vs ~{:,.0f} expected (GEH {:.1f})', x.geh_night_n, x.geh_night_exp,
                   x.geh_night) ELSE 'fewer actuations at night' END
  WHEN 'level' THEN {{ level }}
  WHEN 'rapid' THEN format('{:,} of {:,} ONs follow the previous ON within {:.2g} s ({:.0f} per 1,000; '
                           || 'limit {:.0f})', x.n_fast, x.N, x.gap_lim, x.fast_rate, l.lim_rapid)
       || CASE WHEN x.pair_share >= 0.9 THEN '; counts each vehicle twice' ELSE '' END
       || CASE WHEN v.sev_rule = 'F' THEN ', spread evenly with traffic - flickers as vehicles pass, check zone setup'
               ELSE '' END
  WHEN 'spiky' THEN format('{:.0f} bursts{} in 10-min counts at 3 x its usual level or more (largest GEH {:.0f}); '
                           || '~{:,.0f} vehicles a day', x.spk_bursts, CASE WHEN x.hours > 24.5 THEN ' a day' ELSE '' END,
                           x.spk_gmax, x.spk_daily)
  WHEN 'volume' THEN format('busiest 5 min {:,.0f} veh/h/lane vs limit {:,.0f}', x.q5_use, l.lim_volume)
  WHEN 'rise' THEN {{ rise }}
  WHEN 'prof' THEN CASE WHEN x.flat THEN format('counts do not follow the day: ~{:.0f} an hour by day, ~{:.0f} at night, while '
       || 'the rest of the signal drops from ~{:,.0f} to ~{:,.0f} (noise, not traffic)',
       x.flat_d, x.flat_n, x.flat_rd, x.flat_rn) ELSE
       format('night level {:.0f} % of its busiest hours; phase mates ', 100 * x.tod_r ^ 2)
       || CASE WHEN x.tod_mate_r IS NOT NULL THEN format('{:.0f} %', 100 * x.tod_mate_r ^ 2) ELSE 'none' END END
  END AS txt
FROM fnd f JOIN xf x USING (dev, det) JOIN finV v USING (dev, det) JOIN limits l USING (dev, det) JOIN win w USING (dev)
LEFT JOIN eps e USING (dev, det)
{% if lc %}
{# note 154: lane closures and timing-plan switches, reported (severity 'info') next to the faults #}
{% set rng = lib.hm('r.a_ns') ~ " || '-' || " ~ lib.hm('r.z_ns') %}
-- @step lctx
SELECT r.dev, r.det, 'closure' AS key, 99 AS ki, CASE WHEN r.switch THEN 'Timing plan' ELSE 'Lane closure' END
       AS cat, -1.0 AS sc, 'info' AS sev,
       CASE WHEN r.switch THEN {{ rng }} || CASE WHEN r.kind = 'd' THEN ': silent' ELSE ': counts low' END
                               || ' while coordination pattern ' || r.pat || ' is in effect'
                               || coalesce('; its traffic moves to ' || r.glab, '')
            ELSE {{ rng }} || format(': ~{:,.0f} of its vehicles moved to ', r.loss) || r.glab END AS txt, r.a_ns AS o
FROM lcres r
-- @step lcx mat
SELECT dev, det, bool_and(switch) AS all_switch FROM lcres GROUP BY dev, det
{% endif %}
-- @step cats
SELECT dev, det, string_agg(cat || ' (' || sev || ')', '; ' ORDER BY s DESC, ki0, cat) AS health_categories,
       string_agg(cat || ': ' || txt, '; ' ORDER BY s DESC, ki0, cat) AS parts, bool_or(lane) AS lane_cat
FROM (SELECT dev, det, cat, max(sc) AS s, min(ki) AS ki0,
             CASE WHEN bool_or(sev = 'bad') THEN 'bad' WHEN bool_or(sev = 'info') THEN 'info' ELSE 'suspect' END AS sev,
             string_agg(txt, '; ' ORDER BY ki, o, txt) AS txt, bool_or(key IN ('stuck', 'rapid', 'volume', 'prof', 'spiky')) AS lane
      FROM {{ '(SELECT dev, det, key, ki, cat, sc, sev, txt, 0::BIGINT AS o FROM evid UNION ALL SELECT * FROM lctx)' if lc else '(SELECT *, 0::BIGINT AS o FROM evid)' }} GROUP BY dev, det, cat)
GROUP BY dev, det
