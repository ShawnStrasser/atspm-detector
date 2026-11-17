{# note 158 (option silent_not_held, health_duckdb): a detector held ON sends no new actuations, so a silence over the same
   hours (a whole phase dropped, every input lost, a quiet run) would report one fault twice.  The silence keeps only its
   longest stretch of 5-min bins that no hold of the detector touches (continuous ONs >= 15 min = stuck on / held ON together,
   a Count zone held ON, a mostly-ON run); its expected count and the phase drop's counts shrink with it, and a silence
   with no such stretch left is no silence.  Silence at other times is still reported -> out (src, same columns). #}
{% macro silent_held(src, out) %}
{# the holds: hiv, and every hold the stuck-on check reports (a shorter type limit than hiv's 15 min) #}
-- @step shh mat
SELECT dev, det, s, f FROM hiv UNION ALL SELECT dev, det, s, f FROM efl WHERE label <> 'queue'
-- @step shb mat
SELECT d.dev, d.det, g.b, g.n,
       EXISTS (SELECT 1 FROM shh h WHERE h.dev = d.dev AND h.det = d.det AND h.f > g.b * 300.0 AND h.s < (g.b + 1) * 300.0) AS held
FROM {{ src }} d JOIN g5 g ON g.dev = d.dev AND g.det = d.det AND g.b >= d.drop_b0 AND g.b < d.drop_b1
WHERE d.drop_b1 > d.drop_b0
  AND EXISTS (SELECT 1 FROM shh h WHERE h.dev = d.dev AND h.det = d.det AND h.f > d.drop_b0 * 300.0 AND h.s < d.drop_b1 * 300.0)
-- @step shr mat
SELECT o.dev, o.det, r.a, r.z, r.obs FROM (SELECT DISTINCT dev, det FROM shb) o LEFT JOIN (
  SELECT dev, det, min(b) AS a, max(b) + 1 AS z, sum(n) AS obs FROM (
    SELECT *, b - row_number() OVER (PARTITION BY dev, det ORDER BY b) AS grp FROM shb WHERE NOT held)
  GROUP BY dev, det, grp
  QUALIFY row_number() OVER (PARTITION BY dev, det ORDER BY max(b) + 1 - min(b) DESC, min(b)) = 1) r USING (dev, det)
-- @step {{ out }} mat
SELECT x.* REPLACE (
    CASE WHEN c.dev IS NULL THEN x.drop_lam WHEN c.a IS NULL THEN 0.0
         ELSE x.drop_lam * (c.z - c.a) / (x.drop_b1 - x.drop_b0) END AS drop_lam,
    CASE WHEN c.dev IS NULL THEN x.drop_b0 WHEN c.a IS NULL THEN -1 ELSE c.a END AS drop_b0,
    CASE WHEN c.dev IS NULL THEN x.drop_b1 WHEN c.a IS NULL THEN -1 ELSE c.z END AS drop_b1,
    CASE WHEN c.dev IS NULL THEN x.s_dropout WHEN c.a IS NULL THEN 0.0
         WHEN x.drop_lam * (c.z - c.a) / (x.drop_b1 - x.drop_b0) < 30 THEN 0.0
         ELSE least(x.s_dropout, 0.35 + 0.65 * least((x.drop_lam * (c.z - c.a) / (x.drop_b1 - x.drop_b0) - 30) / 70, 1)) END
      AS s_dropout{% if opt['phase_drop'] %},
    CASE WHEN c.dev IS NULL OR NOT x.ph_drop THEN x.ph_obs WHEN c.a IS NULL THEN 0 ELSE c.obs END AS ph_obs,
    CASE WHEN c.dev IS NULL OR NOT x.ph_drop THEN x.ph_exp WHEN c.a IS NULL THEN 0.0
         ELSE x.ph_exp * (c.z - c.a) / (x.drop_b1 - x.drop_b0) END AS ph_exp{% endif %})
FROM {{ src }} x LEFT JOIN shr c USING (dev, det)
{% endmacro %}
