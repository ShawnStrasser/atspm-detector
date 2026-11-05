{# Shared SQL pieces of the vehicle health templates (each the helper of health_duckdb.py it replaces).  Import with
   {% import "_lib.sql" as lib with context %}. #}
{# health's allowed events: codes ALLOWED, detector events 81 / 82 only on channels 1-64 (p: a table alias + '.') #}
{% macro good(p='') %}{{ p }}EventId IN ({{ ALLOWED|join(',') }}) AND ({{ p }}EventId NOT IN (81, 82) OR {{ p }}Parameter BETWEEN 1 AND 64){% endmacro %}
{# a DOUBLE with NaN as NULL (_nn) #}
{% macro nn(x) %}(CASE WHEN isnan(({{ x }})::DOUBLE) THEN NULL ELSE ({{ x }})::DOUBLE END){% endmacro %}
{# the time ON a detector's quiet bin is judged on: time of unknown state counts too (note 154q, _qocc) #}
{% macro qocc(a='') %}({{ a }}occ + {{ a }}unk){% endmacro %}
{# rows of src (dev, s, f: ns from the origin, s <= f; and the columns cols) split at the 5-min bins: cols, b (INT), t
   (its ns in the bin).  Integers: the parts of a time add up to it exactly (design 3.2).  clamp: f may be the window
   end itself (finalize), whose bin does not exist; the extractor's times all lie before it. #}
{% macro split5(src, cols, clamp=false) %}
WITH o AS (SELECT o.*, (o.s // {{ NS5 }})::INT AS b0, {% if clamp %}least(o.f // {{ NS5 }}, w.nb5 - 1){% else %}(o.f // {{ NS5 }}){% endif %}::INT AS b1
           FROM ({{ src }}) o{% if clamp %} JOIN win w USING (dev){% endif %})
SELECT {{ cols }}, b0 AS b, least(f, (b0 + 1) * {{ NS5 }}) - s AS t FROM o WHERE b1 = b0
UNION ALL
SELECT {{ cols }}, b, least(f, (b + 1) * {{ NS5 }}) - greatest(s, b * {{ NS5 }})
FROM (SELECT *, unnest(range(b0, b1 + 1))::INT AS b FROM o WHERE b1 > b0){% endmacro %}
{# health_core._score: 0 below lo, .35 at lo rising to 1 at hi; NULL stays NULL (_sc1) #}
{% macro sc1(x, lo, hi) %}(CASE WHEN ({{ x }}) IS NULL OR isnan({{ x }}) THEN NULL WHEN ({{ x }}) < {{ lo }} THEN 0.0 ELSE 0.35 + 0.65 * least(greatest((({{ x }}) - {{ lo }}) / ({{ hi }} - {{ lo }}), 0.0), 1.0) END){% endmacro %}
{# health_v4.score: 0 below the suspect limit, .35 at it, .75 at the bad limit, 1 at most; NULL = not checked (_sc) #}
{% macro sc(x, lo, bad) %}(CASE WHEN ({{ x }}) IS NULL OR isnan({{ x }}) OR ({{ lo }}) IS NULL OR isnan({{ lo }}) THEN NULL WHEN ({{ x }}) < ({{ lo }}) THEN 0.0 ELSE least(0.35 + 0.40 * greatest((({{ x }}) - ({{ lo }})) / greatest(({{ bad }}) - ({{ lo }}), 1e-9), 0.0), 1.0) END){% endmacro %}
{# signed GEH: + when m < c (under-counting) (_sgeh) #}
{% macro sgeh(m, c) %}(CASE WHEN ({{ m }}) < ({{ c }}) THEN 1 ELSE -1 END * sqrt(2 * (({{ m }}) - ({{ c }})) ^ 2 / greatest(({{ m }}) + ({{ c }}), 1e-9))){% endmacro %}
{# GEH without sign #}
{% macro geh(a, b) %}sqrt(2 * ({{ a }} - {{ b }}) ^ 2 / greatest({{ a }} + {{ b }}, 1e-9)){% endmacro %}
{# a period's severity on its check's own scale: 0.5 at lo, 0.8 at hi, on log strength (_conf) #}
{% macro conf(x, lo, hi) %}(CASE WHEN ({{ x }}) > 0 AND ({{ lo }}) > 0 THEN 1.0 / (1.0 + exp(-ln(4.0) * ln(({{ x }}) / ({{ lo }})) / ln(greatest(({{ hi }}) / ({{ lo }}), 1.05)))) WHEN ({{ lo }}) IS NOT NULL THEN 0.0 END){% endmacro %}
{# health_v4._dur: '45 s', '12 min', '2 h 5 min' (_dur) #}
{% macro dur(x) %}{% set m = 'round_even((' ~ x ~ ') / 60.0, 0)::BIGINT' %}(CASE WHEN ({{ x }}) < 300 THEN format('{:.0f} s', ({{ x }})) WHEN {{ m }} < 60 THEN {{ m }} || ' min' ELSE ({{ m }} // 60) || ' h' || CASE WHEN {{ m }} % 60 > 0 THEN ' ' || ({{ m }} % 60) || ' min' ELSE '' END END){% endmacro %}
{# a local time (epoch ns) as text (_hm) #}
{% macro hm(ns, fmt='%a %H:%M') %}strftime(make_timestamp(({{ ns }}) // 1000), '{{ fmt }}'){% endmacro %}
{# seconds from the window start t -> epoch ns #}
{% macro at_ns(t) %}w.s_ns + round(({{ t }}) * 1e9)::BIGINT{% endmacro %}
{# {name}_1 .. {name}_n: src with k5's running sum col of detector who at each boundary column of bounds (_at) #}
{% macro at(name, src, who, bounds, col) %}
{% for b in bounds %}
-- @step {{ name }}_{{ loop.index }} mat
SELECT x.*, k.{{ col }} AS {{ col }}_{{ b }} FROM {{ src if loop.first else name ~ '_' ~ loop.index0 }} x JOIN k5 k ON k.dev = x.dev AND k.det = x.{{ who }} AND k.k = x.{{ b }}
{% endfor %}
{% endmacro %}
{# the expected-ONs run with the most expected ONs (stage 1 dr and dc share it) #}
{% macro cap() %}{% if opt['no_recovery_cap'] %}JOIN win w ON w.dev = r.dev WHERE r.z - r.a <= w.nb5 - 12 AND s.n_on >= 5{% endif %}{% endmacro %}
