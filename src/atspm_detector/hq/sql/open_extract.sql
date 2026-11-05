{# What is still open at the window end of a batch run, from the extractor's steps (open_carry.sql: from a store's carry):
   keys in ns from the window start, decisions in today's double seconds on the run's origin. #}
{% macro at(x) %}({{ x }} + {{ ORG }} - w.s_ns){% endmacro %}
-- @step oxc
SELECT c.dev, c.det, {{ at('c.s') }} AS s, c.s / 1e9 AS sd, {{ at('c.mk') }} AS mk FROM xoc c JOIN win w USING (dev)
-- @step oxp
SELECT p.dev, p.ph, p.st, {{ at('p.tag') }} AS tag, p.ad, p.mk / 1e9 AS mkd FROM xpc p JOIN win w USING (dev) WHERE p.op AND p.st < 3
-- @step oxl mat
SELECT l.dev, {{ at('l.lz') }} AS lz, l.lz / 1e9 AS lzd FROM xlast l JOIN win w USING (dev)
{# each detector's last ON, its too-fast pair pending #}
-- @step oxt
SELECT p.dev, p.det, {{ at('p.dn') }} AS dn, p.ta FROM xtf p JOIN win w USING (dev) WHERE p.tb IS NULL
