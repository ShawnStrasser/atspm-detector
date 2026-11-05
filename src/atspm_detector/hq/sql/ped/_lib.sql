{# Shared pieces of the ped templates.  Import with {% import "ped/_lib.sql" as plib %}. #}
{# silences: consecutive events of any code >= 29 s apart, a margin under the 30-s hole limit (design 4.2), from 10-s buckets
   (key, a, z: first and last event in epoch ns; events in one bucket are never that far apart).  prev: a SELECT of more
   (key, a, z), the event before the chunk.  Rows: key, n0 (the event before), n1 (the event after), epoch ns.  Integers: one
   text for a batch and an update, whatever the chunking. #}
{% macro silences(src, key, prev=none) %}SELECT {{ key }}, z AS n0, nt AS n1
FROM (SELECT {{ key }}, z, lead(a) OVER (PARTITION BY {{ key }} ORDER BY a) AS nt
      FROM (SELECT {{ key }}, a, z FROM {{ src }}{% if prev %} UNION ALL {{ prev }}{% endif %}))
WHERE nt - z >= 29000000000{% endmacro %}
{# the window filter of the ped modules: rows with a signal and a time (codes: and those EventIds), [start, end) where
   given #}
{% macro window(start, end, codes=none) %}DeviceId IS NOT NULL AND Timestamp IS NOT NULL{% if codes is not none %} AND EventId IN {{ codes|intlist }}{% endif %}{% if start is not none %} AND Timestamp >= TIMESTAMP '{{ start|tstext }}'{% endif %}{% if end is not none %} AND Timestamp < TIMESTAMP '{{ end|tstext }}'{% endif %}{% endmacro %}
