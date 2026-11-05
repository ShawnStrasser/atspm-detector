{# One update of a store (incremental.update, design 3.1): the same extractor as a batch run over each signal's new
   events, with the carry bound to the store's, and what the next update needs. #}
{% include "windows.sql" %}
{% include "detectors.sql" %}
{% include "extract/input.sql" %}
{% include "extract/signal.sql" %}
{% include "extract/phase.sql" %}
{% include "extract/detector.sql" %}
{% include "extract/overlap.sql" %}
{% include "extract/rows.sql" %}
{% include "extract/emit.sql" %}
{% include "extract/carry.sql" %}
{% if ped %}
{% include "extract/ped.sql" %}
{% endif %}
