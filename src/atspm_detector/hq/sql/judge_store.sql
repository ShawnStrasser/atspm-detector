{# A judge on a store (incremental.judge, design 3.1): the windows from the store, the detectors table, what is open from
   the carry, then what a batch run does from its store tables on. #}
{% include "windows.sql" %}
{% include "detectors.sql" %}
{% include "open_carry.sql" %}
{% include "judge_steps.sql" %}
