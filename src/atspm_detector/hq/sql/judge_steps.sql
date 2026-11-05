{# The steps from the store's rows to the outputs (design 3.1), a batch run and a judge on a store alike: the store's rows
   read back, finalize at the window end, the judge on the adapters -- 5-min bins, stage 1, the measures, the assembled
   table and its verdict, the outputs. #}
{% import "judge/verdict.sql" as vd with context %}
{% import "judge/signal_silent.sql" as ss with context %}
{% import "judge/phase_drop.sql" as pd with context %}
{% import "adapters.sql" as ad with context %}
{% include "store_read.sql" %}
{% include "finalize.sql" %}
{{ ad.cov5() }}
{{ ad.dets() }}
{% include "judge/bins.sql" %}
{% include "judge/stage1.sql" %}
{% include "judge/measures.sql" %}
{% include "judge/held_on.sql" %}
{% include "judge/hourly.sql" %}
{% if opt['drop_end'] %}
{% include "judge/low_runs.sql" %}
{% endif %}
{% include "judge/spiky.sql" %}
{% include "judge/holds.sql" %}
{% if opt['mate_drop'] or opt['count_rise'] %}
{% include "judge/mate_hours.sql" %}
{% endif %}
{% include "judge/assemble.sql" %}
{# the signal-wide checks rewrite the silence of xfa -> (xfs ->) xf #}
{% if opt['signal_silent'] %}
{{ ss.signal_silent('xfa', 'xfs' if opt['phase_drop'] else 'xf') }}
{% endif %}
{% if opt['phase_drop'] %}
{{ pd.phase_drop('xfs' if opt['signal_silent'] else 'xfa', 'xf') }}
{% endif %}
{% if opt['lane_closure'] %}
{% include "judge/closure.sql" %}
{{ vd.verdict('fin', 'xfc', true) }}
{% else %}
{{ vd.verdict('fin', 'xf') }}
{% endif %}
{% include "judge/outputs.sql" %}
{% if periods %}
{% include "judge/periods.sql" %}
{% endif %}
{% include "judge/final.sql" %}
