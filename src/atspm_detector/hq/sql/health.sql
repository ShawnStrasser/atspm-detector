{# A batch run of the vehicle health (health_duckdb.build, design 3.1): the windows and the detectors table; the extractor
   over each signal's window as one chunk with nothing carried in, writing the store tables (temp tables here; store.sql
   has their layout); then what a judge on a store does (judge_steps.sql), what is open at the window end coming from the
   extractor's steps.
   Context (health_duckdb._context): opt, periods, windows, clock, start, end, probabilities, configured, OVFN, TFC, ORG,
   TE, store, phase_only, the limits LJ / R, their VALUES rows T, derived numbers D and the constants. #}
{% include "windows.sql" %}
{% include "detectors.sql" %}
{% include "extract/input.sql" %}
{% include "extract/signal.sql" %}
{% include "extract/phase.sql" %}
{% include "extract/detector.sql" %}
{% include "extract/overlap.sql" %}
{% include "extract/rows.sql" %}
{% include "extract/emit.sql" %}
{% include "open_extract.sql" %}
{% include "judge_steps.sql" %}
