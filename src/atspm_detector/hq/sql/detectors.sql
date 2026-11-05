{# The detectors table (health_duckdb._inputs): the classifier's answers per detector -- phase, function, lane span,
   class probabilities; note 154ac: the configured function (optional column configured_function), read by the
   pulse-mode check only.  The extractor reads each detector's phase (vehicle starts' colour, green overlap) here too; an
   update (phase_only) reads nothing else, and the function only when overlaps are kept for some functions.
   Context: probabilities, configured (the detectors table has p_* / configured_function columns), phase_only, OVFN. #}
{% import "_lib.sql" as lib with context %}
-- @step dcl mat
{% if phase_only %}
SELECT w.dev, d.Detector::INT AS det, {{ lib.nn('d.phase') }} AS ph{% if OVFN is not none %}, CAST(d.function AS VARCHAR) AS fn{% endif %}

{% else %}
SELECT w.dev, d.Detector::INT AS det, {{ lib.nn('d.phase') }} AS ph, CAST(d.function AS VARCHAR) AS fn,
       {{ lib.nn('d.lanes') }} AS lanes, {% for c in PROBCOLS %}{{ lib.nn('d.' ~ c) if probabilities else 'NULL::DOUBLE' }} AS {{ c }}, {% endfor %}{{ "NULLIF(trim(CAST(d.configured_function AS VARCHAR)), '')" if configured else 'NULL::VARCHAR' }} AS cfn
{% endif %}
FROM @DETECTORS@ d JOIN win w ON CAST(d.DeviceId AS VARCHAR) = CAST(w.DeviceId AS VARCHAR)
{# the detectors whose green overlap is kept: those with a phase whose function is in OVFN (None: every one; design
   decision 6).  The extractor writes it, finalize completes it. #}
-- @step xovd mat
SELECT dev, det, ph FROM dcl WHERE ph IS NOT NULL{% if OVFN is not none %} AND fn IN {{ OVFN|inlist }}{% endif %}

