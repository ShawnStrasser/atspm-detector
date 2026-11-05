{# The findings layer (findings.tables): the days of the windows, then the vehicle and ped rows of whichever of the two
   ran.  Context: vehicle, ped, and findings.CONTEXT (labels, patterns and the Confidence of the notes). #}
{% include "findings/days.sql" %}
{% if vehicle %}
{% include "findings/vehicle.sql" %}
{% endif %}
{% if ped %}
{% include "findings/ped.sql" %}
{% endif %}
-- @step frows
{% if vehicle %}SELECT * FROM vrows{% endif %}{% if vehicle and ped %} UNION ALL {% endif %}{% if ped %}SELECT * FROM prows{% endif %}

-- @step srows
{% if vehicle %}SELECT * FROM vstat{% endif %}{% if vehicle and ped %} UNION ALL {% endif %}{% if ped %}SELECT * FROM pstat{% endif %}

