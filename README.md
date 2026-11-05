# atspm-detector

Input signal controller hi-res events; for every vehicle detector it returns the **phase**, **function**,
**lanes**, **setback** from the stop bar and **health**. For pedestrian detectors it outputs the phase/overlaps and health. No cabinet print needed, it uses a small AI model trained on controller data and hand labels from Oregon DOT.

Use it to **build** a signal's ATSPM detector configuration, to **validate** what already have, or for vehicle and pedestrain detector health monitoring.

* Tested on 115 signals it never trained on: phase right for 98.0 % of detectors, function 91.7 % (30 minutes or more of log).
* Fast and light: one second per signal on CPU.

Functions: Stopbar Presence, Advance Count, Stopbar Count, Yellow Red Count, plus non-ATSPM detector classification Mid, Bike and Other.

## Install

```bash
pip install atspm-detector   # Python 3.11 or newer
```

## Usage

Input: `DeviceId`, `Timestamp` (local time), `EventId`, `Parameter`. Process signals one at a time or all at once.

```python
import pandas as pd
from atspm_detector import predict

events = pd.read_parquet("events.parquet")
results = predict(events)
results[["Detector", "phase_pred", "phase_prob", "function_pred", "function_prob",
         "lanes", "distance_ft", "health_status"]].round(2).head()
```

| Detector | phase_pred | phase_prob | function_pred | function_prob | lanes | distance_ft | health_status |
|---|---|---|---|---|---|---|---|
| 2 | 2 | 0.99 | Advance | 0.90 | 2 | 164.0 | ok |
| 3 | 2 | 1.00 | Advance | 0.64 | 1,2 | 153.0 | ok |
| 4 | 2 | 1.00 | Presence | 0.40 | 1,2 | 0.0 | ok |
| 5 | 2 | 1.00 | Advance | 0.53 | 1 | 152.0 | ok |
| 6 | | | | | | | not_enough_data |

From the bundled 30-minute sample; detector 6 had one actuation, too few to judge.

One row per detector: phase and function (each with a probability), lanes, setback in ft, and `health_status`
(ok / watch / suspect / bad / not_enough_data / lane_closure / timing_plan). Under 5 actuations there is no answer.
`predict` also takes a parquet or CSV path or glob, and `start` / `end` times.

## How it works

* The model sees detector on / off, signal phasing and calls. It scores each
  detector against every phase that turned green, from behavior alone. 
* Gradient-boosted trees and a small temporal convolutional neural network pick phase and function; a last step keeps at most one detector
  of each ATSPM function per lane.
* Health is checked last. In fact, the health check requires knowing the detector phase and functions to work optimally.

## Detector health

Each detector gets a status and findings in plain words, judged against what is normal for its type, on a window of
30 minutes to 92 days. Checks: stuck on, counts fall, misses vehicles at night, busy at night, too fast / too many,
spiky counts, no daily pattern, counts too high. Lane closures and timing plans are reported, not counted as faults.
Ped push buttons are judged the same way. Not field-verified.

`predict` already runs health on its own answers. To run it yourself, give `health` each detector's phase,
function and number of lanes. Use `predict`'s output as below, or your own detector configuration with the same
columns. It runs in DuckDB, on tables in your connection:

```python
import duckdb
from atspm_detector import health

detectors = results.assign(phase=results.phase_pred, function=results.function_pred,
                           lanes=results.lanes.str.count(",") + 1)   # lanes: how many lanes the detector covers
con = duckdb.connect()
con.register("events", events)
con.register("detectors", detectors)
findings, status = health(con, "events", "detectors", summary=True)
```

`findings`:

| DeviceId | DetectorType | Detector | Start | End | Label | Confidence | Detail |
|---|---|---|---|---|---|---|---|
| d2d0de72-... | Vehicle | 6 | 2026-10-06 12:00:00 | 2026-10-06 12:30:00.9 | not_enough_data | 0.0 | only 1 actuations in 0.5 h - too few to judge |
| d2d0de72-... | Vehicle | 8 | 2026-10-06 12:00:00 | 2026-10-06 12:30:00.9 | not_enough_data | 0.0 | too few actuations to judge |

`status`, one row per detector and day:

| DeviceId | DetectorType | Detector | Date | Status |
|---|---|---|---|---|
| d2d0de72-... | Vehicle | 2 | 2026-10-06 | ok |
| d2d0de72-... | Vehicle | 3 | 2026-10-06 | ok |

`predict(events, return_findings=True)` returns the same findings in one step. One row per finding. On a real log a fault reads, for example, `Label` stuck on with `Detail` "held ON 3 times, 5 h in
total". Give a day or more of log for the full set of checks.

![Stuck on: a Presence detector held ON three times in one day](https://raw.githubusercontent.com/ShawnStrasser/atspm-detector/main/docs/images/health_stuck_on.png)

![Busy at night: a detector ON 62 % of the time from 1 to 5 am](https://raw.githubusercontent.com/ShawnStrasser/atspm-detector/main/docs/images/health_busy_at_night.png)

## Outputs

* **Findings**: one row per finding (`DeviceId`, `DetectorType`, `Detector`, `Start`, `End`, `Label`, `Confidence`,
  `Detail`). `summary=True` also gives a status per detector and day.
* **Time bins** (`incremental`, `live`, `health_bins`): one row per bin per problem with `Score`, how much of the
  bin was failing (about 1 for a complete failure). No row means no problem. Rows already released never change.
* **Dashboard ranges**: the plain average of `Score` over a period (sum of `Score` divided by the bins of the period).

| Average of `Score` | Good | Suspect | Bad |
|---|---|---|---|
| Vehicle detector, 3 days | under 0.2 % | to under 2 % | from 2 % |
| Vehicle detector, 7 days | under 0.1 % | to under 1 % | from 1 % |
| Ped button, 3 days | under 0.05 % | to under 0.5 % | from 0.5 % |
| Ped button, 7 days | under 0.02 % | to under 0.2 % | from 0.2 % |

## More

Embedding, incremental and every-run health, and the limits: [docs/embedding.md](https://github.com/ShawnStrasser/atspm-detector/blob/main/docs/embedding.md).
Changes: [CHANGELOG.md](https://github.com/ShawnStrasser/atspm-detector/blob/main/CHANGELOG.md).

## License

MIT
