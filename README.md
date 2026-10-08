# atspm-detector

Detector configuration from the hi-res log. Give it a traffic signal controller's high-resolution event log and, for
every vehicle detector, it tells you the **phase** the detector calls, its **function** (Advance, Presence, Count,
Yellow-Red or other), the **lanes** it covers, its **setback** from the stop bar and its **health**. No
channel-to-phase table or wiring sheet is needed: the answers come from how each detector behaves. Use it to
**build** the ATSPM detector configuration for a signal, or to **check** the one you already have.

* **Tested on 115 signals it never trained on:** phase right for **98.0 %** of detectors, function (ATSPM classes)
  **91.7 %**, with 30 minutes or more of log.
* **More accurate than hand-kept configuration:** against the controllers' own timing, the model got the phase wrong
  on 1.3 % of detectors and the hand configuration on 2.6 %. Against cabinet prints, the model's function disagreed
  on about 6 % of detectors, the hand labels on 9.7 %.
* **Fast and light:** runs on an ordinary CPU (no GPU), about a second per signal.

## Contents

* [Install](#install)
* [Usage](#usage)
* [Input](#input)
* [How much data](#how-much-data)
* [How it works](#how-it-works)
* [Accuracy](#accuracy)
* [Detector health](#detector-health)
* [License](#license)

## Install

```bash
pip install atspm-detector
```

Python 3.10 or newer. Dependencies: numpy, pandas, DuckDB and onnxruntime.

## Usage

```python
import duckdb
from atspm_detector import predict

events = duckdb.read_parquet("events.parquet").df()   # or pd.read_parquet / pd.read_csv
events.head()
```

```text
                               DeviceId               Timestamp  EventId  Parameter
0  d2d0de72-c8aa-45ca-ae1a-8fc626c5afc9 2024-12-03 12:00:00.000      150          7
1  d2d0de72-c8aa-45ca-ae1a-8fc626c5afc9 2024-12-03 12:00:00.000      320          0
2  d2d0de72-c8aa-45ca-ae1a-8fc626c5afc9 2024-12-03 12:00:00.000      318          3
3  d2d0de72-c8aa-45ca-ae1a-8fc626c5afc9 2024-12-03 12:00:00.000      316        120
4  d2d0de72-c8aa-45ca-ae1a-8fc626c5afc9 2024-12-03 12:00:00.600       13          6
```

```python
results = predict(events)
results[["Detector", "phase_pred", "phase_prob", "function_pred", "function_prob",
         "lanes", "distance_ft", "health_status"]].round(2).head()
```

```text
   Detector  phase_pred  phase_prob function_pred  function_prob lanes  distance_ft    health_status
0         2           2        0.99       Advance           0.90     2        164.0               ok
1         3           2        1.00       Advance           0.64   1,2        153.0               ok
2         4           2        1.00      Presence           0.40   1,2          0.0               ok
3         5           2        1.00       Advance           0.53     1        152.0               ok
4         6        <NA>         NaN           NaN            NaN   NaN          NaN  not_enough_data
```

(Output from the 30-minute sample log that ships with the package. Detector 6 had one actuation, too few to judge.)

`results` has one row per detector. The main columns:

| Column | What |
|---|---|
| `DeviceId`, `Detector` | signal and detector channel |
| `phase_pred`, `phase_prob` | phase and its probability (`phase_2nd`, `phase_2nd_prob`: runner-up) |
| `function_pred`, `function_prob` | Advance, Presence, Count, Yellow_Red, Mid, Bike or Other, and its probability |
| `lanes`, `phase_n_lanes` | lane(s) the detector covers, number of lanes on its phase |
| `distance_ft` | setback from the stop bar, in feet |
| `night_speed_mph` | typical night speed on the phase (blank without night data) |
| `health_status`, `health_reason` | ok / watch / suspect / bad / not_enough_data, with a plain-English reason |
| `status`, `review_flag` | "ok", or why an answer is missing or weak; `True` = worth a look |
| `n_actuations`, `minutes_of_data` | the evidence behind the answer |

A detector with fewer than 5 actuations gets no answer (the model's raw opinion stays in `phase_guess` and
`function_guess`). `predict(events, return_phases=True)` also returns one row per phase (lanes, volumes, night speed).

**One signal or many.** One call can take one signal or many: `DeviceId` separates them. Each signal is processed on
its own, with its own time window, so its answers are identical whether it is sent alone or in a batch. `predict` also
takes a parquet or CSV path or glob, and `start` / `end` times; for building it into an application see
[docs/embedding.md](https://github.com/ShawnStrasser/atspm-detector/blob/main/docs/embedding.md).

## Input

One row per event, in the standard Indiana hi-res format. Give it the whole log: event codes it does not use are
filtered out.

| Column | What |
|---|---|
| `DeviceId` | signal id (any text or number) |
| `Timestamp` | controller local time |
| `EventId` | Indiana hi-res event code |
| `Parameter` | |

## How much data

| Sample length | Phase | Function | Notes |
|---|---|---|---|
| 5 minutes | 94.1 % | 86.4 % | works, but weaker; no lanes |
| 30 minutes | 97.7 % | 91.7 % | **recommended minimum**; lanes from here on |
| 1 hour | 98.0 % | 91.8 % | |
| 3 hours | 98.2 % | 92.9 % | best for phase and function (3 to 24 hours) |
| 24 hours | 98.4 % | 92.5 % | full health check |

Detector health runs on any length, but its time-of-day checks (busy at night, unusual daily pattern) need a full
calendar day, and night speed needs data between midnight and 5 am. Figures are from the 115-signal test under
[Accuracy](#accuracy).

## How it works

* **In:** detector on / off events, signal colours and phase calls from the log. The model never sees a phase or
  channel number: it judges each detector against every phase that turned green, from behaviour alone (when it turns
  on relative to green and red, how long it stays on, how it moves with the other detectors).
* **Models:** gradient-boosted trees and a small neural network score every detector against every phase and every
  function. A final step looks at the whole intersection at once and keeps at most one detector of each ATSPM
  function per lane. Small further models give lanes and setback; night speed comes from the setback and the travel
  time of night-time vehicles. Health is checked last, by rules, and never changes a phase or function answer.
* **Out:** one row per detector, as above. Everything runs on the CPU with onnxruntime (8.5 MB of weights).

## Accuracy

Measured on **115 locked signals** never used for training or tuning, scored once after the model was final, with
30 minutes to 24 hours of log. Phase truth is the controllers' own timing (21,341 detector samples at 113
signals). Function truth is the configuration export checked against cabinet prints (87 signals). Detectors with no
actuations, or whose phase never turned green in the sample, are not scored.

**Phase: 98.0 %** of detectors right (95 % interval 97.0-98.9 %). The main remaining error is between through phases
that are green together almost all the time (2 and 6, 4 and 8).

**Function: 91.7 %** (88.7-94.0 %). The score counts ATSPM errors: an ATSPM detector given the wrong ATSPM class or
called non-ATSPM, or a non-ATSPM detector called ATSPM. Where the print and the hand label disagreed and the model
picked between them, those detectors are left out of this figure. By true class:

| Advance | Presence | Count | Yellow-Red | Other (non-ATSPM) |
|---|---|---|---|---|
| 93.9 % | 95.2 % | 96.6 % | 89.5 % | 81.8 % |

* **Lanes:** the number of lanes on a phase is exactly right for 82.5 % of phases and within one lane for 97.5 %
  (298 phases on the locked signals' prints).
* **Setback:** median error 36 ft on the locked signals (59.9 % within 50 ft; Advance detectors, 25 signals);
  23.9 ft in cross-validation.
* **Night speed:** an approach speed, not the posted speed. Two different nights agree to a median of 1.5 mph.
* **Speed:** 0.8 s per signal for 30 minutes of log and 1.7 s for 24 hours, about 250-400 MB of memory.

**Against hand labels** (signals the model did not train on): on phase, checked against the timing, the model was
wrong on 1.3 % of 4,636 detectors and the hand configuration on 2.6 %; where they differed, the timing backed the
model 104 times and the hand label 24 times. On function, checked against cabinet prints, the model disagreed with
the print on about 6 % of 5,254 detectors and the hand labels on 9.7 %; where model and hand label differed, the
print backed the model 398 times and the hand label 177 times. Prints can be out of date, so read this as "agrees
with the prints more often", not as field truth.

**Detector health:** in a review by a traffic engineer, the check flagged 19 of the 20 detectors judged
faulty and none of the 12 judged fine. It flags (suspect or bad) 0.4 detectors per 100 on 30-minute
samples, 1.2 on 3 hours and 3.9 on 24 hours. There is no field-verified accuracy figure for health.

All labels come from one agency's network; other agencies' conventions may differ.

## Detector health

Each detector gets a status (ok, watch, suspect, bad, or not_enough_data), a reason in plain words, the periods
affected and the categories found. It is judged against what is normal for its type (function x lanes covered x
sample length), never against one fixed limit. The checks: stuck on, goes silent, count drops, misses vehicles at
night, erratic counts, too-fast actuations, chattering, too many for the traffic, erratic time ON, busy at night, unusual daily
pattern and Count zone held ON, each with limits set from healthy detectors of the same type. A detector is
**bad** rather than suspect when one finding is strong, when two independent checks agree, or when a problem lasts
4 hours or more of the day.

![Stuck on: a Presence detector held ON three times in one day](https://raw.githubusercontent.com/ShawnStrasser/atspm-detector/main/docs/images/health_stuck_on.png)

![Busy at night: a detector ON 62 % of the time from 1 to 5 am](https://raw.githubusercontent.com/ShawnStrasser/atspm-detector/main/docs/images/health_busy_at_night.png)

![Unusual daily pattern: an advance loop goes quiet at midday while its partner keeps counting](https://raw.githubusercontent.com/ShawnStrasser/atspm-detector/main/docs/images/health_unusual_day.png)

## License

MIT
