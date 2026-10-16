# atspm-detector

Give it a signal controller's hi-res event log; for every vehicle detector it returns the **phase**, **function**,
**lanes**, **setback** from the stop bar and **health**. No cabinet print needed: the answers come from how each
detector behaves. Use it to **build** a signal's ATSPM detector configuration, or to **check** the one you have.

* **Tested on 115 signals it never trained on:** phase right for **98.0 %** of detectors, function **91.7 %**
  (30 minutes or more of log).
* **More accurate than hand-labled configuration:** phase wrong on 1.3 % of detectors, against 2.6 % by hand.
* **Fast and light:** one second per signal on CPU.

**Contents:** [Detector functions](#detector-functions) · [Install](#install) · [Usage](#usage) · [Input](#input) ·
[How much data](#how-much-data) · [How it works](#how-it-works) · [Accuracy](#accuracy) ·
[Detector health](#detector-health) · [License](#license)

## Detector functions

| Function | ATSPM | What it is |
|---|---|---|
| Presence | yes | Stop-bar presence zone. The presence zones used in training were about 20 ft long at the stop bar. |
| Advance | yes | Advance count zone, upstream. Single lane or spanning several lanes. |
| Count | yes | Stop-bar count zone. Usually set to pulse, one per lane. |
| Yellow_Red | yes | At the stop bar; may span several lanes. Configure it in the field with a 5 mph speed threshold. |
| Mid | no | Zone between the advance and the stop bar. A class of its own because it helps the model. |
| Bike | no | Bicycle detection zone. |
| Other | no | Everything else that drives operations but is not an ATSPM zone. |

## Install

```bash
pip install atspm-detector   # Python 3.10 or newer
```

## Usage

```python
import pandas as pd
from atspm_detector import predict

events = pd.read_csv("events.csv")   # or pd.read_parquet (needs pyarrow)
events.head()
```

```text
                               DeviceId                Timestamp  EventId  Parameter
0  d2d0de72-c8aa-45ca-ae1a-8fc626c5afc9  2024-12-03 12:00:00.000      150          7
1  d2d0de72-c8aa-45ca-ae1a-8fc626c5afc9  2024-12-03 12:00:00.000      320          0
2  d2d0de72-c8aa-45ca-ae1a-8fc626c5afc9  2024-12-03 12:00:00.000      318          3
3  d2d0de72-c8aa-45ca-ae1a-8fc626c5afc9  2024-12-03 12:00:00.000      316        120
4  d2d0de72-c8aa-45ca-ae1a-8fc626c5afc9  2024-12-03 12:00:00.600       13          6
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

From the bundled 30-minute sample; detector 6 had one actuation, too few to judge. One row per detector; main columns:

| Column | What |
|---|---|
| `phase_pred`, `phase_prob` | phase and its probability (`phase_2nd`, `phase_2nd_prob`: runner-up) |
| `function_pred`, `function_prob` | function (table above) and its probability |
| `lanes`, `phase_n_lanes` | lane(s) the detector covers, number of lanes on its phase |
| `distance_ft`, `night_speed_mph` | setback from the stop bar (ft); typical night speed on the phase |
| `health_status`, `health_reason` | ok / watch / suspect / bad / not_enough_data, and why |
| `status`, `review_flag` | "ok", or why an answer is missing or weak; `True` = worth a look |

Under 5 actuations: no answer (guess in `phase_guess`, `function_guess`). `return_phases=True` also returns a phase table.

**One signal or many.** `DeviceId` separates signals. Each is processed on its own, so its answers are the same alone
or in a batch. `predict` also takes a parquet or CSV path or glob, and `start` / `end` times. To build it into an
application, see [docs/embedding.md](https://github.com/ShawnStrasser/atspm-detector/blob/main/docs/embedding.md).

## Input

One row per event, standard Indiana hi-res format: `DeviceId` (any signal id), `Timestamp` (controller local time),
`EventId`, `Parameter`. Give it the whole log; unused event codes are dropped.

## How much data

| Sample length | Phase | Function | Notes |
|---|---|---|---|
| 5 minutes | 94.1 % | 86.4 % | works, but weaker; no lanes |
| 30 minutes | 97.7 % | 91.7 % | **recommended minimum**; lanes from here on |
| 1 hour | 98.0 % | 91.8 % | |
| 3 hours | 98.2 % | 92.9 % | best for phase and function (3 to 24 hours) |
| 24 hours | 98.4 % | 92.5 % | full health check |

On the 115 test signals. Health runs on any length; busy at night needs a full day, count drops, night misses and spiky
counts 3 hours; night speed needs midnight to 5 am.

## How it works

* **In:** detector on / off, signal colors and phase calls. The model never sees a phase or channel number; it scores
  each detector against every phase that turned green, from behavior alone.
* **Models:** gradient-boosted trees and a small neural network score each detector against each phase and function.
  A final step looks at the whole intersection to keep at most one detector of each ATSPM function per lane. Small
  models give lanes and setback. Health is checked last, by rules, and never changes an answer. All on the CPU with
  onnxruntime (8.5 MB of weights).

## Accuracy

Measured once, on the final model, on **115 locked signals** never used for training or tuning (30 minutes to 24 hours
of log). Phase truth: controller timing (21,341 detector samples, 113 signals). Function truth: configuration export
checked against cabinet prints (87 signals). Detectors with no actuations or no green on their phase are not scored.
All labels come from one agency; other agencies' conventions may differ.

* **Phase: 98.0 %** (95 % interval 97.0-98.9 %). Most errors are between through phases nearly always green together
  (2 and 6, 4 and 8).
* **Function: 91.7 %** (88.7-94.0 %), ATSPM errors only, excluding detectors whose label the model picked (print vs
  hand label). By true class: Advance 93.9 %, Presence 95.2 %, Count 96.6 %, Yellow_Red 89.5 %, Other 81.8 %.
* **Lanes:** lane count per phase exact for 82.5 % of phases, within one for 97.5 % (298 phases).
* **Setback:** median error 36 ft (59.9 % within 50 ft; Advance, 25 signals); 23.9 ft in cross-validation.
* **Night speed:** an approach speed, not the posted speed; two nights agree to a median of 1.5 mph.
* **Speed:** 0.8 s per signal for 30 minutes of log, 1.7 s for 24 hours; 250-400 MB of memory.

**Against hand labels** (signals not trained on): phase wrong on 1.3 % of 4,636 detectors (hand labels 2.6 %); function
off the cabinet print on about 6 % of 5,254 (hand labels 9.7 %). Where model and hand label differed, the timing backed
the model 104 times to 24, the print 398 to 177. Prints can be out of date.

**Health:** flagged 14 of 20 detectors a traffic engineer judged faulty, and 3 of the 12 judged fine.
Flagged (suspect or bad) per 100 detectors: 0.7 on 30 minutes, 0.9 on 3 hours, 4.4 on 24 hours. Not field-verified.

## Detector health

Each detector gets a status (ok, watch, suspect, bad, not enough data), a plain-words reason, the periods affected and
the categories, judged against what is normal for its type (function x lanes x sample length). Six checks:

1. **Stuck on**: held ON longer than its type allows, and not a queue. On 24 hours suspect from 5 to 15 minutes by
   type; bad from 10 minutes for a bike zone, 20 for Count and Yellow_Red, 30 for Advance, 35 for Mid, 60 for Presence
   and Other. Several detectors (not two inputs of one zone) held ON together, starting and ending within 5 seconds:
   one common failure, bad, with a note for the signal. A Count zone: also held ON in light traffic.
2. **Counts fall**: goes silent while its phase mates keep counting, or its hourly count drops against theirs (GEH 10
   suspect, 13 bad) while its own occupancy drops too.
3. **Misses vehicles at night**: far fewer actuations from 9 pm to 5 am than its phase mates suggest (GEH 8 suspect,
   10.5 bad). A night silence with fewer than 100 vehicles expected is suspect at most.
4. **Busy at night**: much busier from 1 to 5 am than its type, and busier than its phase mates (needs a full day).
5. **Too fast / too many**: ONs closer together than vehicles can follow (under 0.8 s per lane at the stop bar,
   0.6 s elsewhere; a zone spanning 2+ lanes keeps the 1-lane limit), or more vehicles than a lane can carry (busiest
   5 minutes over 2,100 per hour per lane suspect, 2,360 bad).
6. **Spiky counts**: bursts in 10-minute counts at least 3 times its own level of the surrounding 2 hours (and GEH 5 or
   more) that its phase mates do not share: 2 in a day suspect, 3 bad. Not checked under 200 vehicles a day; under 500,
   suspect at most unless a burst is very large.

Bike zones get no counts-fall or night check. **Bad** = a finding past its bad limit, two kinds of evidence, or a problem
lasting 4+ hours of the day (busy at night, silent 4 h); otherwise **suspect** (a presence or other zone that only
flickers evenly with traffic stays suspect). **Watch** = borderline, or no healthy phase mate to compare with. The limits
are in `weights/health/health_limits.json` as [suspect, bad] per check and type; under about 8 hours of data stuck on,
goes silent and too many keep the 1.0 limits. Flagged (suspect or bad) on 755 training signals: 4.4 % of detectors on 24
hours (2.3 % suspect, 2.1 % bad), 0.9 % on 3 hours, 0.7 % on 30 minutes. Not field-verified.

![Stuck on: a Presence detector held ON three times in one day](https://raw.githubusercontent.com/ShawnStrasser/atspm-detector/main/docs/images/health_stuck_on.png)

![Busy at night: a detector ON 62 % of the time from 1 to 5 am](https://raw.githubusercontent.com/ShawnStrasser/atspm-detector/main/docs/images/health_busy_at_night.png)

## License

MIT
