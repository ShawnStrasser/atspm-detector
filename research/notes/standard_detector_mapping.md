# Standard detector channel -> phase wiring (332 / 332S / 336 cabinets)

Transcribed 2026-09-17 from the table supplied by the user; 336 column added 2026-09-23 (the user
has less confidence in the 336 column; the 336 is less common and its input file is harder to read).
The channel -> phase default is the same for all three cabinet types; they differ in the input-file
slot. Only the 332S has channels 29-40; the 336 has only channels 1-4, 7-10, 13-18, 21-24, 27-28.

Slot notation: `I`/`J` = input file, number = slot, `U`/`L` = upper/lower. On the print's INPUT FILE
drawing a slot's upper input is the `D`/`E` terminals and its lower input the `J`/`K` terminals
(e.g. `I1-D`/`I1-E` = I1U, `I1-J`/`I1-K` = I1L).

| Det | Phase | 332 slot | 332S slot | 336 slot | | Det | Phase | 332 slot | 332S slot | 336 slot |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | I1U | I1U | I1U | | 21 | 7 | J5U | J6U | I7U |
| 2 | 2 | I2U | I3U | I2U | | 22 | 8 | J6U | J8U | I8U |
| 3 | 2 | I2L | I3L | I2L | | 23 | 8 | J6L | J8L | I8L |
| 4 | 2 | I3U | I4U | I9U | | 24 | 8 | J7U | J9U | I10L |
| 5 | 2 | I3L | I4L | - | | 25 | 8 | J7L | J9L | - |
| 6 | 2 | I4U | I5U | - | | 26 | 8 | J8U | J10U | - |
| 7 | 3 | I5U | I6U | I3U | | 27 | 5 | J9U | J1L | I5L |
| 8 | 4 | I6U | I8U | I4U | | 28 | 7 | J9L | J6L | I7L |
| 9 | 4 | I6L | I8L | I4L | | 29 | 1 | - | I2U | - |
| 10 | 4 | I7U | I9U | I10U | | 30 | 1 | - | I2L | - |
| 11 | 4 | I7L | I9L | - | | 31 | 2 | - | I5L | - |
| 12 | 4 | I8U | I10U | - | | 32 | 3 | - | I7U | - |
| 13 | 1 | I9U | I1L | I1L | | 33 | 3 | - | I7L | - |
| 14 | 3 | I9L | I6L | I3L | | 34 | 4 | - | I10L | - |
| 15 | 5 | J1U | J1U | I5U | | 35 | 5 | - | J2U | - |
| 16 | 6 | J2U | J3U | I6U | | 36 | 5 | - | J2L | - |
| 17 | 6 | J2L | J3L | I6L | | 37 | 6 | - | J5L | - |
| 18 | 6 | J3U | J4U | I9L | | 38 | 7 | - | J7U | - |
| 19 | 6 | J3L | J4L | - | | 39 | 7 | - | J7L | - |
| 20 | 6 | J4U | J5U | - | | 40 | 8 | - | J10L | - |

As a lookup table alone (332/332S) this matches 92.6% of training labels (91.6% statewide); 273 of
418 signals match 100%.

```python
DEFAULT_PHASE = dict(zip(range(1, 41),
    [1,2,2,2,2,2,3,4,4,4,4,4,1,3,5,6,6,6,6,6,7,8,8,8,8,8,5,7,1,1,2,3,3,4,5,5,6,7,7,8]))
SLOT_TO_DET = {   # cabinet type -> {slot: detector channel}
    "332":  {"I1U":1,"I2U":2,"I2L":3,"I3U":4,"I3L":5,"I4U":6,"I5U":7,"I6U":8,"I6L":9,"I7U":10,"I7L":11,
             "I8U":12,"I9U":13,"I9L":14,"J1U":15,"J2U":16,"J2L":17,"J3U":18,"J3L":19,"J4U":20,"J5U":21,
             "J6U":22,"J6L":23,"J7U":24,"J7L":25,"J8U":26,"J9U":27,"J9L":28},
    "332S": {"I1U":1,"I3U":2,"I3L":3,"I4U":4,"I4L":5,"I5U":6,"I6U":7,"I8U":8,"I8L":9,"I9U":10,"I9L":11,
             "I10U":12,"I1L":13,"I6L":14,"J1U":15,"J3U":16,"J3L":17,"J4U":18,"J4L":19,"J5U":20,"J6U":21,
             "J8U":22,"J8L":23,"J9U":24,"J9L":25,"J10U":26,"J1L":27,"J6L":28,"I2U":29,"I2L":30,"I5L":31,
             "I7U":32,"I7L":33,"I10L":34,"J2U":35,"J2L":36,"J5L":37,"J7U":38,"J7L":39,"J10L":40},
    "336":  {"I1U":1,"I2U":2,"I2L":3,"I9U":4,"I3U":7,"I4U":8,"I4L":9,"I10U":10,"I1L":13,"I3L":14,
             "I5U":15,"I6U":16,"I6L":17,"I9L":18,"I7U":21,"I8U":22,"I8L":23,"I10L":24,"I5L":27,"I7L":28},
}
```
