# 34 — Voyage (VD) numbering audit of the cabinet-print records (2026-09-24)
Question (user): old prints number detectors in Voyage "VD" order; every controller in the data is MaxTime. Were VD
numbers ever taken as channels? Code: `research/code/cabinet/vd_audit.py scan|audit|card|fix`; outputs in
`%DC_WORK%/cabinet/`: `vd_scan.csv`, `vd_audit.csv`, `card_health.parquet`, `card_channels.parquet`. Locked signals skipped.
## VD -> MaxTime is a fixed relabelling of the same physical slot
VD 1-8 = phases 1/3/5/7 pairs (VD1->1, 2->13, 3->7, 4->14, 5->15, 6->27, 7->21, 8->28), VD 9-28 = phases 2/4/6/8 in order
(VD9-13 -> 2-6, 14-18 -> 8-12, 19-23 -> 16-20, 24-28 -> 22-26), VD 29-40 = same number. Checked three ways:
* config descriptions "VD#n": **2,459 / 2,459** channels agree.
* VD labels drawn per slot on cabinet / layout sheets (332 front view, 332S input file, 336 slot table): **3,070 labels on
  105 prints, 0** off the slot SLOT_TO_DET predicts (332, 332S and the one 336 table all consistent).
* input-file entries where the two readings differ (5,233 with a printed phase marker): marker = timing phase of the
  SLOT_TO_DET channel **94.9 %**, of the Voyage-order channel 26.5 %; on VD-era prints alone 92-100 % vs 21-34 %; no print
  where the Voyage order wins. **The physical slot -> MaxTime table applies to Voyage-era cabinets; no open question.**
## Which prints use VD numbering (210 of 761 records; 197 finished)
cabinet template only (VD drawer labels, loops via input file) 93 · config descriptions "VD#n" only 61 · VD layout
tables / bubbles / "Voyage" legends 42 · no text layer, reader flagged `vd_remapped` 14. Bubble phase test (zone "pX/n":
does channel n or VD(n)'s channel carry phase p in the timing?) marks 13 prints as VD-numbered bubbles: 03043, 05046,
06004, 06005, 06030, 08037, 08093, 08117, 08130, 08154, 11011 (weak) and the unfinished **01022, 06031** (their scripted
zone labels are VD numbers: whoever reads them must remap). Legends "max time or voyage number" (radar) are ambiguous;
there the phase test says MT everywhere. Newer tables headed "Voyage Detector #" often hold MT numbers (03099, 03102).
## Per detector (3,241 on 197 finished VD prints; 2,631 labelled)
Derivation (first of): input-file slot 1,366 · slot named in reason 228 · MT table / xlsm 370 · VD number cited 90 ·
VD layout table 50 · zone bubble 288 · config description 170 · counts 6 · unstated 63.
Verdicts: **confirmed 1,674** (slot / VD / MT table consistent, timing = print phase, live, behaviour not failed) ·
**plausible 953** (consistent but dead, phase unknown, or tied by description/bubble only) · **suspect 4** · **wrong 0**.
* All 199 detectors citing "VD n" sit on VD2MT[n]; every scripted VD-bubble channel was rejected by its reader.
* 1,911 bubble-as-channel labels over all 761 records: MT reading supported by phase 1,586, both 249, neither 52, VD only
  24 — all 24 explained (reader already rejected the bubble, overlap phases, or an MT/xlsm table names the channel).
* Suspect: 08037 d24 (tied to zone 8A/24 by liveness; 8A/24 = VD24 = ch22, dead; config calls ch24 VD#26) -> flag
  `vd_suspect` (already low); 06024 d20, 2B132 d16/d17 (print vs timing phase, not VD-related; already low).
* 11011 checked by hand: VD reading would swap the NB lanes, but behaviour (RT-lane presence ch25 hold 0.21 vs ch23 0.79;
  ch24 upstream arrivals on red) supports the reader's MT reading — kept.
Fixes: 0 channels changed; 1 flag via cab_record (`sweep_changes.csv`, rule `vd_audit`; backup `backup_vd_audit/`).
## Card rule (cleansing only, never a model input)
One input-file slot = one card, upper + lower output. `card` pairs the channels per SLOT_TO_DET, runs dq_core health
(newest window) and marks `card_suspect` when both outputs are in use (on the print, live in Dec 2024, or a specific
config description) and both are dead, or both erratic (faults / stuck-on / chatter). Signals with every channel dead
are excluded. 735 signals, 8,980 two-output slots, 4,281 with both outputs in use: **card_suspect 302 slots (604
channels) on 143 signals** — both dead 272 (both on the print 157; 41 were live in Dec 2024 = died together since),
both erratic 30. Not yet merged into v3 / dq (the orchestrator owns the rebuild; join on DeviceName + detector).
