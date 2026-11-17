# 158 One fault reported once; ped recall never alerts (2026-11-10, folded into 2.1.0 before publishing)

User request (3 items): a held detector also reported silent (10075, 11-06..11-08); walks without presses firing on
recall newly turned on; a stuck button's sudden stop worded "lost data, not stuck".

## Vehicle: silent_not_held (judge/silent_held.sql, last step of the silence chain xfa -> xfs -> xfh -> xf)
* A detector's own quiet runs already skip held bins (quiet = < 1 s ON); the overlaps came from the phase-level silences
  that overwrite drop_b0 / b1 (whole phase dropped, and in principle signal silent).  Saved report findings Nov 4-9: 6-16
  stuck on x silent overlaps per window; network Nov 6-8: 11.
* Fix: a silence keeps its longest stretch of 5-min bins no hold of the detector touches (hold >= 5 min, hib Count zone
  held ON bins, mostly-ON runs); drop_lam / ph_exp scaled by its share, ph_obs recounted, nothing left = no silence.
* Network Nov 6-8: overlaps 11 -> 0; 6 detectors lose silent (04119 d2/4/5/9/10/11, all inside their holds), 01028 d16
  whole phase dropped cut to Sun 11:25-21:00; no status change.  The 10075 windows in the report (Nov 9, 10) had holds
  ending 22:01 and silence from 22:05: no overlap there.
* drop_not_held (user: "apply the same fix", double labels confuse): hiv (held_on.sql) = holds >= 15 min (stage 1's
  silence-in-a-hold length; 5-min holds took a normal presence queue hold and moved 08171 d15's drop point -> a new false
  drop), hib bins, mostly-ON runs; hheld = the clock hours they touch.  gh: such an hour is not covered for the detector
  (shares, drop point, GEH, night GEH) and flags hd; lvr and mhf (mate drop): no low hour, ends a run.  Basic detectors
  (no function): lvl0 leaves out the 15-min bins a hold >= 15 min touches.  silent_held now reads hiv + every reported
  stuck-on hold (efl, label <> queue: type limits under 15 min).
* Network Nov 6-8 (on5 -> on6): counts dropped over a hold 12 -> 2 (08019 d22: day-long drop touching two 12-min holds at
  the hour's edge), misses vehicles at night 8 -> 5 (whole-day rows, now judged without the held hours); basic detectors
  10 + 10 left (their share fell outside the held bins too; their night check unchanged).  7 detectors lose counts
  dropped, 1 night miss, 1 silent; 08019 d23 silent -> counts dropped (daytime, between its holds); 9 detector-days bad ->
  suspect, 11026 d3 ok -> watch (no healthy mate left: below the report's 0.5).

## Ped
* stop_not_held: ped_stop.stop_sql(held=) -- 15-min bins touched by a stuck on / long hold break the zero run and leave
  the level.  Synthetic: stop moved from 21:15 (inside a 6-h hold) to 03:00 (its end).  Network: no stop overlapped.
* walks_need_on: f_nop carries last_ev (the input's last 89/90 before the run's end, carry ONs included); not 'on' ->
  config note "P# walks every cycle with no presses: recall (inferred)" (findings label "recall (inferred)", Confidence
  0; the ATSPM report drops Confidence-0 rows), none when the phase already has the "on ped recall" note or the input is
  stuck.  The sudden stop of the same silence is still judged on its own.  Network: the one finding (04015 ped 2) has
  last_ev 'on' and stays ("its last button event is an ON with no OFF: stuck ON").
* Wording: _reason passes stuck=True to stop_reason when the input has a stuck-on finding and the stop's last_ev is 'on'.
Tests: test_health (held in a phase drop), test_ped_walks (recall note, last_ev, ON evidence, stop over a hold, wording);
walks tests on the old rule run with walks_need_on off.  pytest 260 (slow included), atspm-detector-check passes, no re-freeze.
