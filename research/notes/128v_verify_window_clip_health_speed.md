# 128v Adversarial check of final_v7_next (note 128) - all PASS (2026-10-07)
Copies of both src trees (dc_work/s128v/pkg) so prod was never imported in place; prod md5 of all 100 files
before = after. Scripts research/code/final128v/, outputs dc_work/s128v/. CPU, 4 threads, locked_v2 asserted absent.

**New signals**: 10 non-locked pool signals not in any earlier parity / bench / robustness / clip set (6-32 detector
channels, seed 1281), Sun 2026-09-20 06:00 -> 09-21 06:00; 30 min (16:30-17:00), 3 h (13-16), 24 h.

1. Parity prod vs next, default settings: 50 calls (exact window + no window), 42 columns, 893 detector rows, 268
   phase rows, warnings: all identical (DataFrame.equals), incl. every health and status column. Same 50 under
   pandas 3.0.6 (fresh venv, both packages): identical. PASS.
2. Far bounds (next): 90 calls (start 1 h / 365 d early, 2000 -> 2030) = prod with the far bound left out, 90/90
   identical. Extra: window wholly before / after data -> 0 rows (as prod); start > end -> 0 rows; far start + mid
   end = end only; mid start + far end = start only; tz-aware df + far UTC start = no start; two signals in one call
   (one starts 1.5 h late) = each alone, joint and chunked. Start 29 min early kept, 31 min clipped (designed step;
   end 29 min late changes health text vs 31 min, as prod would). PASS.
3. Far-start speed: 30 min, start 365 d early 33.5 s -> 0.52 s; 3 h 48.2 s -> 1.10 s; 30 min 2000 -> 2030:
   prod still running at 11.5 min (stopped) -> 0.40 s. PASS.
4. Status text: min_prob 0.99, phase prob 0.98974: "0.99 (need >= 0.99)" -> "0.989 (need >= 0.99)"; status the
   only changed column. 9.998-min sample: "only 9.9 min of data". Cosmetic: minutes_of_data column still shows 10.0
   next to "only 9.9 min". PASS (minor note).
5. Health stage alone (_health_events + assess inside predict, warm, best of 2 interleaved runs x median of 5;
   load 2 %), prod -> next s: 30 min .130/.130/.158 -> .047/.046/.058; 3 h .136/.193/.307 -> .050/.071/.104;
   24 h .200/.214/.592 -> .078/.086/.214 (x01 8 ch / x05 16 ch / x09 32 ch). 2.6-3.0x faster everywhere; 3 h
   busiest 0.104 s (target 0.1 met on average only, as note 128 says). Whole call 3 h .84 -> .71 s mean. PASS.
6. Fresh venv from the rebuilt wheel (pandas 3.0.6, duckdb 1.5.6, ort 1.30, no pyarrow): wheel 69 files = src
   byte for byte; check 7/7, no warnings except the expected FutureWarning test line; CLI with a far start runs.
   PASS.
7. Robustness battery (124v robust_a/b/c/d, run on prod and next copies, logs diffed): every result identical
   prod vs next - tz (Indiana, UTC, fixed offset, csv offsets, no-zone parquet), bad rows (df + csv), apostrophe
   paths / ids / unicode, batching + chunking + glob, channel 0, >64, stuck ON (2 kinds), empty df / file, one
   detector, no begin-green, no 43/44, disallowed + fault codes added (no change), 90-min hole, start > end.
   7 days 1 thread (n08, 2.0 M rows): 9.1 s, peak 882 MB < 1 GB, answers = prod. PASS.
   (robust_c copy: syntax error in its unused health_scan fixed by skipping it; it read old s124v outputs.)

Not re-run: the research-scorer 4,180 check and the 132-case set (note 128); items 1, 2 and 7 cover new data.
Verdict: final_v7_next ready to swap in after the exam. Nothing fixed, nothing in final_v7_next / _prod changed.
