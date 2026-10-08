# Research status — the single source of truth for whoever works next

Rules and the ordered plan live in `AGENTS.md`. This file holds the STATE: what is done, what is
running, what is next, what the user still has to decide. **Every agent updates this file at the end
of every step** (and before handing back), so the next session can pick up without the chat history.
Keep it short; put detail in `research/notes/`.

_Last updated: 2026-10-07 by the note-113 subagent (behaviour-only lead neighbours replace channel adjacency; PACKAGE v6b built); before that 2026-10-07 by the note-112 subagent (decoder without channel adjacency; analysis only, nothing built); before that 2026-10-06 by the note-109 subagent (simpler architecture: one network, single-seed trees; analysis only); before that 2026-10-06 by the note-110 subagent (health check artefact fixes v110 + review sheet rebuilt, 36 rows; analysis only); before that 2026-10-06 by the note-108b subagent (health review v3 rebuilt: charts match explanations + profile on partial samples; sheet + analysis only); before that 2026-10-06 by the note-108 subagent (health baseline of normal per type -> per-type limits + time-of-day profile + spot-check sheet; analysis only); before that 2026-10-06 by the note-104b subagent (health review v3 re-laid in a short form; sheet + charts only); before that 2026-10-06 by the note-106 subagent (vehicle paths / lanes for function: path features, function<->lanes iteration, joint decode; CPU only); before that 2026-10-06 by the note-107 subagent (arrow-rule re-check of all multi-lane print phases -> lane truth v3; labels only); before that 2026-10-06 by the note-105 subagent (lane truth from user answers + lanes refit + package v5c; CPU only); before that 2026-10-06 by the note-104 subagent (health review v2 answers -> resolver v104 + review sheet v3; CPU only); before that 2026-10-06 by the note-102 subagent (wide-right-lane fix attempt + lane review v2; CPU only); before that 2026-10-06 by the note-100/100b/100c subagent (siba distillation / small-student screen, fold 0; RunPod dc95a, outputs copied back); before that 2026-10-05 by the note-98 subagent (speed vs beta, fast variants, package v5_fast; CPU only); before that 2026-10-05 by the note-96 subagent (context-aware health study + proposal + review sheet v2; analysis only); before that 2026-10-05 by the note-93 subagent (health vs accuracy; YR without Count; analysis only); before that 2026-10-05 by the note-94 subagent (user review v1 answers + rulings of Oct 5 -> labels v4p, locked key v2; labels only); before that 2026-10-05 by the note-90 subagent (FINAL package v4f); before that 2026-10-05 by the note-86 subagent; before that 2026-10-04 by the note-89 subagent (exclusion groups decided by data; final labels v4o)._

## ORCHESTRATOR RESUME POINT (read first; keep current after every step)
SHIPPED LOCALLY 2026-10-08 15:30: commit dffdad7 (atspm-detector 1.0.0, model/ removed, tag beta-final_v2), remote = ShawnStrasser/atspm-detector. NOT PUSHED: user said hold until the ped health review is done. After push: release v1.0.0 (gh, if user says), LinkedIn. paper/ never committed.
RELEASE READY 2026-10-08 05:15: atspm-detector 1.0.0 = dc_work/final_v7_prod (still detector_classifier internally; sync_from_prod.py renames) staged in dc_work/ship_stage (apply_ship.py --apply on "ship"; deletes model/). Locked exam done (note 119). User to-do + open question (4-h health rule) in USER_INPUT. Running: scrub of user id / local paths in research/ before the user commits. LinkedIn post + carousel in review/linkedin (post only after PyPI is live). Paper local in paper/.
USER DECISIONS 2026-10-08 ~03:40: package name atspm-detector (import atspm_detector), VERSION 1.0.0 (not 7.0.0); user will rename the GitHub repo (assume ShawnStrasser/atspm-detector, confirm); REMOVE the beta (model/) from the repo entirely and every mention of a beta in package / README (say "the final model"); the LinkedIn post MAY compare with the beta to show gains. Outputs must work with the user's atspm package (PyPI atspm 2.4.0, a local checkout of ATSPM_Aggregation_package) -> exporter (note 131). Running: wf_bd1045cc-66f (128 speed, 129 ped 2026) and wf_4481a95f-989 (130 hand-label evidence, 131 atspm compat, LinkedIn post redo). Next: final package = final_v7_next + atspm exporter + version 1.0.0 + pandas warning fix (check.py:250) -> swap -> ship_stage sync with beta removal (delete model/, repoint research rpath, AGENTS/README/REPORT without beta) -> verify.
UPDATE 2026-10-08 01:30: done 120 (fast switch gone), 121, 122 (nothing added), 123/123b, 124/124v (health v4d in final_v7_prod), 125-127. RUNNING: exam agent (119, on final_v7_prod - do not modify it until exam finishes) and wf_bd1045cc-66f (128 far-start + health speed in COPY final_v7_next -> swap after exam; 129 ped on 2026 data + fresh ped sheet). After both: swap final_v7_next -> final_v7_prod, ship_stage sync_from_prod.py, paper exam + health numbers, results page exam, USER_INPUT (exam, health v4 in package, 4-h persistence rule question, ped sheet, PyPI to-do, paper to-do).
AUTONOMOUS RUN (from 2026-10-07 23:30, user away until morning; replies via USER_INPUT). Running: wf_018e54eb-5df (119 locked exam AUTHORISED by user Oct 7 + verify; 120 remove fast switch (user decision); 121 health v4 refinements from user's v4 comments, cautious severity rule; 122 red/green deeper; 123 ped button health on Dec 2024 data + 123b sheet; 124 adopt health v4 into final_v7_prod; verify) and wf_90e75eb5-e06 (125 pull Sept 2026 ped events 21/22/23/45/89/90 - user asked; 126 stage PyPI publishing in dc_work/ship_stage mirroring ATC-Signal-Replay, README manual-only + docs/embedding.md; 127 TRB paper draft in user's voice, local only in paper/ excluded via .git/info/exclude). Then: 125+123 -> ped study on 2026 data; fill exam numbers into paper; sync final package into ship_stage; update results page with exam; USER_INPUT summary + user to-do (PyPI trusted publisher etc.). Ship (replace model/) and commit remain the user's call.
ROUND 2 DONE (2026-10-07 ~23:00, wf_56bdd955-499): PRODUCTION PACKAGE dc_work/final_v7_prod (detector_classifier 7.0.0 wheel) fixed + re-verified, all PASS, ready for ship decision (swap into model/ only on user "ship"). Health v4c sheet review/health_review_v4.xlsx (40 rows, 15 categories, one chart per category) released to user; open for user via sheet: 4 user-bad cases now suspect (re-trigger rule), row 11 night expectation. Results page updated with v7 speed. Next: wait for user answers on health v4; after that, adopt health v4 into the package.
NOTES 114-118 (2026-10-07 18:30-21:00, CPU, workflow wf_beccea06-ff7): 114 pk_unhealthy removed (no loss) -> health fully out of classification. 115 PRODUCTION PACKAGE dc_work/final_v7_prod = pip library detector_classifier 7.0.0 (v6b + 114 + vectorisation fixes; decisions identical; check 6/6; bench 115c: v7 full 3 h .89 s / 263 MB, le2h .66 / 246, 30 min .58 / 242 vs beta 1.19 / 1076). 115v found defects -> round 2 workflow wf_56bdd955-499 (115d package fixes, 118c health fixes, re-verify, repair) RUNNING. Health v4 (116 ToD night-level chosen over full-profile outlier methods, 117 red/green, 118a/b) sheet NOT released until 118c verified. Results page rebuilt (architecture merged, old architecture artifact deleted). User decisions Oct 7: production = build, swap into model/ only on "ship"; runtime = Python library + small edge device; twin input removed. Next: when round 2 lands -> update results page speed (v7), USER_INPUT, release health_review_v4.
NOTE 113 DONE (2026-10-07 ~10:30, CPU only): user decision (Oct 7) adjacency must go -> CANDIDATE PACKAGE dc_work/final_v3_candidate_v6b = v6 with decode_v3 + decode_trees refit (full data) on LEAD neighbours (lag_*: detectors that follow one another within 1-8 s more than the green state explains; similarity.build_lead_window) instead of adj_*; decode.py has no channel arithmetic; dead yr_* block removed; docstrings / CLI help / blend.json / model_card top block fixed (default profile full). OOF 3 seeds vs v6: >= 30 E +0.00 [-0.08,+0.09] (.9846), 5 min -0.42* (.9634); vs noadj +0.20* / +0.22*; fast decode_trees vs v6 +0.32* (.9811), 5 min -0.31*. Shuffled control = noadj. co-timing (same second), 1-s phi, 2nd-order phi = noadj (dropped). New check.py check 3b (3 random channel bijections onto 1..64 x phase renumberings + 10 min + no-network): FAILS on v6 (adjacency; decoded prob up to 0.17; ranker+net clean), passes v6b (2.7e-8) -> no other channel / order leak on the sample. check 6/6 after --freeze; sample answers = v6 (17/17). Bench 3 h v6 vs v6b tie (machine shared, paired only). Function not re-scored (pred_phase held fixed in its OOF; phase unchanged at >= 30). Rebuild: final113 pairs113 -> f113 oof/shuf/score/fit -> assemble113 weights -> check --freeze -> assemble113 card. model/ = final_v2 still. ADOPTED by orchestrator 2026-10-07 16:45 (user required adjacency removed): v6b = THE candidate. USER_INPUT + both artifacts updated. Next: wait for user health_review_v3 answers.
NOTE 112 DONE (2026-10-07 ~08:40, CPU only, analysis): v6 decoders WITHOUT the 5 channel-adjacency features (adj_*, |channel difference| <= 2; both decode_v3 and decode_trees have them), fit111 recipe, 3 seeds: phase >= 30 E -0.19 [-0.26,-0.12]* (.9827 vs .9846; R -0.20*), 5 min -0.64*; fast decode_trees -0.57* (5 min -1.55*); seed spread <= 0.01 pt -> over the 0.1-pt limit -> v6b NOT built, v6 stays the candidate. yr_* dead block = 0.016-0.027 s per 3-h call (1.7-2.2 %), safe to remove in any later package. Open (orchestrator/user): keep the relative-channel adjacency (state it plainly: reads channel DIFFERENCES, not numbers) or accept -0.2 pt; untested: behaviour-only neighbour substitute. Code final112/, work s112/, note 112.
NOTE 111 DONE (2026-10-06 ~21:10, CPU only): CANDIDATE = PACKAGE dc_work/final_v3_candidate_v6 (replaces v5c as candidate; not shipped, model/ = final_v2) = note-109 design C: phase TCN removed, the 3 w32 siba members' pair phase head is the phase network (one shared pass with function); ranker / function trees / stackers (mean3 / single / nonet) / setback P50 at ONE seed; decoder refit on ranker-s0 0.5 / siba head 0.5; decode_trees (fast profiles) on ranker s0 alone; lanes / health = v5c. 21 trained models vs v5c 36. Parity siba phase head vs torch PASS (max 9.5e-6, 0 winners differ; s111/siba_phase_parity111.json); check.py 5/5 after --freeze; card final_v6_note111 (s111/card111.json incl. bench). OOF >= 30 E vs v5c: phase .9846 (+0.03 n.s.; 5 min +0.36*), function .9307 (-0.01 n.s.; 5 min -0.16*); le2h vs v5c le2h phase +0.04, function -0.08* (score111.json). Bench 3 h warm r8/r11/ev/ch: beta .45/.39/.56/.52 s (259-299 MB), v5c 1.34/1.01/1.60/1.78 (427-581), v6 .99/.76/1.28/1.41 (310-414; 1.95-2.71x beta), v6 le2h .68/.56/.93/.89 (249-279; 1.44-1.71x beta) (f98/bench summary_c111_*). Rebuild: fit111 dec/dectrees/stack/oofnonet -> assemble111 init/weights/card -> check.py --freeze/check. Default profile decided by orchestrator (2026-10-07): full by default, le2h = edge switch (both in package; per full-and-light rule, no user question). Note-109 chain_nf result: ranker-free A2 phase on unfiltered siba ge30 E .9848 (+0.05 n.s. vs v5c, = v6) and slower (no tree filter, +0.3-1.1 s at 3 h) -> dropped, v6 stands. USER_INPUT + results artifact updated with v6 (Oct 7 3:52 am). Next: wait for user health_review_v3 answers.
NOTE 109 DONE (2026-10-06 ~19:00, CPU + local A1000 inference only; simpler architecture): the siba w32 net's own phase head replaces the phase TCN (trees+siba head -> decoder: phase >= 30 E +0.04 n.s., 5 min +0.38*); tree / stacker / ranker seeds 3 -> 1 free (function -0.01, phase +0.03); one net member costs function -0.12..-0.17 (5 min -0.4..-0.6*); no decoder -0.23*; lane rule keep (5 min -0.17* without); setback 1 seed per group = tie. Bench (copy s109/pkg): no phase TCN -27..-30 % at 3 h, + 1 member -41..-45 %. Recommend design C (1 net kind x3 + 1-seed LightGBMs: 36 -> 21 models, ties v5c). Open: ranker-free phase A2 on unfiltered siba (s109/chain_nf.sh still running at hand-back; results -> s109/phase/score2-3.log). Code final109/, note 109. Nothing shipped.
NOTE 110 DONE (2026-10-06 ~19:00, CPU only): resolver v110 (research/code/health/h110_resolve.py; work dc_work/health110) = v108 + six fixes from the 108b charts: F1 erratic yardstick = same-function mates, else 'no yardstick' watch (never the mixed whole signal); F2 count drops time-of-day aware (type hourly share) AND own counts must halve; F3 stuck = total time / number of long ONs per type; F4 queue per ON, > 60 min cleared when healthy mates are queued (>= 75 % of it above their type's p95 % ON), and (orchestrator) every queue clear needs the phase's Advance/Count still counting >= 0.5x; F5 function p < .7 -> least strict plausible type's limit, said in the row; F6 profile finding from 12 h, 3-12 h watch. v108 reproduced exactly with fixes off. Suspect+bad per 100 v108 -> v110: 30 min .588 -> .554, 3 h 1.483 -> 1.404 (watch +1.0, F6, kept watch-only by decision), 24 h 3.958 -> 3.796. User rows: 3 and 12 -> bad (several long ONs), 7 -> ok, 9 -> suspect (Q1b), 5 -> ok. review/health_review_v3.xlsx REBUILT, 36 rows (30 + 6 fix examples; backup review/_backup/health_review_v3_20261006_175608.xlsx; 12/12 user entries carried; 'Our reply' updated rows 3, 5, 6, 7, 9, 12; 127/127 recomputed = resolver; 36 PNGs opened; QA dc_work/health110/qa_v110.csv). Row 34 (whole phase ON, zero counts 2 h) now not a queue -> suspect. Nothing adopted; health rules still WAIT on the user's answers to this sheet.
NOTE 108b DONE (2026-10-06 ~18:00, CPU only): review/health_review_v3.xlsx + charts REBUILT (same 30 rows; code research/code/health/h108c_review.py; backup review/_backup/health_review_v3_20261006_172751.xlsx; user's 12 entries carried cell by cell; new 'Our reply' column rows 1-9, 11, 12). Every 'Why' number re-computed from events with health_core (91/91 = resolver) and drawn on its chart (evidence panel per finding); 30/30 PNGs opened, QA log dc_work/health108/qa_v3.csv. Charts expose check artefacts (orchestrator to decide, nothing changed): erratic falls back to the whole signal when mates are twins (row 6); count drop uses the whole signal so night-busy detectors 'drop' at dawn (rows 7, 26); stuck scores only the longest ON (row 3); Q1 never clears > 60 min (row 9, looks like congestion); function error -> type limit fires (row 20). Profile on partial samples (held out): recall 3 h 26 % / 6 h 36 % / 12 h 60 % per window, healthy flagged 1.4 % at all lengths -> usable from 12 h, 3-6 h watch at most. Health rules still WAIT on the user's answers to this sheet (rows 9-30 unanswered).
NOTE 108 DONE (2026-10-06 ~15:45, CPU only, analysis; user: find what NORMAL is per type before setting limits): types = model function x lane span (+ volume band; Count pulse/normal measured from clean ONs; NO 'long zone' type anywhere). Baseline per type (dc_work/health108/pct_table.csv, plots/type_*.png): 2+ lanes ON->ON < 1 s p99.5 1.5-2.6x 1-lane (chatter not); volume p99.5 33-145 vs old 150; queue pattern (time ON up, counts down at high traffic) measured in 2.8-15.5 % of Presence/Other samples. Resolver v108 (research/code/health/h108_resolve.py, proposal: per-type limits p99.8, profile p99.5): Q1 queue only with healthy mates (R1b dropped), Q2 like-with-like (time ON vs healthy same-class mates; replaces R2/R3), Y no-yardstick -> watch, D1 missing-OFF data note, time-of-day profile REPLACES corr / night_day / N5 (catches 81 / 74 / 57 %). Suspect+bad per 100 package/v104/v108: 30 min .60/.47/.59, 3 h 1.80/1.51/1.48, 24 h 5.05/3.85/3.96 (p99.5 everywhere 1.10/2.55/5.02). v1 Y flagged 36 -> 34/37; v2 OK'd kept 4/8. review/health_review_v3.xlsx REBUILT as a 30-row spot-check of every v108 flag (backup review/_backup/health_review_v3_20261006_153355.xlsx; row-1 answer + 9/11 earlier comments carried). Nothing adopted; health rules WAIT on the answers to this sheet (open: quantile p99.8 vs p99.5).
NOTE 104b DONE (2026-10-06 ~15:30): review/health_review_v3.xlsx rebuilt short (same 25 rows; header block once, one 'why' line per row, Det with phase/function/lane; charts = every same-phase detector labelled, flagged thick, counts + % ON, expected dashed, flagged period shaded). Backup review/_backup/health_review_v3_20261006_145801.xlsx; the user's one entry (row 1 '?' + comment) carried over verbatim. Open points seen in the charts: R2 uses traffic counts not peers' time ON (his row-1 question), row 7 no healthy phase mate, row 14 R1b looks stuck. Health rules still WAITING on the v3 answers.
NOTE 106 DONE (2026-10-06 13:15, CPU only, analysis; user idea 'function should see vehicle paths / lanes'): v5b 2026 OOF (reproduced .9308 E). (1) 26 prediction-free ON-time lead-lag path features (up/downstream partners, travel time, propagation share, chain position, lane-mates): stacker +0.03 n.s., trees->stacker +0.01 n.s.; trees ALONE +0.21* (shuffled +0.01) -> real signal, already captured (trees carry lagsib/lagany features, stacker has lanes D). (2) lanes D fed the stacker OOF instead of the trees, 2 rounds, 3 seeds: vs v5b +0.04 / +0.03 n.s. (vs its own 2026-trees-lanes baseline +0.08 [+0.00,+0.16]); n_lanes exact .8481 -> .8466 / .8482; not nested. (3) joint per-phase lanes+roles decode (nested weights): strong lanes -0.19*, weak lanes +0.01 n.s. with lane count -1.6*; never beats v5b lane count. Note-102 'wrong function' phases fixed 0-1/12 by any idea. Nothing adopted, packages untouched. Optional (accuracy-neutral): lanes D fed stacker probs. Code research/code/final106, work dc_work/s106.
NOTE 103 DONE (2026-10-06 13:00, CPU only, analysis): one LightGBM per task (decider context folded into the scorer). Six-fold 2026 OOF, >= 30 min E vs full v5b: phase A2 (TCN + 1 LGB) +0.05 n.s. at every length (+0.15* at 5 min), A1 (1 LGB, no net) -0.75*, A1o (2 stages) -0.49*, no nets -0.59*; function A2 (siba + 1 LGB) -0.22* (3 h tie, 5 min -0.58*), A1 -1.14*, A1o -0.92*, no nets -0.81*. Deciders cost <= 0.02 s per 3-h call; phase A2 needs the TCN on every candidate (~+0.9 s at 3 h). Code final103/, work s103/. Nothing shipped. Open: A2 phase with a cheap TCN candidate filter (untested).
NOTE 107 DONE (2026-10-06): arrow-rule re-check of every multi-lane truth phase (+23 one-lane phases the model calls 2+) on the prints: 1,114 phases / 452 signals, agree 91.5 %; 67 high-confidence corrections -> lane truth v3 (dc_work/final_v3_work/f107/ln8; research/labels/lane_truth_changes_v3.csv, 28-row list lane_truth_arrow_list_v3.csv): 57 overlap (OLA-OLD) right-turn lanes the truth counted (old count accepted), 10 wide lanes (1 arrow over 2 loop columns). v5c lane OOF re-scored, no retrain: exact .8509 -> .8381 (-1.28*), lenient .8850 -> .8844 (n.s.); model = old count on 70 % of corrected samples -> lane error is NOT mostly label error. Open (orchestrator/user): should an overlap RT lane whose detectors call the phase count as a lane? (53/57 such); lanes refit on v3 not run. Truth v3 unused by any model; packages untouched.
NOTE 105 DONE (2026-10-06 ~12:10): user lane answers (lane review v1 rows 1-10, v2, TR+R sheet) -> corrected lane truth (research/labels/lane_truth_corrections_user_v2.csv + lane_truth_changes_v2.csv; f105/ln8): rules U (user answers, 'also acceptable' kept as accepted alt), A (arrow rule = lanes drawn as arrows; fits all 6 TR+R answers; 5 more TR+R phases -> 1 lane), O (overlap OLA-OLD lanes not counted, counting accepted; 6 more phases). 30 phases / 61 det rows changed; no phase/function label change (labels stay v4q). Refit (note-102 base recipe): n_lanes exact .8505 -> .8509 (+0.04 n.s.), lenient +0.09 n.s., changed phases +1.45*, detector-lane exact -0.02 n.s., ATSPM >= 30 E .9308 -> .9307 = tie. PACKAGE dc_work/final_v3_candidate_v5c = v5b + lanes refit (check 4/4; sample answers = v5b, confidences moved). Open for orchestrator: adopt v5c as candidate?; 09037/2B421 prints exist in cabinet_locked/pdf (answer to the user's '?').
NOTE 104 DONE (2026-10-06 ~11:30): health review v2 answers read (8 OK'd, R1 row 3 rejected, 9 '?' mostly chart confusion: 'Nx' erratic wording, scaled traffic line, R9 silence not shown). Implemented his ideas as resolver v104 (research/code/health/h104_resolve.py, dc_work/health104): R1' (healthy peers, occupancy corr, no light-traffic hold), R1b (possible congestion -> watch), N1' (pulse/normal by median ON), N3 NEW 'erratic time ON' (count-type; suspect at p99.5), N5 NEW night-relative watch. Suspect+bad per 100 package/v96/v104: 3 h 1.83/1.40/1.54, 24 h 5.20/3.73/4.01. OK'd v2 rows unchanged; row 3 now flagged. Finding: 26 % of N3 detectors are missing-OFF chains (logging), 66 % real long ONs. Sheet review/health_review_v3.xlsx (25 rows, unscaled charts, 'Your earlier comment' column). Nothing adopted; WAITING on the v3 answers before any health rule goes into a package (no other work depends on it).
NOTE 102 DONE (2026-10-06 10:45): wide-right-lane fix (user, lane review v1) NOT achieved: lanes D 2026 retrain (f102/ln8, v4q + research/labels/lane_truth_corrections_user_v1.csv) + interval pair features W / medium labels Wm / shared-neighbour T / decoder stack-exemption and Advance no-span: n_lanes exact .8481 base, all variants within +-0.1 n.s.; pattern (83 phases) +0.45 / +0.90 n.s.; ATSPM through lanes ~0 (-0.01 / -0.03 n.s.). Nothing adopted, packages untouched. Sheet review/lane_count_review_v2.xlsx (19 rows / 23 phases, one per signal, 2 NEW shaded; v1-answered phases and blind under-counts left out) waits for the user; v1 rows 11-25 unanswered (most reappear in v2). Possible next step (orchestrator): label check of TR + R split 2-lane prints (01030-type misreads).
NOTE 100c DONE (2026-10-06 06:45): PACKAGES dc_work/final_v3_candidate_v5b (= v5 + 3 full-data width-32 siba members x100_w32full{,_s1,_s2} + stackers refit on the w32 OOF; v3fit100) and final_v3_candidate_v5b_fast (v5_fast code + weights_v5b). check 4/4 and 5/5 PASS. Accuracy = v5 (OOF >= 30 E .9307 / R .9397; le2h .9275 = v5 le2h). Speed 3 h warm 1.02-1.78 s vs v5 1.53-2.65 (beta .39-.57); peak 400-546 vs 519-786 MB; 30 min faster than beta. Rebuild: pkg100 pkgsingle/pkgstacker -> assemble100 init/func/siba/card -> check.py --freeze/check. Open: make v5b / v5b_fast the candidate (orchestrator adopted); note-101 tables still describe v5.
NOTE 100b DONE (2026-10-06 05:20): plain w32 siba in the v5 recipe (2026-only, six folds x 3 seeds: x100_w32{,_s1,_s2}_f0-5 in tcn53/fpreds; final100/f100.py; dc_work/s100/b). w32 x3 + v5 stacker = v5: >= 30 E .9307 vs .9305 (+0.02 [-0.09,+0.14]), R +0.02; 5 min -0.07 n.s.; 10 min -0.20*. Package 3 h warm -26..-34 % (1.03-1.78 s vs 1.51-2.65), RAM -18..-27 %. To ship: 3 full-data w32 refits (s95_sibafull recipe --width 32) -> assemble95 siba export -> stacker refit -> check.py --freeze/check -> bench98. Decision for orchestrator/user.
NOTE 100 DONE (2026-10-06 01:30, fast screen fold 0): distil siba mean3 into one small TCN (neural/tcn100_distill.py, time100.py, final100/s100.py; dc_work/s100). Fixed 0.6/0.4 blend >= 30 E vs mean3 .8999: distilled w56 / w32 (3 seeds) +0.08 / +0.11 n.s.; PLAIN (no teacher) w56 / w32 +0.40* / +0.44*; w16 loses net-alone accuracy. CPU at 3 h (ONNX, 4 thr): w32 15x cheaper than mean3, w56 7x. Distillation is not needed; small nets suffice. Caveat: one fold, fixed blend, pod accum 1. Next if wanted: plain w32 six folds x 3 seeds -> stacker refit -> compare to v4f. Pod dc95a released by this agent (all copied, md5-verified).
NOTE 101 DONE (2026-10-05 23:50): final morning table beta vs v5 vs v5 fast -> dc_work/final_v3_work/final_table101.json (code final101/table101.py). Paired >= 30 min (beta OOF only on STG + REL): phase .9856 / .9872 / .9849 E; function .8852 / .9370 / .9338 E. Analysis only.
User away until Mon Oct 5 morning; keep working autonomously (RunPod until credit ends, then local GPU).
NOTE 97 DONE (2026-10-05, fast test fold 0 / seed 0): decided phase to the function side. A neutral (frame already decoder-built; v4f decoder top -0.00 [-0.02,+0.02] >= 30 E). B siba + decoder-top flag channel (neural/tcn97_dec.py, x97_dec_f0) = baseline seeds 1/2 in the 0.6 blend (+0.30 / -0.02); not promoted. GPU released.
User reads ONLY review/USER_INPUT.md: short, timestamped [Mon D, h:mm pm Pacific]; WINS section; RUNNING NOW block.
Monday: put a SHORT summary at the top of USER_INPUT (best numbers vs beta, what changed, decisions needed).

SPEED VS BETA + FAST PACKAGE (note 98, 2026-10-05, DONE): v4f / beta 3 h warm 4.7-6.2x, peak 2.1-2.7x (24 h 3.0-4.4x / 1.8-2.5x;
30 min ~1x / 0.5-0.9x) -> over the user's 2x rule. Variants from saved OOF (>= 30 min E vs full .9816 / .9197): le2h = both networks only
<= 2 h (beta-like): phase -0.16*, function -0.33* (vs beta: phase -0.02 n.s., function +4.13*); no networks: -0.59* / -0.87*; 1 siba -0.17*
function. PACKAGE dc_work/final_v3_candidate_v5_fast (copy of v4f + note-95 health fix): fast profiles weights_v4f/fast.json (default
le2h, 'full' = v4f; --profile / DC_FAST_PROFILE), package.json weight-set switch (weights_v4f -> weights_v5; DC_WEIGHTS), decode_trees +
stacker nonet / single, ORT spinning off (CPU-s 30 -> 6 per 3-h call, same answers). check.py 5/5. Bench le2h: 3 h 1.45-1.79x time,
1.09-1.15x RAM; 24 h <= 2.00x / 1.16x. RE-BASE ON V5: refit decode_trees (p98_phase fitdec recipe) + stackers nonet / single (f98_func
pkgstacker recipe) on the v5 OOF inputs -> assemble98.py extras --base weights_v5 -> package.json -> check.py --freeze / check -> bench98.
Decision for the user (orchestrator): le2h (within 2x) vs full.

NOTE 95 DONE (2026-10-05 23:50; notes 95 / 95b). Labels v4q = current (pointers flipped). FINAL CANDIDATE PACKAGE
`dc_work/final_v3_candidate_v5` = every learned part refit on 2026-only data (Sept-2026 staging; Dec-2024 never used), v4q;
check 4/4; health_score NaN for not_enough_data; onnxruntime no-spin. Six-fold OOF, Sept-2026 windows, >= 30 min: phase E .9843
[.9811,.9870] / R .9863 (761 signals; vs v4f shared rows +0.01 n.s.); function (v4q truth) E .9305 [.9214,.9388] / R .9395
(vs v4f -0.11 [-0.33,+0.09]; without n1_model_decided .9309, +0.05 n.s.) = TIE on less data. 3 h warm 1.6-2.9 s, peak <= 0.89 GB.
Fast package `final_v3_candidate_v5_fast` re-based on weights_v5 (check 5/5; le2h 3 h 0.57-0.94 s; phase -0.20, function -0.30).
SWA (95b) not adopted. Rebuild: pool95 build -> phase95 fit/fitranker/decode/fitdec -> func95 fit -> GPU q95 + pod (siba) ->
func95 stack/score -> pkg95 pkg* -> export95_phase -> assemble95 init/health/phase/func/siba/card -> check.py --freeze.
Held fixed (caveat): frame pred_phase and lanes/pick/twin OOF chain (older OOF). RunPod pod dc95a: my work finished and copied
back (orchestrator terminates). model/ = final_v2 still.

LABELS v4p (note 94, 2026-10-05, labels only, NOT yet the pointer; v4o stays default): v4o + user review-v1 answers (research/labels/function_label_corrections_user_v1.csv) + 05999 print (5CE055) + N1 print-vs-hand rule (model OOF agreement; 659 scored truth rows model-decided, flag n1_model_decided) + N2 dummy ch 40/41; LABEL_SETS['v4p']. Locked key function_labels_locked_v2 (training rules, labels only; 168 N1 rows n1_pending_exam). Open: 05999 d23 user-vs-print, 2B061 whole signal?, 04034 d9/d24 + 598 Function-vs-text rows, prints 5CE166/04CA156/5CE069 not yet read.

LABELS (note 81, 2026-10-03): v4l = scoring truth AND training labels for all research (rpath.LABELS_CURRENT;
atspm_score.V3S; v3_retrain.LABEL_SETS["v3s"] = v4l, old table "v3s_orig"). NEW FUNCTION HEADLINE on v4l, champion
note-77 OOF, >= 30 min: .9190 E [.9078,.9288] / .9289 R [.9186,.9384]; 10 min .9030 E; 5 min .8936 E (the .9169/.9171
numbers below are on v3s truth). Label rules R4 user answers (research/labels/function_label_corrections_v4l.csv),
R5 low-conf behaviour-tied readings out, R6 < 15-ON unused inputs = noise: cabinet/v4l_rules.py apply.
CURRENT CHAMPION (>=30 min; dc_work/cand64 + s67): phase .9818 E / .9842 R (trees 3 seeds + GRU p3 blend + decoder);
function .9169 E / ~.928 R = trees 229 feat 3 seeds + TCN siba (6 folds x 3 seeds) -> context stacker (s67, 3 seeds)
-> D lanes + lane-confidence gate .9 + stack pick (nonatspm_ap) ; short windows: twin decode. vs beta: phase +0.16,
function +3.98. Weekday samples ~0.3 pt lower (note 66).
Adopted for lock (speed, note 73/73b): ONNX tree ensembles, GRU pair batch 64, GRU only on candidates with tree p>=.01.
Measured (note 75 package, warm): 3 h 2.5-3.0 s typical / 4.0-5.1 s busiest (target <= 4-5 s); peak RAM <= 0.8 GB per signal.
FINAL PACKAGE (note 90, 2026-10-05): `dc_work/final_v3_candidate_v4f` (not shipped; model/ = final_v2) = v4e + phase network
TCN ad_all (weights/phase_tcn.onnx, 15-ch raster shared with siba; decoder refit on the TCN-blend OOF, ad_all_e100n76) + function
trees / lanes D / setback / stackers refit on the final labels v4o. Headline >= 30 min: phase .9816 E / .9841 R (vs GRU -0.02 n.s.);
function (v4l truth) .9197 E [.9087,.9293] / .9291 R (vs v4e +0.03 / -0.00 n.s.). 3 h warm 2.4-2.8 s typical, 3.2-3.8 s busiest
(24 h busiest 5.0-5.3 s), ~0.5 s faster than v4e; peak <= 0.89 GB; check.py 4/4. Siba members = FINAL v4o refits x86_sibafull4o
seeds 0 / 1 / 3 (seed 2 left out: outlier, agree90.json), swapped in 2026-10-05 09:00 (siba90.py), references re-frozen.
Rebuild: p90_phase decode/fitdec --arm tcn_ad76; oof90 stack/score/pkg*; assemble90 phase / func / card; check.py --freeze.

CHAMPION PACKAGE (note 84, 2026-10-04): `dc_work/final_v3_candidate_v4b` (not shipped; model/ = final_v2) = v4 + THREE
full-data siba v4l members (stacker 'mean3'; 'single' fallback if one member's files remain) + siba filter ON by default.
OOF v4l >= 30 min .9191 E [.9079,.9291] / .9290 R (= champion .9190; +0.17 [+0.06,+0.29] vs single+filter); 10 / 5 min
.9036 / .8940 E. Speed 3 h warm 2.99 / 2.47 s typical, 4.06 / 3.42 s busiest (24 h busiest 5.41 s), peak <= 1.0 GB.
Note 84b: all three seeds filtered (OOF x74_sibaflt_s1/_s2): mean3+filter .9192 E / .9291 R, = note 84 (+0.01),
vs champion +0.02 [-0.05,+0.10], vs single+filter +0.14..+0.45* at every length; 5 min filter cost -0.09* (noise-size).
check.py 4/4 PASS, references re-frozen. Rebuild: final84 fit84 stacker3 -> export84 export / parity -> assemble84 ->
check.py --freeze. Swap into model/ only at the end of the search.

MONDAY TABLE (note 85, 2026-10-04): beta vs v4b, v4l, >= 30 min paired -> dc_work/final_v3_work/monday_table.json; phase .9802 -> .9818 E (+0.16*); function .8761 -> .9195 E (+4.34*, 75 % of rows); YR +20.2, non-ATSPM +10.8; 3 h warm 0.5 s vs 2.5-4.1 s.

LABEL DISCRETION SHEET (note 87, 2026-10-04): review/label_discretion_review_v1.xlsx + _charts/ = 1,001 rows / 485 signals,
every training/dev label dropped or changed by a rule or agent (1,973 detectors; Dec-role 621, check fail 405, unhealthy 166,
DQ 114, misconfigured 70, R5 68, R1 66, stack 41, unusual 21 signals/438 ...); NOT yet put in USER_INPUT (orchestrator).
Finding: --clean drops 233 labelled rows on dq health = fault events 84-88 (user-banned) -> fix + refit; locked key lacks
the label-only rules (R1/R3/R5/R6/stack/sweep) -> apply before the locked confirmation.

FAULT EVENTS REMOVED (note 88, 2026-10-04): DONE. dq_core actuation-only (83-88 never read), card rule + hb_data
presumed-healthy actuation-only, legacy producers patched. Labels research/labels/function_labels_v4m.parquet = v4l with
fault-free dq (truth identical; LABEL_SETS["v4m"]): +228 detectors / +7,379 training rows. Six-fold refit (229 trees x 3 +
mean3 stacker, v4b recipe): >= 30 min E .9193 / R .9292 vs v4b .9192 / .9291 = +0.01 [-0.04,+0.06] (vs current-v4l control
+0.04 [-0.01,+0.10]): neutral -> adopt for compliance. PACKAGE dc_work/final_v3_candidate_v4d (v4c name left to note 86):
v4b + v4m trees / lanes D / stackers / setback P50; check.py 4/4 PASS; speed = v4b. Rebuild: final88 oof88 fit -> pkgfunc,
pkglanes, pkgsbfeats, pkgsetback, pkgsingle, pkgstacker -> assemble88 -> check.py --freeze. OPEN (orchestrator): point
LABELS_CURRENT / LABEL_SETS["v3s"] at v4m and note 86's oof86 at the v4m trees (f76/function_c_v4m) + build on v4d;
siba nets still train on func_rows_v4l -> tcn53/func_rows_v4m.parquet ready for the next GPU refit; health limits of
notes 38/43/79 were set on the fault-shaped healthy reference (note-83 calib re-run: no move).

EXCLUSION GROUPS BY DATA (notes 89 / 89b, 2026-10-04): DONE, replaces the user review of the note-87 sheet. Each group's
training rows restored, trees refit, fixed v4l truth: no group hurts beyond noise. Round 1 (base v4m): eight groups (Dec-role,
check fail, unhealthy, DQ, misconfigured, R5, 21 unusual signals, R6+sweep+re-read flips) together .9200 E / .9297 R (+0.05 /
+0.03 n.s.) -> v4n. Round 2 (base v4n): not_checkable (+657 detectors) 3 seeds .9205 / .9300 (+0.05 / +0.04 n.s.) -> v4o.
FINAL TRAINING LABELS = research/labels/function_labels_v4o.parquet (restored training values in <col>_train, plain
columns = v4m so scoring sets are unchanged; g89_restored names the group). POINTERS FLIPPED: rpath.LABELS_CURRENT /
DEC_ROLE_CURRENT, v3_retrain LABEL_SETS["v3s"], final88/oof88 pkg stages (DC_PKG_LABELS default v4o, trees
f76/function_c_v4o), siba func_rows dc_work/tcn53/func_rows_v4o.parquet (no GPU job started). Training rows 306,079 ->
357,754; still excluded with data 2,196 -> 466. Not touched (user rules): R1, YR identical, stack relabel, R2/R4.
SANITY SHEET READY (not yet announced): review/label_sanity_20.xlsx + _charts/ (20 random still-excluded labels; Y/N/?).
REVIEW SHEETS RE-LAID (note 91, 2026-10-05): v1 / v1b / label_sanity_20 now say "det 15: P5 Presence" per detector (Label,
Model says, How sure, What this row asks); v1's 4 typed answers carried and verified; originals in review/_backup/.
Rebuilds via review72 / review81 / sanity89 carry typed answers and refuse to drop any (evaluation/review91.py = re-layout).

HEALTH vs ACCURACY + YR WITHOUT COUNT (note 93, 2026-10-05, analysis only, nothing changed): v4f package health on the OOF
windows (>= 30 min, v4l truth): function E ok 92.4 / not_enough_data 89.8 / suspect 89.8 / bad 75.0 %; flagged - ok -7.2 pt
[-10.9,-3.4]; phase ok 98.3 / suspect 96.7 / bad 97.3 (-1.4 pt). health_score ranks flagged rows (71 -> 90 %) but is 1.0 for
not_enough_data too (package caveat: filter on status or give it NaN; not changed). YR on phases without Count: 445 scored rows
(0.24 %), 33.9 % (12052 video called Count); out of training -0.02 n.s., out of scoring +0.13 -> keep. Partner label errors
04016 d23 / 04040 d28 (Presence -> Count) for the next label pass. Optional sheet review/yr_without_count_review.xlsx (10 rows,
not announced). Work %DC_WORK%/s93/, code research/code/final93/.
RUNNING / NEXT (in order):
-0b. DONE note 87b (subagent, CPU only, 2026-10-04; numbered 87b because another agent's note 87 = label discretion review).
   A. Phase net in the current pipeline (saved six-fold OOF, same pool / folds / labels / coverage): >= 30 min E GRU p3 .9818,
   TCN ad_all .9816 (-0.02 [-0.07,+0.04]), r05_all .9814 (-0.03); R -0.03; 5 min -0.11 n.s.; fold 0 seed-1 TCN same. TIE.
   3 h network time (ONNX CPU, kept pairs) GRU 0.38 / 2.08 s vs TCN ad 0.13 / 0.74 s typical / busiest (~2.8x faster).
   User rule (tie -> faster): TCN ad_all. Swap NOT started: needs GPU full-data ad_all refit (~1.5-2 h, tcn53.py as now) +
   optional 6 fold re-infers with the fixed partner tie-break (~1 h), then CPU: 6 extra raster channels in the package,
   decoder refit on TCN-blend OOF, ONNX, check.py --freeze, bench. Orchestrator to queue after note 86 frees the GPU.
   B. Joint phase + function decider (iterate once, nested): phase +0.05 [+0.02,+0.08] (beats shuffled control; under the
   0.3-pt bar) not adopted; function -0.05 [-0.09,0.00] dropped. Code research/code/final87/, work dc_work/s87/.
-1. note 86 (subagent, LOCAL A1000; research/notes/86_siba_v4l_folds_stacker.md). DONE: 18 siba fold nets x86_siba4l on v4l
   (filtered OOF); mean3 stacker refit on it (v4m trees): >= 30 min .9193 E / .9291 R vs v4d .9193 / .9292, -0.00 [-0.11,+0.11]
   E = NEUTRAL (Count +0.24*); package dc_work/final_v3_candidate_v4e = v4d + that stacker (check 4/4, speed = v4d).
   DONE: phase TCN full refit tcn53/models/ad_all76_full.pt -> ONNX s86/phase/ad_all76_full.onnx (parity 9.5e-6; not in a
   package yet); ad_all fold re-inference on note-76 inputs: TCN vs GRU still a tie (-0.02 [-0.07,+0.04] E >= 30 min).
   DONE: final v4o siba refits x86_sibafull4o{,_s1,_s2} -> s74/full/x86_sibafull4o*_full.onnx (parity <= 3.6e-5); seed 2
   converged poorly (train loss .653 vs .56) -> seed 3 DONE (x86_sibafull4o_s3, loss .586, parity 1.4e-5). Package agent
   picks 3 members by agreement.
   DONE note 92b (research/notes/92b_siba_11class_screen.md): 11-class siba head, 2-fold screen -> NOT ADOPTED (>= 30 min
   E vs trees-only subclasses +0.01 [-0.51,+0.54]; more upstream zones called Advance 419 vs 396). GPU idle, queue empty
   (optional unfiltered x86 infers stay #OFF in q74/queue74.txt).
   Optional unfiltered x86 infers paused (#OFF lines in q74/queue74.txt; only f0 ran).
0. DONE note 74 / 74b (subagent, LOCAL A1000, 2026-10-01 19:17 -> 2026-10-04 01:40; GPU idle now, queue empty).
   All verdicts on the champion (stacker + gate .9), >= 30 min E: sibm -0.04 (5 min -0.21*) dropped; siba+cw2 -0.02 dropped;
   specialist decider (siba+sibm+cw2) +0.03 [-0.03,+0.08], = seed diversity, not adopted; GRU head vs TCN (seed 0, v4l)
   -0.17 [-0.34,0.00], 5/10 min -0.39*/-0.41*, slower -> TCN kept. Steps 1-3 re-checked on v4l: same verdicts.
   PRODUCTION NET: full-data siba refit tcn53/models/x74_sibafull4l_full.pt (v4l labels; v3s twin x74_sibafull_full.pt),
   43 epochs, seed 0, post-note-76 inputs; ONNX dc_work/s74/full/*.onnx via neural/export74.py (parity 1.1e-5).
   + seeds 1 / 2 (x74_sibafull4l_s1_full.pt / _s2_full.pt, ONNX same folder, parity PASS) for the 3-member + filter test.
   siba candidate filter (tree p >= .01): accuracy-neutral (-0.01 [-0.07,+0.05] E), keeps ~27-32 % of pairs -> adopt
   (expected ~-0.7 / -1.3 s at 3 h). Queue tooling for reruns: dc_work/tcn53/q74/queue74c.ps1 (+ queue74.txt, STOP).
1. DONE note 69: siba adopted (champion function >=30 min .9169 E, +0.26 over fj). RunPod $3.51 left, no pods,
   volume rvwhubuncy kept (never let balance hit 0). Weekend = note 74 on LOCAL GPU (detached queue): sibm, siba+cw2,
   specialist stacker, GRU backbone. Post each result to USER_INPUT (timestamped, WINS only if CI > 0).
2. fj/siba candidate filter (idea 2) needs one re-inference (local GPU ok).
3. DONE note 75 (2026-10-02): integrated package `dc_work/final_v3_candidate_v3` (not shipped; model/ = final_v2).
   CPU refits: decoder (filtered blend), 229 trees, lanes D, stacker, setback sb7; phase trees + GRU reused; siba =
   fold-0 seed-0 research model (ONNX) until a GPU full refit; all trees ONNX (2e-14). Parity vs OOF: same models ->
   identical final function (741/741); check.py 3/3 PASS. 3 h warm 2.5-3.0 s typical, 4.0/5.1 s busiest, peak <= 0.8 GB.
   OPEN (orchestrator): one siba member costs -0.12 pt >= 30 min / ~-0.35 at 5-10 min (stacker 'single' refit) vs
   3 members +~5 s busiest 3 h; siba filter (idea 2) needs a re-inference.
   Swap-in later: export75.py siba --members <refit tag> -> check.py --freeze -> copy package into model/.
3b. DONE note 76 (2026-10-02): features phase- AND channel-order-free (ties averaged / kept, never broken by a number);
   research copies in `research/code/phasefree/` (rpath puts them before model/; model/ = final_v2 still has the old
   tie-break). Refit six folds: phase >= 30 min E -0.004 [-0.035,+0.025], function (arm c) +0.011 [-0.025,+0.047]:
   neutral, ADOPTED. Package final_v3_candidate_v3 rebuilt (fit76.py + assemble76.py: phase ranker, decoder, function
   trees, lanes D, stackers refit; references re-frozen); pre-fix copy dc_work/final_v3_candidate_v3_pre76.
   check.py now tests EVERY probability under renumbering: 3/3 PASS; 42/42 bench renumberings identical (<= 4.6e-7).
   New default for every research build: pool = f76/phase/pool.parquet, function frame = f76/frame_v6e_c (lag tables
   f76/lag_{dec,stg}.parquet); OOF = f76/phase/full, f76/function_c, f76/s74/f76c (stack). GPU refits of siba must
   use research neural/tcn53.py (content partner tie-break).
3d. DONE note 83 (2026-10-04): package `dc_work/final_v3_candidate_v4` (v3 kept): v4l refits (function trees, lanes D,
   stacker 'single' 47 cols = 15 health cols out, setback P50; phase ranker / decoder / GRU unchanged), siba = full-data
   v4l x74_sibafull4l (pair + head ONNX, head masks filtered candidates), health in whole ticks (chatter .30 -> .263,
   rapid unchanged), dead check removed, partner-undercount info note, SIBA_FILTER switch OFF (predict siba_filter /
   --siba-filter / DC_SIBA_FILTER=1). check.py 4/4 PASS; parity A 741/741. OOF v4 recipe >= 30 min .9181 E / .9279 R
   (-0.09 / -0.10 vs champion, single-member cost; 10 / 5 min -0.37 / -0.29). 3 h warm 2.4-3.3 s typical, 4.3 / 5.2
   busiest (filter on 2.0-2.3 / 2.7-3.3). Rebuild: final83 fit83 (func, lanetruth, lanes, stacker, sbfeats, setback;
   F76_ARM=c) -> export83 export / parity -> assemble83 -> check.py --freeze. OPEN: filter on/off from the GPU result.
3c. DONE note 77 (2026-10-02): lanes D (pair model on BOTH orientations, mean), lanes decode / ln6 / ln7 / twin pairs in a
   canonical behavioural order (content_key), decode ties by probability content, setback pair model both orientations,
   night-speed / health partner ties -> no channel-order dependence left. OOF inside the champion: function >= 30 min E
   +0.012 [-0.015,+0.041], R +0.009; setback neutral -> ADOPTED. Package rebuilt (fit77 + assemble77; pre-fix copy
   final_v3_candidate_v3_pre77); check.py now 4 checks incl. channel reversal / shift: 4/4 PASS; 102-run battery clean.
   3 h warm 2.4-3.2 s typical, 4.1 / 5.0 s busiest (unchanged). New OOF defaults: lanes f77/ln8/lanes_D.func, pick
   lanes/ln6_pick_D229o + ln7_stackhealth_D229o, stack health f77/stackhealth_ff, twins f77/pairs_short, stack
   f77/s74/f77 (of77.scoring_frame77 / setup_new), setback f77/sb7. Research lane_output / ln6 / ln7 / atspm_decode /
   s62 / sb5_setback / health_core now order-free by default (ln6_pick.ORDER_FREE).
4. GRU vs TCN only for the final function network (local GPU if no credit).
5. Open user questions in USER_INPUT: radar/long zones labelled Other (1.9 pt); YR without Count (43 phases).
   User's partial answers in review/function_label_review_v1.xlsx — fold in at next label pass, never overwrite.
   Note 81: the 4 answered v1 rows are in research/labels/function_label_corrections_v4l.csv (applied as R4); add new
   answers there + re-run `DC_CAB_STORE=cabinet_v4l python cabinet/v4l_rules.py apply`.
   Second batch READY (not yet announced to the user): review/function_label_review_v1b.xlsx (+ _charts/), 32 rows on
   29 signals, champion vs v4l, p >= .8, nothing from v1; 6 detectors repeating a v1 row's pattern left out
   (dc_work/rev81/review81_same_pattern_as_v1.csv) -> they inherit the v1 answer. Rebuild: `python evaluation/review81.py
   rows` then `review --write` (delete the v1b xlsx + charts first).
5b. Note 82: health review sheet READY (not yet announced to the user): review/health_review_v1.xlsx (+ _charts/), 70 rows
   = 14 checks x 5 (3 clear + 2 borderline, w40 26-28 Sep, training signals), sheet "Checks" lists all 16 checks with
   fires per 100 detectors at 3 h / 24 h; partner-ratio check shown as PROPOSED. Rebuild: `python health/h82_review.py`
   (gate metrics `h82_gate.py` first; delete the charts folder first). Open finding: chatter / rapid limits compared in
   float seconds on the 0.1-s tick (gap of exactly 0.3 s counted only sometimes) - package unchanged, orchestrator call.
6. Exam on locked signals only when the user says go.

## Shipped

`model/` = **final_v2** (commit `e28c40f`, restructured in `656d22b`): LightGBM ranker → joint
decoder, plus a GRU blended in (ONNX runtime) for samples under 2 h; 5-class function head.
Locked-signal exam (143 signals): phase **.9815** at 30 min, **.9862** with days; function **.868**
on the user's corrected labels (.755 on the raw config). Thresholds and usage: `README.md`.

## Done (see research/notes/00 … 14)

| # | what | outcome |
|---|---|---|
| 00–12 | first wave: LightGBM pipeline, official phase labels, more data, final_v1 | phase .986 (days) |
| 13 | GRU refit + blend, ONNX | +1.2 pt phase at 30 min → final_v2 |
| 14 (Track A) | function: user-corrected labels; expert features; lane decoding; Other-as-rejection | labels +11 pt; features +0.16; lanes failed (see caveat below); rejection = flag only |

**Track A caveats to carry forward.** (1) The lane experiment judged "two presence zones on one
phase" as a label violation — wrong: a phase has 1–3 lanes, so the rule is <= 1 Presence / Count /
Advance **per lane**, up to 3 lanes; Yellow_Red may span lanes. Redo the lane check with that rule
before concluding anything. (2) Same-lane pairing from timing alone reached only AUC .74; 44
signals carry RL/CL/LL lane text in the channel description (`data/detector_plans.parquet`) — a
ready-made lane label set to validate against. (3) The round-1 corrections were all on locked
(test) signals, so they fixed *scoring* but no *training* label — round 2 is the one that changes
what the model learns.

## Label override table — the truth for function

`research/labels/function_labels_v2.parquet` (+ README): the user's corrections beat the config
file; rows marked `?` are excluded from training AND scoring; Bike / Bike Loop / Departure → Other.
**Every future function stage must read this table, not the config file.** Merge new review rounds
into it the same way (keep a `source` column: config / review_round1 / review_round2 …).

## Running now

Launched 2026-09-23 by the orchestrator (session 3); each subagent writes its own note, the
orchestrator folds the result into this file.

| work | owner | output | state |
|---|---|---|---|
| Track B **B3** (dropout + augmentation) → **B4** (mixed lengths 5 min…6 h), fold 0, TCN; six folds for whatever is kept | subagent, GPU, detached chain from `dc_work/repo_local` | `research/notes/16_neural_trackB_b3b4.md` | **done — both dropped, six folds skipped.** B3 best (dropout .2 + context-channel dropout .3) blend .9759 at 30 min = +0.08 pt vs 3-seed TCN mean (net alone +0.43, absorbed by the trees; jitter was a no-op, `sample_plan` already randomises starts). B4 (5 min…6 h) worse at 30 min (−0.39 pt vs mean) and 1 h, only gains at 66 h where the net is off. `best_args` still `--arch tcn`. Chain: `research/code/neural/trackb_b3b4.py` (use it, not `run_trackb.sh`, whose gate double-appends flags). Track B GPU total 2.3 h. |
| **Lanes redo** with the per-lane rule (label-violation recount, same-lane cues on the RL/CL/LL signals, decode only if grouping improves) | subagent, CPU (threads <= 6) | `research/notes/17_lanes_redo.md` | **done** — labels are fine under the per-lane rule (1.4 % of phases imply > 3 lanes; A3's "25 %" was wrong); new same-lane cue AUC .895, combined .883 (A3 .736); constrained decode +0.8 pt 5-class / −1 pt A/P/C ⇒ **dropped**, argmax stays; `n_lanes` kept as a review aid (`dc_work/trackA/lr4_nlanes.parquet`). Cabinet lane labels plug in at `dc_work/trackA/nlanes_labels.parquet` (DeviceId, target, n_lanes), validation only. Track A a1–a4 scripts now ported to `research/code/trackA/` (DC_WORK/DC_REPO paths, compile-checked, mapped in `research/code/README.md`). |
| **Cabinet-print pilot**: lanes per phase from the intersection diagram, 10 training signals (share is read-only; copies in `dc_work/tmp/cabinet_pilot/`) | subagent | `review/cabinet_prints_pilot.md` (git-ignored) for the user | **done** — 10/10 found, 49/59 phases high confidence, 52/59 agree with label-implied lanes; loop circles match channel descriptions on 8/10. Awaiting user go/no-go + 6 questions (`review/USER_INPUT.md` item 2). PDF copies in `dc_work/tmp/cabinet_pilot/pdf/`, cached share listing there too. |

| **B9** LightGBM neighbour-trace summary features (hand-built sibling context, aimed at concurrent-pair errors), fold 0 first, CPU (threads <= 6) | subagent | `research/notes/18_trackB_b9_neighbour_trees.md` | **done — dropped.** 42 `nb_*` columns: ranker alone +0.32 pt at 30 min (real: 2 seeds, shuffled control back to baseline; concurrent-pair ranker errors −13 %), but after the decoder +0.08 and in the blend **+0.02 pt** — the decoder already recovers neighbour context from the neighbours' probabilities. Lesson: judge ranker-side ideas after the decoder / in the blend from the start. Untried: a lagged (advance→stop-bar) neighbour graph. Code `research/code/trackB/b9_*.py`. |

| **B5** sibling-detector context inside the network (set-attention over the signal's detectors), fold 0 first, headroom analysis first | subagent, GPU | `research/notes/19_trackB_b5_sibling_net.md` | **done — dropped.** Headroom first: fold-0 blend at 30 min has 170 errors (53 % concurrent pairs); the net alone is right on only 18, decoded trees on 47, *any* model on 73 ⇒ the network side can add at most ~0.26 pt. b5a/b5b (attention over sibling embeddings, +first-pass probs) blend .9731/.9729 vs .9750 mean; net alone +0.24/+0.34 but on rows the trees already get right. 0.9 GPU-h. ONNX works but needs per-signal input `[D,K,9,T]`. |
| **B6** trees + GRU + TCN three-way blend, **B8** one stacking test, both from existing predictions | subagent, CPU (threads <= 6) | `research/notes/20_trackB_b6b8_blend_stack.md` | **done — both dropped.** B6 three-way (.5/.25/.25) +0.11 pt mean at 30 min; the same with two nets of one kind does as well (TCN errors correlate with GRU errors φ .74, as much as two seeds; trees vs net .43). B8 stacking +0.01 after decoder, −0.12 before. **Cut-off finding for final_v3:** blend − trees = +0.35 pt at 3 h, +0.31 at 6 h, +0.20 at 24 h (six folds, GRU) — the 120-min net cut-off leaves ~0.3 pt on the table at 3–6 h; a 6-h TCN pass costs ~2 s/signal. Settle at the final re-fit on six folds (OOF set has no 2/4/12 h windows). |

| Last Track B checks: **B10** one wider/deeper TCN variant, **B7** Mamba/S4 time-boxed ~3 h, then the **six-fold TCN baseline** + net cut-off (120 vs 360 min) on six folds, for the end-of-search TCN-vs-GRU comparison | subagent, GPU | `research/notes/21_trackB_b10_b7_sixfold.md` | **done — Track B closed** (note written by the orchestrator). B10 +0.07 pt, B7/S4D +0.10 pt at 30 min: dropped. **Six-fold TCN vs GRU blend: −0.13 pt at 30 min (TCN behind on 5 of 6 folds), +0.05 to +0.10 pt past 3 h.** Blend beats trees at EVERY length on six folds (+0.3–0.4 pt at 3–6 h, +0.2–0.3 at 24 h/full) ⇒ final_v3 candidate: raise/remove the 120-min net cut-off. GPU idle. |
| **Phase residual-error audit** (label error / concurrent-pair ambiguity / low data / genuine), all six folds OOF; top ~60 label suspects → `review/phase_label_suspects_round1.xlsx` (optional user review) | subagent, CPU (threads <= 6) | `research/notes/22_phase_residual_errors.md` | **done.** Six-fold OOF errors, 30 min / full: likely label error 18.7 % / 26.0 %, switch/additional-call target 3.0 / 2.6 %, concurrent pair 13.7 / 15.0 %, low data (<10 act.) 29.2 / 3.5 %, genuine 35.5 / 52.9 %. Fixing labels ⇒ ~+0.5 pt at both lengths. Genuine errors are mostly through read as left-turn (2→1, 6→1, 2→5), not concurrent pairs. 59 suspects → `review/phase_label_suspects_round1.xlsx` (user item 4, optional). |

| **Cabinet-print mining, step 1**: tooling (`research/code/cabinet/`), check the share's `Detection_Configuration_*.xlsm` files as a structured source, redo the 10 pilot signals with the user's spot-check fixes | subagent | `research/notes/23_cabinet_mining_tooling.md`; records `dc_work/cabinet/` | **done.** Tooling `research/code/cabinet/cab_*.py`; pilot v2 gets every user spot-check item right (orchestrator verified 01005/01017 records). 163 print detectors: 131 high / 26 medium / 6 low; tiers 2 complete_high, 7 mixed, 1 incomplete. xlsm workbooks: 150 of 761 training signals (mostly radar), phase 94 % vs timing, function text = config's own source ⇒ channel list/cross-check only. ~4 s scripted + ~1 min visual per signal. Open items batched for the user later: 01011 camera zones 42/46 (config says bikes, bike-level counts), spanning tied pairs 110–150 ft upstream labelled Presence in config (→ Other/mid per guide). |
| **Cabinet scale-up stage 1**: scripted pass on all ~761 non-locked training signals + visual batch lists (`dc_work/cabinet/batches/`) | subagent | records `dc_work/cabinet/signals/` | **done.** 750/761 PDFs (11 not on the share); input file parsed 455, blank (video/radar sites) 252, no text layer 43; cabinets 332 500 / 332S 156 / 336 61 / unreadable 33. 35 visual batches in `dc_work/cabinet/batches/` (25 ordinary × ~25, 3 × 7 for 336, 7 × 8–9 complex) + `VISUAL_PASS_INSTRUCTIONS.md`; records written only via `research/code/cabinet/cab_record.py`; runner `cab_scale.py`. `batch_nodata.txt` (62 signals, no hi-res data in either window) **skipped** — no training value (orchestrator decision). DQ must be re-run with print-derived locations after the visual pass. |
| **Cabinet visual pass**: batches 01–04 running in parallel (01 includes the 10 pilot stop-bar re-reads). Orchestrator checks batch 01 before launching 05–35, ~4 at a time | subagents | `dc_work/cabinet/signals/*.json` (status finished) | running |
| **Detector DQ toolkit** (`research/code/cabinet/dq_*.py`): stop-bar vs advance, mid vs sum of advance, same-lane order, saturation (150/5 min/lane), bike plausibility, dead/stuck | subagent, CPU | `research/notes/24_detector_dq_toolkit.md` | **done.** `dq_core.run(dets)`: 8 checks → `dq_score`, `suspect_config_or_health` (8 % of config-labelled detectors on 20 test signals); 0.3 s/signal. Findings: dead detectors change between 2024 and 2026 (use newest window); the peak-undercount rule holds for Presence, reversed for Count; side-by-side advance pairs firing together (01006 d22/23, d8/9; 01011 d8/9) and one loop on two inputs (2B062 d34/d42) = config questions for the user, to batch with the print review later. |
| **Whole-intersection Other**: how many active channels are unlabelled and what the function model calls them; experiment: Other split into subtypes (Mid, Bike, …) as training classes, collapsed at output | subagent, CPU | `research/notes/25_whole_intersection_other.md` | **done.** 19–24 % of active channels have no function label (median 2–4/signal, as busy as labelled ones); never in function training; the current head calls **~50 % of them a performance-measure class** (24 % at p ≥ .80) — the user's concern confirmed. 7-class head (Mid, Bike) costs nothing (5-class .7922 vs .7911, A/P/C +0.5 pt; Bike P/R .74/.67, Mid .59/.52) ⇒ goes in with the print-label retrain. Adding all unlabelled rows as Other fixes 68 % of them but costs 5 pt A/P/C (some are real unlisted perf detectors) ⇒ **must be done on print-complete signals only**. Script `research/code/trackA/w1_whole_intersection.py`. |

**Cabinet-print plan (user decision 2026-09-23, rules in `research/labels/cabinet_print_guide.md` and
AGENTS.md):** after step 1, the orchestrator checks the pilot v2 against the user's spot-check, then
scales to all ~700 training signals in batches (several subagents in parallel), then consolidates:
intersection tiers (complete_high / complete_mixed / incomplete), overrides list
(`review/print_label_overrides.xlsx`), dead-detector list for maintenance
(`review/dead_detectors_from_prints.csv`), medium/low-confidence list for the user. Then retrain
function (and phase checks) on complete_high intersections with whole-intersection Other labels.
The round-2 function review is superseded by this. User rules added 2026-09-23: Bike and Mid are
reported output classes; stop-bar Presence = 0–20 ft upstream of the stop bar, Count = on/just past
it, Yellow_Red = next to the count zones (crops saved to `dc_work/cabinet/crops/`); **the user wants
to spot-check a sample of Count / Yellow_Red readings** — prepare that (small, with crops) once the
first visual batches are done. Locked signals are NOT mined (ask the user at the
end whether to mine them for exam truth).

Pattern so far: every gain on the network alone or the ranker alone (B2, B3, B9: +0.3–0.4 pt) is
absorbed by the trees + decoder in the blend. Only ideas that fix what the blend gets wrong
(concurrent pairs on short samples) can move the headline.

Plan while the user reviews (several hours): phase-only work continues. After B3/B4 finishes, B5
(sibling context inside the network, ~1 week) is next on the GPU; consider the parked "preload rasters
into GPU memory" speed-up first, since B5 is the long one. B9's result tempers B5's expected gain
(the decoder + blend already absorb hand-built neighbour context), so screen B5 cheaply on fold 0
before committing the full week. Function retrain waits for round-2 labels.

Questions to the user and their answers live in `review/USER_INPUT.md` (git-ignored) — read it
before starting work.

**Track B, B1 and B2 are finished** — numbers, verdicts and caveats in `research/notes/15_neural_trackB.md`.

| step | verdict | fold-0 blend at 30 min | rule |
|---|---|---|---|
| B1 TCN backbone | **kept** | .9737 vs GRU .9745 (−0.08 pt), 11× faster to train, 2.6× faster on CPU | within 0.3 pt ⇒ adopt |
| B2 three-seed ensemble | **dropped** | +0.10 pt over the mean of three seeds | needs ≥ 0.3 pt |

Two corrections to the previous hand-over. (1) The GRU's fold-0 inner-validation score is **.9528**, not
.9185; the TCN's .9501 is inside the GRU's own seed spread, so inner validation was never evidence for
the TCN — the held-out table is. (2) The first night's chain died at 18:20 on a launcher bug (a path
embedded in a `python -c` string is not converted from `/c/...` by Git Bash), costing ~11.5 GPU-hours;
also `research/code/rpath.py` had left `research/code/` behind `model/` on `sys.path`, so **every**
research script failed on `from common import DC_WORK`. Both are fixed.

**Carry forward:** the TCN blend's seed sd at 30 min is **~0.2 pt** (0.19 pt on the quantity stage 13
measured as 0.069 pt for the GRU), so judge B3/B4 against ~0.2 pt and fit two seeds before believing a gain.

**What `model/` would need to ship the TCN** (not done, nothing in `model/` was edited): `gru_onnx.py`
is already architecture-agnostic — it feeds `[N,9,T]` to an ONNX session and reshapes to `[D,K]`, so the
TCN needs **no code change**, only `model/weights/gru.onnx` replaced by the TCN export (identical input
/output names `x`/`score` and dynamic axes). `gru_input.py` and the `blend.json` settings (0.5, "before",
120 min) are unchanged; the 120-min cut-off is worth revisiting separately now the net is 2.6× cheaper.
Docstrings in `gru_onnx.py`/`gru_blend.py` say "3-layer bidirectional GRU" and would need rewording, the
weights file could keep or change its name (`WEIGHTS_FILE` in `gru_blend.py`), `model/check.py`
references must be re-frozen, and the shipped net must come from a `--final` fit on all training
signals, not a fold model.

## Next (in order)

1. Track B: B3, B4, B9 done and dropped (notes 16, 18); B5, B6, B8 running (see above). Remaining after
   those: B7 (Mamba, time-boxed), B10 (wider/deeper, last). Older text of this item, kept for the tooling
   pointers: **B3** (dropout + augmentation) → **B4** (mixed sample lengths), fold 0 each, on the
   TCN; then all six folds for whatever was kept, so the blend numbers sit on the stage-13 rows.
   Everything is in place: `research/code/neural/trackb_{train,infer,eval,eval_full,decide,export}.py`,
   `dc_work/run_trackb.sh` (fix its `spec()` helper the way `run_trackb_b2.sh` does before reusing it),
   state in `dc_work/trackB/state/` (`best_args.txt` = `--arch tcn`, `best_seed0.txt` = `tb_tcn_f0_oof`),
   and `trackb_eval.py` reproduces the stage-13 fold-0 blend exactly, so any candidate is one command.
   A TCN fold is ~12–20 min of GPU plus ~4 min of inference, so B3 + B4 + six folds is about 3 h.
2. When the user returns round-2 function corrections (`review/function_disagreements_round2.xlsx`,
   586 rows, 193 signals): merge into the override table, retrain the function head on corrected
   labels **with** the expert features (`dc_work\trackA\models\function_v5_expert\`), re-score.
3. ~~Lane structure, done right~~ — done (note 17): decode dropped, `n_lanes` a review aid. Next on
   lanes depends on the user's cabinet-print go/no-go (`review/USER_INPUT.md` item 2): per-loop lane
   labels would give true same-lane pairs to validate/train the grouping on.
4. Only after ALL of the above (and include the net cut-off: 120 vs 360 min, six folds, see note 20): re-fit the best configuration with a GRU backbone too, compare,
   and then — and only then — propose a `final_v3` via the stop rule in `AGENTS.md`.

## Cabinet-print label mining

**PLAN AFTER THE USER'S 2026-09-24 REVIEW (supersedes the stop-rule timing below).** The user reviewed
label-check failures: the checks compared against the wrong partners (radar occlusion: far/advance
zones undercount; multi-lane advance < sum of lane stop-bar zones; loops: advance >= same-lane stop
bar), the YR "off in red" rule is wrong (YR exists to catch yellow/red entries), coordination makes
advance favour green (a tendency), radar presence may have internal delay (identify presence by holding
the call through red at peak), radar ETA zones are Other. **Two layers:** data CLEANSING may use prints /
technology / lanes / card structure; the MODEL sees only hi-res data (no technology input; mixed-tech
signals are normal); condition-dependent features are computed only when the sample allows (NaN
otherwise) so short samples still get answers. **Do NOT use the exam yet** — the search is not done.
Order: (1) rebuild label checks + 15-row spot-check for the user (retrain waits for the user's OK);
(2) VD (Voyage) number audit of training prints + card rule (both outputs of a slot fail together —
cleansing only); (3) read the 186 locked signals' prints into a SEPARATE store `dc_work/cabinet_locked/`
(labels only, no scoring); then move half of NEWTEST (random) + keep 43 TEST + other half locked;
(4) retrain on cleaned labels, then retest lanes (drop was premature), a radar-ETA class, health-as-feature;
(5) detector HEALTH (secondary objective, bounded): rule-based score (dead, sudden drop to 0 while the
signal is active, stuck, chatter, day/night shape by type, count variance range, correlation with
siblings breaking) + a time-boxed self-supervised anomaly model (inject synthetic faults, predict which
detector/hours are bad; idea from the user's imputer notebook), output separately from phase/function;
(6) setback distance by physics: free-flow speed from stop-bar COUNT ON times ~15 s into green (20-ft
cars, percentile), × advance→stop-bar time for isolated off-peak vehicles; stop-bar detectors = 0 (clip).
Final report must show accuracy before AND after label correction and by function type.
**VD audit DONE (note 34): 0 wrong channels** (1,674 confirmed / 953 plausible / 4 suspect of
2,631 labelled on 197 VD prints); VD n → MaxTime channel is a fixed slot relabelling and the physical
slot → MaxTime table holds for old Voyage cabinets (printed phase matches it 94.9 % vs 26.5 % for
Voyage order) — the user's open question is answered by the data. Card rule: 302 slots / 604 channels
`card_suspect` (both outputs dead or erratic) → merged into the label-check cleansing.
**Label-check rebuild (v2) DONE (note 29):** 847 of 11,357 fail (7.5 %); rules AUC .86–.99
(advance-before-stop-bar .99, presence holds call through red at peak .94); card faults merged; 60
phase flags on 48 signals. **USER REVIEW PENDING** (`review/USER_INPUT.md` items 1–3: spot-check
`review/spotcheck_label_checks.xlsx`; not_checkable rows usable?; loop count rule info-only?) →
**retrain PAUSED** until answered. cab_final step 8 stalled because the user had the dead-list csv open
— re-run `cab_final.py --skip-dq` when closed; new failures sheet is `review/label_check_failures_new.xlsx`
(replace the old one after). Locked-print reading continues (not affected). **Locked prints:**
scripted pass done (note 35; opt-in `--locked-labels-only`, separate store `dc_work/cabinet_locked/`,
182/186 PDFs, 10 visual batches); **ALL 10 locked visual batches DONE; consolidation + split DONE (note 36).** Locked answer key
`research/labels/function_labels_locked_v1.parquet` (3,677 rows / 182 signals; labels only). Split
seed 20260924: 71 NEWTEST released (`newtest_released.csv`), `locked_v2.csv` = 43 TEST + 72 NEWTEST =
115 (AGENTS.md updated). Released rows appended to v3 (now 16,759 rows; `released_from_newtest`,
validated = pending), `funcframe_v6` (456,033 rows), `folds_v4.csv`. Uncoded-zone rule also applied to
the training store (38 detectors). **cab_final.py --skip-dq must be re-run** once the user closes
`review/dead_detectors_from_prints.csv` (PermissionError). **Retrain (after the user's review):**
`v3_retrain.py --frame v6 --require-validated` (+ validate released rows first).
**2026-09-28 night:** label-check v3 DONE (note 29): fail 4.5 % (was 7.5 %), new status `unhealthy`
(78, actuation-based, excluded from training but not a label verdict), FYA/protected-permissive phases
detected (343) and exempted from the red rules, order+occupancy pair check, fault events gone;
`review/field_issues_for_staff.xlsx` (859 rows); new 10-row spot-check + 15-row health spot-check for
the user (USER_INPUT). Defaults adopted (user didn't object): not_checkable rows with high-confidence
print labels and healthy → train; loop count rule info-only; setback = loop-only add-on.
**Function retrain on label-check v3 DONE (note 28):** first.all.wi +3.92 pt 7-class [+2.51,+5.56]
vs current head (.8710 → .9102), core-four +2.45, AGR +2.28, all folds positive; WI non-PM 72.6 → 92.8 %;
FYA phases still ~5 pt harder (research item). Admitting not_checkable-high rows adds +0.85 pt.
Refit in `dc_work/final_v3_work/function_v3b/`. Follow-up DONE: released rows validated (787 pass / 52 fail
of 1,095), not_checkable with field_issue excluded, re-run +3.85 pt [+2.41,+5.49] (AGR +2.13), final
function model `dc_work/final_v3_work/function_v3c/` (current best). To-do: note 29 still says
released rows pending; card/DQ tables never run on released signals; overrides/dead lists exclude them.
**Setback for all advance DONE (note 41 setback):** single `distance_ft` (+ setback_confidence),
predicted-function inputs, OOF median 20 ft / 71 % within ±25 % (loop 17, video 23, radar 47;
>200 ft 55 ft; high-confidence 3 ft); ~96 % coverage; agency-specific (retrain per agency). Report all,
flag low confidence (orchestrator call). Needs numpy port of LightGBM/sklearn parts for model/.
**Lane output DONE (note 42):** `research/code/lanes/lane_output.py` (numpy/pandas): n_lanes exact
.80 (30 min) → .88 (full), ±1 .99; same-lane precision/recall .96/.94 at full; detector lane set right
.88–.93; lane 1 = busiest lane (not geometric). Goes into final_v3 (orchestrator call). Running: label checks v4 + retrain + feature test,
health v3.
**Health v3 DONE (note 43):** rules + predicted phase/function; choppiness vs same-phase (15-min),
bike-aware silence, stuck-on with bad periods and per-episode recovery; false alarms 1.0 % at 2 h;
output `dc_work/health/health_v3.parquet`. Training cleansing: exclude `bad`, drop bad periods of
`suspect` (orchestrator call). User review (optional) `review/spotcheck_health.xlsx` (10 rows).
Still running: label checks v4 + retrain + feature test.
**Label check v4 + honest retrain DONE (notes 44, 45):** permissive phases from behaviour (AUC .95),
right-turn exemption, 80 `misconfigured` excluded from training, 2B143 rulings. **HONEST headline
(all labelled, only >= 5 actuations): new function model +2.70 pt [+2.10,+3.33] vs current head
(.8423 → .8693), core-four +1.34**; healthy-only +2.72; the clean-label subset (+3.90) overstates by
~1.2 pt — always quote A. Refit `dc_work/final_v3_work/function_v3d/` (current best). Label-check
features: +0.04 overall → dropped (small real gain on permissive phases only). Next: option-2 test
(classify → health w/ model confidence → drop bad periods → reclassify), HELD until the user's health
spot-check answers are in; option 3 (joint model) infeasible without real health labels.
**User authorised (2026-09-29) a one-shot pull of the latest staging data** (for health and
training): 7 days, all 945 signals, into `dc_work/data/staging_2026_w40/` — running. Also running:
health v4 (user ideas incl. rolling function-consistency) + option-2 test.
**Health v4 + option 2 DONE (note 46):** kept recurring-spike rule (cleared only with >= 4 days),
recovery vs best-tracking sibling, multi-lane rapid limits; false alarms 0.95 % at 2 h; user answers
19/19; stable across Sept 18–21 → 26–28 (94 % bad stay bad). Rolling function-consistency: not a
rule; kept as informational "watch" field (orchestrator call). Model confidence as health input:
dropped. **Option 2 (classify → health → drop bad periods → reclassify): DROPPED** (function
−0.02 to −0.04 pt, phase −0.16…+0.05 pt; ceiling ~0.1–0.17 pt). Function accuracy by health status:
ok .887 / suspect .848 / bad .739. Scoring rule approved by the user: always two numbers (realistic,
everything). New data: `dc_work/data/staging_2026_w40/` (Sept 26–28; staging keeps ~3 days); asked the
user about a second one-time pull ~Oct 2. User review: `review/spotcheck_health.xlsx` (8 rows).
**final_v3 candidate v2 BUILT (note 48):** `dc_work/final_v3_candidate_v2/` — phase_v3 + net all lengths
(K=4), function_v3d, lanes, setback, health v5 (no fault events); numpy only; check.py 3/3. Six-fold
OOF vs final_v2: phase 30 min −0.03 (n.s.), 6 h +0.39 [+0.23,+0.58], full +0.13 (n.s.); function
+2.70 everything / +2.87 realistic, core-four +1.34/+1.50. Runtime 0.8 s (30 min) – 3.1 s (24 h)/signal.
Open: function OOF used stage-12 phase inputs (not phase_v3); setback only measured at 66 h; locked
confirmation NOT run (user decides when). Running: permissive-phase recognition test.
**Permissive-phase features DROPPED (note 50):** a hi-res-only permissive score works (OOF AUC .98
all phases, .82–.88 within left turns) but adds nothing to the function head overall (+0.00) and the
FYA-subset gain is inside re-fit noise (shuffled control +0.31 on that subset) — so note 45's "+0.31
on FYA" is unconfirmed too. Possible by-product: per-phase "permissive" output. Running: function
refit on phase_v3 inputs (function_v3e).
**Function on phase_v3 inputs DONE (note 49):** trees-only input = note 45 (no change); blend input
+0.11 pt (mostly 5 min) → function_v3e; vs final_v2 head +2.78 everything / +2.96 realistic.
Orchestrator call: swap candidate to v3e + blend input (running).
**PLAN APPROVED 2026-09-30 — component validation (the candidate was built by accretion; re-validate
every part on the current labels/data).** Criterion: optimal performance (keep complexity only if it
pays beyond noise; phase bar 0.3 pt, function 1 pt; seed re-fits; both scoring numbers; 5 min / 30 min /
6 h / full). Use the TCN for all network research; GRU only at the end. Order:
1. Function error review (where the ~13 % errors are) — directs everything else.
2. Trees optimised alone (phase + function): feature groups, joint decoder, 3-seed averaging (a training
   detail — keep only if it pays).
3. TCN optimised alone: 0.5 s steps first (0.1 s only if 0.5 helps), input channels, + engineered features,
   phase-only / function-only / joint heads. Abandon function-on-TCN early if one fold shows no promise.
4. Head-to-head: trees vs TCN vs combined, for phase and function.
5. Lanes & setback: separate models vs outputs of the main model.
6. Pending: lanes as function inputs, radar-ETA class, health as input. (Health rules were already
   validated one at a time — notes 38/40/43/46/47 — do NOT redo.)
7. GRU vs TCN on the final set-up; deterministic night-time speed (advance distance / travel time);
   then the user's error-review spreadsheet (grouped by signal, ranked by confidence, charts + print
   links, excluding known mislabels and explained cases such as switch/additional call phases or
   matching overlap calls).
**Progress (2026-09-30 evening):** step 1 DONE (note 51): realistic error 11.6 % = label noise 21 %
(stale Dec-2024 roles 0.87 pt), ambiguous-by-definition 44 % (superseded live loops, YR≡Count
duplicates, advance-presence/long zones, Mid/series), model-fixable 36 % (short samples 1.5 pt, core-class
confusions 2.4 pt, mostly radar); longer than 3 h gains nothing. **User decided (2026-09-30 night; AGENTS.md 'ATSPM classes, one per lane'):** stacked detectors keep their real
class, per-lane uniqueness of ATSPM classes in the output, set-based scoring, ATSPM-only function score,
YR≡Count duplicates excluded. Building `review/stacked_detectors_review.xlsx` for the user. Next: implement
the rule (labels + per-lane decode + scoring) and resume the function work. **Data:** user asked (2026-09-30) for a DAILY staging pull every morning until enough data, limited to
train+test DeviceIds (895), and pruning of other devices — task `dc_staging_daily`, ends 2026-10-14.
Dec-role filter DONE (note 52; `v3_retrain.py --dec-role`): re-score +1.24 pt realistic. **Orchestrator
call: use only the function/phase-differs rule** (the 'absent from export' rule is weak evidence, half the
rows, mostly correct) — apply when the function retrain resumes. Running: TCN phase optimisation
(note 53), daily data pull set-up.
**Per-lane ATSPM rule IMPLEMENTED (note 54):** greedy per-lane decode (`research/code/lanes/atspm_decode.py`),
gated to windows >= 30 min (lane model weak below); stack-aware ATSPM-only scoring. Re-score of v3e (no
retrain): decode +0.18 pt overall (n.s.), +0.84 at full length; headline ATSPM .8892 everything / .9015
realistic with v3s truth + decode + Dec rule. Labels v3s (`stack_labels_v3s.py`) + retrain command ready;
**retrain WAITS for the user's review of review/stacked_detectors_review.xlsx** (then run with the user's
answers; blanks = pending). Open for the user: the decode gives a loser its next FREE class (may be
another ATSPM class) — scored better (.8892) than "loser → Other" (.8837); ask.
**Paper:** the user will publish (user 2026-09-30: agency and vendor names MAY appear in the paper and ledger; still never credentials, server/table/user names, raw label or event files). An experiment ledger (`research/ledger.md`: one row per experiment —
hypothesis, set-up, data/eval set, result with CI, verdict, note link) is started now and kept current;
the write-up comes after the final architecture. The process must be reusable by other agencies
(cleansing/data checks are agency-specific but the problems are general).
**Next candidate assembly:** final_v3 package = function_v3c + phase_v3 (780-signal trees/decoder/GRU)
+ net at all lengths K=4 + health (no fault events) — assemble when the user's spot-checks are back.
**User review in progress (2026-09-24 end of day):** spot-check rows 1–5 confirmed correct; rows
6–36 + questions 2 (not_checkable usable?) and 3 (loop count rule info-only?) on Monday. Function
retrain PAUSED until then. **Weekend jobs (independent of the review):** phase retrain with the 71
released signals (note 37) — **DONE, verdict NEUTRAL**: blend Δ vs stage 13 on the same rows −0.09 …
+0.05 pt, every 95 % CI contains 0 (30 min −0.01); trees unchanged within recipe noise. Final trees
+ decoder fitted on 780 signals (`dc_work/final_v3_work/phase_v3/*_v5.*`). **Final GRU fit FAILED (host
OOM, "bad allocation")** — at the time another, non-project process (`run_report.py`, system Python,
16.6 GB) held RAM; not ours, left alone. To finish: when RAM is free, `python
research/code/neural/phase_v3_chain.py` from `dc_work/final_v3_work/phase_v3/repo_snapshot/` (resumes at
train_final → export). **Armed:** `dc_work/final_v3_work/phase_v3/wait_and_finish.py` (detached; checks
free RAM every 10 min, runs the chain once >= 14 GB free, gives up after 72 h; logs
`dc_work/logs/phase_v3_wait.log`, `phase_v3_chain3.log`). Note: phase fold map vs folds_v4 disagree on 22 old NEWTRAIN signals (phase OOF
kept the stage-13 map); detector health model — **DONE
(note 38):** rules (`research/code/health/health_core.py`, hi-res only, 0.06–0.16 s/signal) catch
99 % of dead detectors at 6 h, 1.1 % false alarms at 2–6 h, 41 % → 96 % of known-problem detectors at
6 h → 66 h; synthetic-fault LightGBM better on synthetic faults but worse on real ones ⇒ rules only
(pending user spot-check `review/spotcheck_health.xlsx`, USER_INPUT item 6). Orchestrator call: group
silences reported once per signal as likely cabinet/comms. **Actuation-level follow-up (note 40):**
kept a `rapid` rule (re-triggers < 0.5–1 s, bursts) — +2.0 pt catches at 2 h at +0.3 pt false alarms,
nothing at 24 h+; flow–occupancy (Greenshields), histogram distance, rhythm etc. dropped (no better than
loosening the old rules). Kept in health_core (0.07 s/signal at 2 h, 0.48 s at 66 h). **User 2026-09-28: NO fault events
(83–88) in health or validation** — removed from health_core (note 38 before/after: false alarms ~same;
catches on rebuilt sets 25–28 % at 2–6 h, 62 % at 24 h, 96 % at 66 h); new spot-check
`review/spotcheck_health.xlsx` with chart + print links. To-do for final_v3: production
`model/health.py` still has a `controller_fault_events` flag → drop it in the candidate; research-only
`hb_model` still uses n_fault. The user said 2C009 d25 / 2C036 d2 show no actuations on their side —
our log copy has 5,337 ONs for 2C009 d25 over 66 h: check which data/window the user looks at. Limitation: a detector silent the whole
window has no log row — needs the channel list to be called dead; setback
distance by the user's physics method (note 39) — **DONE:** covers 47 % of advance detectors, 27 ft
median error (loops with a same-lane stop-bar partner 25 ft; radar 90, video 59); needs ~24 h; speed
is the bottleneck (count zones are pulse → no speed); asked the user (USER_INPUT item 5): offer as a
loop-only add-on or park.
Consolidation also: uncoded video stop-bar zones → by position (Presence 0–20 ft before / Count on-past);
14036 & 10067 uncoded long radar zones from the stop bar → Presence. Locked-store
consolidation to-do: uncoded long radar zones running upstream from the stop bar (e.g. 2B012 d2/8/16/22,
labelled Other/long_zone) → Presence (consistent with the user's long-presence ruling).
Readings to add to the NEXT user spot-check (not urgent, labels only): 04088 turn-lane/side-street
"Count" bars drawn at the upstream end of the presence zones (labelled Advance by position);
2B344 zone table swaps ch 15/27 vs diagram + config (went with position). The locked store needs its
own consolidation later (cab_final refuses locked mode by design). 2B471/2B490 have no controller
timing (no phase target) — they can't be scored for phase.


User questions and answers: `review/USER_INPUT.md` (nothing open as of 2026-09-23 evening).

**VISUAL PASS COMPLETE (2026-09-23 night): all 688 usable signals visual_done** (73 remaining =
62 no data + 11 no PDF). 68 signals gave few/no labels (32 no diagram / "see TRS#", 36 stale) —
`review/prints_missing_or_stale.csv`. **Asked the user (USER_INPUT items 1–4):** upstream P-coded
radar zones Other vs Advance (~250); ATSPM "A" stop-bar zones Presence vs Count (~15); 2B338 d10;
whether TRS plans / newer prints exist. **PAUSED until answered: the function retrain on print labels**
(answers change labels). **Consolidation DONE** (note 27; `research/code/cabinet/cab_final.py`, rulings in
`dc_work/cabinet/final_rulings.csv` — answering a user question = one line there + re-run): tiers
complete_high 124 / complete_mixed 383 / incomplete 181; 43 unusual_layout. 13,204 print detectors.
`research/labels/function_labels_v3.parquet`: 15,479 rows, 715 signals, train_use 12,057, columns
`label_print_first` and `label_print_agree` (differ on 283 rows). Overrides 594
(`review/print_label_overrides.xlsx`), dead 936 on the maintenance list (46 all-silent/unusual signals
left out as stale prints — orchestrator OK'd). Unexplained channels at incomplete signals stay unlabelled.
Radar-over-loops refinement restored 25 live loops. **User answered Q1–4 (2026-09-23 late):** P-coded upstream radar = Other; ATSPM "A" stop-bar
~20 ft = Presence; lone only-upstream loop = Advance (2B338 d10 + similar); stale/no-diagram signals
stay out. APPLIED (pending_user = 0; user_ruling now outranks all label sources). Dead list refined
(`dead_list()` in cab_final.py): 712 rows / 239 signals, 403 ATSPM-function. **Running: full function
retrain grid** (`v3_retrain.py all`, results → note 28). **User decisions (late):** training/test
labels = `label_print_first` (print wins wherever high confidence; "agree" variant rejected as
circular); no training/scoring on rows without data (>= 5 actuations; retrain agent verifying);
dead list: keep not-in-timing detectors marked `in_timing = no`, drop abandoned + stale-print
signals; overrides xlsx sorted by a review-priority weight (agent reworking both files).
`review/USER_INPUT.md` is PLAIN TEXT (user request; first line `EDITING` = user is in it, don't touch).
**User decision (end of day 2026-09-23):** validate every label with the user's behaviour rules
(presence occupied through red + flatlines at peak vs same-lane advance; count off in red, grows with
volume; YR late on first cars vs co-located count; advance leads stop bar, counts more at peak; mid ≈
sum of advance off-peak; bike far below vehicles; health). **Train and test ONLY on detectors that
pass**; failures → `review/label_check_failures.xlsx` for the user (expects <= ~10 % to fail).
**Label validation DONE** (note 29; `research/code/cabinet/label_check.py`, step 7 of cab_final):
first pass 967 of 12,297 checkable fail (7.9 %); rules kept with separation AUC .77–.98, YR
first-car check dropped (AUC .68). Orchestrator decisions: retrain uses pass-only
(`train_use_validated`); health short-fault count = info only; 2C023 not unusual; two weak rules
("Count grows at peak", "Presence undercounts at peak") demoted to info. **Final: 883 of 12,274
fail (7.2 %; clear print 5.2 %, config 20.2 %); train_use_validated = 10,389 (pass-only).** Review
list `review/label_check_failures.xlsx` (883 rows). **Function retrain on validated labels DONE (note 28): first.all.wi = new function
candidate** — FIXb 7-class .9179 vs current head .8765, **+4.14 pt [+2.62, +5.81]**, core-four +2.73
[+1.22, +4.44], positive on all 6 folds, seed sd .03; label-neutral AGR set +2.49 pt; ~1/5 of the gain
is recalibration; whole-intersection Other predicted non-PM 64.7 % → 93.7 %; Bike P/R .88/.88, Mid
.97/.88. v2 control −0.17 (harness is neutral). **final_v3 candidate package DONE (note 32)** in
`dc_work/final_v3_candidate/` (function head first.all.wi, net cut-off 120 → 360 min, GRU kept; note 32). check.py passes (3/3); expert features = training frame to 3e-16; **6-h sample costs 6.0 s/signal (> 5 s)** — options in note 32 (`max_chunks=4` 2.1 s, needs an OOF check; or cut-off 180). Locked signals not run.
**Pieces test DONE (note 33):** GRU on K=4 evenly spaced 30-min pieces = full-length within 0.06 pt
at 3 h / 6 h / 24 h / full (blend − trees +0.35 / +0.32 / +0.14 / +0.18 pt), 2.1 s/signal at any
length. **Decision: net on at ALL lengths, max_chunks=4 above 120 min (no cut-off)** — APPLIED to the
candidate (check.py 3/3; s/signal 30 min 0.55 / 3 h 2.12 / 6 h 2.11 / 24 h 2.46 vs final_v2 0.55 /
0.11 / 0.13 / 0.36; 24-h peak RAM now 1.92 GB vs final_v2 1.57 after fetching only the pieces' intervals + dropping the
onnx session after use; outputs bit-identical; arena-off variant (1.68 GB, +0.6 s) not adopted). In-sample
smoke vs final_v2 (not evidence): 3 h −6 of ~522 detectors, 6 h +1, 66 h −3 — watch in the locked run.
**Candidate is otherwise READY; only the locked-signal confirmation remains (paused on USER_INPUT item 0).**
**ASKED (USER_INPUT item 0, awaiting answer; the locked-signal confirmation is PAUSED until then):** (a) OK to run the one
confirmation; (b) whether to first mine the cabinet prints of the 186 locked signals (same pipeline) so
the exam's function truth is print-validated like training — otherwise the exam uses config + round-1
labels, which fail validation ~20 % of the time. **Retrain STOPPED** (killed cleanly, no results; note 28) until validation
lands; then run `v3_retrain.py all --min-on 5 --clean --require-validated --variants
first.high.wi,first.high.nowi,first.mixed.wi,first.mixed.nowi,first.all.wi,v2.all.nowi` (~1.2 h) and
analyse with `research/code/trackA/v3_analyse.py` (copied from the stopped agent's scratchpad; it sets
MIN_ON and CLEAN but NOT REQ_VAL — add it). Early unfiltered seed-0 peek, not a result: FIX acc7 −0.8 pt
vs baseline (CI −4.3..+2.0), whole-intersection Other rows predicted Other 89 % vs 68 %. Then then lanes with
print lane labels, distance first look, final_v3 assembly. User is away until tomorrow morning —
continue autonomously. **Lanes with print lane labels DONE** (note 30): same-lane pair GBM AUC .965 (8,178 pairs,
425 signals; best single cue = 1-min detrended off-peak count corr .947); n_lanes per phase .81–.87
exact / .98–.99 within ±1. **Function effect: none** (constraint decode −1.5 to −2.3 pt; lane features
add nothing over a probability stacker) — lanes dropped for function a third time; keep as review /
DQ aid. Aside: recalibrating the v2-trained head onto print labels gains ~2.7 pt — belongs to the v3
retrain. **Setback distance DONE** (note 31): LightGBM 25 ft median error / 65 %
within ±25 % on 572 advance detectors (7 ft for loops with a same-lane stop-bar partner); mostly
learns the agency's design standard (phase mean green = 48 % of gain); physics-only travel-time
estimate 26 ft with a partner. Asked the user (USER_INPUT item 1) whether a rough estimate is useful;
recommended: later add-on, partner-only, not in the main model.
Retrain filters added: exclude dq_suspect rows (496 in train_use) from training + scoring; 7 signals
flagged unusual only in record notes (01064, 04073, 10086, 12032, 13025, 2C023, 2C042) treated as
unusual. **To-do after the retrain:** make cab_build read `unusual_layout` from record notes (or set
signal_flags on those 7) and rebuild v3 so the table itself carries the flag. Training data: v2 7,514
usable detectors / 407 signals → v3 12,057 / 638 (+60 %). Overrides: 594 of 5,234 dual-labelled
(559 real disagreements, 10.7 %).
Retrain harness built and smoke-tested (note 28; `research/code/trackA/v3_retrain.py all`, ~1.5 h
CPU). **Frame extension DONE:** `funcframe_v5` (701 signals, 432,890 rows; v4 rows byte-identical),
`dc_work/folds_v3.csv` (709 signals, new signals keep their stage-12/13 NEWTRAIN folds), phase preds for
new rows = stage-12 OOF trees (slightly stronger on Sept data than the old rows' source — known
mismatch). Coverage: complete_high 124/124, complete_mixed 383/383, incomplete 175/181. `v3_retrain.py`
defaults to `--frame v5`. **Retrain plan (after the user answers Q1–3 AND the frame extension is done):**
function head on v3, 7 classes, expert features, compare label_print_first vs label_print_agree,
complete_high only vs complete_high+mixed, whole-intersection Other rows in/out; unusual_layout
excluded from training, scored separately; six signal-grouped folds, 3 seeds.
**Cabinet visual pass — history (2026-09-23 evening).** 35 batches in `dc_work/cabinet/batches/`
(25 ordinary × ~24, 26–28 = 336 cabinets × 7, 29–35 = complex × 8–9). **Done: 01–23, 26** (~600 signals;
recent batches 70–77 % high confidence; batch 26 = 336 prints: **336 slot table 0 mismatches** vs live
channels). **Also done: 24, 26, 27, 28 (336 table 0 mismatches on all three 336 batches). 25, 29–32 done too. Running: 33, 34,
35 (the last) + the **consolidation build** (`research/code/cabinet/cab_final.py`, label table
`research/labels/function_labels_v3.parquet`, `review/print_label_overrides.xlsx`,
`review/dead_detectors_from_prints.csv`, note 27) — it tests on finished batches and then waits to be
messaged for the final pass once 33–35 are done.** Batch 33 done. End-question additions: 12052 "A"(ATSPM)-coded
per-lane stop-bar zones −2…18 ft (labelled Presence); upstream P-coded radar zones labelled Other leave
some phases with no Advance (ask whether P-coded upstream zones should be Advance when they are the
only upstream detection). Stale-site list for the end questions also includes
12055/12056 (thermal video; zone table numbers don't match live channels 1–8/33–44). Consolidation check: 2C009 channels 4/11/18/25/35/40 are config "Queue Dummy" but
real zones on the print — make sure cab_build does not drop them as derived. For the end question batch: newer prints/zone configs for stale sites whose
live channels no longer match the print (12004, 12053, 12015, 2B021 + the "see TRS#" prints)? Final pass: flag 2B060 `unusual_layout` (side streets rewired after the print). Consolidated so far: 633 signals, 13,319 detector rows. **Nothing paused — no open
question affects running work** (AGENTS.md "Waiting on the user"). End-of-mining question list
(minor, marked for bulk relabel): lone ~75 ft set-back loop that is the approach's only upstream loop
(2B338 d10; now Other/setback); upstream P-coded radar zones = Other/advance_presence (252 zones, only
6 called Advance by the config). For the final question batch: prints that say "see TRS#…"
(no diagram, ~2–3 per batch) — does the user have access to those TRS detection plans? Next: 28–35 (~4 at a time; each
agent prompt = the standard visual-pass prompt, 336/complex ones add the 336-table caution).
Usable ≈ 688 signals = 761 non-locked − 11 no PDF − 62 no hi-res data (`batch_nodata.txt`, skipped).
**Rules the user settled (all in the guide + the final sections of `VISUAL_PASS_INSTRUCTIONS.md`):**
loops never stop-bar Count; stop-bar Count only radar/video; **a zone the print codes YR is
Yellow_Red whatever the technology** (else Yellow_Red = radar); table-coded P zones are Presence even
drawn long — only a WRITTEN extent > 20 ft makes a zone Other; series loop → Mid; set-back / queue /
long call zones → Other; radar over loops → radar wins, loops Other/superseded_by_radar (radar
dilemma/flasher zones are the exception: Other); ATSPM diagram sheet = the detectors to label;
check every page; clearly labelled zones = high; crops show label + stop bar; signal flag
`unusual_layout` (live radar over live loops, multi-intersection, heavily rewired) → excluded from
training, scored separately. Orchestrator rulings: undrawn radar CO/YR/P zones labelled from the
xlsm code at low confidence (`position_not_drawn`); upstream P-coded radar zones = Other/advance_presence.
**Bulk fixes (logged in `dc_work/cabinet/sweep_changes.csv`) — re-run ALL on 24–35 at the end:**
sweep of 01–12 (note 26; high 2,692 → 3,382; 30 of 68 "not drawn" prints had zones on another sheet);
`fix_long_presence.py --batches 20-35 --skip "" --apply` (234 → Presence on 01–19; **also convert the
61 whose only Presence wording is the config description** — orchestrator decision); `fix_video_yr.py
--batches 24-35 --apply` (22 video zones → Yellow_Red on 01–23; validator now allows YR on non-loop).
**Then the consolidation step:** `cab_build.py build` → tiers (complete_high / complete_mixed /
incomplete, + unusual_layout column), whole-intersection Other rows, DQ re-run with print locations
(`dq_core.py`), dead-detector list for maintenance (`review/dead_detectors_from_prints.csv`), overrides
list (`review/print_label_overrides.xlsx`: every hand label a high-confidence print label + agreeing
OOF model prediction overrides), then retrain function on complete_high (7-class, expert features,
whole-intersection Other) and report tier counts to the user.
**Later work (user idea):** predict detector setback distance — `distance_ft` is collected; later test
e.g. free-flow advance→stop-bar travel time as an estimator, or a second model.
**GPU:** the B10/B7/six-fold chain died in the reboot at INFER fold 2; relaunched detached
(`repo_local/research/code/neural/b10b7_chain.py`, resumable). Its agent is gone: when
`dc_work/trackB/eval/` has the eval_full outputs, someone must write note 21 (B10 fold 0 +0.07 pt
at 30 min, B7/S4D +0.10 pt — both failed the 0.3 bar, from `logs/trackB_b10b7_chain.log`). **If the session died (e.g. a reboot):** re-launch visual agents
for any batch whose signals are not all `visual_done` (records are saved per signal, so a batch just
resumes), and check the B10/B7/six-fold GPU chain (`dc_work/logs/trackB_*`, note 21 not yet written)
— it is resumable; re-run its chain script. Orchestrator rule (instructions addendum): undrawn radar
CO*/YR*/P* zones are labelled from their xlsm code at LOW confidence, flag `position_not_drawn`
(batch 08 left them null — the sweep must apply the rule there). Done 09–11 too. Sweep also: fix
`cab_record.py finish` to accept `_approach.png` for `position_not_drawn` Count/YR (batch 11 saved
approach views under `_d<det>.png` — exclude any `position_not_drawn` crop from spot-checks); CO*-bar
at the upstream end of an A* zone = Advance (consistent across batches 03/11). Sweep items so far:
`position_not_drawn` radar sites (labels from xlsm/description, assumed stop-bar position — batch 05
saved 36 Count "crops" that show only the approach: exclude them from any spot-check / high tier);
no-diagram prints ("see TRS#", ~2–3 per batch) are null; bike zones counting like vehicles flagged. Rules added to `VISUAL_PASS_INSTRUCTIONS.md` as
they came up (series loops, multi-zone channels, presence_20_75, VD# ≠ channel on video prints, radar
A*/CO* by position). **Before consolidation, a consistency sweep over all finished records is
required:** re-check batches 01–05 video prints for VD#-as-channel errors (scripted parser mis-lists
them), unify series-loop / A*-CO* / setback labelling, then bulk-apply the user's answers.

## Decisions pending from the user

* None open (see `review/USER_INPUT.md`). Round-2 review superseded by the print mining; cabinet
  prints = go (2026-09-23).
* (Expert features +0.16 pt: nothing ships during the search — they ride along with the round-2
  function retrain and are judged at the end.)

## Parked ideas (user's, not yet tried)

* **Preload the training rasters into GPU memory.** The TCN is small (700 k params, ~2 GB used of
  8 GB) and GPU utilisation drops to 0 % between steps while the CPU builds batches — training is
  data-bound, not compute-bound. Building each fold's rasters once and keeping them resident on the
  GPU could cut epoch time substantially. Try when GPU throughput becomes the bottleneck.

## Uncommitted work (the user commits)

Untracked/modified as of session 3: `research/STATUS.md`, `research/notes/17_lanes_redo.md`,
`18_trackB_b9_neighbour_trees.md`, `research/code/trackA/`, `research/code/trackB/`,
`research/code/neural/trackb_b3b4.py`, `research/code/README.md`. End-of-search clean-up item: the
agency's name appears in ~20 tracked files (model card, `common.py` wiring table, notes 01/04,
README) — pre-existing; decide at the end whether a name mention counts as "agency-specific".

## Never forget

Locked signals (43 TEST + 143 NEWTEST, paths in `AGENTS.md`) are never trained/tuned on. One-time
downloads only. No polling, no login prompts, quiet waiting, sample-based charts, fastest verified
runtime for scoring. Nothing ODOT-specific committed. No Co-Authored-By lines.

## 2026-09-30 evening — user answers folded in
* Loser class: next-free-class (user agreed). Stacked spreadsheet retired; no per-row answers needed.
* New pick rule inside the per-lane decode (user): (1) unhealthy detector loses (health_core v5, hi-res only);
  (2) advance radar spans all lanes — never its own lane; healthy lane-by-lane advance loops beat it;
  (3) where stop-bar Count zones exist, the advance whose counts track them best wins. Scoring unchanged
  (any member of a stacked group correct). Radar-over-loop signals re-admitted.
* USER_INPUT rewritten: nothing open. Running: TCN note 53 (GPU); function retrain v3s + pick rule (note 55).
## 2026-09-30 night — note 55 done (function v3s retrain + stacked pick rule)
* v3s retrain: run `run_51ba131222_exclude_min5_clean_valnc_h3_drfp` (frame v6e), lanes `ln5_lanes_v3s`. ATSPM .8910 / .9036
  (everything / realistic), +0.18 / +0.21 pt vs v3e, CI above 0 -> the function baseline from now on (nothing shipped).
* Pick rule in `lanes/atspm_decode.py` (`pick=`; inputs `lanes/ln6_pick.py` -> `lanes/ln6_pick_v3s.parquet`; eval
  `evaluation/atspm_pick55.py`): stack-scoped version costs -0.06 pt ATSPM, picks the lane-by-lane member in 88 % of
  true-stack span contests (greedy 79 %), halves extra lanes; health key inert; lane-count plausibility flat. Default
  decode stays greedy (orchestrator decides whether pick becomes the default). health_core KeyError 'pulse_frac' on 164
  short windows (bug, not fixed).

## 2026-09-30 night — note 56 done (pick = default, stack-relative health, span v2)
* health_core KeyError 'pulse_frac' fixed (research copy; quiet windows with no detector at min_on).
* Default decode is now the stack-scoped pick (`atspm_decode.stack_pick`, `atspm_score.py` PRIMARY `pick_stack`,
  `--no-pick` = note 54): ATSPM .8904 / .9031 (-0.06 pt vs greedy, accepted). Key (a) replaced by the stack-relative
  health flag (`lanes/ln7_stackhealth.py` -> `lanes/ln7_stackhealth_v3s.parquet`: chi vs a shared reference, member vs
  partner): precision .26 vs .19, false-flag 1.1 % vs 1.7 %, ATSPM neutral; 13025 det 39 flagged, 28 wins its windows.
* Span v2 (side-by-side coincidence): flag precision .81 -> .92 (A/P/C .53 -> .64), +0.05 pt ATSPM vs pick, but fewer
  lane-by-lane wins (768 -> 750 / 868) -> NOT default; option `span_key` / `sp_*` columns. Eval `evaluation/atspm_pick56.py`.
* Open for the orchestrator: the pick's cost is the loser rule (in-stack loser -> next free ATSPM class, e.g. radar
  Advance -> Yellow_Red; stacked A->wrongA 53 -> 72). Test "in-stack loser -> best non-ATSPM class" (needs the user's
  OK to revise "next free class").
* Note 56b: in-stack loser -> non-ATSPM. All classes -0.45 pt (dropped); Advance/Presence only -0.03 pt [-0.07, -0.00]
  but stacked members .7869 -> .8048 and 13025 det 39 -> Other. Default unchanged (next free class); option
  `stack_loser="nonatspm_ap"` awaits the orchestrator's call.
* 2026-10-01 night, orchestrator: in-stack loser default -> "nonatspm_ap" (user's stated stack behaviour: the loser
  is Other; +1.8 pt on stacked members, -0.03 pt overall, within the seed spread). Non-stack losers keep next free class.
  Default flipped in atspm_decode.py line 178; earlier numbers reproduce with stack_loser="next_free".
## 2026-10-01 — note 57 done (validation plan step 2: trees optimised alone)
* Phase (folds_v4, 772 signals): full 261-feature 3-seed bag + decoder .9696 E / .9723 R at 30 min. Decoder +1.5 pt (keep).
  Phase-call 43/44 features carry the trees (-12 pt without). Six feature groups dropped jointly (261 -> 68) cost 0.12 pt at
  30 min; 3-seed bag +0.1 pt: both real, under the 0.3 bar. Simplest by the bar: 68 features, 1 seed, + decoder.
* Function (v3s, pick decode, stack_loser nonatspm_ap for every number): full 442-feature 3 seeds .8901 E / .9029 R. Lag
  (-1.85) and sibling (-0.44) families pay; px expert real but 0.16; 213 of 442 columns add nothing (229-feature head -0.04).
  Trees-only phase input -0.13, ATSPM weighting / pooled rule nothing, 3 seeds +0.10. Simplest: 229 features, 1 seed.
* OOF for step 4 in `%DC_WORK%/trees57/` (both simplest and full arms). Orchestrator call: keep rule = the bar (simplest arms)
  or "beyond noise" (full 3-seed arms, +0.12 phase / +0.11 function).
* 2026-10-01, orchestrator decision on note 57 keep rule: the goal is optimal performance, so complexity stays when it
  pays beyond noise (CI clear of 0); the 0.3 / 1-pt bars screen NEW additions only. Trees side for step 4:
  PHASE = all 261 features, 3 seeds, decoder (dropping six groups costs -0.12 [-0.24,-0.01]; 3 seeds +0.09-0.13, CI>0).
  FUNCTION = 229 features (six families dropped, -0.04 [-0.14,+0.06], n.s.), 3 seeds (+0.10 [0.06,0.14]), blend phase input.
  Paper item: trees lose 12 pt at 30 min without phase-call events 43/44 — agencies that do not log them need a fallback.
## 2026-10-01 — note 58 done (validation plan step 5: lanes and setback)
* Lanes (v3s print lanes, 504 signals; inputs v6e phase + note-57 229-feature function arm): a function-free grouping fails
  (n_lanes exact .715 vs .846, ATSPM -3.4 pt) — lanes need the function head's roles. Function model alone (no lane cues)
  or a count rule lose too. NEW joint pair model D = lane cues + pair type + the function head's 229 features and probs of both
  detectors: n_lanes .854 (+0.80 [+0.34,+1.27]), lane set .922 (+1.08), pair acc .923 (+2.3), permuted control = baseline;
  ATSPM through the stack-pick decode .8914 / .9042 vs as was .8897 / .9027 (+0.17 [+0.07,+0.27]; vs refit A +0.09 [+0.00,+0.18]).
  Kept as the lane model by the keep rule. Refit of the old model on v3s truth changes nothing.
* Setback (by sample length, first time below 66 h): as is (60 features) 25.7 / 22.4 / 22.6 / 20.4 ft medAE at m30 / h6 /
  h24 / full, ~70 % within 50 ft; function / lane inputs pay +1.9 pt within 50 ft [+0.3,+3.7], travel-time block +1.6 -> kept
  as is. Function-free fallback = own-behaviour-only model (-1.6 pt). Radar ~65 ft.
* OOF for steps 4 / 7: `%DC_WORK%/lanes/ln8/lanes_D.func.parquet` (+ A.func, B.func, B.free, C1.rule; ln5 format, windows
  >= 30 min) and `%DC_WORK%/trackA/setback/sb7/oof.parquet`. ln6 pick inputs were not rebuilt from the new lanes (approximation).
* 2026-10-01, orchestrator on note 58: lane model = joint model D (lane outputs gain beyond noise); setback as is.
  Pick inputs get rebuilt on D lanes in the step-4 pipeline (exact numbers there, not now).
## 2026-10-01 — note 59 done (validation plan step 6: pending additions)
* Exact baseline with pick inputs rebuilt on D lanes (`lanes/ln6_pick_D229`, `ln7_stackhealth_D229`): ATSPM .8914 E / .9042 R
  (old inputs -0.003, harmless). Single seeds .8903-.8905. Reference for step 7.
* Screen (six folds, seed 0 vs base seed 0, shuffled control per family; bar 1 pt): NOTHING clears it -> all dropped.
  (a) lanes as inputs, properly nested (training rows' lanes from function + D refits without the outer fold): +0.02 overall,
  +0.17 [+0.04,+0.29] at >= 30 min but -0.37 at 5/10 min; control flat. (b) radar-ETA class (29 detectors; + advance_presence
  338) -0.10 / -0.08. (c) health as input, function-free (no circularity) -0.05, = control. (d) off-peak-only features -0.04,
  off-peak weighting -0.08 (Sept sample is two-thirds weekend: off-peak ~ whole window).
* Code `trackA/s59_step6.py`; work `%DC_WORK%/s59/`. Open for the orchestrator: a >= 30-min-only lane-feature head would keep
  ~+0.2 pt without the short-window cost (not tried: under the bar, would be tuning).
* 2026-10-01, orchestrator on note 59: all four step-6 additions dropped (screen rule). Step-7 reference = .8914 E / .9042 R.
  Deferred, not tuned: re-test off-peak features once the weekday daily pulls (Oct 1-14) are in, since Sept was 2/3 weekend.
## 2026-10-01 — note 60 done (step 7 part: deterministic night-time speed)
* `trackA/sp1_night_speed.py` (numpy only, not trained): advance setback / travel time to the D-lane stop-bar mate, isolated
  00-05 h vehicles, green at both, chance-match background test, mode-anchored median + n + p25/p75. Answers 23 % of phases
  with an Advance at 66 h, 19-20 % at 24 h, 17 % with 2 h of night, 0 without night; through phases with a stop-bar mate
  78-84 %. Shifted-stop-bar control 0.1 %. Median 30 mph (approach speed, 68-75 % in 25-55); error = setback error (14 %);
  two nights agree to 1.5 mph. No posted-speed truth in any export. Side output, nothing shipped. Open for the
  orchestrator: offer to the user as a mainline add-on with the setback, or park.
* 2026-10-01, orchestrator on note 60: night speed kept as a side output in the final candidate (NaN with reason
  when no night / no stop-bar mate). Shown to the user with the final review, not a decision now.
## 2026-10-01 — note 61 done (ATSPM-among-ATSPM confusions)
* Diagnosis on the note-59 baseline (.8914 / .9042): A->wrongA = 3.39 pt (R) — YR->Count .79, Adv->Pres .66, Count->YR .55,
  Pres->Adv .43. YR<->C is radar, half on 5/10-min windows, 631 swaps; radar YR and Count twins log the same vehicles (no
  speed-filter signature). A<->P (loop / video) misses behave as the other class (top features reversed inside the errors).
* Targeted features (`trackA/s61_atspm_conf.py`, work `%DC_WORK%/s61/`): rl (release / red onset) dropped; tw (stop-bar twin
  asymmetries, 15 cols) 3 seeds .8919 E / .9045 R (+0.05 n.s.), YR<->C errors -0.09 pt [+0.02,+0.16] (CI clear, 3 seeds agree),
  A<->P +0.03 pt worse. Under the 1-pt bar. Open for the orchestrator: keep tw (targeted gain, no overall loss) or drop by the bar.
* 2026-10-01, orchestrator on note 61: tw dropped (overall n.s.), rl dropped. Asked user (USER_INPUT): score radar
  YR<->Count swaps between near-identical twins as correct? Scoring-only question; no training paused on it.
## 2026-10-01 — note 62 done (function on short samples, 5 / 10 min)
* Twin decode (`trackA/s62_short.py`, work `%DC_WORK%/s62/`): on 5/10-min windows, within stop-bar zones whose ONs co-actuate
  both ways (chance-corrected match >= .4, hi-res only) at most one Count and one Yellow_Red; loser swaps Count<->YR, else keeps.
  No refit. 3 seeds .8928 E / .9057 R (+0.14 [+0.10,+0.18]); non-twin control -0.37; nested threshold picks .4/.5.
* Lane gate stays 30 min: pair same-lane accuracy m5 .71 / m10 .85 / m30 .92; ln5 decode at m10 -1.4 pt there.
* Training mix: a second head with short rows weighted 4, used only for samples <= 10 min: +0.08 [+0.05,+0.12] (3 seeds);
  one weighted head for all lengths or a non-ATSPM prior on short rows: nothing.
* Combined (3 seeds): ATSPM .8936 E / .9064 R, +0.22 [+0.17,+0.27]; m5 .8607 -> .8710, m10 .8732 -> .8788; >= 30 min unchanged.
  Under the 1-pt bar, CI > 0. Open for the orchestrator: adopt (cost = one extra LightGBM head + a numpy rule), or only the
  twin decode (+0.14, no model). Overlaps the open user scoring question on radar YR<->Count twin swaps.
* 2026-10-01, orchestrator on note 62: ADOPT both (routed short head <=10 min, weight 4 + short-window twin decode,
  threshold .4): .8936 E / .9064 R, +0.22 [+0.17,+0.27], 3 seeds. New function reference for later steps.
## 2026-10-01 — note 63 done (TCN function head + trees: honest blend, stackers, error analysis; fold 0 only)
* Headline >= 30 min (user rule): fj (joint head) blended with the trees at tree weight 0.6 (nested signal-grouped CV inside
  fold 0, chosen in 14/15 splits): +0.89 [+0.19,+1.71] E / +0.76 [+0.14,+1.50] R vs trees .8922 / .8987; 5/10 min +0.76;
  all windows +0.85. ff (function-only head) about half (+0.49 [+0.02,+1.04]). Additive to note 62 (fj on top +0.84 >= 30,
  +0.74..+0.86 at 5/10 min). Control: trees blended with another tree arm -0.30..+0.07 -> the gain is the net's information.
* Stackers (LGB / logistic; LGB + context = health, pick inputs, D lanes, lane- / phase-mates' probs, user idea) +0.23..+1.10
  at >= 30 min with CIs ~2.5 pt wide; none beats the fixed w 0.6. Context stacker worth a retest only with six-fold rows.
* Fixes: non-pulse stop-bar zones the trees called Other / wrong stop-bar class (radar Presence, radar / video Count);
  breaks: Advance<->Other (video), radar pulse YR->Count at 5/10 min. Count error 8.2->5.8 %, Presence 6.5->4.8 %.
* Open for the orchestrator: six-fold function-head (fj) run. Fold-0 >= 30-min +0.89 is just under the 1-pt bar, clear of noise.
* 2026-10-01 noon, orchestrator on note 63: promote fj (joint head) + fixed blend tree weight 0.6 to six folds x 3 seeds
  (fold 0 >=30 min +0.89 [+0.19,+1.71]); ff dropped; context stacker revisit only with six-fold rows. RunPod = yes,
  waiting on user sign-in; local GPU running fj folds 1-2 seed 0 (run_fj_more.ps1).
* Note 61b (user cues, judged on the >= 30-min headline): lane-spanning (sp) and downstream-lag (lg) features for YR vs Count.
  Cues are real in the log (YR later than its twin 56-57 % vs Count 16-22 %) but add nothing at >= 30 min (sp+lg +0.01
  [-0.20,+0.19], YR<->C flat) -> dropped; gains only at 5 min (+0.33..+0.40, CI > 0). Note-61 tw also fails at >= 30 min
  (-0.02 n.s.; short-only +0.17 [+0.03,+0.32]). Spanning YR taking Count in the decode: 33 rows (0.01 pt), no change.
  For the note-62 short-window agent: features in `%DC_WORK%/s61/feats.parquet` (tw_ / sp_ / lg_ columns, KEY-aligned).
## 2026-10-01 — note 64 done (integrated candidate scorer + final error-review builder)
* `evaluation/cand64.py all` (one command; re-run as fj folds arrive, ~3 min after the first run) -> `%DC_WORK%/cand64/`
  (per-row OOF phase_rows / function_rows incl. lanes D, setback sb7, health v5, night speed; headline.json). Reproduces notes
  57 / 48 / 62 / 63 exactly. Headline >= 30 min: phase .9818 [.9787,.9849] E / .9842 R; function ATSPM .9009 [.8894,.9114] E /
  .9137 R; 5 / 10 min phase .9646 / .9726 E, function .8662 / .8776 E. fj (fold 0 only so far): +0.89 [+0.18,+1.71] at >= 30.
* `evaluation/review64.py` (dry run default; `--preview N` into dc_work; `--write` = review/, NOT run): 651 rows / 250
  signals before a confidence cut, 234 at >= .8, 134 at >= .9 (`cand64/review_dryrun.json`). Open for the orchestrator: the
  cut, and whether to wait for fj six folds (`--arm fj`) before writing the sheet.
* 2026-10-01, orchestrator on note 64: review sheet waits for fj on all six folds; then --min-conf 0.9, collapse
  same-signal same-pattern rows (e.g. 03104's 11 radar Other) into one item. Current headline >=30 min (cand64):
  phase .9818 E / .9842 R; function .9009 E / .9137 R (trees, no fj).
* 2026-10-01, note 64 addendum: final_v2 (published beta) added to cand64 as a reference arm (`cand64.py v2ref`, saved OOF, headline.json `final_v2_ref*`). Candidate - final_v2, E, >= 30 min: phase +0.16 [+0.07,+0.26], function +2.72 [+1.93,+3.51] (cand+fj on folds 0,1,3,4,5: +2.96); 5 / 10 min: phase +0.05 / -0.01 n.s., function +3.64 / +3.25.
## 2026-10-01 — note 65 done (fj six folds x 3 seeds on RunPod; cand64 headline)
* Cloud: 1x RTX 5090 community ($0.69/h), pod 11:28-14:01 PDT, **spent $1.78** (balance $20.00 -> $18.22), terminated
  (API: no pods). 18 runs (2 validation + 16 queue), 4 at a time, 19 s/epoch (local ~60). Upload 2.1 GB took ~55 min at
  ~90 KB/s per stream (parallel chunks). fpreds `tcn53/fpreds/fj[_s1|_s2]_f{0..5}` complete. Copies, logs and checkpoints
  in `tcn53/cloud65/`. Package fix needed: tcn53_func train must mkdir `runs/`.
* Validation fold 0: cloud runs agree with local as closely as two local runs agree (.85 top-class). Single-seed spread is
  ~0.5 pt in the blend, so note 63's fold-0 seed was lucky (+0.77 vs cloud +0.34 / +0.21 at 0.5 blend).
* Six-fold fj blend (w 0.6, 3 seeds averaged): >= 30 min .9009 -> .9048 E, +0.39 [+0.23,+0.56] E / +0.38 R; 5 min +1.13,
  10 min +0.87, all +0.56. Per seed +0.36 / +0.32 / +0.35. Per fold +0.22..+0.67. Open for the orchestrator: adopt or drop
  by the 1-pt bar (CI > 0 everywhere; ~+1 pt on 5/10 min).
* 2026-10-01 14:15, orchestrator on note 65: ADOPT fj blend (tree weight 0.6, 3 seeds) for function at all lengths:
  >=30 min +0.39 [+0.23,+0.56], every fold and seed positive; promoted-then-confirmed beyond noise. Package bug
  (runs/ mkdir) to fix in cloud_pkg. Next CPU: six-fold context stacker + Count/Presence regression vs final_v2.
## 2026-10-01 — note 68 stub (GRU joint function head gj): STOPPED by the orchestrator, nothing scored
* `tcn53_func.py --arch gru` ready (fp32 GRU backbone: bf16 autocast put cuDNN's GRU on a ~14x slow path; infer also
  writes phase-head `ppreds/`; runs/ mkdir fixed). RunPod network volume **rvwhubuncy** (dc-cloud-pkg, 20 GB, US-NE-1, kept,
  ~$1.4/month) holds the package + deps + worker scripts. **Spent $0.50** (balance $18.225 -> $17.725); all pods terminated.
* 2026-10-01 15:15, PLAN (orchestrator, shown to user): phase work stopped (ceiling). Function: (1) tune the
  function net on RunPod (note 69), per-class reporting; (2) specialists + tree combiner only if variants differ by
  class beyond noise (user idea; six-fold stacker result decides feasibility); (3) CPU: Count/Presence slip (note 67),
  extra weekday days (note 66); (4) stop after two rounds < noise (~Oct 3); (5) lock, integrated score, GRU vs TCN on
  final net (~Oct 4); (6) error-review sheet (~Oct 5); (7) exam on user go. Few planned variants, all folds/seeds agree.
* 2026-10-01 15:40, USER: two use cases — FULL (best accuracy, desktop) and LIGHT (<= ~1 GB RAM; start from beta-sized
  trees + cheap gains). Track RAM peak + s/signal per component; build both at lock time; one if full is small enough.
  Note 70 launched: re-measure error decomposition on the current candidate (note 51's split is stale).
## 2026-10-01 — note 70 done (function error decomposition redone on the current candidate; analysis only)
* `evaluation/err70.py` (note 51's rules + B5 stack_extra, D = Bike/Mid<->PM) on cand64 function_rows (trees + fj), >= 30 min.
  Realistic 8.26 pt error (acc .9174): (a) label 1.52 (18 %), (b) ambiguous by definition 3.21 (39 %), (c) model-fixable
  3.30 (40 %), (d) 0.23. Everything 9.52: 2.97 / 3.10 / 3.23 / 0.22.
* vs note 51 restated on today's score (v3e, R 10.26): label / scoring fixes -1.11 (mostly (a): Dec stale .90 -> .30), model
  -0.89 (mostly (b): Other-like subtypes 2.52 -> 1.44, Mid .85 -> .64). (c) flat: PM<->PM 1.44 -> .93 but ->Other .65 -> 1.12
  (stop-bar Presence / Advance demoted to Other by the lane decode / pick; undecoded tree argmax right on 38 %, p < .5 on 69 %).
* Levers left in (c): ->Other demotions 1.1 pt, 30 min / 1 h misses 1.1, PM<->PM .9. Ceiling ~.95 R with clean labels + half
  of (c); (b) 3.2 pt only with a definition / scoring decision. Remaining (a): config-only confident .49, Dec stale radar
  Count not caught by the filter .33. Error rows `%DC_WORK%/err70/errors_*.parquet`.
## 2026-10-01 — note 72 done (label review sheet v1 for the user)
* `evaluation/review72.py` -> `review/function_label_review_v1.xlsx` (+ `_charts/`): 38 rows / 30 signals, likely-wrong labels only (20 config-only label contradicted, 18 model p >= .9; 31 function, 7 phase); definitional rows (open question), unhealthy, user-ruled excluded. Built 15:15 PDT in 52 s. p >= .8 would add ~29 items (v1b not built). Waiting on the user's L / M / ? answers -> next label table.
## 2026-10-01 — note 67 done (six-fold decider; Count / Presence vs final_v2; decode demotions)
* `evaluation/s67_decider.py`, work `%DC_WORK%/s67/` (base reproduces cand64 trees+fj exactly). All results >= 30 min, E / R.
* A: the context stacker (probs + 42 hi-res context cols, OOF over folds, 3 seeds) beats the fixed 0.6 blend by +0.48 [+0.28,+0.68]
  E / +0.35 [+0.16,+0.54] R, all folds positive. Shuffled-context control +0.23; context - control +0.26 [+0.13,+0.38].
  Probs-only stacker +0.26.
* B: Count -0.79 [-2.54,+1.10] / Presence -0.30 vs final_v2 = noise. The cause is the per-lane decode (Count -> YR on radar
  pulse, Presence -> Other); final_v2 has no decode. Audit with note 70: 1.43 pt of true ATSPM (R) are demoted with argmax
  right, 55 % lost to a same-truth lane-mate (D merged lanes). Mechanism: one-lane conflict 50 %, stack-loser 56b 27 %,
  spanning 20 %.
* Fixes, nested: class weights 0, stack-loser next_free +0.04, tau +0.06, D lane-confidence gate +0.25 (blend) / +0.42
  [+0.25,+0.60] (stacker).
* Combined stacker + gate: .9048 -> .9134 E, +0.86 [+0.59,+1.13] (R +0.72), all folds positive; Count / Presence above
  final_v2; YR -1.0, non-ATSPM -2.3 vs candidate.
* Open for the orchestrator: adopt (each piece CI > 0; combined under the 1-pt bar). Gate .9 is at the grid edge.
* 2026-10-01 ~15:30, orchestrator on note 67: ADOPT context stacker (3 seeds) + lane-confidence gate (one-per-lane binds
  only where D lane_conf >= .9): >=30 min .9048 -> .9134 E (+0.86 [+0.59,+1.13]), R +0.72, every fold. Drop class
  weights / tau / loser change. Follow-ups: widen gate grid (> .9); nested-stacking caveat noted — locked exam checks it.
* 2026-10-01, note 67b (orchestrator follow-up): wider gate grid on the context stacker is flat .90-.97 (.9136 E >= 30), .99
  .9130; decode off .9127. Nested gate +0.42 [+0.23,+0.61] E / +0.40 R (unchanged); decode off +0.36 [+0.04,+0.66], less
  steady and costs YR 3.5 / non-ATSPM 4.4 pt. Keep gate .9.
## 2026-10-01 — note 71 done (peak memory / time per signal; analysis only)
* One signal per call, fresh process: peak final_v2 3 h 0.15-0.18 GB (no GRU above 2 h), 30 min up to 1.39 GB (GRU); candidate
  3 h / 6 h / 24 h 0.44-1.47 GB (busiest 43 ch 1.45-1.47), 0.43 GB at GRU pair batch 64. Note 48's 1.92 / 1.57 GB were per
  20-signal batch. fj ONNX +0.9 / +2.7 s, ~0.2 GB; D lanes ~0.1-0.2 s (est.). Champion fits a 4 GB app on 3 h; for LIGHT,
  pair batch 64 (output identity to be checked) and ONNX not torch for fj. `evaluation/bench71.py`, `%DC_WORK%/bench71/`.
* 2026-10-01, orchestrator on note 71: per-signal RAM fits the user's 4 GB app at 3 h (candidate <=1.5 GB busiest;
  pair batch 64 -> 0.43 GB, verify identical output at lock). Plan: one model; ship fj as ONNX; light model only if
  final package exceeds ~1.5 GB per signal.
* 2026-10-01, USER: speed target <= 4-5 s per signal for a 3-h sample (candidate measured 6-17 s under contention,
  beta ~2 s). Primary model first; then speed work (ONNX fj, fewer pieces, int8, distillation, TCN vs GRU speed);
  if the full model can't meet it, a separate fast model. Report s/signal at 3 h for every candidate.
  YR-without-Count: 43 labelled phases / 16 signals (mostly video whole-intersections) asked to user.
## 2026-10-01 — note 66 done (more weekday data per detector for function)
* Mon 28 - Wed 30 Sept from the daily pull (the frame's stg = Fri 18 - Mon 21 Sept; w40 days were in no frame), 22 windows
  mirroring stg, features via the candidate package on whole-log tables (229 cols identical to frame v6e on 953 rows),
  `trackA/s66_moredata.py`, work `%DC_WORK%/s66/`. +190,522 training rows (+67 %), training only, cand64 trees arm scoring.
* >= 30 min: 3 seeds .9009 -> .9013 E, +0.04 [-0.04,+0.13] (R same); seed 0 +0.09 [-0.00,+0.19]; 5 min +0.18, 10 min +0.02
  -> dropped (noise-sized): function is not data-limited per detector.
* Secondary: same detectors on weekday windows score -0.36 [-0.54,-0.19] pt lower than on Sept windows at >= 30 min (full
  -0.80); weekday training recovers ~0.1. Note-59 off-peak features re-tested with weekday data: -0.02 vs control -0.03, dropped.
* Open for the orchestrator: none needed; optional one-line to the user that the Sept headline is slightly optimistic for
  weekday samples (~0.3 pt).
* 2026-10-01 ~17:30: harness flagged the note-69 cloud agent ("Security Weaken"). Reviewed: no API-key leak, SSH
  config uses accept-new + dedicated known_hosts; likely trigger = /usr/bin/sleep used to bypass the tool's sleep block
  in fetch69b.sh. Agent told to stop and never work around harness/permission restrictions; asked what else it changed.
* 2026-10-01 17:45, USER: away until Monday morning (Oct 5). Continue autonomously; RunPod until credit runs out,
  then local. Partial answers in review/function_label_review_v1.xlsx (a few rows) — read at the next label pass,
  don't overwrite. Monday: put a SHORT summary at the top of USER_INPUT (best numbers vs beta, changes, decisions).
## 2026-10-01 — note 73 done (per-component timing profile of the champion, one signal; no model change)
* `evaluation/prof73.py`, work `%DC_WORK%/prof73/`. Package wrapped + research proxies (fj ONNX, D lanes, ln6/ln7 pick inputs,
  s67 stacker / decode, sp1). Warm total, s: typical 30 min 2.36 / 3 h 5.56, busiest (43 ch) 30 min 3.51 / 3 h 9.10; cold +1.1-1.4
  (imports .6-.7, model parsing .42). At 3 h: GRU ONNX 2.78 / 5.19, fj ONNX 1.01 / 1.89, numpy trees ~0.85, all else ~1 s.
* GRU pair batch 64 = 512 bit-identical (p_gru and full predict() output, 6 / 6 cases). fj exported to ONNX (one graph incl.
  pooling + head, `%DC_WORK%/prof73/fj_f0.onnx`): vs torch on real inputs |dlogp| <= 1.2e-5 -> PASS 1e-4.
* Speed ideas (measured on both 3-h signals): (1) GRU only on candidates with tree p >= .01 (19-27 % of pairs): -4.2 / -2.1 s,
  pre-decoder top phase unchanged 65/65 -> needs a six-fold OOF check (CPU, saved OOF); (2) same filter for fj: ~-1.4 / -0.75 s
  est., needs fj re-inference; (3) LightGBM -> ONNX TreeEnsemble (converter in prof73): 50-100x, |diff| 2.4e-7, ~-0.8 s.
  int8 quantisation: no gain (GRU), 18x slower (fj). Projection 1+3: busiest 3 h ~4.0 s, typical ~2.6 s.
* 2026-10-01 ~18:00, orchestrator on note 73: champion 3 h warm 5.6 s typical / 9.1 s busiest (networks 66-78 %).
  Approved: six-fold OOF check of GRU candidate filter (tree prob >= .01; -4.2 s busiest). Adopted for lock: ONNX
  tree ensembles (50-100x faster, identical to 2e-7), GRU pair batch 64 (bit-identical). int8 dropped. fj filter
  needs a cloud re-inference; fj phase head replacing GRU needs fj phase OOF (ppreds now written).
* 2026-10-01, note 73b (orchestrator approved idea 1 check; idea 3, ONNX tree ensembles, ADOPTED for lock time): GRU only on
  candidates with tree p >= thr, six-fold OOF through the cand64 phase pipeline (decoder refit): thr .003 / .01 / .03 keep
  42 / 32 / 25 % of GRU pairs; >= 30 min E -0.007 / -0.001 [-0.024,+0.019] / +0.003, R -0.004 / +0.002 / +0.009 pt, all CIs ~+-0.02 and
  containing 0; 5 / 10 min n.s. Top phase changes on 0.18 % of >= 30-min det-windows at .01 (function bound; not re-scored).
  K = 2 pieces not testable on p3 OOF (note 33, older GRU: -0.05..-0.23 pt). Recommend thr .01 at lock (~-65 % GRU time).
* 2026-10-01 ~18:45, orchestrator on note 73b: ADOPT GRU candidate filter tree prob >= .01 at lock (phase neutral,
  CIs ±0.02 pt; ~-65 % GRU time). Projection with ONNX trees: 3 h warm ~4.0 s busiest / ~2.6 s typical. fj filter
  (idea 2) to test in the next cloud/local fj re-inference.
* 2026-10-01 17:50: second harness flag on the note-69 cloud agent ("Interfere With Workloads"): its guard terminated
  ALL pods on the account in a loop (only its own existed; verified). Rule: terminate pods by explicit id only.
  All pods down; 42 checkpoints copied local (md5 verified); balance $3.60; volume kept.
* 2026-10-01, adopted combo (context stacker 3 seeds + gate .9 on all folds) vs final_v2, paired >= 30 min E (141,842 rows, 383 sig.): Advance .9092 -> .9386, Presence .9391 -> .9440, Count .8941 -> .9129, YR .6416 -> .8389, non-ATSPM .7263 -> .8303, overall .8739 -> .9139 (+4.00 [+3.24,+4.76]).
## 2026-10-01 — note 69 done (function net tuned on RunPod; sibling attention wins)
* 13 variants screened on folds 0 + 3 x seed 0 (cand64 fixed blend, vs fj seed 0; seed noise +-0.06 pt): only sibling
  context pays (sibm +0.31, siba +0.21 at >= 30 min; ~+0.7-0.9 at 5/10 min). Finer rasters (0.5/0.2/0.1 s), aug, capacity,
  long windows, label smoothing, K = 12 pieces: flat or worse -> dropped. cw2 = Advance specialist (+0.49) but nonATSPM -0.64.
* siba six folds x 3 seeds: >= 30 min .9048 -> .9077 E, +0.29 [+0.17,+0.41] (R +0.27), 5/10 min +0.86/+0.90, every fold and
  seed positive. Inside the note-67 champion (stacker + gate .9): .9143 -> .9169 E, +0.26 [+0.10,+0.42], fold 1 -0.42.
* Open for the orchestrator: adopt siba in place of fj (blend and stacker); sibm (cheaper) only screened, not promoted.
  All 44 variant OOF preds kept (`tcn53/fpreds|ppreds/x69_*`) for a specialist / combiner study.
* RunPod: H100 SXM + RTX PRO 6000 (+ 10-min MIG pod for checkpoint copy). **Spent $14.05** (balance $17.56 -> $3.51), no
  pods left, volume rvwhubuncy kept; all 42 runs finished, checkpoints copied to `tcn53/cloud69/models/` (md5 verified).
* 2026-10-01 18:05, orchestrator on note 69: ADOPT siba (sibling attention) in place of fj in blend + context stacker:
  champion >=30 min .9143 -> .9169 E (+0.26 [+0.10,+0.42]), R +0.26; 5/6 folds positive. Weekend (local GPU,
  detached runs): sibm 6x3, siba+cw2 6x3, then specialist context stacker (siba + sibm + cw2 + trees), then GRU
  backbone of the final head (6 folds x 1 seed first). Note 74.
* 2026-10-02, orchestrator on note 75: package dc_work/final_v3_candidate_v3 built (check.py 3/3; parity 741/741).
  3 h warm: 2.5-3.0 s typical, 4.0-5.1 s busiest; RAM 0.54-0.79 GB. DECISION: ONE siba network (speed target
  4-5 s; three cost ~+5 s busiest) at -0.12 pt >=30 min; production net = full-data siba refit queued at end of q74.
  Phase-renumbering tie-break (<= .0055 prob, no answer changes) also in final_v2 — note only.
* 2026-10-03 09:15, note 74 steps 2-3: siba+cw2 dropped (-0.02 n.s.); specialist decider not adopted (+0.03 n.s.,
  diversity not specialism, 3x network cost). Step 4 GRU backbone running (~15 h), then siba full refit + filter.

## 2026-10-03 — note 78 done (plateau diagnosis + ceiling; analysis only, nothing trained for production)
* Champion R >= 30 error 7.18 pt: label 1.46 / definitional 3.24 / model-fixable 2.26 / other .22; everything set 8.29.
  The +1.1 pt since note 70 all came out of the model-fixable bucket; label and definitional buckets unchanged.
* Mislabel: label-caused error 0.4-1.5 pt (central ~0.9); print-high vs config disagree 10.3 %; behaviour-check fail 3.0 %.
* Learning curve (trees): ~+1.1 pt per doubling of labelled SIGNALS, not flat; more data per detector did not help (note 66).
* Ceiling (R >= 30): .935-.94 today's labels; .94-.955 with label fixes; .955-.97 if long / advance-presence zones are scored
  as their nearest class (open user question); ~.975 hard top. Further network / tuning work: a few tenths at most.
* Outputs %DC_WORK%/err78/; code evaluation/err78.py, lc78.py. User summary: last 8 lines of note 78 (orchestrator posts).

## 2026-10-03 — note 78b done (more print-labelled signals: feasibility, read-only)
* Outside frame + hold-outs: 118 devices; 36 with Sept 18-21 detector data (35 of them lack an official timing export), 65 no data.
  35 have prints (scripted: ok 29 / blank 1 / no text 5) -> ~26 usable signals, ~400-450 high-confidence labels, ~5-6 agent-hours.
* Gain on note 78's slope: +0.03-0.06 pt (champion), unmeasurable vs the 1-pt screening rule. Events already local (no event pull);
  phase truth would need a one-time timing-export re-try (user OK). Recommendation: do not pursue for accuracy. Nothing paused.

## 2026-10-03 — note 79 done (detector health: evaluation set + real performance; analysis only, package untouched)
* Eval set `%DC_WORK%/health79/evalset.parquet` (code health/h79_evalset.py): dead / stuck / chatter / intermittent / degraded,
  tiers A independent (112 alive-Dec print-dead, 43 share fell > 5x since Dec, 25 user answers), B partly, C circular (66-h rules).
* Package health (prod call, scored on 26-28 Sep = independent of the 18-21 Sep labels), 30 min / 3 h / 24 h: FA .33 / 1.16 / 3.38 %;
  chatter 42 / 67 / 83 %; stuck 2 / 9 / 26 %; intermittent (6) 4 / 17 / 67 %; undercount 1 / 2 / 8 %; dead 0 % (invisible without a
  channel list; with it 39 / 94 / 97 %). Precision not measurable (no confirmed-healthy set); labelled-universe ranking only.
* Candidates: partner count ratio with per-function-pair limits = real (tier-A recall +7-8 pt, CI > 0) but +1 pt FA -> offered as an
  information note, not adopted (orchestrator). Green-miss asymmetry, within-sample drift: dropped. Learned scorer: too few labels.
* Cheapest user help to make health measurable: a short list of known-bad (type, dates) and known-good detectors for 26-28 Sep, and
  per-signal channel lists. Nothing paused.
* 2026-10-03, orchestrator on note 79: add partner-ratio undercount check as an INFO note in health_reason (not a
  status change) — do at next package rebuild. Asked user: optional channel list input for health (dead channels);
  small known-bad/known-good list. Paused on answer: dead-channel output only.
* 2026-10-03, orchestrator: build review v1b (p >= .8, ~29 items) AFTER note 80's label update lands, so the sheet
  reflects the new labels; ready for the user Monday.

## 2026-10-03 — note 79b done (does health help classification? analysis only, nothing changed)
* Inside the note-77 champion, six folds, 3 seeds: stacker without the 16 health columns -0.003 [-0.03,+0.02] pt (>= 30 E), shuffled
  +0.003, pick without the health key +0.001 (6 rows), both removed 0.000; every class / fold within noise; 10-min -0.06 fails its
  shuffled control. On flagged detectors the classifier is ~0.2 pt better WITHOUT health. Answer: health does not help
  classification. Orchestrator may drop it from the stacker + pick (refit 46-col stacker; arm a is that OOF). Nothing paused.
* 2026-10-03, orchestrator on note 79b: DROP the 15 health columns from the context stacker at next package rebuild
  (no measurable cost; decouples function from health). KEEP the stack-pick health key (user's explicit rule
  "unhealthy one loses"; harmless). Health stays a separate output + label-cleansing filter.

## 2026-10-03 — note 80 done (label pass v4l: released NEWTEST in the training pipeline, incomplete prints re-read)
* Labels `research/labels/function_labels_v4l.parquet` (v3s schema, `truth_v3s` = v4l truth); change list
  `research/labels/function_label_changes_v4l.csv` (339 rows, priority-sorted, cause per row). Store copy `%DC_WORK%/cabinet_v4l/`
  (training store + the 71 released records; hold-out locked_v2 asserted); old stores, review/ and v3 / v3s untouched.
* Champion >= 30 min E: rescore v3s -> v4l +0.20 [+0.06,+0.38] (measurement: wrong truth removed); trees + stacker refit on v4l
  +0.03 [-0.02,+0.09] (noise). Refits: f76/function_c_v4l, f77/s74/f80v4l; scoring `evaluation/s80.py` (score / fit / stack / report).
* RESUME (all steps idempotent, in order): `python cabinet/v4l_store.py setup` (skips if setup.done) -> re-reads via
  `DC_CAB_STORE=cabinet_v4l cab_record.py finish` (checkpoints `%DC_WORK%/lab80/reread/done/<DN>.txt`; lists list_1..3.txt; all 130
  done) -> `DC_CAB_STORE=cabinet_v4l python cabinet/v4l_expl_sweep.py --apply` -> `DC_CAB_STORE=cabinet_v4l python cabinet/cab_final.py`
  -> `DC_CAB_STORE=cabinet_v4l python cabinet/stack_labels_v3s.py` -> `python cabinet/v4l_rules.py apply` / `diff` ->
  `F76_ARM=c python evaluation/s80.py score|fit|stack|report` (delete the v4l output folders first to force a refit).
* Open for the orchestrator: adopt v4l as truth + training labels (recommended; to switch, point atspm_score.V3S /
  v3_retrain.LABEL_SETS["v3s"] at it); build review v1b on v4l; reader rule questions in note 80 section 6. Nothing paused.

## 2026-10-03 — note 81 done (v4l adopted as truth + training labels; review sheet v1b; user's v1 answers folded in)
* Every research scorer / trainer reads `rpath.LABELS_CURRENT` (= v4l); `t57_function.stage_fit` refuses to write a
  non-v3s fit into the v3s trees folders (set OUT / BASE_RUN like s80.stage_fit). v3s kept on disk for comparisons.
* v4l_rules.py: R4 user answers (2B091 out, 2C009 d5 / d19 phase confirmed -> `phase_user_confirmed`), R5 68 low-conf
  behaviour-tied readings out (24 signals), R6 10 noise inputs out; training rows 11,390 -> 11,322; truth unchanged.
* Headline on v4l (>= 30 min): .9190 E / .9289 R (row-identical to note 80's rescore). Nothing trained.
* Review v1b: review/function_label_review_v1b.xlsx, 32 rows / 29 signals (work: dc_work/rev81/). v1 read-only, md5
  unchanged. Nothing paused.

* 2026-10-03, orchestrator on note 81: v4l adopted everywhere (headline >=30 min .9190 E / .9289 R). v1b announced.
  Rule: when the user answers a v1/v1b row, apply the same answer to its listed same-pattern detectors
  (dc_work/rev81/review81_same_pattern_as_v1.csv). <15-ON rule kept as built.
* 2026-10-03, orchestrator on note 82: health sheet announced. At next package rebuild: (1) compare gaps in whole 0.1 s
  ticks (chatter 0.3 s, rapid 1.0/0.5 s) and recalibrate fire rates; (2) remove the "dead channel" check (log-only rule,
  user); (3) add "undercounts vs partner" as info note with the 20-actuation minimum. Tune limits from the user's Y/N.
* 2026-10-03, note 74 step 4: GRU backbone of siba -0.17 [-0.34,0.00] at >=30 min, slower -> TCN kept (end-of-search
  GRU check closed for function). Steps 1-3 verdicts hold on v4l (note 74b). siba full refit then filter check running.
* 2026-10-03 22:30, orchestrator: production network = x74_sibafull4l (v4l labels; consistent with the v4l stacker).
  Caveat: stacker learned on OOF from v3s-trained siba fold nets; a v4l six-fold siba re-run would make it fully
  consistent (~18 GPU-runs) — only if the locked-exam / time allows.

## 2026-10-04 — note 84 done (candidate v4b: 3 siba members + filter ON; not shipped, locked untouched)
* `dc_work/final_v3_candidate_v4b` (v4 untouched): members x74_sibafull4l / _s1 / _s2 (parity vs torch 2.1e-5, filtered
  5.5e-5), stacker 'mean3' (3-seed-mean OOF, 47 cols, v4l) + 'single' fallback, SIBA_FILTER default ON. check.py 4/4 PASS.
* OOF (six folds, v4l, filtered net seed 0 only): >= 30 min E mean3+filter .9191 vs single+filter .9174, +0.17
  [+0.06,+0.29]; R +0.17; 10 / 5 min +0.38 / +0.31; filter on mean3 -0.00; vs champion +0.01 n.s.
* Speed 3 h warm mean3+filter 2.99 / 2.47 typical, 4.06 / 3.42 busiest -> rule met -> mean3 is the default.
  Nothing paused. Optional: filtered OOF for seeds 1 / 2 (GPU ~2 h) to make the mean3+filter number fully clean.

## 2026-10-04 — note 83 done (candidate v4 package; not shipped, locked untouched)
* `dc_work/final_v3_candidate_v4`: every decision since note 77 (v4l refits, full-data v4l siba + stacker 'single'
  without health, health tick fix / no dead check / partner info note, siba filter switch OFF). check.py 4/4 PASS.
* OOF (six folds, v4l): >= 30 min E .9181 [.9070,.9279] / R .9279 [.9170,.9372] vs .9190 / .9289 (-0.09 [-0.18,0.00] /
  -0.10 [-0.18,-0.01]); 10 / 5 min -0.37 / -0.29 (CI < 0) = one siba member instead of three; health drop -0.01 n.s.
* Speed 3 h warm: typical 3.25 / 2.41 s, busiest 4.25 / 5.19 s; filter on 2.29 / 1.98, 2.67 / 3.33; peak <= 0.79 GB.
* Found + fixed: importing f76_function resets t57_function.OUT (v3s trees) -> fit83 asserts the stacker's tree dir.
  Nothing paused. Orchestrator: filter decision from the GPU check; swap into model/ only at the end of the search.

* 2026-10-04 early, orchestrator on note 83: candidate v4 built (check 4/4). OOF -0.09 vs 3-member champion (one siba).
  Plan once the GPU filter check lands: if the filter is neutral, turn it ON and test whether 3 siba members fit the
  4-5 s target with the filter (busiest 3.33 s with one member + filter) — that would win back the 0.09 pt.
* 2026-10-04 ~02:30, orchestrator on note 74b: siba candidate filter ADOPTED (neutral, ~-70 % siba time). GPU: full
  siba v4l refits seeds 1, 2 queued; then package v4 test of 3 members (mean3 stacker) + filter vs the 4-5 s target.
* 2026-10-04, orchestrator on note 84: CHAMPION PACKAGE = dc_work/final_v3_candidate_v4b (mean3 siba + both filters,
  v4l labels, no health columns in stacker). OOF >=30 min .9191 E (= research champion); 3 h warm 2.5-3.0 s typical,
  3.4-4.1 s busiest; RAM <= 1.0 GB; check 4/4. model/ still final_v2 (ship only on user word). Next: Monday summary.
* 2026-10-04 ~06:00, orchestrator on note 84b: v4b default confirmed with all seeds filtered (.9192 E / .9291 R).
  Started note 86 (GPU, ~21 h): siba fold nets on v4l + stacker refit, for train/production label consistency.
* 2026-10-04 ~15:00, USER: every discretionary label decision (exclusions by behaviour checks/rules, automatic changes)
  must be user-reviewable — sheet being built (label_discretion_review_v1). Phase TCN vs GRU on current pipeline + joint
  phase/function decider: note 87 running.
* 2026-10-04, USER RULE: if two options tie on accuracy (within noise), pick the FASTER one (e.g. TCN over GRU).
* 2026-10-04 evening, note 87: label-discretion sheet posted (1,001 rows). VIOLATION found: --clean DQ flag used fault
  events 84-88 -> 233 rows wrongly out of training; note 88 fixing + refit. Asked user: apply label-only rules to the
  locked exam key before the exam (no scoring).
* 2026-10-04, orchestrator on note 87b: phase network -> TCN ad_all (ties GRU .9816 vs .9818, 2.8x faster; user rule).
  Full-data phase TCN refit appended to the note-86 GPU queue; package swap (6 channels, decoder refit, ONNX, check,
  bench) after. Joint phase+function decider not adopted (+0.05 phase, -0.05 function).
* 2026-10-04, orchestrator on note 88: fault events removed from all live code; labels v4m current (+228 detectors back);
  package v4d = clean (check 4/4, .9193 E). Plan: note 89 decides exclusion groups -> final label table (v4n) ->
  final full-data refits of siba (x3) + phase TCN on v4n -> package rebuild -> user "ship"/"exam".
* 2026-10-04: note-88 follow-ups: rpath.LABELS_CURRENT -> final table from note 89 (v4n) when it lands (v4m until then);
  health limits from notes 38/43/79 were calibrated on a fault-event-shaped healthy reference — re-check them after
  the user's health-sheet answers. model/ final_v2 still reports controller_fault_events; goes away at ship.
* 2026-10-04 evening, orchestrator on note 89: all 8 exclusion groups restored (none hurt) -> labels v4n (1,123 still
  excluded vs 2,196). not_checkable group being tested next; then final siba x3 full refits on the final table (GPU,
  after note 86 + phase TCN), package rebuild. Stacker keeps note-86 OOF (v4l fold nets) — documented small mismatch.
* 2026-10-04 20:00: FINAL training labels = v4o (only 466 labelled detectors never trained). Final siba x3 on v4o queued
  after phase TCN on the GPU; then package rebuild (v4f) with phase TCN + v4o fits.

## 2026-10-05 — note 90 done (FINAL package v4f; not shipped, locked untouched)
* `dc_work/final_v3_candidate_v4f`: phase TCN ad_all (package raster = research render53 exactly, parity 0.0 / probs 1e-6),
  decoder 216 trees on the re-inferred TCN OOF; function refits on v4o. Phase tie with GRU (-0.02 [-0.07,+0.04] E), function
  = v4e (+0.03 [-0.06,+0.12] E). check.py 4/4. 3 h warm v4f / v4e paired: typical 2.80 / 3.38, busiest 3.78 / 4.38 s.
* 09:00: siba members swapped to v4o seeds 0 / 1 / 3 (seed 2 outlier by member agreement); parity 3e-6; check 4/4; speed same.
* 09:30: function OOF re-scored with the v4f (TCN) phase as the frame's phase input: >= 30 min E .9197 +0.01 [-0.02,+0.03],
  R .9292 +0.01 (oof90.py tcnphase) -> the v4f function headline stands.

* 2026-10-05, note 90: FINAL package dc_work/final_v3_candidate_v4f (phase TCN, v4o function refits), check 4/4;
  phase .9816 E (tie with GRU), function .9197 E / .9291 R; 3 h warm 2.4-2.8 s typical / 3.2-3.8 s busiest.
  Pending: swap in three v4o siba members when note 86's GPU queue lands them.

* 2026-10-05, note 92 (subagent, CPU): Other split into non-ATSPM subclasses (Upstream_presence, Setback_presence, Long_zone,
  Departure; training labels only, ATSPM labels untouched) on the v4f recipe: >= 30 min E -0.07 [-0.18,+0.04] (kway) / -0.04
  (sum) vs v4f .9197, shuffled control -0.13*; definitional errors 3,120 -> 3,198 / 3,040 rows. NOT adopted; v4f unchanged.
  Subclasses learnable (exact .71 / .73 / .36 / .40). Code research/code/final92/sub92.py, work dc_work/x92.
* 2026-10-05, note 92: Other subclasses (trees + stacker only) not adopted (-0.07 n.s.). Next: 2-fold GPU screen with
  the siba head ALSO trained on the 11-class labels (after the seed-3 refit), then decide.
* 2026-10-05 ~09:45: FINAL PACKAGE dc_work/final_v3_candidate_v4f COMPLETE: TCN phase, v4o function refits, siba v4o
  seeds 0/1/3 (seed 2 outlier). check 4/4; phase .9816 / function .9197 E; 3 h warm 2.5-2.8 / 3.3-3.7 s. Awaiting
  user "ship" / "run the exam". Open: 11-class siba screen (note 92b, GPU).
* 2026-10-05, note 92b: 11-class siba head not adopted. Research complete; final = v4f. GPU idle. Awaiting user.
* 2026-10-05 15:30, USER answers: (1) new print-vs-hand rule for train AND test labels: keep hand label if model
  agrees; overwrite with print only if model agrees with print. (2) health: don't bother removing; final output =
  prediction + health score so the app can filter; report accuracy by health. (3) YR-without-Count: decide with data
  first, sheet only for unresolved. (4) channels 40/41 dummy -> ignore. (5) exam key gets same label rules. (6) never
  ask "ship/exam" until research is actually done.
* 2026-10-05, note 82b (subagent, CPU): health review sheet v1 charts redrawn simply (no step plots; one panel, plain
  title; bars / ON timelines / histograms with limit line); same 70 PNG names and links; 4 user answers kept identical;
  backups review/_backup/*20261005_152234*. Sheet ready for the user's review again.
* 2026-10-05, note 82c (subagent, CPU): health review charts redrawn as LINE charts with hours of context around each
  flagged period (user feedback on 82b); same 70 PNG names / links; 4 user answers identical; backups *20261005_153624*.
* 2026-10-05 ~16:20, USER: 05999 det 23 -> exclude. 2B061: drop phase 4 only. Loops hand-labelled "advance presence"
  -> Advance (2B331 d22, 2B417 d8, 04026 d8/9/21 and the general case). Channel-text vs Function column conflicts ->
  model-agreement rule. Read the 3 renamed prints (5CE166, 04CA156, 5CE069) only if device matching is certain.
  FINAL TRAINING: use only 2026 data (drop the 2024/2025 pulls) for a cleaner report. Stop asking low-value
  single-detector questions. Order: health "day chart" column first, then labels, then 2026-only retrain.

* 2026-10-05, note 82d (subagent): health_review_v1.xlsx got a second chart column 'Day chart' (whole flagged day, 15-min
  lines); existing cells untouched; the 11 answers the user has typed so far identical; backup *20261005_155545*.
* 2026-10-05, USER (corrected): health sheet "N (its ok)" = detector OK (check fired wrongly); unanswered rows = not reviewed, NOT false alarms. Post the health study
  result in USER_INPUT when done. Wants the retrain cost explained before any RunPod top-up.
* 2026-10-05, note 96 (subagent, CPU, analysis only): context-aware health (user's 4 ideas). Occupancy per type + phase
  traffic reference + resolver R1-R9 / new watch N1 on top of the v4f findings (h96_*.py, dc_work/health96). Suspect+bad per
  100: 3 h 1.83 -> 1.40, 24 h 5.20 -> 3.73 (+ watch .47 / 1.12); user answers Y 37 -> 36 flagged, ? 15 -> 5, N 0. Sheet
  review/health_review_v2.xlsx (19 rows) for the user. ADOPTED for the next package build: R7 (single borderline statistical
  finding -> watch) and R9 (two-pass: detectors bad in pass 1 removed from phase mates' yardstick). PAUSED on the v2 answers:
  R1 (stuck in congestion), R2 (long-zone occupancy follows traffic), R4 (multi-lane volume), N1 (new occupancy note).
  R3 dropped unless v2 supports it. Unanswered v1 rows = not reviewed (not false alarms); "N (its ok)" = detector OK.
  No package touched yet.
* 2026-10-05 ~18:00, USER (away until Tue 07:00): finish the 2026-only final model by morning (local GPU + RunPod, funds
  added: $13.33), show test results (six-fold OOF; locked exam still needs an explicit "run the exam"), speed/resource
  comparison vs the beta, and if > 2x time or memory also a simplified variant optimising accuracy vs efficiency.
  Artifact with final results + architecture linked in USER_INPUT. Quality over rushing. Plan: note 95 (retrain:
  phase TCN local after test B, siba on RunPod + SWA experiment), note 98 (speed + fast variant, CPU), note 97 B
  (decided-phase channel, local GPU now). Orchestrator assembles the morning artifact.
* 2026-10-05, note 99 (subagent, CPU, analysis only): lane-count review sheet review/lane_count_review.xlsx (25 rows, 11 over / 14 under) of phases where OOF lanes D disagrees with the print lane count (majority over >= 30-min Sept samples). 291 / 2,515 phases wrong, of which 121 under-counts are blind (missing lane only spanning zones or a dead loop) and left out. Awaiting the user's answers; nothing changed.
* 2026-10-05 18:45: RunPod BLOCKED by the permission system for the retrain agent (credential access). Not worked around;
  surfaced to user. Fallback: all local; siba seed-0 test runs only; stacker on single-seed OOF.
* 2026-10-05 ~19:00, note 98: v4f vs beta at 3 h 4.7-6.2x time, 2.1-2.7x RAM -> FAST variant le2h (both networks only
  on samples <= 2 h): 1.5-1.8x time, ~1.1x RAM; phase -0.16, function -0.33 vs full; vs beta function +4.1, phase tie.
  Package dc_work/final_v3_candidate_v5_fast (check 5/5); re-base onto v5 is in the retrain agent's final steps.
* 2026-10-05 ~19:20, note 97: decided phase into function — A neutral (function side already uses the decided phase), B
  (siba + decided-phase channel, fold 0) within seed spread (+0.30 / -0.02 vs seeds 1/2). Not promoted. GPU -> q95.
* 2026-10-05 ~19:30, USER: explicitly ordered RunPod use. Orchestrator created pod 36dnz5k3frzgnz (dc95a, RTX PRO 6000,
  $2.09/h, volume rvwhubuncy); SSH handed to the retrain agent; orchestrator terminates by id when copy-back verified.
  USER: the "fast version" is the same model with networks skipped above 2 h — present it as a setting, not a model.
  Distillation not yet tried (int8 tried in note 73: no gain).
* 2026-10-05 23:25, note 95: siba done on cloud (18 folds + 3 full, copied back, verified). Function v5 (2026-only) vs v4f
  on 2026 rows: -0.11 [-0.33,+0.09] E, tie; R -0.00. SWA single run does not replace 3 copies (-0.36 vs mean3).
  Pod dc95a kept for the distillation test (note 100) until <= 02:30 (balance $7.95 at $2.10/h).
* 2026-10-05 23:55: FINAL PACKAGE v5 DONE: dc_work/final_v3_candidate_v5 (2026-only, labels v4q, check 4/4, no-spin):
  phase .9843 E / .9863 R, function .9305 E / .9395 R (v4q truth, 2026 windows); ties v4f. 3 h warm 1.6-2.9 s, <= 0.89 GB.
  Fast setting le2h (dc_work/final_v3_candidate_v5_fast, check 5/5): 3 h 0.57-0.94 s / 0.29-0.32 GB; -0.20 / -0.30.
  Next: note 101 final table vs beta; note 100 distillation (pod until 02:00); orchestrator builds the morning artifact.
* 2026-10-06 01:10: pod 36dnz5k3frzgnz terminated by id (all results copied back, verified). No pods left; balance $4.28;
  network volume rvwhubuncy kept. Note 100 (distillation) scoring on CPU.
* 2026-10-06 01:20, note 100: a plain width-32 siba net matches the 3-member mean (fold 0 blend +0.44 vs mean3) at ~1/15
  of its CPU cost (3 h: .035 / .31 s vs .52 / 4.8 s). Distillation itself not needed. Running six-fold w32 on local GPU
  (2026 setup) + stacker refit; progress by 06:00.
* 2026-10-06 03:40, USER asked to use remaining RunPod funds: pod 7i64g3ahns5g0e (dc100b) created for plain-w32 seeds 1/2
  x 6 folds; hard copy-back by 05:15; orchestrator terminates by id. Local GPU: w32 seed 0 (folds 0-2 done at 03:29).
* 2026-10-06 03:36: pod 7i64g3ahns5g0e terminated unused — the permission system denied the agent SSH to it (an
  orchestrator message can't grant that); not worked around. Balance $3.99. w32 continues locally (seed 0 ~04:45).
  For the user: agents need an explicit permission rule to SSH/use RunPod pods.
* 2026-10-06 03:55: orchestrator ran the w32 seeds 1/2 bundle on pod jn20ytctzdkh5j itself (user: "use it"); DATA_OK,
  12 jobs running. Fetch to dc_work/s100/cloud_b_out, md5 verify, terminate by id. Hard stop for cost ~05:30.
* 2026-10-06 04:30: cloud w32 seeds 1/2 done (12/12 ok), fetched + md5 OK to dc_work/s100/cloud_b_out/out; pod
  jn20ytctzdkh5j terminated; balance $2.59; no pods. Scoring w32 stacker vs v5 (note 100b).
* 2026-10-06 05:15, note 100b: w32 x3 ties v5 (+0.02 [-0.09,+0.14] at >= 30 min) and is ~1/3 faster, -25 % RAM ->
  ADOPTED (tie -> faster). Building package v5b (3 full w32 refits local GPU, stacker refit, check, bench, fast setting).
* 2026-10-06 06:40: CANDIDATE = dc_work/final_v3_candidate_v5b (+ v5b_fast, le2h default), check 4/4 and 5/5. Accuracy
  = v5 (.9307 E function, phase .9843). 3 h warm 1.02-1.78 s / 400-546 MB (fast .57-.95 s / 293-329 MB); beta
  .39-.57 s / 246-279 MB. Results page https://claude.ai/artifact/VFzAUxabD3NSCabmk9S98b updated. model/ = beta.
* 2026-10-06, note 102b: TR + R stop-bar split (corner widening) in the print-lane truth: 31 phases / 27 signals; 15 counted
  1 lane (13 as read + 2 user answers), 16 still counted 2 lanes (7 high-confidence prints, e.g. 01072 P8 = twin of the
  corrected P4), + 3 L/R stem splits. review/tr_r_lane_examples.xlsx (10 rows) for the user. Re-labelling the 2-lane ones
  (and any lane re-fit) waits on his answer.
* 2026-10-06 09:30, note 104: health v3 sheet ready (25 rows); resolver v104 (R1', R1b, N1', N3, N5) waits on user's v3
  answers. Orchestrator call: "ON logged again without OFF" on count zones becomes its own data-quality note, not N3.
* 2026-10-06, note 105 adopted: CANDIDATE = dc_work/final_v3_candidate_v5c (v5b + lane model refit on corrected lane truth
  v2; check 4/4; lane count .8509, function unchanged). Note 107 running: arrow-rule re-check of all multi-lane prints.
* 2026-10-06, orchestrator on note 107: user said overlap lanes may be counted or not -> keep the v2 count as truth with the
  other accepted (lenient). Apply only the 10 wide-lane misread fixes (truth v3b = v2 + those 10). No refit needed now
  (10 phases); include in the next lane refit. Lane error is mostly the model, not labels (91.5 % of prints agree).
* 2026-10-06, note 103: one LightGBM per task — without networks -0.75 phase / -1.14 function vs full (worse than the
  current no-network design); with networks phase ties, function -0.22*. Deciders cost ~1 % of time. Keep current design.
* 2026-10-06, note 106: vehicle-path features / function<->lanes iteration / joint lane+role decode — none beat v5b (the
  full model already carries path info via lead-lag features and lanes D context). Remaining wrong-function lane errors:
  low-volume lane loops that every function model calls Bike/Mid/Other. Nothing adopted.
* 2026-10-06 ~15:40, orchestrator on note 108: limits = per-type p99.8 + profile p99.5 (flag rates level). Profile check
  replaces "busier at night" + "doesn't follow traffic" (24 h only). Spot-check sheet health_review_v3 posted to user.
  Watch: Presence 2+ lanes fire rate 7.55 / 100 at 24 h (n=212) — revisit after user answers.
* 2026-10-06 ~16:30, USER away until Tue/Wed morning: (1) health_review_v3 rebuilt with verified charts (note 108b
  agent); (2) then simpler-architecture search (note 109, local GPU/CPU only, NO RunPod). Morning: post both in USER_INPUT.
* 2026-10-06 night, note 109: design C adopted (one network kind: 3 w32 siba members do phase + function; 1-seed tree
  models; setback 1 seed per group): ties v5c at every length, 27-30 % faster at 3 h, 36 -> 21 trained models.
  Building package v6 (note 111).
