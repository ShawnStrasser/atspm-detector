# 21_trackB_b10_b7_sixfold — last backbone checks and the six-fold TCN (2026-09-23)

Chain `research/code/neural/b10b7_chain.py` (resumable; relaunched once after a machine reboot killed it
at fold-2 inference), S4D backbone `research/code/neural/trackb_ssm.py`, logs
`dc_work/logs/trackB_b10b7_chain.log` / `.out2`, results `dc_work/trackB/eval/{b10,b7,final6_tcn,final6_gru13}.json`.
Gate as in note 16: fold-0 blend before the decoder at 30 min, seed 0 vs `tb_tcn_f0_oof` >= +0.3 pt, then a
second seed vs the three-seed TCN mean .9750. Variants fixed before launch.

## B10 — wider TCN (96 → 192 channels, 2.68 M weights): dropped
Fold 0, seed 0, blend before decoder: 5 min .9508 (+0.16), 10 min .9628 (+0.07), **30 min .9745 (+0.07)**,
1 h .9743 (+0.32), 3 h .9829 (+0.21) vs TCN seed 0. Fails the 30-min screen; the 1-h gain is against the worst
of three TCN seeds (note 15), i.e. inside the seed spread.

## B7 — state-space backbone (bidirectional S4D, pure PyTorch, 0.50 M weights): dropped
`mamba-ssm` was not attempted on Windows; S4D written in-house instead. Fold 0: 5 min .9544 (+0.52),
10 min .9646 (+0.24), **30 min .9747 (+0.10)**, 1 h .9753 (+0.43), 3 h .9818 (+0.09). Fails at 30 min. The
5-min and 1-h gains are single-seed against TCN seed 0 (the weakest); not pursued (screening rule: dropped,
not tuned).

## Six-fold TCN vs the stage-13 GRU (same harness, same 1,532,182 rows, coverage .970)

| six folds, OOF | 5 min | 10 min | 30 min | 1 h | 3 h | 6 h | 24 h | full |
|---|---|---|---|---|---|---|---|---|
| LightGBM (trees + decoder) | .9287 | .9423 | .9623 | .9620 | .9762 | .9770 | .9797 | .9808 |
| GRU net alone | .9273 | .9475 | .9639 | .9658 | .9709 | .9706 | .9671 | .9668 |
| TCN net alone | .9248 | .9436 | .9601 | .9620 | .9685 | .9673 | .9657 | .9668 |
| **GRU blend** (before decoder) | .9519 | .9614 | **.9747** | .9747 | .9797 | .9801 | .9817 | .9828 |
| **TCN blend** (before decoder) | .9508 | .9608 | **.9734** | .9739 | .9803 | .9805 | .9824 | .9838 |
| TCN − GRU, blend (pt) | −0.11 | −0.06 | **−0.13** | −0.07 | +0.05 | +0.04 | +0.07 | +0.10 |
| GRU blend − trees (pt) | +2.32 | +1.92 | +1.24 | +1.26 | +0.35 | +0.31 | +0.20 | +0.20 |
| TCN blend − trees (pt) | +2.21 | +1.86 | +1.11 | +1.19 | +0.41 | +0.35 | +0.27 | +0.30 |

Per fold at 30 min, GRU / TCN: .9745/.9737, .9724/.9709, .9785/.9767, .9735/.9712, .9771/.9785, .9721/.9691
— the TCN is behind on 5 of 6 folds, by 0.13 pt on average, single seed each; the TCN blend's seed sd is
~0.2 pt (note 15), so the two backbones are not separable at 30 min but the sign leans GRU. Past 3 h the TCN
blend leads slightly on every length.

## Net cut-off (production switches the net off above 120 min)
On six folds the blend beats the trees alone at **every** length: +0.35 / +0.31 / +0.20 / +0.20 pt (GRU) and
+0.41 / +0.35 / +0.27 / +0.30 pt (TCN) at 3 h / 6 h / 24 h / full. The stage-13 argument ("with hours of data
both sit on the label-noise floor") does not hold on the OOF rows: the network never hurts and keeps adding
0.2–0.4 pt. **Candidate for final_v3: raise or remove the cut-off** — cost is the only reason to keep it
(a 6-h TCN pass ~2 s/signal on CPU, note 20; GRU ~2.6× that). The OOF window set has no 2 / 4 / 12 h windows.

## Verdict
Track B is closed: B2, B3, B4, B5, B6, B8, B9, B10 and B7 all failed the 0.3-pt bar (notes 15–21); the
network side's headroom at 30 min is ~0.26 pt (note 19). For final_v3: keep the GRU at short windows
(six-fold lead 0.13 pt at 30 min) unless CPU cost decides otherwise; turn the network on beyond 2 h (TCN or
GRU, +0.3 pt at 3–6 h, +0.2–0.3 pt at a day). GPU time: B10 0.7 h, B7 1.5 h, six folds ~2.5 h.
