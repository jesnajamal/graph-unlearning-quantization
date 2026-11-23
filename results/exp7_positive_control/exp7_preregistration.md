# Experiment 7: positive control for quantization reversal. Pre-registration
Saved: 2026-11-20 19:22:22, before any result of this experiment. Device: cpu (None).

## Why
Experiment 6 (quantization stage, seeds 40-44) classified all 36 configurations as PRESERVED (primary CiteSeer INT4
per-channel recomputed: reversal fraction -1.0 %, 95 % CI -2.9 % to +0.8 %). Experiment 6 is not changed or re-analysed here.
This experiment asks whether the same pipeline detects a reversal that is built in by construction.

## Dataset and fixed settings
CiteSeer only; 2-layer GCN, hidden 128, dropout 0.5, Adam lr 0.01, weight decay 0.0005,
200 epochs; 10 % training nodes, 10 % of them deleted; split code identical to Experiment 6.
Selection seeds [50, 51, 52, 53, 54], evaluation seeds [60, 61, 62, 63, 64] (all fresh).

## Models per seed
Original; Retrain-1..5; FT (Experiment 6 fine-tuning, lr 0.005, 100 steps, unchanged);
positive-control variants, all started from the Original with the FT objective (retained labels, reduced graph, Adam lr 0.005,
weight decay 0.0005, dropout on), with projection after every optimizer step:
* biases are held at the Original's values;
* in each row of each weight matrix the entry with the largest |w| is held fixed and every other entry stays strictly
  inside (-max, +max) of that row, so the per-row INT4 scale cannot change;
* a share p of the remaining weights is confined to its INT4 per-channel rounding cell of the Original,
  [(c - 0.5 + 0.05) s, (c + 0.5 - 0.05) s] with c = round(w_O / s), s = row max|w_O| / 7.
  PC-full p = 1, PC-half p = 0.5 (random subset per seed), PC-none p = 0.
* Fallback PC-full-INT3: p = 1 with INT3 cells (about 2.3x wider), used for validation only if PC-full fails V1.
For PC-full (PC-full-INT3) this guarantees that INT4 (INT3) per-channel quantization, recomputed or frozen-Original scales,
gives exactly the quantized Original's weights; the notebook asserts this.

## Dose selection (selection seeds only)
Step grid [100, 200, 400, 800]. Band = 2 x sqrt(mean over seeds and Retrain pairs of squared AUC differences) (descriptive tolerance;
the pairs are correlated). For each variant, the smallest dose that passes: mean dist <= band; improvement over Original > 0 on
>= 4 seeds; utility >= Original - 0.02. If none passes, the largest dose is used and the variant is
labelled "did not reach Retrain in FP32 on the selection seeds". Frozen in frozen_doses.json, never recomputed.

## Evaluation seeds
FP32 gate exactly as in Experiment 6 (improvement CI above 0, >= 4 seeds, within band, utility, Original fails band).
Every model is quantized whether or not it passes (this is a validation experiment; gate status is reported with each result).
Quantization, matched noise, measurements, band_Q / effective band, reversal fraction, excess over noise, node-level
recovery with raw shifts and the instability flag (RMS < 0.01), the verdict order and the clipping label are
identical to the Experiment 6 quantization stage.

## Validation criteria (decided in advance)
V1  PC-full passes the FP32 gate on the evaluation seeds (the built-in forgetting exists).
V2  PC-full at INT4 per-channel, recomputed AND frozen scales: verdict REVERSAL and the lower 95 % bound of the reversal
    fraction > 0.5. (Expected: about 1, since its quantized weights equal the quantized Original's.)
V3  PC-none at INT4 per-channel recomputed: not REVERSAL (the extra restrictions alone do not create a reversal).
V4  FT replicates Experiment 6 at INT4 per-channel recomputed: not REVERSAL.
Fallback: if V1 fails for PC-full, V1 and V2 are applied to PC-full-INT3 at INT3 per-channel instead, and this is reported.
Descriptive (no pass/fail): PC-half reversal fraction; PC-full at INT8 and INT3 and per-tensor; node-level specificity.

## Interpretation
V1-V3 all met: the pipeline can detect a reversal of the size that matters, which supports reading Experiment 6 as a real
negative result for this setting. V2 failed with V1 met: the pipeline lacks power and Experiment 6 must be qualified.
V1 failed: a positive control of this form could not be built; report this, PC-full is still quantized and reported as exploratory.
Scope: the measured membership attack, not formal privacy guarantees. PC-full is an engineered update, not a proposed method.
