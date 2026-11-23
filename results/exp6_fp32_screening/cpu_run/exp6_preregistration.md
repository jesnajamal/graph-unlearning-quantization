# Experiment 6: minimum effective fine-tuning unlearning update (FP32 screening only). Pre-registration
Written before any Experiment 6 result was computed (2026-11-20, ~11:40 Asia/Dubai).
Quantization is NOT part of this stage; it is gated on the screening outcome below.

## Frozen settings (from Experiment 5; no further architecture selection)
Cora and CiteSeer (Planetoid, row-normalised features); 2-layer GCN, hidden 128, dropout 0.5; Adam lr 0.01,
weight decay 5e-4, 200 epochs, last epoch used; 10 % of nodes are training nodes (random); deletion of 10 % of the
training nodes. Implementation: `fastgcn.py`, a sparse-input implementation of the same PyG GCN (outputs checked
against PyG GCNConv with identical weights: max difference about 1e-7), used only for speed.

## Seeds
Selection seeds 20-24 are used only to choose the dose. Evaluation seeds 30-34 (fresh) are used only to assess the frozen dose.
Each seed draws its own training nodes, deleted nodes, held-out split and initialisations.

## Models per seed
* Original: all training labels, full training graph.
* Retrain-A, Retrain-B: deleted nodes' labels removed AND their incident edges removed from the graph; two initialisations.
* Fine-tuning unlearning (FT): start from the Original; loss on the retained training labels only; message passing on
  the same reduced graph as Retrain (safeguard 4); same optimiser family (Adam, weight decay 5e-4, dropout on).
  Learning rates {0.001, 0.005, 0.01} x doses (steps) {1, 2, 5, 10, 20, 50, 100, 200}: 24 configurations.
* Drift control: identical fine-tuning on the full training data and full graph (nothing deleted), same grid.

## Measurements (identical inference conditions)
* Membership: every model is queried on the same full graph; score = probability of the true label; members = all
  deleted nodes D; non-members = held-out nodes H_eval (held-out nodes are split into H_cal / H_eval per seed).
  AUC, signed AUC, advantage 2|AUC - 0.5|, distance to Retrain-A = |AUC - AUC(Retrain-A)|.
* Paired improvement over Original = dist(Original) - dist(config), on the same D and H_eval.
* Calibrated attack: threshold on retained training nodes vs H_cal; TPR(D) - FPR(H_eval).
* Utility: accuracy on held-out nodes, on the reduced graph (the graph after deletion), for every model.
* Update size: ||W - W_Original||; median |dW| / quantization step and share of |dW| > step/2 for INT4 and INT3,
  per-channel and per-tensor, with the step taken from the Original's weights (frozen scales).

## Selection rule (selection seeds only)
band = 2 * RMS over selection seeds of (AUC(Retrain-A) - AUC(Retrain-B)).
Precondition: the Original must clearly fail the band: mean dist(Original) > band AND dist(Original) > band on >= 4/5 seeds.
A configuration passes if (a) mean dist(config) <= band, (b) paired improvement over Original > 0 on >= 4/5 seeds,
(c) mean utility >= mean utility(Original) - 0.02. Among passing configurations choose the smallest mean update norm.
If none passes, the dataset is reported as "fine-tuning does not reach Retrain" and no dose is frozen.

## Evaluation rule (evaluation seeds, frozen dose; all doses are still reported)
Success if: paired improvement over Original has a 95 % t-interval above 0; improvement > 0 on >= 4/5 seeds;
mean dist(frozen dose) <= band (band from the selection seeds); mean utility >= utility(Original) - 0.02;
and the Original fails the band on the evaluation seeds (mean dist(Original) > band).

## Update-size classification (for the gating decision)
"Small" update: median |dW| / INT4 per-channel step (frozen Original scale) < 0.5 in the frozen dose.

## Decision
* Success with a small update -> full quantization matrix (INT8/INT4/INT3; per-channel / per-tensor; frozen vs
  recomputed scales; matched-noise control; quantized Original and Retrain) under a new pre-registration.
* Success with only a substantial update -> quantize the smallest successful dose and measure update survival.
* No success on the evaluation seeds -> stop and investigate the unlearning procedure.
