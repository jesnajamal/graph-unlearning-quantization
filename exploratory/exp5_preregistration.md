# Experiment 5: signal-strength screen. Pre-registration
Written before any Experiment 5 result was computed (2026-11-20, ~08:15 Asia/Dubai).
Purpose: find a setting where node-membership forgetting is measurable. Only Original and Retrain models are
trained; no unlearning method and no quantization.

## Implementation
Standard 2-layer GCN (PyG `GCNConv`, symmetric normalisation with self-loops), full-batch, Adam lr 0.01,
weight decay 5e-4, 200 epochs, no early stopping (the last epoch is used: no checkpoint selection on any
evaluation data). Planetoid Cora and CiteSeer with row-normalised features. This replaces the authors' GIF GCN
(fixed 16 hidden units) because hidden size and dropout are screening variables; it is documented as such.

## Grid (16 configurations)
dataset {Cora, CiteSeer} x hidden {64, 128} x dropout {0, 0.5} x labels {many: random 90 % of nodes for
training, as in the GIF code; few: random 10 % of nodes for training}.
Deletion: 10 % of the training nodes (drawn from training-node IDs); their labels are removed from training and
their edges from the training graph. Retrain-A and Retrain-B: trained on the reduced data from two different initialisations.

## Membership attack (identical inference conditions, paired)
* All models are queried on the same full graph; score = predicted probability of the true label.
* Members = the deleted nodes D. Non-members = held-out nodes (never used for training), split once per seed into
  H_cal and H_eval. The SAME D and H_eval are used for every model of a seed (paired comparison).
* AUC(D vs H_eval) per model; signed gap = AUC(Original) - AUC(Retrain-A); attack advantage = 2|AUC - 0.5|.
* Calibrated attack: threshold chosen on retained training nodes vs H_cal only; TPR(D) - FPR(H_eval) reported.
* Retrain noise = |AUC(Retrain-A) - AUC(Retrain-B)| on the same D and H_eval.
* Utility: test accuracy of the Original on held-out nodes (full graph).

## Screening (seeds 0, 1, 2) and pass criteria (per configuration)
1. signed gap > 0 on 3/3 seeds;
2. mean signed gap >= 3 x mean Retrain noise;
3. mean signed gap >= 0.10 AUC (0.15 is the stretch target, reported but not required);
4. utility: mean Original test accuracy >= 0.9 x the best mean test accuracy of that dataset (same label regime) in the grid.

## Selection and confirmation
Per dataset, among passing configurations choose the one with the largest mean signed gap (ties: higher utility).
Freeze it and confirm on 5 NEW seeds (10-14) that were not used for screening. Confirmation report: mean signed
gap with 95 % CI, seeds with positive gap (k/5), mean Retrain noise, advantages of both models, calibrated
TPR-FPR, utility. A configuration is "confirmed" if the 95 % CI of the signed gap excludes 0, the gap is positive on
>= 4/5 seeds, and mean gap >= 3 x mean Retrain noise.

## Decision
* Confirmed configuration found -> use it (with retained-data fine-tuning as the unlearning method) to return to the
  quantization question, with a new pre-registration.
* None found -> reconsider the attack (e.g. a loss- or shadow-model-based attack) before increasing model complexity.
