# Experiment 8b: diagnostic on saved Experiment 8 checkpoints. Pre-registration
Saved: 2026-11-22 14:17:43, before any result of this diagnostic. Experiment 8 fingerprint 5bf4efb8184d2460.
Experiment 8 and its pre-registered answers are not changed. All analyses here are descriptive (no pass/fail).

D1 replay check: RC-p0.00 is retrained from the saved Original with the Experiment 8 seed, bounds (biases frozen, row maximum
fixed), optimizer (Adam lr 0.005, weight decay 0.0005) and 400 steps, on CPU. Reported: max |W_replay - W_saved|.
If it exceeds 1e-4, D2 is labelled approximate.
D2 (weight-decay hypothesis for E3): per weight entry, over the replay, S_data = sum of |data gradient| and S_wd = sum of
|weight decay x w| (the two parts of the gradient Adam receives). For the top f = [0.01, 0.05, 0.1, 0.25] of entries by
|W_RC0 - W_O| / INT4 bin (the Experiment 8 TF ranking) and for the rest: share with S_wd > S_data ("weight-decay-gradient
dominance"), median S_data / S_wd, share moving toward zero (|W_RC0| < |W_O|), per layer and overall.
Sensitivity (actual optimizer updates): each Adam step's update of each entry (before projection) is attributed to weight decay
if it points toward zero and against (or without) the data gradient, to the data if it follows the data gradient and points
away from zero, and is left ambiguous when both agree; share of entries whose weight-decay-attributed movement exceeds the
data-attributed movement.
Hypothesis under test (stated by Claude after Experiment 8): the top movers are mostly weight-decay dominated. It is reported
as consistent with the data only if both the gradient-dominance share and the update-attribution share are above 0.5 for the
top movers and clearly higher than for the rest; otherwise as not supported. Neither measure alone shows that weight decay
caused the movement: Adam's momentum and adaptive scaling act on the combined gradient.
D3 (reversion, no re-optimisation): from saved RC-p0.00 and FT models, revert to the Original (a) the top-f movers,
(b) all but the top-f movers, (c) the same number of random entries (5 draws), and two code-based splits:
(d) Original-grid crossings: entries whose INT4 per-channel code differs when BOTH weights use the Original's scale (the
boundary-crossing mechanism analysis), and (e) the remaining in-cell entries; (f) recomputed-scale code changes: entries whose
code differs when each model uses its own scale (a deployment-quantization diagnostic), and (g) the rest. Biases keep the
model's own values. Measure forgetting achieved phi = 1 - |AUC - AUC(Retrain-1)| / |AUC(Original) - AUC(Retrain-1)| on the same
attack as Experiment 8. Paired over seeds, 95 % t-intervals. Reverting changes to their FP32 Original values is not the same as
quantizing: quantization maps weights to discrete values and does not restore them.
Limits: D3 removes changes without letting the network re-adapt, so it measures contribution in place, not necessity.
Pre-run clarifications after review: D2 is labelled weight-decay-gradient dominance and an optimizer-update sensitivity check
was added; D3 separates Original-grid crossings from recomputed-scale code changes, and no reversion is described as
equivalent to INT4 quantization. Objectives, replay check, seeds and controls are unchanged.
