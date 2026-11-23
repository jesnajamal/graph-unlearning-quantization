# Forgotten at full precision, remembered at low bits?

Experiments on whether post-training quantization undoes graph unlearning.

For LLMs it has been shown that 4-bit quantization can bring back a lot of the knowledge that unlearning removed (Zhang et al., ICLR 2025). I wanted to know if the same happens for graph neural networks, so I tested it on small GCNs with a membership-inference check: after unlearning, does a deleted node still look like a training member, and does that change once the model is quantized?

Short answer so far: ordinary fine-tuning-based unlearning kept its forgetting under INT8, INT4 and INT3 in every configuration I tried. Quantization did undo forgetting when I forced part of the unlearning update to stay inside the rounding cells of the original weights, and the amount undone grew with the share of weights confined.

This is ongoing work for a preprint. Everything was run on Kaggle/Colab (CPU is enough).

## Setup in one paragraph

Cora and CiteSeer (Planetoid), 2-layer GCN (hidden 128, dropout 0.5), only 10% of nodes labelled, 10% of those deleted. Unlearning = fine-tuning on the retained labels with the deleted nodes and their edges removed. A model counts as "forgotten" if its membership AUC (deleted vs held-out nodes) is within a band estimated from 5 independently retrained models. Quantization is simulated weight-only symmetric round-to-nearest, INT8/4/3, per-channel or per-tensor, with scales recomputed per model or frozen to the original model's. Every experiment had a pre-registration file written before it ran, and doses were chosen on different seeds from the ones used to evaluate them.

## Notebooks

| Notebook | What it does |
| --- | --- |
| `01_fp32_dose_screening.ipynb` | Finds the fine-tuning dose that reaches the retrained model in FP32 (Exp 6a) |
| `02_quantization.ipynb` | Quantizes the unlearned model and compares it with quantized retrained models (Exp 6b) |
| `03_positive_control.ipynb` | Unlearning confined to INT4 rounding cells, as a positive control (Exp 7) |
| `04_confinement_curve.ipynb` | Varies how much of the update is confined (Exp 8) |
| `05_diagnostic_replay.ipynb` | Replays Exp 8 from its checkpoints to look at which weight changes matter (Exp 8b) |

Each notebook installs `torch_geometric`, writes its own pre-registration, saves per-seed CSVs as it goes, can be resumed, and zips its output at the end. On Kaggle, run with Accelerator = None and use "Save & Run All (Commit)" so the outputs are kept. `05` needs the output of `04` attached as an input.

`exploratory/` has the earlier signal-strength screen (Exp 5) that picked the few-label setting.

## Main numbers

All from `results/`, 5 seeds each, 95% t-intervals.

| | CiteSeer | Cora |
| --- | --- | --- |
| Membership AUC gap, original vs retrained (Exp 5) | 0.257 | 0.175 |
| Fine-tuning, INT4 per-channel: share of forgetting undone (Exp 6b) | -1.0% (-2.9 to 0.8) | 1.4% (0.3 to 2.4) |
| Configurations where fine-tuning stayed within the band (Exp 6b) | 12 / 12 | 24 / 24 |
| 25% of weights confined: share undone by INT4 (Exp 8) | 15.4% (14.4 to 16.4) | 15.4% (13.3 to 17.5) |

The 25%-confined model passes the retrain check in FP32 and fails it after INT4 on both datasets. On CiteSeer the FP32 pass is borderline (3 of 5 seeds inside the band), which is shown seed by seed in the results.

## Things to keep in mind

- One small architecture, two citation graphs, one simple confidence-based attack. Not a privacy guarantee.
- The positive control in Exp 7 did not work as pre-registered (the constrained model never forgot enough), so it is reported as a failed control.
- The confined variants in Exp 7/8 are analysis tools, not unlearning methods.
- Exp 8b is exploratory. Some of my initial guesses about the mechanism did not hold up and the notes in its results say so.
- Model checkpoints are not in the repo (about 100 MB per run); the notebooks regenerate them.

## Results folder

`results/<experiment>/` holds the pre-registration (where saved), the gate decision, the per-seed CSVs and the summary tables that the numbers above come from. For Exp 6a there are two runs, one on CPU and one on a Kaggle GPU; they picked the same CiteSeer dose and gave very similar results, while the Cora decision depended on the band estimate.

## Requirements

Python 3.10+, `torch`, `torch_geometric`, `scikit-learn`, `scipy`, `pandas`, `matplotlib` (see `requirements.txt`).

## References

- Zhang et al. Catastrophic Failure of LLM Unlearning via Quantization. ICLR 2025.
- Cheng et al. GNNDelete: A General Strategy for Unlearning in Graph Neural Networks. ICLR 2023.
- Wu et al. GIF: A General Graph Unlearning Strategy via Influence Function. WWW 2023.
- Kipf and Welling. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.

## Author

Jesna Jamal
