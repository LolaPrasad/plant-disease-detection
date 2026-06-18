# Cross-Dataset Generalization Analysis for Plant Disease Detection Under Domain Shift

> **Research Question:** How does domain shift affect the performance and explainability of EfficientNetB0, MobileNetV2, and ResNet50 for plant disease classification?

Models are trained exclusively on the **PlantVillage** controlled-lab dataset and evaluated on both PlantVillage and the real-world **PlantDoc** dataset to quantify performance degradation caused by domain shift.

---

## Key Results

| Model | PlantVillage Acc | PlantDoc Acc | Relative Drop |
|---|---|---|---|
| EfficientNetB0 | **99.76%** | 21.9% | 78.0% |
| MobileNetV2 | 99.32% | 21.0% | 78.9% |
| ResNet50 | 98.19% | **24.8%** | **74.7%** |

**ResNet50 is most robust to domain shift** despite lower in-domain accuracy.

---

## Project Structure

```
.
├── configs/                  # YAML hyperparameter configs (base + per-model)
├── datasets/                 # Dataset classes, transforms, DataLoader factory
├── models/                   # Model definitions, checkpoint helpers
├── training/                 # Training loop (AMP, early stopping, scheduler)
├── evaluation/               # Metrics, inference runner, domain shift analysis
├── explainability/           # Grad-CAM generation, sample selection, comparison grids
├── visualization/            # All publication figures (300 DPI PNG + PDF)
├── utils/                    # Config loader, seed, logger
├── scripts/                  # Entry-point CLI scripts
│   ├── download_data.py
│   ├── prepare_datasets.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_gradcam.py
│   ├── compare_domains.py
│   └── generate_figures.py
├── tests/                    # Unit tests (pytest)
├── outputs/
│   ├── checkpoints/          # Saved model weights (*_best.pt, *_last.pt)
│   ├── metrics/              # JSON + CSV metrics, training history
│   ├── figures/              # Publication figures (PNG + PDF)
│   ├── predictions/          # Per-image prediction CSVs
│   └── gradcam/              # Grad-CAM heatmaps + comparison grids
├── requirements.txt
└── pyproject.toml
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Dissertation
pip install -r requirements.txt
pip install -e .
```

**Python 3.10+ required.** Tested on Python 3.13 / PyTorch 2.9 (CPU + MPS).

### 2. Verify GPU / device

```python
import torch
print(torch.cuda.is_available())      # NVIDIA GPU
print(torch.backends.mps.is_available())  # Apple Silicon
```

---

## Datasets

### PlantDoc (already present)

```
src/data/PlantDoc-Dataset/
    train/<class>/
    test/<class>/
```

15 classes mapped to PlantVillage equivalents; 13 non-overlapping classes excluded. See [`datasets/plantdoc.py`](datasets/plantdoc.py) for the full mapping.

### PlantVillage

**Option A — Kaggle download (automatic):**

```bash
# Set credentials first:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

python scripts/download_data.py
python scripts/prepare_datasets.py   # splits into train/val/test (70/15/15)
```

**Option B — Manual:**

1. Download from https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Unzip into `src/data/plantvillage_raw/`
3. Run `python scripts/prepare_datasets.py`

The processed splits land at `src/data/PlantVillage_processed/{train,val,test}/`.

---

## Training

Train one model:

```bash
python scripts/train.py --model efficientnetb0
python scripts/train.py --model mobilenetv2
python scripts/train.py --model resnet50
```

Override hyperparameters at the command line:

```bash
python scripts/train.py --model resnet50 --epochs 20 --batch-size 16 --lr 5e-5
```

Disable mixed-precision (for CPU runs):

```bash
python scripts/train.py --model efficientnetb0 --no-amp
```

**Outputs per model:**
- `outputs/checkpoints/<model>_best.pt` — best validation checkpoint
- `outputs/checkpoints/<model>_last.pt` — last epoch checkpoint
- `outputs/metrics/<model>_training_history.json`
- `outputs/metrics/<model>_config.json` — exact config snapshot
- `outputs/logs/<model>/` — TensorBoard event files

**View training curves in TensorBoard:**

```bash
tensorboard --logdir outputs/logs
```

---

## Evaluation

### Experiment 1 — In-domain (PlantVillage test set)

```bash
python scripts/evaluate.py --model efficientnetb0 --dataset plantvillage
python scripts/evaluate.py --model mobilenetv2    --dataset plantvillage
python scripts/evaluate.py --model resnet50       --dataset plantvillage
```

### Experiment 2 — Cross-domain (PlantDoc)

```bash
python scripts/evaluate.py --model efficientnetb0 --dataset plantdoc
python scripts/evaluate.py --model mobilenetv2    --dataset plantdoc
python scripts/evaluate.py --model resnet50       --dataset plantdoc
```

### Both at once

```bash
python scripts/evaluate.py --model efficientnetb0 --dataset both
```

**Outputs per run:**
- `outputs/metrics/<model>_<dataset>_metrics.json`
- `outputs/metrics/<model>_<dataset>_report.txt`
- `outputs/predictions/<model>_<dataset>_predictions.csv`

---

## Experiment 3 — Domain Shift Analysis

After evaluating all models on both datasets:

```bash
python scripts/compare_domains.py
```

**Outputs:**
- `outputs/metrics/publication_tables.csv` — LaTeX-ready performance table
- `outputs/metrics/domain_shift_summary.csv` — absolute + relative drops
- `outputs/metrics/statistical_tests.csv` — McNemar pairwise comparisons

---

## Grad-CAM Explainability

Single model:

```bash
python scripts/generate_gradcam.py --model efficientnetb0
```

All models + cross-model comparison grids:

```bash
python scripts/generate_gradcam.py --model all --compare
```

**Outputs:**
- `outputs/gradcam/<model>/correct_pv/` — correctly predicted PlantVillage samples
- `outputs/gradcam/<model>/correct_pd/` — correctly predicted PlantDoc samples
- `outputs/gradcam/<model>/wrong_pd/`   — misclassified PlantDoc samples
- `outputs/gradcam/comparisons/`        — multi-model side-by-side grids
- `outputs/gradcam/gradcam_manifest.csv`

---

## Publication Figures

```bash
python scripts/generate_figures.py            # all figures
python scripts/generate_figures.py --only curves   # training curves only
python scripts/generate_figures.py --only cm       # confusion matrices only
python scripts/generate_figures.py --only domain   # domain shift charts only
```

**Figures produced (PNG + PDF at 300 DPI):**

| File | Description |
|---|---|
| `fig_domain_gap_bars` | Grouped bar: PV vs PD accuracy + F1 per model |
| `fig_accuracy_drop` | Absolute + relative accuracy drop dual-axis |
| `fig_perclass_f1_heatmap` | F1 heatmap: 10 classes × PV/PD × 3 models |
| `fig_class_degradation_ranking` | Per-class F1 drop ranked (horizontal bar) |
| `fig_performance_radar` | Radar: 4 metrics on PlantDoc per model |
| `training_curves_<model>` | Loss + accuracy per epoch |
| `confusion_matrices/cm_<model>_<dataset>` | Normalised confusion matrix |

---

## Reproducibility

Every experiment is seeded (`seed: 42` in `configs/base.yaml`). To reproduce from scratch:

```bash
# 1. Prepare data
python scripts/prepare_datasets.py

# 2. Train all models
python scripts/train.py --model efficientnetb0
python scripts/train.py --model mobilenetv2
python scripts/train.py --model resnet50

# 3. Evaluate all models on both datasets
python scripts/evaluate.py --model efficientnetb0 --dataset both
python scripts/evaluate.py --model mobilenetv2    --dataset both
python scripts/evaluate.py --model resnet50       --dataset both

# 4. Domain shift analysis
python scripts/compare_domains.py

# 5. Grad-CAM
python scripts/generate_gradcam.py --model all --compare

# 6. All figures
python scripts/generate_figures.py
```

---

## Testing

```bash
pytest tests/ -v
```

22 unit tests covering: dataset loading, label mapping, model initialisation, evaluation metrics, Grad-CAM generation.

---

## Configuration

All hyperparameters live in `configs/`. Model-specific YAMLs merge on top of `configs/base.yaml`.

Key settings (`configs/base.yaml`):

```yaml
training:
  epochs: 10
  batch_size: 32
  learning_rate: 1.0e-4
  early_stopping_patience: 5
  mixed_precision: true
```

---

## Citation

If you use this codebase, please cite:

```
[Your name]. Cross-Dataset Generalization Analysis for Plant Disease Detection
Under Domain Shift. MSc Dissertation, [University], 2026.
```

---

## License

MIT
