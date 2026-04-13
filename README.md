# Manufacturing Feasibility Classifier

A plug-and-play Streamlit application for binary classification of manufacturing parts as **feasible** or **infeasible** from 3-D point cloud data (`.ply` format).

---

## Overview

The system evaluates **60 configurable ML pipelines** built from five interchangeable modules:

| Module | Options | Description |
|--------|---------|-------------|
| **PP** Preprocessing | PP-A … PP-E\* | Raw, normalise, RANSAC plane removal, canonical frame alignment, voxel downsample |
| **FE** Feature Engineering | FE-A … FE-N | Geometric stats, voxel occupancy, projection histograms, plane-relative stats, PCA latent, autoencoder latent |
| **IB** Imbalance Handling | IB-0 … IB-3 | None, class_weight, SMOTE, SMOTE + class_weight |
| **FS** Feature Selection | FS-0 … FS-3 | None, SelectKBest, RFE, PCA reduce |
| **CL** Classifier | CL-1 … CL-7 | Random Forest, SVM, XGBoost, GradientBoosting, MLP, Logistic Regression, KNN |

The primary evaluation metric is **F1 score on the infeasible (minority) class**, reflecting the real-world cost of classifying a bad part as good.

---

## Key Design Decisions

### Reference Plane Handling
Each point cloud contains a large reference plane fixture that must be separated from the part geometry of interest. The system handles this at two levels:

- **PP-C / PP-E** — RANSAC plane removal only (no orientation change)
- **PP-C\* / PP-E\*** — RANSAC removal + connected-component segmentation (retains only the largest part cluster) + rotation into a canonical plane-aligned coordinate frame (+Z = plane normal). This enables FE-K, FE-L, FE-M, FE-N which compute plane-relative features such as height distribution above the plane, lateral spread, and point density gradient.

### Autoencoder
FE-I, FE-J, and FE-N use a shallow autoencoder trained **unsupervised** (no labels used) on all point clouds. The encoder (input → 256 → 64) is extracted and its bottleneck embeddings used as features. Implemented with `sklearn.neural_network.MLPRegressor` — no PyTorch required.

### Pipeline Caching
To keep runtimes manageable across 60 pipelines:
1. Each unique **PP module** is run once; results are cached.
2. Each unique **(PP, FE) pair** is extracted once; feature matrices are cached.
3. The **autoencoder** is trained once and shared across all AE-group pipelines.

---

## Installation

```bash
# Clone / copy the project
cd feasibility_app

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app.py
```

Then in the browser:

1. **Data tab** — Enter the paths to your `feasible/` and `infeasible/` directories and click **Load Dataset**. The app will display class distribution and point-count statistics.

2. **Sidebar — Pipeline Selection** — Choose to run *All 60*, a specific *Group*, or a *Custom* set of pipeline IDs.

3. **Sidebar — Parameters** — Adjust RANSAC thresholds, voxel size, autoencoder settings, and train/test split as needed.

4. **Click ▶ Run Pipelines** — Results appear in the **Pipeline Results** tab, sorted by F1 (infeasible) descending, with a bar chart, scatter plot, and confusion matrix for the best pipeline.

5. **Export** — Download the full results table as a CSV.

---

## Dataset Format

```
feasible/
    part_001.ply
    part_002.ply
    ...
infeasible/
    part_301.ply
    part_302.ply
    ...
```

- All files must be `.ply` format (ASCII or binary).
- Labels are inferred from directory: `feasible/ → 1`, `infeasible/ → 0`.

---

## Project Structure

```
feasibility_app/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── README.md
└── pipeline/
    ├── __init__.py
    ├── ingestion.py         # Module 0 — load .ply files
    ├── preprocessing.py     # Module 1 — PP-A … PP-E*
    ├── features.py          # Module 2 — FE-A … FE-N
    ├── autoencoder.py       # Unsupervised AE (sklearn-based)
    ├── components.py        # Modules 3–6 — IB, FS, CL, Evaluation
    ├── catalog.py           # 60 pipeline definitions
    └── runner.py            # Orchestration with caching
```

---

## Runtime Expectations

| Pipeline group | Typical time (500 clouds) |
|----------------|--------------------------|
| Baseline (P-001–P-004) | < 1 min |
| Plane Removal / Aligned Frame | 2–5 min |
| Plane-Relative Features | 3–8 min |
| Full hand-crafted features | 5–15 min |
| PCA / AE latent | 10–25 min (AE trains once) |
| All 60 pipelines | 30–90 min |

> Tip: Use **By Group** or **Custom** mode to run subsets during development. The Kitchen Sink pipeline (P-060) is the slowest single pipeline.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `f1_infeasible` | F1 for class 0 (infeasible) — **primary sort metric** |
| `recall_infeasible` | Recall for infeasible class — critical for catching bad parts |
| `roc_auc` | ROC-AUC across both classes |
| `f1_macro` | Macro-averaged F1 |
| `accuracy` | Overall accuracy (use cautiously — class imbalance) |

---

## Extending the System

To add a new preprocessing step, feature extractor, or classifier:

1. Implement the function in the relevant module (`preprocessing.py`, `features.py`, `components.py`).
2. Register it in the module's `PREPROCESSORS` / dispatch dict.
3. Add new `PipelineConfig` entries to `catalog.py`.
4. No changes to `runner.py` or `app.py` are needed.
