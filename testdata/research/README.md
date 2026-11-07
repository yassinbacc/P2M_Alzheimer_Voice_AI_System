# Research / legacy files

Old experiments, backup datasets, and exploratory training code.  
**Not used by the production pipeline** (`scripts/train.py`, `scripts/predict.py`, etc.).

## Contents

| Path | Description |
|------|-------------|
| `ml/pipeline/extract.py` | Legacy feature extractor → `features.csv` |
| `ml/training/research_model.py` | Full `model.py` experiment (many plots, SHAP, pairwise) |
| `ml/training/trial_model.py` | Early XGBoost on legacy features |
| `scripts/extract_features.py` | Runner for legacy extract |
| `data/features_archive/` | Old CSV backups and `extract - Copy.py` |
| `artifacts/figures/` | PNG plots from research_model |
| `artifacts/reports/` | Old CV tables, GradientBoosting reports, pairwise results |

## Running research code (optional)

From project root:

```bash
python research/ml/training/research_model.py
```

You may need to fix paths inside that script if it still points at old locations.
