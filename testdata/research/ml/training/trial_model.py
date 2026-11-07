"""
XGBoost Classifier for Alzheimer's Disease Detection
Features: MFCCs, Acoustic & Linguistic features
Target: patient_type → AD / MCI / Control
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
print("=" * 60)
print("  XGBoost - AD / MCI / Control Classifier")
print("=" * 60)

from pathlib import Path

_archive = Path(__file__).resolve().parents[2] / "data" / "features_archive" / "features.csv"
_legacy = _archive if _archive.exists() else Path(__file__).resolve().parents[2] / "data" / "features_archive" / "features_V1.csv"
df = pd.read_csv(_legacy)
print(f"\n[OK] Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. Prepare Features & Target
# ─────────────────────────────────────────────
X = df.drop(columns=["sample", "patient_type"])
y_raw = df["patient_type"]

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

print(f"\nClass distribution:")
for cls, count in zip(class_names, np.bincount(y)):
    print(f"   {cls:10s} -> {count} samples")

# ─────────────────────────────────────────────
# 3. Build XGBoost Pipeline
# ─────────────────────────────────────────────
# Class weights to handle imbalance (MCI is smallest)
class_counts = np.bincount(y)
total = len(y)
sample_weights = np.array([total / (len(class_counts) * class_counts[yi]) for yi in y])

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    objective="multi:softprob",
    num_class=3,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", xgb_model)
])

# ─────────────────────────────────────────────
# 4. Cross-Validation Evaluation
# ─────────────────────────────────────────────
print("\nRunning 10-Fold Stratified Cross-Validation...")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

y_pred = cross_val_predict(pipeline, X, y, cv=cv, method="predict")
y_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")

acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, y_proba, multi_class="ovr", average="macro")

print(f"\n{'='*60}")
print(f"  Overall Accuracy : {acc*100:.2f}%")
print(f"  ROC-AUC (macro)  : {auc:.4f}")
print(f"{'='*60}")

print("\nPer-Class Report:")
print(classification_report(y, y_pred, target_names=class_names))

print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
cm = confusion_matrix(y, y_pred)
header = f"{'':>10}" + "".join(f"{c:>10}" for c in class_names)
print(header)
for i, row in enumerate(cm):
    print(f"{class_names[i]:>10}" + "".join(f"{v:>10}" for v in row))

# ─────────────────────────────────────────────
# 5. Feature Importance (fit on full data)
# ─────────────────────────────────────────────
print("\nTop 15 Most Important Features:")
pipeline.fit(X, y)
importances = pipeline.named_steps["xgb"].feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

for i, (feat, imp) in enumerate(feat_imp.head(15).items(), 1):
    bar = "#" * int(imp * 300)
    print(f"  {i:>2}. {feat:<25} {imp:.4f}  {bar}")

feat_imp.to_csv("feature_importance.csv", header=["importance"])

print("\n[OK] Done! feature_importance.csv saved.")
print("=" * 60)