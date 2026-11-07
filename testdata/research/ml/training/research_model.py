import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from collections import Counter
from itertools import combinations

# ML
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate, learning_curve
)
from sklearn.preprocessing import (
    LabelEncoder, RobustScaler, label_binarize
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    XGB_OK = False
    print("⚠ xgboost not installed — pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LGB_OK = True
except ImportError:
    LGB_OK = False
    print("⚠ lightgbm not installed — pip install lightgbm")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False
    print("⚠ imbalanced-learn not installed — pip install imbalanced-learn")

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("⚠ shap not installed — pip install shap")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STYLE SETUP
# ─────────────────────────────────────────────
PALETTE    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
CLASS_CMAP = sns.color_palette(PALETTE)
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
from pathlib import Path

from ml.config import FEATURES_ULTIMATE_CSV, ensure_dirs

_RESEARCH_ROOT = Path(__file__).resolve().parents[2]
ensure_dirs()
(_RESEARCH_ROOT / "artifacts" / "figures").mkdir(parents=True, exist_ok=True)
(_RESEARCH_ROOT / "artifacts" / "reports").mkdir(parents=True, exist_ok=True)

CSV_FILE = str(FEATURES_ULTIMATE_CSV)
OUTPUT_DIR = str(_RESEARCH_ROOT / "artifacts" / "reports") + os.sep
FIGURES_OUTPUT = str(_RESEARCH_ROOT / "artifacts" / "figures") + os.sep

RANDOM_STATE = 42
CV_FOLDS     = 5
TARGET_COL   = "patient_type"
DROP_COLS    = ["sample", "transcript", "duration", TARGET_COL]

# ════════════════════════════════════════════════════════════
# HELPER — save figure
# ════════════════════════════════════════════════════════════
def save(name):
    folder = FIGURES_OUTPUT if name.endswith(".png") else OUTPUT_DIR
    path = os.path.join(folder, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {name}")

# ═════════════════════════════════════════════
# STEP 1 — LOAD & INSPECT
# ═════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  STEP 1 — LOADING DATA")
print(f"{'═'*60}")

df = pd.read_csv(CSV_FILE)
print(f"  Shape        : {df.shape}")
print(f"  Distribution :")
print(df[TARGET_COL].value_counts().to_string())

# ── FIG 1: Class distribution bar chart ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = df[TARGET_COL].value_counts()
axes[0].bar(counts.index, counts.values, color=PALETTE[:len(counts)], edgecolor="white", linewidth=1.5)
axes[0].set_title("Class Distribution", fontweight="bold")
axes[0].set_ylabel("Count")
for i, (idx, v) in enumerate(counts.items()):
    axes[0].text(i, v + 0.5, str(v), ha="center", fontweight="bold")

axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
            colors=PALETTE[:len(counts)], startangle=140,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Class Proportions", fontweight="bold")

plt.suptitle("Dataset Class Distribution", fontsize=14, fontweight="bold", y=1.02)
save("01_class_distribution.png")

# ─────────────────────────────────────────────
# STEP 2 — PREPARE FEATURES
# ─────────────────────────────────────────────
print(f"\n{'═'*60}")
print(f"  STEP 2 — FEATURE PREPARATION")
print(f"{'═'*60}")

feature_cols = [c for c in df.columns if c not in DROP_COLS]

# Drop cols with >40% missing
missing_pct  = df[feature_cols].isnull().mean()

# ── FIG 2: Missing data heatmap ──────────────────────────
top_missing = missing_pct[missing_pct > 0].sort_values(ascending=False).head(40)
if len(top_missing) > 0:
    fig, ax = plt.subplots(figsize=(14, max(4, len(top_missing) * 0.3)))
    ax.barh(top_missing.index, top_missing.values * 100, color="#C44E52", alpha=0.8)
    ax.axvline(40, color="black", linestyle="--", linewidth=1.5, label="40% threshold")
    ax.set_xlabel("Missing %")
    ax.set_title("Features with Missing Values", fontweight="bold")
    ax.legend()
    save("02_missing_data.png")

keep_cols    = missing_pct[missing_pct < 0.4].index.tolist()
dropped_miss = [c for c in feature_cols if c not in keep_cols]
if dropped_miss:
    print(f"  Dropped {len(dropped_miss)} cols with >40% missing")
feature_cols = keep_cols

std_vals      = df[feature_cols].std()
keep_cols     = std_vals[std_vals > 0].index.tolist()
dropped_const = [c for c in feature_cols if c not in keep_cols]
if dropped_const:
    print(f"  Dropped {len(dropped_const)} constant cols")
feature_cols = keep_cols

X     = df[feature_cols].copy()
y_raw = df[TARGET_COL].copy()

le = LabelEncoder()
y  = le.fit_transform(y_raw)
print(f"  Classes      : {list(le.classes_)}")
print(f"  Feature cols : {len(feature_cols)}")
print(f"  Samples      : {len(X)}")

class_counts  = Counter(y)
total_samples = len(y)
class_weight  = {
    cls: total_samples / (len(class_counts) * count)
    for cls, count in class_counts.items()
}
print(f"  Class weights: {class_weight}")

# Impute & scale
imputer   = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
scaler    = RobustScaler()
X_scaled  = scaler.fit_transform(X_imputed)

# ── FIG 3: Feature statistics overview ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(std_vals[feature_cols], bins=40, color="#4C72B0", edgecolor="white")
axes[0].set_title("Feature Std Dev Distribution", fontweight="bold")
axes[0].set_xlabel("Std Dev")

axes[1].hist(missing_pct[feature_cols] * 100, bins=30, color="#DD8452", edgecolor="white")
axes[1].set_title("Missing Data % per Feature", fontweight="bold")
axes[1].set_xlabel("Missing %")

axes[2].hist(X_scaled.mean(axis=0), bins=40, color="#55A868", edgecolor="white")
axes[2].set_title("Feature Means (after scaling)", fontweight="bold")
axes[2].set_xlabel("Mean value")

plt.suptitle("Feature Statistics", fontsize=14, fontweight="bold")
save("03_feature_statistics.png")

# ═════════════════════════════════════════════
# STEP 3 — FEATURE SELECTION
# ═════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  STEP 3 — FEATURE SELECTION")
print(f"{'═'*60}")

print(f"  Stage 1: Mutual information...")
mi_scores = mutual_info_classif(X_scaled, y, random_state=RANDOM_STATE)
mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

top_120   = mi_series.head(120).index.tolist()
print(f"  → Top 120 by mutual information selected")

# ── FIG 4: MI score distribution ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(len(mi_series)), mi_series.values, color="#4C72B0", linewidth=1.5)
axes[0].axvline(120, color="#C44E52", linestyle="--", linewidth=2, label="Top 120 cutoff")
axes[0].set_title("Mutual Information Scores (all features)", fontweight="bold")
axes[0].set_xlabel("Feature rank")
axes[0].set_ylabel("MI score")
axes[0].legend()

mi_top30 = mi_series.head(30)
axes[1].barh(mi_top30.index[::-1], mi_top30.values[::-1], color="#4C72B0", alpha=0.85)
axes[1].set_title("Top 30 Features by Mutual Information", fontweight="bold")
axes[1].set_xlabel("MI Score")
save("04_mutual_information.png")

print(f"  Stage 2: Random Forest importance...")
X_top120 = X_scaled[:, [feature_cols.index(c) for c in top_120]]

rf_sel = RandomForestClassifier(
    n_estimators=200, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1
)
rf_sel.fit(X_top120, y)

rf_importance = pd.Series(
    rf_sel.feature_importances_, index=top_120
).sort_values(ascending=False)

top_80 = rf_importance.head(80).index.tolist()
print(f"  → Top 80 by RF importance selected")

fi_df = pd.DataFrame({
    "feature":    rf_importance.head(80).index,
    "importance": rf_importance.head(80).values,
    "mi_score":   [mi_series[f] for f in rf_importance.head(80).index]
})
fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)

final_features = top_80
final_idx      = [feature_cols.index(c) for c in final_features]
X_final        = X_scaled[:, final_idx]
print(f"  Final features: {len(final_features)}")

# ── FIG 5: Feature importance (RF) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

top30 = fi_df.head(30)
colors_bar = plt.cm.Blues(np.linspace(0.4, 0.9, 30))[::-1]
axes[0].barh(top30["feature"][::-1], top30["importance"][::-1], color=colors_bar)
axes[0].set_title("Top 30 Features — RF Importance", fontweight="bold")
axes[0].set_xlabel("Importance")

axes[1].scatter(fi_df["mi_score"], fi_df["importance"], alpha=0.7,
                color="#4C72B0", edgecolors="white", linewidth=0.5, s=60)
axes[1].set_xlabel("Mutual Information Score")
axes[1].set_ylabel("RF Importance")
axes[1].set_title("MI Score vs RF Importance", fontweight="bold")
# Annotate top 10
for _, row in fi_df.head(10).iterrows():
    axes[1].annotate(row["feature"][:15], (row["mi_score"], row["importance"]),
                     fontsize=7, alpha=0.8,
                     xytext=(4, 4), textcoords="offset points")

save("05_feature_importance.png")

# ── FIG 6: Correlation heatmap (top 25 features) ─────────
top25_names = final_features[:25]
top25_idx   = [final_features.index(c) for c in top25_names]
X_top25     = X_final[:, top25_idx]
corr_matrix = np.corrcoef(X_top25.T)

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask,
    xticklabels=[f[:12] for f in top25_names],
    yticklabels=[f[:12] for f in top25_names],
    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 7},
    linewidths=0.5, ax=ax,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"}
)
ax.set_title("Feature Correlation Matrix (Top 25 Features)", fontweight="bold", fontsize=14)
plt.xticks(rotation=45, ha="right")
save("06_correlation_heatmap.png")

# ── FIG 7: Class-wise feature distributions (top 6) ──────
top6 = final_features[:6]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, feat in zip(axes.flatten(), top6):
    feat_idx = final_features.index(feat)
    for cls_idx, cls_name in enumerate(le.classes_):
        vals = X_final[y == cls_idx, feat_idx]
        ax.hist(vals, bins=25, alpha=0.6, label=cls_name,
                color=PALETTE[cls_idx], density=True, edgecolor="white")
    ax.set_title(feat[:25], fontweight="bold", fontsize=9)
    ax.legend(fontsize=7)
    ax.set_ylabel("Density")
plt.suptitle("Class-wise Distributions — Top 6 Features", fontsize=14, fontweight="bold")
save("07_feature_distributions_by_class.png")

# ── FIG 8: Box plots top 10 features by class ────────────
top10 = final_features[:10]
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, feat in zip(axes.flatten(), top10):
    feat_idx = final_features.index(feat)
    data_by_class = [X_final[y == c, feat_idx] for c in range(len(le.classes_))]
    bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(le.classes_, fontsize=8)
    ax.set_title(feat[:20], fontsize=8, fontweight="bold")
plt.suptitle("Feature Box Plots by Class (Top 10 Features)", fontsize=14, fontweight="bold")
save("08_boxplots_by_class.png")

# ── FIG 9: PCA 2D projection ─────────────────────────────
pca   = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_final)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cls_idx, cls_name in enumerate(le.classes_):
    mask_cls = y == cls_idx
    axes[0].scatter(X_pca[mask_cls, 0], X_pca[mask_cls, 1],
                    label=cls_name, alpha=0.7, s=40,
                    color=PALETTE[cls_idx], edgecolors="white", linewidth=0.4)
axes[0].set_title(f"PCA (var explained: {pca.explained_variance_ratio_.sum():.1%})", fontweight="bold")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[0].legend()

# Scree plot
pca_full  = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_final)
cumvar    = np.cumsum(pca_full.explained_variance_ratio_)
n_to_95   = np.argmax(cumvar >= 0.95) + 1
axes[1].plot(cumvar, color="#4C72B0", linewidth=2)
axes[1].axhline(0.95, color="#C44E52", linestyle="--", label=f"95% @ PC{n_to_95}")
axes[1].axvline(n_to_95, color="#C44E52", linestyle="--")
axes[1].fill_between(range(len(cumvar)), cumvar, alpha=0.15, color="#4C72B0")
axes[1].set_title("PCA Scree Plot — Cumulative Variance", fontweight="bold")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance Explained")
axes[1].legend()

save("09_pca_projection.png")

# ── FIG 10: t-SNE 2D projection ───────────────────────────
print(f"  Computing t-SNE (may take a moment)...")
tsne   = TSNE(n_components=2, perplexity=min(30, len(X_final)//4),
              random_state=RANDOM_STATE, max_iter=1000)
X_tsne = tsne.fit_transform(X_final)

fig, ax = plt.subplots(figsize=(8, 6))
for cls_idx, cls_name in enumerate(le.classes_):
    mask_cls = y == cls_idx
    ax.scatter(X_tsne[mask_cls, 0], X_tsne[mask_cls, 1],
               label=cls_name, alpha=0.75, s=50,
               color=PALETTE[cls_idx], edgecolors="white", linewidth=0.5)
ax.set_title("t-SNE 2D Projection of Feature Space", fontweight="bold", fontsize=13)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(title="Class")
save("10_tsne_projection.png")

# ═════════════════════════════════════════════
# STEP 4 — DEFINE MODELS
# ═════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  STEP 4 — MODEL DEFINITIONS")
print(f"{'═'*60}")

models = {}

models["RandomForest"] = RandomForestClassifier(
    n_estimators=500, max_depth=None,
    min_samples_split=2, min_samples_leaf=1,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
models["ExtraTrees"] = ExtraTreesClassifier(
    n_estimators=500, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1
)
models["SVM_RBF"] = SVC(
    kernel="rbf", C=10, gamma="scale",
    class_weight="balanced", probability=True,
    random_state=RANDOM_STATE
)
models["SVM_Linear"] = SVC(
    kernel="linear", C=1.0,
    class_weight="balanced", probability=True,
    random_state=RANDOM_STATE
)
models["GradientBoosting"] = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=4, subsample=0.8, random_state=RANDOM_STATE
)
models["LogisticRegression"] = LogisticRegression(
    C=1.0, class_weight="balanced",
    max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"
)

if XGB_OK:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1
    )
if LGB_OK:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1
    )

print(f"  {len(models)} models defined")

# ═════════════════════════════════════════════
# STEP 5 — CROSS VALIDATION
# ═════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  STEP 5 — CROSS VALIDATION ({CV_FOLDS}-fold stratified)")
print(f"{'═'*60}")

cv         = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

for name, model in models.items():
    print(f"\n  ▶ {name}...")

    if SMOTE_OK:
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE,
                            k_neighbors=min(3, min(class_counts.values())-1))),
            ("model", model)
        ])
    else:
        pipeline = model

    scores = cross_validate(
        pipeline, X_final, y,
        cv=cv,
        scoring={
            "accuracy":          "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro":          "f1_macro",
            "f1_weighted":       "f1_weighted",
        },
        n_jobs=-1 if not SMOTE_OK else 1,
        return_train_score=True
    )

    cv_results[name] = {
        "acc_mean":       scores["test_accuracy"].mean(),
        "acc_std":        scores["test_accuracy"].std(),
        "bal_acc_mean":   scores["test_balanced_accuracy"].mean(),
        "bal_acc_std":    scores["test_balanced_accuracy"].std(),
        "f1_macro_mean":  scores["test_f1_macro"].mean(),
        "f1_macro_std":   scores["test_f1_macro"].std(),
        "f1_w_mean":      scores["test_f1_weighted"].mean(),
        "f1_w_std":       scores["test_f1_weighted"].std(),
        "train_acc_mean": scores["train_accuracy"].mean(),
    }

    r = cv_results[name]
    print(f"    Accuracy         : {r['acc_mean']:.3f} ± {r['acc_std']:.3f}")
    print(f"    Balanced Accuracy: {r['bal_acc_mean']:.3f} ± {r['bal_acc_std']:.3f}")
    print(f"    F1 Macro         : {r['f1_macro_mean']:.3f} ± {r['f1_macro_std']:.3f}")
    print(f"    Train Accuracy   : {r['train_acc_mean']:.3f}  (overfit check)")

results_df = pd.DataFrame(cv_results).T.sort_values("bal_acc_mean", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"))

# ── FIG 11: Model comparison bar chart ───────────────────
metrics = ["acc_mean", "bal_acc_mean", "f1_macro_mean", "f1_w_mean"]
labels  = ["Accuracy", "Bal. Accuracy", "F1 Macro", "F1 Weighted"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, metric, label in zip(axes.flatten(), metrics, labels):
    vals  = results_df[metric]
    stds  = results_df[metric.replace("mean", "std")]
    colors_m = [PALETTE[i % len(PALETTE)] for i in range(len(vals))]
    bars = ax.bar(vals.index, vals.values, yerr=stds.values,
                  color=colors_m, edgecolor="white", linewidth=1.5,
                  capsize=5, alpha=0.85)
    ax.set_title(label, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Score")
    for bar, v in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.suptitle("Model Comparison — Cross-Validation Results", fontsize=15, fontweight="bold")
save("11_model_comparison.png")

# ── FIG 12: Model comparison radar ───────────────────────
def radar_chart(results_df, metrics, labels, title, fname):
    n  = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for i, (name, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics] + [row[metrics[0]]]
        ax.plot(angles, values, "o-", linewidth=2,
                label=name, color=PALETTE[i % len(PALETTE)])
        ax.fill(angles, values, alpha=0.08, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    save(fname)

radar_chart(results_df, metrics, labels,
            "Model Comparison — Radar Chart", "12_model_radar.png")

# ── FIG 13: Overfitting analysis ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x      = np.arange(len(results_df))
width  = 0.35
bars1  = ax.bar(x - width/2, results_df["train_acc_mean"], width,
                label="Train Accuracy", color="#4C72B0", alpha=0.85, edgecolor="white")
bars2  = ax.bar(x + width/2, results_df["acc_mean"], width,
                label="Test Accuracy", color="#DD8452", alpha=0.85, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(results_df.index, rotation=30, ha="right")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Accuracy")
ax.set_title("Train vs Test Accuracy (Overfitting Check)", fontweight="bold")
ax.legend()
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
save("13_overfitting_analysis.png")

# ═════════════════════════════════════════════
# STEP 6 — BEST MODEL DEEP EVALUATION
# ═════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  STEP 6 — BEST MODEL DEEP EVALUATION")
print(f"{'═'*60}")

best_name  = results_df.index[0]
best_model = models[best_name]
print(f"  Best model: {best_name}")

all_y_true = []
all_y_pred = []
all_y_prob = []

for train_idx, test_idx in cv.split(X_final, y):
    X_tr, X_te = X_final[train_idx], X_final[test_idx]
    y_tr, y_te = y[train_idx],       y[test_idx]

    if SMOTE_OK:
        smote = SMOTE(random_state=RANDOM_STATE,
                      k_neighbors=min(3, min(Counter(y_tr).values())-1))
        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

    best_model.fit(X_tr, y_tr)
    y_pred = best_model.predict(X_te)
    y_prob = best_model.predict_proba(X_te) if hasattr(best_model, "predict_proba") else None

    all_y_true.extend(y_te)
    all_y_pred.extend(y_pred)
    if y_prob is not None:
        all_y_prob.extend(y_prob)

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

print(f"\n  Classification Report ({best_name}):")
report_str = classification_report(all_y_true, all_y_pred, target_names=le.classes_)
print(report_str)
with open(os.path.join(OUTPUT_DIR, f"classification_report_{best_name}.txt"), "w") as f:
    f.write(report_str)

report_dict = classification_report(all_y_true, all_y_pred,
                                    target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).T
report_df.to_csv(os.path.join(OUTPUT_DIR, f"classification_report_{best_name}.csv"))

# ── FIG 14: Confusion matrix (counts) ────────────────────
cm = confusion_matrix(all_y_true, all_y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_, yticklabels=le.classes_,
            cmap="Blues", linewidths=0.5, ax=axes[0],
            cbar_kws={"shrink": 0.8})
axes[0].set_title(f"Confusion Matrix — Counts\n({best_name})", fontweight="bold")
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted Label")

# Normalized confusion matrix
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2f",
            xticklabels=le.classes_, yticklabels=le.classes_,
            cmap="Blues", linewidths=0.5, ax=axes[1],
            vmin=0, vmax=1, cbar_kws={"shrink": 0.8, "label": "Rate"})
axes[1].set_title(f"Confusion Matrix — Normalized\n({best_name})", fontweight="bold")
axes[1].set_ylabel("True Label")
axes[1].set_xlabel("Predicted Label")

plt.suptitle(f"Confusion Matrices — {best_name} (5-fold CV)", fontsize=14, fontweight="bold")
save(f"14_confusion_matrix_{best_name}.png")

# ── FIG 15: Per-class metrics bar chart ──────────────────
per_class = report_df.loc[le.classes_, ["precision", "recall", "f1-score"]]
fig, ax   = plt.subplots(figsize=(10, 5))
per_class.plot(kind="bar", ax=ax, color=PALETTE[:3],
               edgecolor="white", linewidth=1.2, alpha=0.85)
ax.set_ylim(0, 1.1)
ax.set_xticklabels(per_class.index, rotation=0)
ax.set_title(f"Per-Class Metrics — {best_name}", fontweight="bold")
ax.set_ylabel("Score")
ax.legend(title="Metric")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)
save("15_per_class_metrics.png")

# ── FIG 16: ROC curves (OvR) ─────────────────────────────
if all_y_prob:
    all_y_prob_arr = np.array(all_y_prob)
    y_bin = label_binarize(all_y_true, classes=list(range(len(le.classes_))))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls_name in enumerate(le.classes_):
        if y_bin.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_bin[:, i], all_y_prob_arr[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2.5, color=PALETTE[i],
                    label=f"{cls_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5)
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves (One-vs-Rest) — {best_name}", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right")
    save("16_roc_curves.png")

    # ── FIG 17: Precision-Recall curves ──────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls_name in enumerate(le.classes_):
        if y_bin.shape[1] > 1:
            prec, rec, _ = precision_recall_curve(y_bin[:, i], all_y_prob_arr[:, i])
            ap           = average_precision_score(y_bin[:, i], all_y_prob_arr[:, i])
            ax.plot(rec, prec, linewidth=2.5, color=PALETTE[i],
                    label=f"{cls_name} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves — {best_name}", fontweight="bold", fontsize=13)
    ax.legend(loc="upper right")
    save("17_precision_recall_curves.png")

    # ── FIG 18: Calibration curves ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls_name in enumerate(le.classes_):
        if y_bin.shape[1] > 1:
            fraction_pos, mean_pred = calibration_curve(
                y_bin[:, i], all_y_prob_arr[:, i], n_bins=10, strategy="uniform"
            )
            ax.plot(mean_pred, fraction_pos, "s-", linewidth=2,
                    color=PALETTE[i], label=cls_name)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curves — {best_name}", fontweight="bold", fontsize=13)
    ax.legend()
    save("18_calibration_curves.png")

# ── FIG 19: Learning curves ───────────────────────────────
print(f"  Computing learning curves...")
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_final, y,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
    train_sizes=np.linspace(0.2, 1.0, 8),
    scoring="balanced_accuracy",
    n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
test_mean  = test_scores.mean(axis=1)
test_std   = test_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(train_sizes, train_mean, "o-", color="#4C72B0", linewidth=2.5, label="Train")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.15, color="#4C72B0")
ax.plot(train_sizes, test_mean, "s-", color="#DD8452", linewidth=2.5, label="CV Test")
ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                alpha=0.15, color="#DD8452")
ax.set_xlabel("Training Set Size", fontsize=12)
ax.set_ylabel("Balanced Accuracy", fontsize=12)
ax.set_title(f"Learning Curves — {best_name}", fontweight="bold", fontsize=13)
ax.legend()
ax.set_ylim(0, 1.1)
save("19_learning_curves.png")

# ── FIG 20: Error analysis — misclassification heatmap ───
errors_df = pd.DataFrame({
    "true":  le.inverse_transform(all_y_true),
    "pred":  le.inverse_transform(all_y_pred),
    "correct": all_y_true == all_y_pred
})
error_pivot = pd.crosstab(errors_df["true"], errors_df["pred"])

fig, ax = plt.subplots(figsize=(8, 6))
np.fill_diagonal(error_pivot.values, 0)  # zero out diagonal to highlight errors
sns.heatmap(error_pivot, annot=True, fmt="d", cmap="Reds",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Error Count"})
ax.set_title(f"Error Analysis — Off-Diagonal Misclassifications\n({best_name})",
             fontweight="bold")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")
save("20_error_analysis.png")

# ═════════════════════════════════════════════
# STEP 7 — ENSEMBLE
# ═════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  STEP 7 — ENSEMBLE (Top 3 models)")
print(f"{'═'*60}")

top3_names  = results_df.index[:3].tolist()
top3_models = [(name, models[name]) for name in top3_names]
print(f"  Combining: {top3_names}")

ensemble = VotingClassifier(estimators=top3_models, voting="soft")

if SMOTE_OK:
    ens_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE,
                        k_neighbors=min(3, min(class_counts.values())-1))),
        ("ensemble", ensemble)
    ])
else:
    ens_pipeline = ensemble

ens_scores = cross_validate(
    ens_pipeline, X_final, y,
    cv=cv,
    scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
             "f1_macro": "f1_macro"},
    n_jobs=1
)

print(f"  Ensemble Accuracy    : {ens_scores['test_accuracy'].mean():.3f} ± {ens_scores['test_accuracy'].std():.3f}")
print(f"  Ensemble Bal. Acc    : {ens_scores['test_balanced_accuracy'].mean():.3f} ± {ens_scores['test_balanced_accuracy'].std():.3f}")
print(f"  Ensemble F1 Macro    : {ens_scores['test_f1_macro'].mean():.3f} ± {ens_scores['test_f1_macro'].std():.3f}")

# ── FIG 21: Model vs Ensemble comparison ─────────────────
ensemble_row = {
    "bal_acc_mean": ens_scores["test_balanced_accuracy"].mean(),
    "bal_acc_std":  ens_scores["test_balanced_accuracy"].std(),
    "f1_macro_mean": ens_scores["test_f1_macro"].mean(),
    "acc_mean":     ens_scores["test_accuracy"].mean(),
}
summary = results_df[["bal_acc_mean", "bal_acc_std", "f1_macro_mean", "acc_mean"]].copy()
summary.loc["Ensemble (Top3)"] = ensemble_row

fig, ax = plt.subplots(figsize=(12, 5))
x   = np.arange(len(summary))
col = [PALETTE[i % len(PALETTE)] for i in range(len(summary))]
col[-1] = "#C44E52"  # highlight ensemble
bars = ax.bar(x, summary["bal_acc_mean"], yerr=summary["bal_acc_std"],
              color=col, edgecolor="white", linewidth=1.5,
              capsize=5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(summary.index, rotation=30, ha="right")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Balanced Accuracy")
ax.set_title("All Models + Ensemble — Balanced Accuracy", fontweight="bold")
for bar, v in zip(bars, summary["bal_acc_mean"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
save("21_model_vs_ensemble.png")

# ═════════════════════════════════════════════
# STEP 8 — SHAP ANALYSIS
# ═════════════════════════════════════════════
if SHAP_OK and XGB_OK:
    print(f"\n{'═'*60}")
    print(f"  STEP 8 — SHAP FEATURE ANALYSIS")
    print(f"{'═'*60}")
    try:
        xgb_final = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1
        )
        xgb_final.fit(X_final, y)
        explainer = shap.TreeExplainer(xgb_final)
        shap_vals = explainer.shap_values(X_final)

        if isinstance(shap_vals, list):
            shap_mean = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            shap_mean = np.abs(shap_vals).mean(axis=0)

        shap_df = pd.DataFrame({
            "feature":   final_features,
            "shap_mean": shap_mean.mean(axis=0) if shap_mean.ndim > 1 else shap_mean
        }).sort_values("shap_mean", ascending=False)

        shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)

        # ── FIG 22: SHAP bar ─────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        top25_shap = shap_df.head(25)
        colors_shap = plt.cm.Oranges(np.linspace(0.4, 0.9, 25))[::-1]
        ax.barh(top25_shap["feature"][::-1], top25_shap["shap_mean"][::-1],
                color=colors_shap)
        ax.set_title("Top 25 Features — Mean |SHAP| Value (XGBoost)", fontweight="bold")
        ax.set_xlabel("Mean |SHAP|")
        save("22_shap_importance.png")

        # ── FIG 23: SHAP beeswarm (native) ───────────────
        plt.figure(figsize=(12, 9))
        shap.summary_plot(
            shap_vals if not isinstance(shap_vals, list) else np.array(shap_vals).mean(axis=0),
            X_final,
            feature_names=[f[:20] for f in final_features],
            max_display=20,
            show=False
        )
        plt.title("SHAP Summary — Feature Impact", fontweight="bold")
        save("23_shap_beeswarm.png")

        print(f"  ✓ SHAP analysis saved")
    except Exception as e:
        print(f"  ⚠ SHAP failed: {e}")

# ═════════════════════════════════════════════
# STEP 9 — PAIRWISE BINARY
# ═════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  STEP 9 — PAIRWISE BINARY CLASSIFICATION")
print(f"{'═'*60}")

pairs        = list(combinations(list(le.classes_), 2))
pair_results = {}

for cls_a, cls_b in pairs:
    mask   = np.isin(y_raw.values, [cls_a, cls_b])
    X_pair = X_final[mask]
    y_pair = (y_raw.values[mask] == cls_b).astype(int)

    if len(X_pair) < 10:
        print(f"  {cls_a} vs {cls_b}: not enough samples")
        continue

    rf_bin = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    cv_bin = StratifiedKFold(
        n_splits=min(CV_FOLDS, min(Counter(y_pair).values())),
        shuffle=True, random_state=RANDOM_STATE
    )
    bin_scores = cross_val_score(rf_bin, X_pair, y_pair,
                                 cv=cv_bin, scoring="balanced_accuracy")
    pair_results[f"{cls_a}_vs_{cls_b}"] = {
        "balanced_accuracy": bin_scores.mean(),
        "std": bin_scores.std(),
        "n_samples": len(X_pair)
    }
    print(f"  {cls_a:8s} vs {cls_b:8s} | "
          f"BalAcc={bin_scores.mean():.3f} ± {bin_scores.std():.3f} "
          f"| n={len(X_pair)}")

pair_df = pd.DataFrame(pair_results).T
pair_df.to_csv(os.path.join(OUTPUT_DIR, "pairwise_results.csv"))

# ── FIG 24: Pairwise results ─────────────────────────────
if pair_results:
    fig, ax = plt.subplots(figsize=(10, 4))
    pairs_list = list(pair_results.keys())
    bal_accs   = [pair_results[p]["balanced_accuracy"] for p in pairs_list]
    stds       = [pair_results[p]["std"] for p in pairs_list]
    n_samps    = [pair_results[p]["n_samples"] for p in pairs_list]

    bars = ax.bar(pairs_list, bal_accs, yerr=stds,
                  color=[PALETTE[i] for i in range(len(pairs_list))],
                  edgecolor="white", linewidth=1.5, capsize=6, alpha=0.85)
    for bar, v, n in zip(bars, bal_accs, n_samps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}\n(n={n})", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Pairwise Binary Classification — Balanced Accuracy", fontweight="bold")
    save("24_pairwise_results.png")

# ── FIG 25: Grand summary dashboard ─────────────────────
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 — model bal_acc
ax1 = fig.add_subplot(gs[0, 0])
vals = results_df["bal_acc_mean"]
ax1.barh(vals.index[::-1], vals.values[::-1],
         color=[PALETTE[i % len(PALETTE)] for i in range(len(vals))], alpha=0.85)
ax1.set_title("Balanced Accuracy", fontweight="bold")
ax1.set_xlim(0, 1)

# Panel 2 — F1 macro
ax2 = fig.add_subplot(gs[0, 1])
f1v = results_df["f1_macro_mean"]
ax2.barh(f1v.index[::-1], f1v.values[::-1],
         color=[PALETTE[i % len(PALETTE)] for i in range(len(f1v))], alpha=0.85)
ax2.set_title("F1 Macro", fontweight="bold")
ax2.set_xlim(0, 1)

# Panel 3 — Confusion matrix (best model)
ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(cm_norm, annot=True, fmt=".2f",
            xticklabels=le.classes_, yticklabels=le.classes_,
            cmap="Blues", linewidths=0.5, ax=ax3, cbar=False)
ax3.set_title(f"Norm. CM — {best_name}", fontweight="bold")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("True")

# Panel 4 — PCA
ax4 = fig.add_subplot(gs[1, 0])
for cls_idx, cls_name in enumerate(le.classes_):
    mask_cls = y == cls_idx
    ax4.scatter(X_pca[mask_cls, 0], X_pca[mask_cls, 1],
                label=cls_name, alpha=0.6, s=25,
                color=PALETTE[cls_idx])
ax4.set_title("PCA 2D", fontweight="bold")
ax4.legend(fontsize=7)

# Panel 5 — t-SNE
ax5 = fig.add_subplot(gs[1, 1])
for cls_idx, cls_name in enumerate(le.classes_):
    mask_cls = y == cls_idx
    ax5.scatter(X_tsne[mask_cls, 0], X_tsne[mask_cls, 1],
                label=cls_name, alpha=0.6, s=25,
                color=PALETTE[cls_idx])
ax5.set_title("t-SNE 2D", fontweight="bold")
ax5.legend(fontsize=7)

# Panel 6 — per-class metrics
ax6 = fig.add_subplot(gs[1, 2])
per_class.plot(kind="bar", ax=ax6, color=PALETTE[:3],
               edgecolor="white", alpha=0.85, linewidth=1)
ax6.set_title(f"Per-class Metrics\n{best_name}", fontweight="bold")
ax6.set_ylim(0, 1.15)
ax6.tick_params(axis="x", rotation=0)
ax6.legend(fontsize=7)

fig.suptitle("Classification Pipeline — Summary Dashboard", fontsize=16, fontweight="bold")
save("25_summary_dashboard.png")

# ═════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  FINAL SUMMARY")
print(f"{'═'*60}")
print(f"  Dataset : {len(df)} samples × {len(final_features)} features")
print(f"  Classes : {list(le.classes_)}")
print(f"\n  ── Individual Models ──")
for name, row in results_df.iterrows():
    bar = "█" * int(row["bal_acc_mean"] * 20)
    print(f"  {name:20s} {bar:20s} {row['bal_acc_mean']:.3f} ± {row['bal_acc_std']:.3f}")

print(f"\n  ── Ensemble ──")
print(f"  Top-3 Ensemble  {ens_scores['test_balanced_accuracy'].mean():.3f} ± {ens_scores['test_balanced_accuracy'].std():.3f}")

print(f"\n  ── Pairwise ──")
for pair, res in pair_results.items():
    print(f"  {pair:25s} {res['balanced_accuracy']:.3f} ± {res['std']:.3f}")

print(f"\n  ── Figures saved ({OUTPUT_DIR}) ──")
figures = [
    "01_class_distribution.png", "02_missing_data.png",
    "03_feature_statistics.png", "04_mutual_information.png",
    "05_feature_importance.png", "06_correlation_heatmap.png",
    "07_feature_distributions_by_class.png", "08_boxplots_by_class.png",
    "09_pca_projection.png",     "10_tsne_projection.png",
    "11_model_comparison.png",   "12_model_radar.png",
    "13_overfitting_analysis.png",
    f"14_confusion_matrix_{best_name}.png",
    "15_per_class_metrics.png",  "16_roc_curves.png",
    "17_precision_recall_curves.png", "18_calibration_curves.png",
    "19_learning_curves.png",    "20_error_analysis.png",
    "21_model_vs_ensemble.png",  "22_shap_importance.png",
    "23_shap_beeswarm.png",      "24_pairwise_results.png",
    "25_summary_dashboard.png",
]
for f in figures:
    print(f"    {f}")