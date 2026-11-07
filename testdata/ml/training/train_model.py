"""
Train & export Alzheimer's voice classifier with:
  - Aggressive feature selection (drop low-importance / redundant features)
  - MCI recall focus (class weights + probability threshold tuning)
  - Multiple model comparison
  - joblib pickle export for inference
"""

import os
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

from ml.config import (
    CLASSIFIER_PKL,
    FEATURES_ULTIMATE_CSV,
    REPORTS_DIR,
    SELECTED_FEATURES_CSV,
    ensure_dirs,
)

ensure_dirs()

CSV_FILE = str(FEATURES_ULTIMATE_CSV)
OUTPUT_DIR = str(REPORTS_DIR) + os.sep
MODEL_PATH = str(CLASSIFIER_PKL)
FEATURES_PATH = str(SELECTED_FEATURES_CSV)

RANDOM_STATE = 42
CV_FOLDS = 5
TARGET_COL = "patient_type"
DROP_COLS = ["sample", "transcript", "duration", TARGET_COL]

MISSING_THRESH = 0.40
MI_MIN_SCORE = 0.005          # drop near-zero MI features
MI_TOP_N = 100                  # cap after MI filter
RF_CUMULATIVE = 0.95            # keep features until 95% RF importance mass
CORR_THRESH = 0.92              # drop one of highly correlated pair
MIN_RF_IMPORTANCE = 0.001       # absolute floor on RF importance


try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

try:
    from xgboost import XGBClassifier

    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    from lightgbm import LGBMClassifier

    LGB_OK = True
except ImportError:
    LGB_OK = False


# -- Feature selection --
def select_features(df: pd.DataFrame, feature_cols: list, y: np.ndarray) -> tuple[list, pd.DataFrame]:
    log = []

    # Missing values
    miss = df[feature_cols].isnull().mean()
    keep = miss[miss < MISSING_THRESH].index.tolist()
    log.append(f"  After missing (<{MISSING_THRESH:.0%}): {len(keep)} (dropped {len(feature_cols) - len(keep)})")
    feature_cols = keep

    # Constant / near-constant
    std = df[feature_cols].std()
    keep = std[std > 1e-8].index.tolist()
    log.append(f"  After variance filter: {len(keep)} (dropped {len(std) - len(keep)} constant)")

    X_raw = df[feature_cols].values
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Mutual information
    mi = mutual_info_classif(X_scaled, y, random_state=RANDOM_STATE)
    mi_s = pd.Series(mi, index=feature_cols).sort_values(ascending=False)
    mi_keep = mi_s[mi_s >= MI_MIN_SCORE].head(MI_TOP_N).index.tolist()
    log.append(f"  After MI (min={MI_MIN_SCORE}, top<={MI_TOP_N}): {len(mi_keep)}")

    idx = [feature_cols.index(c) for c in mi_keep]
    X_mi = X_scaled[:, idx]

    # Random Forest importance (balanced for minority MCI)
    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_mi, y)
    imp = pd.Series(rf.feature_importances_, index=mi_keep).sort_values(ascending=False)
    imp = imp[imp >= MIN_RF_IMPORTANCE]

    # Cumulative importance
    cum = imp.cumsum() / imp.sum()
    n_cum = int((cum <= RF_CUMULATIVE).sum()) + 1
    n_keep = min(max(n_cum, 25), len(imp))
    rf_keep = imp.head(n_keep).index.tolist()
    log.append(f"  After RF importance (>={MIN_RF_IMPORTANCE}, {RF_CUMULATIVE:.0%} mass): {len(rf_keep)}")

    idx_rf = [mi_keep.index(c) for c in rf_keep]
    X_rf = X_mi[:, idx_rf]
    imp_rf = imp.loc[rf_keep]

    # Correlation pruning — drop lower-importance feature in each pair
    corr = np.corrcoef(X_rf.T)
    to_drop = set()
    names = list(rf_keep)
    for i in range(len(names)):
        if names[i] in to_drop:
            continue
        for j in range(i + 1, len(names)):
            if names[j] in to_drop:
                continue
            if abs(corr[i, j]) >= CORR_THRESH:
                if imp_rf[names[i]] >= imp_rf[names[j]]:
                    to_drop.add(names[j])
                else:
                    to_drop.add(names[i])
                    break
    final = [c for c in rf_keep if c not in to_drop]
    log.append(f"  After correlation prune (|r|>={CORR_THRESH}): {len(final)} (dropped {len(to_drop)} redundant)")

    selection_df = pd.DataFrame(
        {
            "feature": imp.index,
            "rf_importance": imp.values,
            "mi_score": [mi_s[f] for f in imp.index],
            "selected": [f in final for f in imp.index],
        }
    ).sort_values("rf_importance", ascending=False)

    for line in log:
        print(line)

    return final, selection_df


def tune_mci_threshold(y_true, y_prob, mci_idx: int) -> float | None:
    """
    Tune MCI probability threshold. Returns None to use plain argmax if tuning
    does not beat the default predictor without hurting macro F1.
    """
    pred_default = y_prob.argmax(axis=1)
    baseline_f1 = f1_score(y_true, pred_default, average="macro", zero_division=0)
    baseline_mci = recall_score(y_true == mci_idx, pred_default == mci_idx, zero_division=0)
    best_t = None
    best_score = baseline_mci * 0.45 + baseline_f1 * 0.55

    for t in np.arange(0.22, 0.48, 0.02):
        pred = predict_with_threshold(y_prob, mci_idx, t)
        mci_rec = recall_score(y_true == mci_idx, pred == mci_idx, zero_division=0)
        macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)
        if macro_f1 < baseline_f1 - 0.03:
            continue
        score = mci_rec * 0.45 + macro_f1 * 0.55
        if score > best_score + 0.01:
            best_score, best_t = score, t

    return best_t


def predict_with_threshold(
    proba: np.ndarray, mci_idx: int, threshold: float | None
) -> np.ndarray:
    pred = proba.argmax(axis=1)
    if threshold is None:
        return pred
    for i in range(len(proba)):
        if proba[i, mci_idx] >= threshold:
            pred[i] = mci_idx
    return pred


def build_models(mci_weight_boost: float = 2.0):
    """Class weights with extra boost for MCI (smallest / hardest class)."""
    counts = Counter(y)
    total = len(y)
    weights = {
        cls: (total / (len(counts) * cnt))
        * (mci_weight_boost if le.classes_[cls] == "MCI" else 1.0)
        for cls, cnt in counts.items()
    }
    print(f"  Sample weights map (by encoded class): {weights}")

    sample_w = np.array([weights[yi] for yi in y])

    models = {}

    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=8,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=6,
        subsample=0.85,
        random_state=RANDOM_STATE,
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models["LogisticRegression"] = LogisticRegression(
        C=0.5,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    models["SVM_RBF"] = SVC(
        kernel="rbf",
        C=5.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
    )

    if XGB_OK:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if LGB_OK:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    return models, sample_w


def main():
    run_training()


def run_training():
    global y, le

    print("=" * 60)
    print("  ALZHEIMER VOICE CLASSIFIER - TRAINING")
    print("=" * 60)

    df = pd.read_csv(CSV_FILE)
    print(f"\nLoaded {CSV_FILE}: {df.shape[0]} samples, {df.shape[1]} columns")
    print(df[TARGET_COL].value_counts().to_string())

    feature_cols_all = [c for c in df.columns if c not in DROP_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    mci_label = int(np.where(le.classes_ == "MCI")[0][0])
    print(f"Classes: {list(le.classes_)}  (MCI index={mci_label})")

    print("\n-- Feature selection --")
    selected_features, selection_df = select_features(df, feature_cols_all, y)
    selection_df.to_csv(FEATURES_PATH, index=False)
    print(f"  Saved feature report: {FEATURES_PATH}")

    X = df[selected_features].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_proc = scaler.fit_transform(imputer.fit_transform(X))

    print(f"\nFinal feature count: {len(selected_features)} (from {len(feature_cols_all)})")

    print("\n-- Model comparison (5-fold CV) --")
    models, sample_weight = build_models(mci_weight_boost=2.5)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    mci_scorer = make_scorer(
    lambda yt, yp: recall_score(yt, yp, labels=[mci_label], average="macro", zero_division=0)
    )

    results = []
    cv_probs = {}

    for name, model in models.items():
        use_sw = name in ("GradientBoosting", "XGBoost", "LightGBM")

        if SMOTE_OK:
            pipe = ImbPipeline(
                [
                    (
                        "smote",
                        SMOTE(
                            random_state=RANDOM_STATE,
                            k_neighbors=min(3, min(Counter(y).values()) - 1),
                        ),
                    ),
                    ("model", model),
                ]
            )
        else:
            pipe = model

        scores = cross_validate(
            pipe,
            X_proc,
            y,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "bal_acc": "balanced_accuracy",
                "f1_macro": "f1_macro",
                "mci_recall": mci_scorer,
            },
            n_jobs=-1 if not SMOTE_OK else 1,
            return_estimator=False,
        )

        results.append(
            {
                "model": name,
                "accuracy": scores["test_accuracy"].mean(),
                "bal_accuracy": scores["test_bal_acc"].mean(),
                "f1_macro": scores["test_f1_macro"].mean(),
                "mci_recall": scores["test_mci_recall"].mean(),
            }
        )
        print(
            f"  {name:22s} acc={scores['test_accuracy'].mean():.3f}  "
            f"bal={scores['test_bal_acc'].mean():.3f}  "
            f"f1={scores['test_f1_macro'].mean():.3f}  "
            f"MCI_rec={scores['test_mci_recall'].mean():.3f}"
        )

    results_df = pd.DataFrame(results)
    results_df["score"] = (
        results_df["mci_recall"] * 0.45
        + results_df["bal_accuracy"] * 0.30
        + results_df["f1_macro"] * 0.25
    )
    results_df = results_df.sort_values("score", ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "train_cv_comparison.csv"), index=False)

    best_name = results_df.iloc[0]["model"]
    print(f"\n  Best model (MCI-focused score): {best_name}")

    # -- Deep eval + threshold tuning on best model --
    best_model = models[best_name]
    use_sw = best_name in ("GradientBoosting", "XGBoost", "LightGBM")

    all_true, all_pred, all_prob = [], [], []
    fold_thresholds = []

    for train_idx, test_idx in cv.split(X_proc, y):
        X_tr, X_te = X_proc[train_idx], X_proc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if SMOTE_OK:
            sm = SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=min(3, min(Counter(y_tr).values()) - 1),
            )
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            est = best_model
            est.fit(X_tr, y_tr)
        else:
            est = best_model
            if use_sw:
                sw = sample_weight[train_idx]
                est.fit(X_tr, y_tr, sample_weight=sw)
            else:
                est.fit(X_tr, y_tr)

        prob = est.predict_proba(X_te)
        t = tune_mci_threshold(y_te, prob, mci_label)
        fold_thresholds.append(t)
        pred = predict_with_threshold(prob, mci_label, t)

        all_true.extend(y_te)
        all_pred.extend(pred)
        all_prob.extend(prob)

    valid_t = [t for t in fold_thresholds if t is not None]
    mci_threshold = float(np.median(valid_t)) if valid_t else None
    all_true = np.array(all_true)
    all_pred_default = np.array([p.argmax() for p in all_prob])
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)

    print(f"\n-- Evaluation: {best_name} (CV, default argmax) --")
    print(classification_report(all_true, all_pred_default, target_names=le.classes_))

    th_str = f"{mci_threshold:.2f}" if mci_threshold is not None else "argmax only"
    print(f"-- Evaluation: {best_name} (CV, MCI threshold={th_str}) --")
    report = classification_report(all_true, all_pred, target_names=le.classes_)
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report_tuned.txt"), "w") as f:
        f.write(f"Model: {best_name}\nMCI threshold: {mci_threshold}\n\n")
        f.write(report)

    print("\n-- Training final model on full dataset --")
    if SMOTE_OK:
        sm = SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=min(3, min(Counter(y).values()) - 1),
        )
        X_fit, y_fit = sm.fit_resample(X_proc, y)
        final_estimator = best_model
        final_estimator.fit(X_fit, y_fit)
    else:
        final_estimator = best_model
        if use_sw:
            final_estimator.fit(X_proc, y, sample_weight=sample_weight)
        else:
            final_estimator.fit(X_proc, y)

    ensemble = None
    top3 = results_df.head(3)["model"].tolist()
    if len(top3) >= 2:
        estimators = [(n, models[n]) for n in top3]
        ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        if SMOTE_OK:
            X_e, y_e = sm.fit_resample(X_proc, y)
            ensemble.fit(X_e, y_e)
        else:
            ensemble.fit(X_proc, y)
        print(f"  Also trained ensemble: {top3}")

    bundle = {
        "version": "1.0",
        "model_name": best_name,
        "label_encoder": le,
        "class_names": list(le.classes_),
        "mci_class_index": mci_label,
        "mci_probability_threshold": mci_threshold,
        "selected_features": selected_features,
        "imputer": imputer,
        "scaler": scaler,
        "classifier": final_estimator,
        "ensemble_top3": ensemble if len(top3) >= 2 else None,
        "feature_selection": {
            "original_count": len(feature_cols_all),
            "selected_count": len(selected_features),
            "mi_min": MI_MIN_SCORE,
            "corr_threshold": CORR_THRESH,
        },
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\n  Saved model bundle: {MODEL_PATH}")
    print(f"  Features used: {len(selected_features)}")
    print("  Top 10 features:")
    for feat in selection_df[selection_df["selected"]].head(10)["feature"]:
        print(f"    - {feat}")

    print("\n-- Load for inference --")
    print(f"  bundle = joblib.load('{CLASSIFIER_PKL}')")
    print("  X = df[bundle['selected_features']]")
    print("  X = bundle['scaler'].transform(bundle['imputer'].transform(X))")
    print("  proba = bundle['classifier'].predict_proba(X)")
    print("  pred = train_model.predict_with_threshold(proba, bundle['mci_class_index'], bundle['mci_probability_threshold'])")


if __name__ == "__main__":
    main()
