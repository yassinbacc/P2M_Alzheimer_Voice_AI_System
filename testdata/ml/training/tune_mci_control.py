"""
Grid-search MCI vs Control tradeoff (class weights + probability threshold).
Saves presets and updates artifacts/models/alzheimer_classifier.pkl
"""

import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, RobustScaler

from ml.config import (
    CLASSIFIER_PKL,
    FEATURES_ULTIMATE_CSV,
    MCI_TRADEOFF_CSV,
    MCI_TRADEOFF_PNG,
    REPORTS_DIR,
    ensure_dirs,
)
from ml.training.train_model import (
    CV_FOLDS,
    DROP_COLS,
    RANDOM_STATE,
    SMOTE_OK,
    TARGET_COL,
    predict_with_threshold,
    select_features,
)

warnings.filterwarnings("ignore")

ensure_dirs()

CSV_FILE = str(FEATURES_ULTIMATE_CSV)
MODEL_PATH = str(CLASSIFIER_PKL)
OUTPUT_DIR = str(REPORTS_DIR)
TRADEOFF_CSV = str(MCI_TRADEOFF_CSV)
TRADEOFF_PNG = str(MCI_TRADEOFF_PNG)
PRESETS_JSON = os.path.join(OUTPUT_DIR, "mci_control_presets.txt")

MCI_WEIGHT_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
THRESHOLD_GRID = [None] + [round(t, 2) for t in np.arange(0.20, 0.46, 0.02)]


def make_class_weights(y, le, mci_mult: float, control_mult: float = 1.0) -> dict:
    from collections import Counter

    counts = Counter(y)
    total = len(y)
    n_classes = len(counts)
    weights = {}
    for cls, cnt in counts.items():
        w = total / (n_classes * cnt)
        name = le.classes_[cls]
        if name == "MCI":
            w *= mci_mult
        elif name == "Control":
            w *= control_mult
        weights[cls] = w
    return weights


def eval_split(y_true, y_pred, le, mci_idx, control_idx):
    mci_rec = recall_score(y_true == mci_idx, y_pred == mci_idx, zero_division=0)
    ctrl_rec = recall_score(y_true == control_idx, y_pred == control_idx, zero_division=0)
    mci_f1 = f1_score(y_true == mci_idx, y_pred == mci_idx, zero_division=0)
    ctrl_f1 = f1_score(y_true == control_idx, y_pred == control_idx, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    # Harmonic mean of the two recalls (MCI vs Control focus)
    if mci_rec + ctrl_rec > 0:
        recall_hmean = 2 * mci_rec * ctrl_rec / (mci_rec + ctrl_rec)
    else:
        recall_hmean = 0.0
    return {
        "mci_recall": mci_rec,
        "control_recall": ctrl_rec,
        "mci_f1": mci_f1,
        "control_f1": ctrl_f1,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "recall_hmean": recall_hmean,
        "recall_gap": abs(mci_rec - ctrl_rec),
    }


def pick_presets(grid_df: pd.DataFrame) -> dict:
    """Choose three operating points on the Pareto surface."""
    presets = {}
    viable = grid_df[grid_df["macro_f1"] >= 0.62]

    # 1) Balanced: best harmonic mean of MCI & Control recall (macro F1 floor)
    pool = viable if len(viable) else grid_df
    presets["balanced"] = pool.loc[pool["recall_hmean"].idxmax()].to_dict()

    # 2) MCI priority: push MCI recall, keep Control >= 0.45
    mci_ok = grid_df[(grid_df["control_recall"] >= 0.45) & (grid_df["macro_f1"] >= 0.62)]
    if len(mci_ok):
        presets["mci_priority"] = mci_ok.loc[mci_ok["mci_recall"].idxmax()].to_dict()
    else:
        presets["mci_priority"] = grid_df.loc[grid_df["mci_recall"].idxmax()].to_dict()

    # 3) Control priority: maximize Control recall (macro F1 floor, MCI recall floor 0.48)
    ctrl_ok = grid_df[(grid_df["mci_recall"] >= 0.48) & (grid_df["macro_f1"] >= 0.65)]
    if len(ctrl_ok):
        presets["control_priority"] = ctrl_ok.loc[ctrl_ok["control_recall"].idxmax()].to_dict()
    else:
        presets["control_priority"] = pool.loc[pool["control_recall"].idxmax()].to_dict()

    return presets


def fit_final_lr(X_proc, y, le, mci_mult: float):
    from collections import Counter

    from imblearn.over_sampling import SMOTE

    cw = make_class_weights(y, le, mci_mult)
    model = LogisticRegression(
        C=0.5,
        class_weight=cw,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    if SMOTE_OK:
        sm = SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=min(3, min(Counter(y).values()) - 1),
        )
        X_fit, y_fit = sm.fit_resample(X_proc, y)
        model.fit(X_fit, y_fit)
    else:
        model.fit(X_proc, y)
    return model


def main():
    print("=" * 60)
    print("  MCI vs CONTROL TRADEOFF TUNING")
    print("=" * 60)

    df = pd.read_csv(CSV_FILE)
    feature_cols_all = [c for c in df.columns if c not in DROP_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    mci_idx = int(np.where(le.classes_ == "MCI")[0][0])
    control_idx = int(np.where(le.classes_ == "Control")[0][0])

    print("\n-- Feature selection (same as train_model) --")
    selected_features, _ = select_features(df, feature_cols_all, y)

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    X_proc = scaler.fit_transform(imputer.fit_transform(df[selected_features]))

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if not SMOTE_OK:
        raise RuntimeError("imbalanced-learn (SMOTE) required for tuning pipeline")

    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    rows = []

    print(f"\n-- Grid: {len(MCI_WEIGHT_GRID)} weights x {len(THRESHOLD_GRID)} thresholds --")

    for mci_mult in MCI_WEIGHT_GRID:
        cw = make_class_weights(y, le, mci_mult)
        lr = LogisticRegression(
            C=0.5,
            class_weight=cw,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        pipe = ImbPipeline(
            [
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE,
                        k_neighbors=min(3, min(__import__("collections").Counter(y).values()) - 1),
                    ),
                ),
                ("model", lr),
            ]
        )
        print(f"  mci_weight_mult={mci_mult:.1f} ...", end=" ", flush=True)

        oof_proba = cross_val_predict(
            pipe, X_proc, y, cv=cv, method="predict_proba", n_jobs=1
        )

        for thr in THRESHOLD_GRID:
            pred = predict_with_threshold(oof_proba, mci_idx, thr)
            m = eval_split(y, pred, le, mci_idx, control_idx)
            thr_label = "argmax" if thr is None else thr
            rows.append(
                {
                    "mci_weight_mult": mci_mult,
                    "mci_threshold": thr_label,
                    "mci_threshold_value": np.nan if thr is None else thr,
                    **m,
                }
            )
        print("done")

    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(TRADEOFF_CSV, index=False)
    print(f"\n  Saved grid: {TRADEOFF_CSV}")

    presets = pick_presets(grid_df)

    # Plot: MCI recall vs Control recall, colored by weight
    fig, ax = plt.subplots(figsize=(10, 7))
    for mci_mult in MCI_WEIGHT_GRID:
        sub = grid_df[grid_df["mci_weight_mult"] == mci_mult]
        ax.scatter(
            sub["control_recall"],
            sub["mci_recall"],
            s=40,
            alpha=0.7,
            label=f"w={mci_mult}",
        )
    for name, row in presets.items():
        ax.scatter(
            row["control_recall"],
            row["mci_recall"],
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=1,
            zorder=5,
            label=f"{name}",
        )
        ax.annotate(
            name,
            (row["control_recall"], row["mci_recall"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Control recall")
    ax.set_ylabel("MCI recall")
    ax.set_title("MCI vs Control recall tradeoff (5-fold OOF)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="equal recall")
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TRADEOFF_PNG, dpi=150)
    plt.close()
    print(f"  Saved plot: {TRADEOFF_PNG}")

    # Print presets
    print("\n" + "=" * 60)
    print("  RECOMMENDED PRESETS")
    print("=" * 60)

    lines = []
    for name, row in presets.items():
        thr = row["mci_threshold"]
        print(f"\n  [{name}]")
        print(f"    mci_weight_mult = {row['mci_weight_mult']}")
        print(f"    mci_threshold   = {thr}")
        print(f"    MCI recall      = {row['mci_recall']:.3f}")
        print(f"    Control recall  = {row['control_recall']:.3f}")
        print(f"    macro F1        = {row['macro_f1']:.3f}")
        print(f"    recall h-mean   = {row['recall_hmean']:.3f}")

        lines.append(f"[{name}] weight={row['mci_weight_mult']} threshold={thr}")
        lines.append(
            f"  MCI rec={row['mci_recall']:.3f}  Control rec={row['control_recall']:.3f}  "
            f"macro_f1={row['macro_f1']:.3f}\n"
        )

    with open(PRESETS_JSON, "w") as f:
        f.writelines(lines)

    # OOF eval for default balanced preset
    active = "balanced"
    prow = presets[active]
    mci_mult = float(prow["mci_weight_mult"])
    thr_val = prow["mci_threshold_value"]
    if np.isnan(thr_val):
        thr_val = None
    else:
        thr_val = float(thr_val)

    cw = make_class_weights(y, le, mci_mult)
    lr = LogisticRegression(C=0.5, class_weight=cw, max_iter=2000, random_state=RANDOM_STATE)
    pipe = ImbPipeline(
        [
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            ("model", lr),
        ]
    )
    oof_proba = cross_val_predict(pipe, X_proc, y, cv=cv, method="predict_proba", n_jobs=1)
    oof_pred = predict_with_threshold(oof_proba, mci_idx, thr_val)
    report = classification_report(y, oof_pred, target_names=le.classes_)

    print(f"\n-- OOF report (active preset: {active}) --")
    print(report)

    with open(os.path.join(OUTPUT_DIR, "classification_report_preset_balanced.txt"), "w") as f:
        f.write(f"preset={active}\nweight={mci_mult}\nthreshold={thr_val}\n\n{report}")

    # Train one classifier per preset (weights differ)
    print("\n-- Training final classifiers per preset --")
    preset_bundle = {}
    for name, row in presets.items():
        w = float(row["mci_weight_mult"])
        thr_raw = row["mci_threshold_value"]
        thr = None if (isinstance(thr_raw, float) and np.isnan(thr_raw)) else (
            None if row["mci_threshold"] == "argmax" else float(thr_raw)
        )
        clf = fit_final_lr(X_proc, y, le, w)
        preset_bundle[name] = {
            "mci_weight_mult": w,
            "mci_threshold": thr,
            "metrics": {
                "mci_recall": float(row["mci_recall"]),
                "control_recall": float(row["control_recall"]),
                "macro_f1": float(row["macro_f1"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "recall_hmean": float(row["recall_hmean"]),
            },
            "classifier": clf,
        }
        print(f"  {name}: weight={w}, threshold={thr}")

    active_row = preset_bundle[active]

    bundle = {}
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)

    bundle.update(
        {
            "version": "1.1",
            "model_name": "LogisticRegression",
            "label_encoder": le,
            "class_names": list(le.classes_),
            "mci_class_index": mci_idx,
            "control_class_index": control_idx,
            "mci_probability_threshold": active_row["mci_threshold"],
            "mci_weight_multiplier": active_row["mci_weight_mult"],
            "active_preset": active,
            "presets": preset_bundle,
            "selected_features": selected_features,
            "imputer": imputer,
            "scaler": scaler,
            "classifier": active_row["classifier"],
        }
    )

    joblib.dump(bundle, MODEL_PATH)
    print(f"\n  Updated bundle: {MODEL_PATH}")
    print(f"  Active preset: {active} (change with bundle['active_preset'] or predict --preset)")


if __name__ == "__main__":
    main()
