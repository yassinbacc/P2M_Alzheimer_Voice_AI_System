"""
Run inference with the trained Alzheimer voice classifier bundle.

Usage:
  python predict.py
  python predict.py path/to/features.csv
  python predict.py --preset mci_priority
  python predict.py features.csv --preset control_priority
"""

import argparse
import sys

import joblib
import pandas as pd

from ml.config import CLASSIFIER_PKL, FEATURES_ULTIMATE_CSV
from ml.training.train_model import predict_with_threshold

BUNDLE_PATH = str(CLASSIFIER_PKL)
DEFAULT_CSV = str(FEATURES_ULTIMATE_CSV)
PRESET_CHOICES = ("balanced", "mci_priority", "control_priority")


def get_preset_config(bundle: dict, preset: str | None) -> dict:
    name = preset or bundle.get("active_preset", "balanced")
    presets = bundle.get("presets")
    if presets and name in presets:
        p = presets[name]
        return {
            "name": name,
            "classifier": p.get("classifier", bundle["classifier"]),
            "mci_threshold": p.get("mci_threshold", bundle.get("mci_probability_threshold")),
            "metrics": p.get("metrics", {}),
        }
    return {
        "name": name,
        "classifier": bundle["classifier"],
        "mci_threshold": bundle.get("mci_probability_threshold"),
        "metrics": {},
    }


def predict_from_features(
    df: pd.DataFrame, bundle: dict, preset: str | None = None
) -> pd.DataFrame:
    cfg = get_preset_config(bundle, preset)
    feats = bundle["selected_features"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} features, e.g. {missing[:5]}")

    X = df[feats]
    X = bundle["scaler"].transform(bundle["imputer"].transform(X))
    proba = cfg["classifier"].predict_proba(X)
    pred_idx = predict_with_threshold(
        proba, bundle["mci_class_index"], cfg["mci_threshold"]
    )
    labels = bundle["label_encoder"].inverse_transform(pred_idx)

    out = pd.DataFrame(
        {
            "predicted": labels,
            "preset": cfg["name"],
            "prob_AD": proba[:, list(bundle["class_names"]).index("AD")],
            "prob_Control": proba[:, list(bundle["class_names"]).index("Control")],
            "prob_MCI": proba[:, list(bundle["class_names"]).index("MCI")],
        }
    )
    if "sample" in df.columns:
        out.insert(0, "sample", df["sample"].values)
    if "patient_type" in df.columns:
        out.insert(1, "true_label", df["patient_type"].values)
    return out


def main():
    parser = argparse.ArgumentParser(description="Alzheimer voice classifier inference")
    parser.add_argument("csv", nargs="?", default=DEFAULT_CSV, help="Feature CSV path")
    parser.add_argument(
        "--preset",
        choices=PRESET_CHOICES,
        default=None,
        help="Operating point: balanced | mci_priority | control_priority",
    )
    parser.add_argument("--list-presets", action="store_true", help="Show preset metrics")
    args = parser.parse_args()

    bundle = joblib.load(BUNDLE_PATH)

    if args.list_presets:
        presets = bundle.get("presets", {})
        print(f"Active preset: {bundle.get('active_preset', 'balanced')}\n")
        for name, p in presets.items():
            m = p.get("metrics", {})
            print(
                f"  {name:18s}  weight={p.get('mci_weight_mult')}  "
                f"threshold={p.get('mci_threshold')}  "
                f"MCI_rec={m.get('mci_recall', 0):.3f}  "
                f"Ctrl_rec={m.get('control_recall', 0):.3f}"
            )
        return

    df = pd.read_csv(args.csv)
    if len(df) > 10 and args.csv == DEFAULT_CSV:
        df = df.sample(10, random_state=42)

    results = predict_from_features(df, bundle, preset=args.preset)
    print(results.to_string(index=False))
    if "true_label" in results.columns:
        acc = (results["predicted"] == results["true_label"]).mean()
        print(f"\nAccuracy on sample: {acc:.1%}")


if __name__ == "__main__":
    main()
