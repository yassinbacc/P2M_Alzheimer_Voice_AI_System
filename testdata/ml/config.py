"""
Central paths for the Alzheimer voice ML project.
All production scripts import paths from here.
"""

import os
from pathlib import Path

# Project root (testdata/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"

# Data
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_CROPPED = DATA_DIR / "cropped"
DATA_VERIFIED = DATA_DIR / "verified"
FEATURES_DIR = DATA_DIR / "features"

FEATURES_ULTIMATE_CSV = FEATURES_DIR / "features_ultimate.csv"

# Artifacts (production outputs only)
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

CLASSIFIER_PKL = MODELS_DIR / "alzheimer_classifier.pkl"
SELECTED_FEATURES_CSV = REPORTS_DIR / "selected_features.csv"
TRAIN_CV_COMPARISON_CSV = REPORTS_DIR / "train_cv_comparison.csv"
MCI_TRADEOFF_CSV = REPORTS_DIR / "mci_control_tradeoff.csv"
MCI_TRADEOFF_PNG = REPORTS_DIR / "mci_control_tradeoff.png"

# Mobile app
MOBILE_APP_DIR = PROJECT_ROOT / "mobile" / "app"

# Class folder names inside raw/cropped/verified
CLASS_FOLDERS = {
    "control": "Control",
    "mci": "MCI",
    "ad": "AD",
}

# API / single-file inference (faster than batch training pipeline)
# Options: tiny, base, small, medium — set WHISPER_MODEL_INFERENCE=tiny for max speed
WHISPER_MODEL_INFERENCE = os.environ.get("WHISPER_MODEL_INFERENCE", "base")


def ensure_dirs() -> None:
    """Create standard directories if missing."""
    for path in (
        DATA_RAW,
        DATA_CROPPED,
        DATA_VERIFIED,
        FEATURES_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
