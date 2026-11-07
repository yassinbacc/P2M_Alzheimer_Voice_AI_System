"""
Single-file inference: WAV -> features -> classifier -> assessment payload.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd

from ml.config import CLASSIFIER_PKL
from ml.inference.audio_validation import (
    MIN_DURATION_SEC,
    build_inconclusive_assessment,
    validate_audio_row,
)
from ml.inference.predict import get_preset_config, predict_from_features
from ml.pipeline.extract_ultimate import extract_all_features

_bundle_cache = None


def load_bundle():
    global _bundle_cache
    if _bundle_cache is None:
        if not CLASSIFIER_PKL.exists():
            raise FileNotFoundError(
                f"Model not found at {CLASSIFIER_PKL}. Run: python scripts/train.py"
            )
        _bundle_cache = joblib.load(CLASSIFIER_PKL)
    return _bundle_cache


def _risk_level_from_score(score: float) -> str:
    if score < 0.35:
        return "Low"
    if score < 0.65:
        return "Moderate"
    return "High"


def _risk_level_fr(level: str) -> str:
    return {"Low": "Faible", "Moderate": "Modéré", "High": "Élevé"}.get(level, level)


def _build_biomarkers(row: dict, pred: str, prob_ad: float, prob_mci: float) -> list[str]:
    markers = [
        f"Classe prédite: {pred}",
        f"Probabilité AD: {prob_ad:.0%}",
        f"Probabilité MCI: {prob_mci:.0%}",
    ]
    if row.get("speech_rate_median") is not None:
        markers.append(f"Débit de parole (médiane): {row['speech_rate_median']:.2f}")
    if row.get("pause_mean_median") is not None:
        markers.append(f"Pause moyenne: {row['pause_mean_median']:.2f}s")
    if row.get("ct_content_score") is not None:
        markers.append(f"Score contenu Cookie Theft: {row['ct_content_score']:.2f}")
    if row.get("filler_ratio") is not None:
        markers.append(f"Taux de mots de remplissage: {row['filler_ratio']:.2%}")
    return markers


def analyze_wav_file(
    audio_path: str | Path,
    patient_id: str,
    preset: str | None = None,
) -> dict:
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    t0 = time.perf_counter()
    y, sr = librosa.load(str(path), sr=None, mono=True)
    duration_sec = float(len(y) / sr)

    if duration_sec < MIN_DURATION_SEC:
        raise ValueError(
            f"Audio too short ({duration_sec:.1f}s). "
            f"Minimum required: {MIN_DURATION_SEC:.0f} seconds."
        )

    row = extract_all_features(str(path), patient_type=None, verbose=True, fast=True)
    feature_sec = time.perf_counter() - t0
    print(f"Feature extraction done in {feature_sec:.1f}s")

    duration_sec = float(row.get("duration", 0))
    transcript = str(row.get("transcript", "") or "")

    validation = validate_audio_row(row, duration_sec)
    total_sec = time.perf_counter() - t0

    if not validation.valid:
        print(f"Audio validation failed: {validation.code} — {validation.message_fr}")
        result = build_inconclusive_assessment(
            patient_id=patient_id,
            audio_name=path.name,
            duration_sec=duration_sec,
            validation=validation,
            transcript=transcript,
            processing_time_sec=round(total_sec, 1),
        )
        result["featureExtractionSec"] = round(feature_sec, 1)
        return result

    bundle = load_bundle()
    df = pd.DataFrame([row])
    pred_df = predict_from_features(df, bundle, preset=preset)

    predicted = pred_df["predicted"].iloc[0]
    prob_ad = float(pred_df["prob_AD"].iloc[0])
    prob_control = float(pred_df["prob_Control"].iloc[0])
    prob_mci = float(pred_df["prob_MCI"].iloc[0])
    preset_name = pred_df["preset"].iloc[0]

    risk_score_val = float(
        np.clip(prob_ad * 0.92 + prob_mci * 0.58 + prob_control * 0.12, 0.0, 1.0)
    )
    confidence = float(max(prob_ad, prob_control, prob_mci))
    risk_level_en = _risk_level_from_score(risk_score_val)
    risk_level_fr = _risk_level_fr(risk_level_en)

    biomarkers = _build_biomarkers(row, predicted, prob_ad, prob_mci)

    now = datetime.now(timezone.utc)
    assessment_id = str(uuid.uuid4())

    return {
        "id": assessment_id,
        "patientId": patient_id,
        "audioFilePath": path.name,
        "recordedAt": now.isoformat(),
        "duration": int(round(duration_sec)),
        "transcription": transcript,
        "riskScore": {
            "score": round(risk_score_val, 4),
            "riskLevel": risk_level_fr,
            "confidence": round(confidence, 4),
            "biomarkers": biomarkers,
        },
        "analysisDetails": {
            "predictedClass": predicted,
            "validationStatus": "valid",
            "validationCode": "ok",
            "prob_AD": round(prob_ad, 4),
            "prob_Control": round(prob_control, 4),
            "prob_MCI": round(prob_mci, 4),
            "preset": preset_name,
            "riskLevelEn": risk_level_en,
            "speech_rate_median": row.get("speech_rate_median"),
            "pause_mean_median": row.get("pause_mean_median"),
            "ct_content_score": row.get("ct_content_score"),
            "lexical_diversity": row.get("lexical_diversity"),
            "words_per_minute": row.get("words_per_minute"),
            "total_words": row.get("total_words"),
            "processingTimeSec": round(total_sec, 1),
            "featureExtractionSec": round(feature_sec, 1),
        },
        "notes": "",
        "createdAt": now.isoformat(),
        "processingTimeSec": round(total_sec, 1),
        "featureExtractionSec": round(feature_sec, 1),
    }
