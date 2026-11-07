"""
Audio quality checks before clinical classification.

Invalid recordings return a dedicated outcome (not Control / MCI / AD).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── Thresholds (tune for Cookie Theft picture description task) ──
MIN_DURATION_SEC = 30.0
MIN_WORD_COUNT = 25
MIN_WORDS_PER_MINUTE = 30.0
MIN_CT_CONTENT_SCORE = 0.12  # ~2 Cookie Theft elements mentioned
MAX_FILLER_RATIO_WHEN_LOW_CONTENT = 0.45
MIN_CT_FOR_FILLER_CHECK = 0.18
MIN_SPEECH_RATE_MEDIAN = 0.12
MIN_WORDS_IF_LOW_SPEECH = 50

FILLERS = {
    "uh", "um", "er", "ah", "like", "basically",
    "literally", "actually", "right", "okay", "well", "euh", "hein",
}

# French/English hints that user is describing a scene (not only noise)
DESCRIPTION_HINTS = re.compile(
    r"\b(boy|girl|woman|mother|cookie|jar|water|sink|stool|falling|kitchen|"
    r"window|dish|plate|stealing|washing|overflow|picture|image|see|looking|"
    r"garçon|fille|femme|mère|biscuit|bonbon|eau|évier|tabouret|tombe|"
    r"cuisine|fenêtre|vaisselle|vole|lave|déborde|image|vois|regarde)\b",
    re.IGNORECASE,
)


@dataclass
class AudioValidationResult:
    valid: bool
    code: str = "ok"
    message_fr: str = ""
    details: dict | None = None


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text.lower()))


def _filler_ratio(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    fillers = sum(1 for w in words if w in FILLERS)
    return fillers / len(words)


def _words_per_minute(word_count: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    return (word_count / duration_sec) * 60.0


def validate_audio_row(row: dict, duration_sec: float) -> AudioValidationResult:
    """
    Run all checks on extracted feature row (after Whisper + fast features).
    """
    details: dict = {
        "durationSec": round(duration_sec, 1),
    }

    if duration_sec < MIN_DURATION_SEC:
        return AudioValidationResult(
            valid=False,
            code="too_short",
            message_fr=(
                f"Enregistrement trop court ({duration_sec:.0f} s). "
                f"Minimum requis : {MIN_DURATION_SEC:.0f} secondes."
            ),
            details=details,
        )

    transcript = str(row.get("transcript") or "").strip()
    words = _word_count(transcript)
    wpm = float(row.get("words_per_minute") or 0) or _words_per_minute(words, duration_sec)
    ct_score = float(row.get("ct_content_score") or 0)
    filler = float(row.get("filler_ratio") or 0) or _filler_ratio(transcript)
    speech_rate = float(row.get("speech_rate_median") or 0)
    pause_mean = float(row.get("pause_mean_median") or 0)

    details.update(
        {
            "wordCount": words,
            "wordsPerMinute": round(wpm, 1),
            "ctContentScore": round(ct_score, 3),
            "fillerRatio": round(filler, 3),
            "speechRateMedian": round(speech_rate, 3),
            "pauseMeanMedian": round(pause_mean, 3),
        }
    )

    if words < MIN_WORD_COUNT:
        return AudioValidationResult(
            valid=False,
            code="insufficient_speech",
            message_fr=(
                f"Pas assez de parole détectée ({words} mots). "
                "Décrivez l’image clairement pendant au moins 30 secondes."
            ),
            details=details,
        )

    if wpm < MIN_WORDS_PER_MINUTE:
        return AudioValidationResult(
            valid=False,
            code="insufficient_speech",
            message_fr=(
                f"Débit de parole trop faible ({wpm:.0f} mots/min). "
                "Parlez de façon continue en décrivant l’image."
            ),
            details=details,
        )

    if ct_score < MIN_CT_CONTENT_SCORE and not DESCRIPTION_HINTS.search(transcript):
        return AudioValidationResult(
            valid=False,
            code="no_image_description",
            message_fr=(
                "La description ne mentionne pas les éléments de l’image (Cookie Theft). "
                "Décrivez ce que vous voyez : personnes, actions, cuisine, etc."
            ),
            details=details,
        )

    if (
        ct_score < MIN_CT_FOR_FILLER_CHECK
        and filler >= MAX_FILLER_RATIO_WHEN_LOW_CONTENT
    ):
        return AudioValidationResult(
            valid=False,
            code="no_image_description",
            message_fr=(
                "Trop de mots de remplissage et peu de contenu lié à l’image. "
                "Décrivez la scène avec des phrases complètes."
            ),
            details=details,
        )

    if speech_rate < MIN_SPEECH_RATE_MEDIAN and words < MIN_WORDS_IF_LOW_SPEECH:
        return AudioValidationResult(
            valid=False,
            code="mostly_noise_or_silence",
            message_fr=(
                "L’audio semble surtout du silence ou du bruit de fond. "
                "Réenregistrez en parlant distinctement près du micro."
            ),
            details=details,
        )

    if pause_mean > 0.85 and words < MIN_WORDS_IF_LOW_SPEECH:
        return AudioValidationResult(
            valid=False,
            code="mostly_noise_or_silence",
            message_fr=(
                "Très peu de parole par rapport aux pauses. "
                "Décrivez l’image de façon continue pendant 30 secondes ou plus."
            ),
            details=details,
        )

    return AudioValidationResult(valid=True, code="ok", details=details)


def build_inconclusive_assessment(
    patient_id: str,
    audio_name: str,
    duration_sec: float,
    validation: AudioValidationResult,
    transcript: str = "",
    processing_time_sec: float | None = None,
) -> dict:
    """Payload for Flutter when audio fails quality checks (no AD/MCI/Control label)."""
    from datetime import datetime, timezone
    import uuid

    now = datetime.now(timezone.utc)
    reason = validation.message_fr or "Analyse non concluante."

    biomarkers = [
        "Résultat: Non concluant (hors classes Control / MCI / AD)",
        f"Motif: {validation.code}",
        reason,
    ]
    if validation.details:
        if "wordCount" in validation.details:
            biomarkers.append(f"Mots détectés: {validation.details['wordCount']}")
        if "ctContentScore" in validation.details:
            biomarkers.append(
                f"Score contenu image: {validation.details['ctContentScore']}"
            )

    analysis = {
        "predictedClass": "Non_concluant",
        "validationStatus": "invalid",
        "validationCode": validation.code,
        "validationMessage": reason,
        "prob_AD": 0.0,
        "prob_Control": 0.0,
        "prob_MCI": 0.0,
        "clinicalLabelsSkipped": True,
    }
    if validation.details:
        analysis["validationDetails"] = validation.details
    if processing_time_sec is not None:
        analysis["processingTimeSec"] = processing_time_sec

    return {
        "id": str(uuid.uuid4()),
        "patientId": patient_id,
        "audioFilePath": audio_name,
        "recordedAt": now.isoformat(),
        "duration": int(round(duration_sec)),
        "transcription": transcript,
        "riskScore": {
            "score": 0.0,
            "riskLevel": "Non concluant",
            "confidence": 0.0,
            "biomarkers": biomarkers,
        },
        "analysisDetails": analysis,
        "notes": reason,
        "createdAt": now.isoformat(),
    }
