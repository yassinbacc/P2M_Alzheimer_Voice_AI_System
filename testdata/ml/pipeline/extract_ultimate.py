import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import whisper
from collections import Counter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Try importing parselmouth (Praat)
# ─────────────────────────────────────────────
try:
    import parselmouth
    from parselmouth.praat import call
    PRAAT_AVAILABLE = True
    print("parselmouth available")
except ImportError:
    PRAAT_AVAILABLE = False
    print("WARN: parselmouth not installed - pip install praat-parselmouth")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
from ml.config import (
    CLASS_FOLDERS,
    DATA_VERIFIED,
    FEATURES_ULTIMATE_CSV,
    WHISPER_MODEL_INFERENCE,
    ensure_dirs,
)

ensure_dirs()
INPUT_FOLDER = str(DATA_VERIFIED) + os.sep
CSV_FILE     = str(FEATURES_ULTIMATE_CSV)
FOLDERS      = CLASS_FOLDERS

# Frame config (batch / training extraction)
FRAME_SEC    = 5.0    # seconds per frame
OVERLAP_SEC  = 2.5    # 50% overlap
MIN_DURATION = 10.0

# Faster frames for live API inference (fewer librosa/pyin passes)
FAST_FRAME_SEC   = 10.0
FAST_OVERLAP_SEC = 7.5

# ─────────────────────────────────────────────
# WHISPER (lazy load for API / imports)
# ─────────────────────────────────────────────
_whisper_models: dict[str, object] = {}


def get_whisper_model(model_name: str = "small"):
    """Load and cache a Whisper model by name (tiny, base, small, ...)."""
    if model_name not in _whisper_models:
        print(f"Loading Whisper model ({model_name})...")
        _whisper_models[model_name] = whisper.load_model(model_name)
    return _whisper_models[model_name]


def transcribe_audio(
    file_path: str,
    *,
    model_name: str = "small",
    fast: bool = False,
    verbose: bool = True,
) -> tuple[str, list]:
    """Transcribe audio; [fast] uses lighter Whisper decode settings."""
    if verbose:
        print(f"    Transcribing ({model_name})...")
    model = get_whisper_model(model_name)
    kwargs: dict = {}
    if fast:
        kwargs = {
            "fp16": False,
            "beam_size": 1,
            "best_of": 1,
            "condition_on_previous_text": False,
            "temperature": 0,
            "verbose": False,
        }
    result = model.transcribe(file_path, **kwargs)
    text = (result.get("text") or "").strip()
    segments = result.get("segments") or []
    return text, segments

# ═════════════════════════════════════════════
# COOKIE THEFT VOCABULARY
# ═════════════════════════════════════════════

# Main elements — each scored 0/1
COOKIE_ELEMENTS = {
    "boy":      [r"\bboy\b", r"\bson\b", r"\bchild\b", r"\bkid\b", r"\blittle (boy|one)\b"],
    "girl":     [r"\bgirl\b", r"\bdaughter\b", r"\bsister\b", r"\blittle girl\b"],
    "woman":    [r"\bwoman\b", r"\bmother\b", r"\bmom\b", r"\blady\b", r"\bwife\b"],
    "cookie":   [r"\bcookie\b", r"\bcookies\b", r"\bbiscuit\b"],
    "jar":      [r"\bjar\b", r"\bcookie jar\b", r"\bcontainer\b"],
    "stool":    [r"\bstool\b", r"\bstep stool\b", r"\bladder\b", r"\bstep\b"],
    "falling":  [r"\bfall\b", r"\bfalling\b", r"\btipping\b", r"\btoppl\b", r"\bunstable\b"],
    "water":    [r"\bwater\b", r"\bwet\b", r"\bflood\b", r"\bdrip\b"],
    "overflow": [r"\boverflow\b", r"\boverflowing\b", r"\bspill\b", r"\bspilling\b", r"\brunning over\b"],
    "sink":     [r"\bsink\b", r"\basin\b"],
    "dishes":   [r"\bdish\b", r"\bdishes\b", r"\bplate\b", r"\bcup\b", r"\bglass\b"],
    "window":   [r"\bwindow\b"],
    "curtains": [r"\bcurtain\b", r"\bcurtains\b"],
    "cabinet":  [r"\bcabinet\b", r"\bcupboard\b", r"\bshelves\b", r"\bshelf\b"],
    "kitchen":  [r"\bkitchen\b", r"\broom\b", r"\bindoor\b"],
    "outside":  [r"\boutside\b", r"\bgarden\b", r"\byard\b", r"\btree\b", r"\bpath\b", r"\bgrass\b"],
    "apron":    [r"\bapron\b"],
    "stealing": [r"\bsteal\b", r"\bstealing\b", r"\bsneak\b", r"\bsneak\b", r"\bhelp\b", r"\breaching\b", r"\bgrab\b"],
}

# Action words — describe events not objects
ACTION_WORDS = {
    r"\bwashing\b", r"\bdrying\b", r"\bstealing\b", r"\bfalling\b",
    r"\boverflowing\b", r"\bspilling\b", r"\breaching\b", r"\bstanding\b",
    r"\bclimbing\b", r"\bbalancing\b", r"\bignoring\b", r"\blooking\b",
    r"\bpointing\b", r"\bgiving\b", r"\btaking\b", r"\bdripping\b",
    r"\btipping\b", r"\bflooding\b", r"\bwiping\b", r"\bstaring\b",
}

# Spatial language
SPATIAL_WORDS = {
    r"\bleft\b", r"\bright\b", r"\bbehind\b", r"\bin front\b",
    r"\bnext to\b", r"\babove\b", r"\bbelow\b", r"\bbeside\b",
    r"\bbackground\b", r"\bforeground\b", r"\bcenter\b", r"\bcorner\b",
    r"\bnear\b", r"\bfar\b", r"\binside\b", r"\boutside\b",
    r"\bon top\b", r"\bunder\b", r"\bbetween\b", r"\baround\b",
}

# Inference markers
INFERENCE_WORDS = {
    r"\bseems\b", r"\blooks like\b", r"\bappears\b", r"\bprobably\b",
    r"\bmight\b", r"\bmust be\b", r"\babout to\b", r"\bgoing to\b",
    r"\bi think\b", r"\bi believe\b", r"\bmaybe\b", r"\bperhaps\b",
    r"\bsupposed to\b", r"\bseems like\b", r"\bi guess\b",
}

# Filler words
FILLERS = {
    "uh", "um", "er", "ah", "like", "basically",
    "literally", "actually", "right", "okay", "well"
}

# Verb tense markers
PAST_RE    = r"\b(was|were|had|did|went|said|came|got|made|knew|thought|saw|told|felt|left|found|gave|took|brought|kept|began|showed|heard|stood|lost|met|ran|sat|fell|grew|drove|wrote|wore|chose|rang)\b"
FUTURE_RE  = r"\b(will|gonna|going to|shall|would|might|could)\b"
PRESENT_RE = r"\b(is|are|am|have|has|do|does|see|look|think|know|feel|want|need|seem|appear|get|go|come|say|tell|show)\b"

# ═════════════════════════════════════════════
# HELPER — SAFE FLOAT
# ═════════════════════════════════════════════
def safe_float(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except:
        return default

# ═════════════════════════════════════════════
# FRAME SPLITTER
# ═════════════════════════════════════════════
def split_into_frames(y, sr, frame_sec=FRAME_SEC, overlap_sec=OVERLAP_SEC):
    """Split audio into overlapping frames."""
    frame_samples  = int(frame_sec   * sr)
    hop_samples    = int((frame_sec - overlap_sec) * sr)
    frames         = []
    start          = 0
    while start + frame_samples <= len(y):
        frames.append(y[start:start + frame_samples])
        start += hop_samples
    # Include last partial frame if > 2s
    if start < len(y) and (len(y) - start) > 2 * sr:
        frames.append(y[start:])
    return frames

# ═════════════════════════════════════════════
# PER-FRAME ACOUSTIC FEATURES
# ═════════════════════════════════════════════
def frame_acoustic_features(y_frame, sr):
    """Extract acoustic features from a single frame."""
    feats = {}

    # ── MFCCs ──
    mfccs      = librosa.feature.mfcc(y=y_frame, sr=sr, n_mfcc=13)
    delta      = librosa.feature.delta(mfccs)
    delta2     = librosa.feature.delta(mfccs, order=2)
    for i, v in enumerate(np.mean(mfccs,  axis=1), 1): feats[f"mfcc{i}"]        = safe_float(v)
    for i, v in enumerate(np.mean(delta,  axis=1), 1): feats[f"mfcc_d{i}"]      = safe_float(v)
    for i, v in enumerate(np.mean(delta2, axis=1), 1): feats[f"mfcc_d2_{i}"]    = safe_float(v)

    # ── Spectral ──
    feats["spectral_centroid"]  = safe_float(np.mean(librosa.feature.spectral_centroid(y=y_frame, sr=sr)))
    feats["spectral_bandwidth"] = safe_float(np.mean(librosa.feature.spectral_bandwidth(y=y_frame, sr=sr)))
    feats["spectral_rolloff"]   = safe_float(np.mean(librosa.feature.spectral_rolloff(y=y_frame, sr=sr)))
    feats["spectral_flatness"]  = safe_float(np.mean(librosa.feature.spectral_flatness(y=y_frame)))
    feats["spectral_contrast"]  = safe_float(np.mean(librosa.feature.spectral_contrast(y=y_frame, sr=sr)))
    feats["zcr"]                = safe_float(np.mean(librosa.feature.zero_crossing_rate(y_frame)))
    feats["rms_energy"]         = safe_float(np.mean(librosa.feature.rms(y=y_frame)))
    feats["rms_std"]            = safe_float(np.std(librosa.feature.rms(y=y_frame)))

    # ── Pitch via pyin ──
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y_frame, sr=sr,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7")
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        feats["pitch_mean"]   = safe_float(np.mean(f0_voiced)  if len(f0_voiced) > 0 else 0.0)
        feats["pitch_std"]    = safe_float(np.std(f0_voiced)   if len(f0_voiced) > 0 else 0.0)
        feats["pitch_range"]  = safe_float(np.ptp(f0_voiced)   if len(f0_voiced) > 0 else 0.0)
        feats["voiced_ratio"] = safe_float(np.mean(voiced_flag) if voiced_flag is not None else 0.0)
    except:
        feats["pitch_mean"] = feats["pitch_std"] = feats["pitch_range"] = feats["voiced_ratio"] = 0.0

    # ── Energy/pause ──
    frame_length = min(2048, len(y_frame) // 4)
    hop_length   = frame_length // 4
    if len(y_frame) >= frame_length * 2:
        frames_e  = librosa.util.frame(y_frame, frame_length=frame_length, hop_length=hop_length)
        energy    = np.sum(frames_e ** 2, axis=0)
        thresh    = np.mean(energy) * 0.5
        is_speech = energy > thresh
        feats["speech_rate"] = safe_float(np.mean(is_speech))
        feats["pause_mean"]  = safe_float(1.0 - np.mean(is_speech))
        feats["pause_count"] = int(np.sum(np.diff(is_speech.astype(int)) != 0))
    else:
        feats["speech_rate"] = feats["pause_mean"] = 0.0
        feats["pause_count"] = 0

    return feats

# ═════════════════════════════════════════════
# AGGREGATE FRAME STATS
# mean, std, min, max, median, range per feature
# ═════════════════════════════════════════════
def aggregate_frame_features(frame_feat_list):
    """
    Given a list of per-frame feature dicts,
    compute mean/std/min/max/median/range for each feature.
    Returns a single flat dict.
    """
    if not frame_feat_list:
        return {}

    keys = frame_feat_list[0].keys()
    agg  = {}

    for key in keys:
        vals = np.array([f[key] for f in frame_feat_list
                         if isinstance(f.get(key), (int, float))])
        if len(vals) == 0:
            continue
        agg[f"{key}_mean"]   = safe_float(np.mean(vals))
        agg[f"{key}_std"]    = safe_float(np.std(vals))
        agg[f"{key}_min"]    = safe_float(np.min(vals))
        agg[f"{key}_max"]    = safe_float(np.max(vals))
        agg[f"{key}_median"] = safe_float(np.median(vals))
        agg[f"{key}_range"]  = safe_float(np.max(vals) - np.min(vals))

    return agg

# ═════════════════════════════════════════════
# PRAAT FEATURES (clinical grade, whole file)
# ═════════════════════════════════════════════
def extract_praat_features(file_path):
    feats = {
        "praat_pitch_mean": 0.0, "praat_pitch_std": 0.0,
        "praat_jitter_local": 0.0, "praat_jitter_rap": 0.0,
        "praat_shimmer_local": 0.0, "praat_shimmer_apq3": 0.0,
        "praat_hnr": 0.0, "praat_f1": 0.0, "praat_f2": 0.0,
        "praat_voiced_fraction": 0.0,
    }

    if not PRAAT_AVAILABLE:
        return feats

    try:
        sound = parselmouth.Sound(file_path)

        # Pitch
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        feats["praat_pitch_mean"] = safe_float(call(pitch, "Get mean", 0, 0, "Hertz"))
        feats["praat_pitch_std"]  = safe_float(call(pitch, "Get standard deviation", 0, 0, "Hertz"))

        # Voiced fraction
        total_dur    = sound.duration
        voiced_dur   = safe_float(call(pitch, "Get total duration of voiced frames"))
        feats["praat_voiced_fraction"] = voiced_dur / total_dur if total_dur > 0 else 0.0

        # Point process for jitter/shimmer
        pp = call(sound, "To PointProcess (periodic, cc)", 75, 600)

        feats["praat_jitter_local"] = safe_float(call(pp, "Get jitter (local)",   0, 0, 0.0001, 0.02, 1.3))
        feats["praat_jitter_rap"]   = safe_float(call(pp, "Get jitter (rap)",     0, 0, 0.0001, 0.02, 1.3))
        feats["praat_shimmer_local"]= safe_float(call([sound, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        feats["praat_shimmer_apq3"] = safe_float(call([sound, pp], "Get shimmer (apq3)",  0, 0, 0.0001, 0.02, 1.3, 1.6))

        # HNR
        harmonicity     = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        feats["praat_hnr"] = safe_float(call(harmonicity, "Get mean", 0, 0))

        # Formants
        formants        = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        feats["praat_f1"] = safe_float(call(formants, "Get mean", 1, 0, 0, "Hertz"))
        feats["praat_f2"] = safe_float(call(formants, "Get mean", 2, 0, 0, "Hertz"))

    except Exception as e:
        print(f"      ⚠ Praat: {e}")

    return feats

# ═════════════════════════════════════════════
# COOKIE THEFT CONTENT FEATURES
# ═════════════════════════════════════════════
def extract_cookie_theft_features(text, segments):
    """
    Extract Cookie Theft picture specific features
    using the full transcript and word-level timestamps.
    """
    text_low = text.lower()
    words    = re.findall(r"\b\w+\b", text_low)
    feats    = {}

    # ── Element detection — each 0/1 ──
    element_scores = {}
    for element, patterns in COOKIE_ELEMENTS.items():
        mentioned = any(re.search(p, text_low) for p in patterns)
        feats[f"ct_{element}"]  = int(mentioned)
        element_scores[element] = int(mentioned)

    # ── Content completeness score ──
    feats["ct_content_score"] = round(sum(element_scores.values()) / len(COOKIE_ELEMENTS), 4)

    # ── Action vs Object ratio ──
    action_count = sum(1 for p in ACTION_WORDS if re.search(p, text_low))
    total_words  = len(words)
    feats["ct_action_count"]  = action_count
    feats["ct_action_ratio"]  = round(action_count / total_words, 4) if total_words > 0 else 0.0

    # ── Spatial language ──
    spatial_count = sum(1 for p in SPATIAL_WORDS if re.search(p, text_low))
    feats["ct_spatial_count"] = spatial_count
    feats["ct_spatial_ratio"] = round(spatial_count / total_words, 4) if total_words > 0 else 0.0

    # ── Inference markers ──
    inference_count = sum(1 for p in INFERENCE_WORDS if re.search(p, text_low))
    feats["ct_inference_count"] = inference_count
    feats["ct_inference_ratio"] = round(inference_count / total_words, 4) if total_words > 0 else 0.0

    # ── Element revisit — how many times same element re-mentioned ──
    revisit_count = 0
    for element, patterns in COOKIE_ELEMENTS.items():
        mentions = sum(len(re.findall(p, text_low)) for p in patterns)
        if mentions > 1:
            revisit_count += mentions - 1
    feats["ct_element_revisit"] = revisit_count

    # ── Time to first key element (boy, woman, cookie) ──
    # Use Whisper segment timestamps
    key_elements_re = [
        r"\bboy\b", r"\bwoman\b", r"\bmother\b", r"\bcookie\b",
        r"\bwater\b", r"\boverflow\b", r"\bfalling\b", r"\bstool\b"
    ]
    first_mention_time = None
    for seg in segments:
        if any(re.search(p, seg["text"].lower()) for p in key_elements_re):
            first_mention_time = seg["start"]
            break
    feats["ct_first_element_time"] = safe_float(first_mention_time if first_mention_time else 0.0)

    # ── Description switches — how many times patient jumps between elements ──
    element_sequence = []
    for seg in segments:
        seg_text = seg["text"].lower()
        for element, patterns in COOKIE_ELEMENTS.items():
            if any(re.search(p, seg_text) for p in patterns):
                if not element_sequence or element_sequence[-1] != element:
                    element_sequence.append(element)
    feats["ct_description_switches"] = max(0, len(element_sequence) - 1)

    # ── Mentioned only main vs peripheral elements ──
    main_elements = ["boy", "girl", "woman", "cookie", "water", "overflow", "stool", "falling"]
    peripheral    = ["curtains", "cabinet", "outside", "apron", "window"]
    feats["ct_main_elements"]       = sum(element_scores.get(e, 0) for e in main_elements)
    feats["ct_peripheral_elements"] = sum(element_scores.get(e, 0) for e in peripheral)

    return feats

# ═════════════════════════════════════════════
# LINGUISTIC FEATURES (whole file)
# ═════════════════════════════════════════════
def extract_linguistic_features(text, segments):
    text_low = text.lower()
    words    = re.findall(r"\b\w+\b", text_low)
    feats    = {}

    if not words:
        return {
            "total_words": 0, "unique_words": 0, "lexical_diversity": 0.0,
            "repetitions": 0, "filler_count": 0, "filler_ratio": 0.0,
            "SpeechErr": 0, "NotRem": 0, "verbTense": 0.0,
            "mean_word_length": 0.0, "sentence_count": 0,
            "words_per_sentence": 0.0, "brunet_index": 0.0,
            "transcript": text
        }

    total_words  = len(words)
    unique_words = len(set(words))
    word_counts  = Counter(words)

    # TTR — lexical diversity
    lexical_diversity = unique_words / total_words if total_words > 0 else 0.0

    # Brunet's index — more robust diversity measure
    brunet_index = total_words ** (unique_words ** -0.165) if unique_words > 0 else 0.0

    # Repetitions
    repetitions = sum(c - 1 for c in word_counts.values() if c > 1)

    # Filler count and ratio
    filler_count = sum(word_counts.get(f, 0) for f in FILLERS)
    filler_ratio = round(filler_count / total_words, 4) if total_words > 0 else 0.0

    # Speech errors — stuttering / consecutive short word repeats
    speech_err = 0
    for i in range(len(words) - 1):
        if words[i] == words[i+1] and len(words[i]) <= 3:
            speech_err += 1
    speech_err += sum(1 for w in words if len(w) == 1 and w not in {"a", "i"})

    # "I don't remember" expressions
    not_rem = len(re.findall(
        r"(i don'?t remember|i can'?t remember|i don'?t recall|i forgot|i'?m not sure)",
        text_low
    ))

    # Verb tense diversity
    has_past    = 1 if re.search(PAST_RE,    text_low) else 0
    has_future  = 1 if re.search(FUTURE_RE,  text_low) else 0
    has_present = 1 if re.search(PRESENT_RE, text_low) else 0
    verb_tense  = round((has_past + has_future + has_present) / 3.0, 4)

    # Mean word length — syntactic complexity
    mean_word_length = round(np.mean([len(w) for w in words]), 4) if words else 0.0

    # Sentence count and words per sentence
    sentences        = re.split(r'[.!?]+', text)
    sentences        = [s.strip() for s in sentences if s.strip()]
    sentence_count   = len(sentences)
    words_per_sentence = round(total_words / sentence_count, 4) if sentence_count > 0 else 0.0

    # Speech rate in words per minute (using total audio duration from segments)
    total_seg_dur = segments[-1]["end"] if segments else 0
    wpm = round((total_words / total_seg_dur) * 60, 2) if total_seg_dur > 0 else 0.0

    feats.update({
        "total_words":        total_words,
        "unique_words":       unique_words,
        "lexical_diversity":  round(lexical_diversity, 4),
        "brunet_index":       round(brunet_index, 4),
        "repetitions":        repetitions,
        "filler_count":       filler_count,
        "filler_ratio":       filler_ratio,
        "SpeechErr":          speech_err,
        "NotRem":             not_rem,
        "verbTense":          verb_tense,
        "mean_word_length":   mean_word_length,
        "sentence_count":     sentence_count,
        "words_per_sentence": words_per_sentence,
        "words_per_minute":   wpm,
        "transcript":         text
    })

    return feats


def extract_ct_content_score_only(text: str) -> dict:
    """Minimal Cookie Theft feature used by the trained classifier."""
    text_low = text.lower()
    element_scores = {}
    for element, patterns in COOKIE_ELEMENTS.items():
        mentioned = any(re.search(p, text_low) for p in patterns)
        element_scores[element] = int(mentioned)
    return {
        "ct_content_score": round(sum(element_scores.values()) / len(COOKIE_ELEMENTS), 4),
    }


def extract_linguistic_minimal(text: str) -> dict:
    """Linguistic features for inference + audio validation."""
    text_low = text.lower()
    words = re.findall(r"\b\w+\b", text_low)
    if not words:
        return {
            "words_per_sentence": 0.0,
            "total_words": 0,
            "filler_ratio": 0.0,
            "lexical_diversity": 0.0,
            "transcript": text,
        }
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = len(sentences)
    wps = round(len(words) / sentence_count, 4) if sentence_count > 0 else 0.0
    filler_count = sum(1 for w in words if w in FILLERS)
    return {
        "words_per_sentence": wps,
        "total_words": len(words),
        "filler_ratio": round(filler_count / len(words), 4),
        "lexical_diversity": round(len(set(words)) / len(words), 4),
        "transcript": text,
    }


# ═════════════════════════════════════════════
# MASTER EXTRACTION FUNCTION
# ═════════════════════════════════════════════
def extract_all_features(file_path, patient_type=None, verbose=True, fast=False):
    """
    Extract features from one audio file.

    fast=True (API / app inference):
      - Skips Praat (not in selected model features)
      - Whisper base + faster decode (override via WHISPER_MODEL_INFERENCE)
      - Larger frames (fewer passes)
      - Only ct_content_score + words_per_sentence from text
    """
    y, sr = librosa.load(file_path, sr=None)
    duration = len(y) / sr

    if verbose:
        mode = "fast inference" if fast else "full"
        print(f"    Duration: {duration:.1f}s — {mode} extraction...")

    row = {
        "sample": os.path.basename(file_path),
        "patient_type": patient_type or "Unknown",
        "duration": round(duration, 2),
    }

    frame_sec = FAST_FRAME_SEC if fast else FRAME_SEC
    overlap_sec = FAST_OVERLAP_SEC if fast else OVERLAP_SEC
    frames = split_into_frames(y, sr, frame_sec=frame_sec, overlap_sec=overlap_sec)
    if verbose:
        print(f"    {len(frames)} frames of {frame_sec}s...")

    frame_feat_list = []
    for frame in frames:
        if len(frame) < sr * 1.0:
            continue
        try:
            frame_feat_list.append(frame_acoustic_features(frame, sr))
        except Exception as e:
            print(f"      WARN Frame error: {e}")

    agg_feats = aggregate_frame_features(frame_feat_list)
    row.update(agg_feats)
    if verbose:
        print(f"    OK Frame features: {len(agg_feats)}")

    if not fast:
        praat_feats = extract_praat_features(file_path)
        row.update(praat_feats)
        if verbose:
            print(f"    OK Praat features: {len(praat_feats)}")
    elif verbose:
        print("    Skip Praat (not used by classifier)")

    whisper_name = WHISPER_MODEL_INFERENCE if fast else os.environ.get("WHISPER_MODEL", "small")
    try:
        text, segments = transcribe_audio(
            file_path,
            model_name=whisper_name,
            fast=fast,
            verbose=verbose,
        )
    except Exception as e:
        print(f"    WARN Whisper error: {e}")
        text, segments = "", []

    if fast:
        row.update(extract_ct_content_score_only(text))
        row.update(extract_linguistic_minimal(text))
        if verbose:
            print("    OK Text features (ct_content_score, words_per_sentence)")
    else:
        row.update(extract_cookie_theft_features(text, segments))
        row.update(extract_linguistic_features(text, segments))
        if verbose:
            print(f"    OK Cookie Theft + linguistic features")

    return row

# ═════════════════════════════════════════════
# COLUMN ORDER FOR CSV
# ═════════════════════════════════════════════
def get_ordered_columns(df):
    meta      = ["sample", "patient_type", "duration"]

    # Frame aggregated acoustic
    base_feats = (
        [f"mfcc{i}"     for i in range(1, 14)] +
        [f"mfcc_d{i}"   for i in range(1, 14)] +
        [f"mfcc_d2_{i}" for i in range(1, 14)] +
        ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
         "spectral_flatness", "spectral_contrast", "zcr",
         "rms_energy", "rms_std",
         "pitch_mean", "pitch_std", "pitch_range", "voiced_ratio",
         "speech_rate", "pause_mean", "pause_count"]
    )
    stats      = ["mean", "std", "min", "max", "median", "range"]
    frame_cols = [f"{f}_{s}" for f in base_feats for s in stats]

    # Praat
    praat_cols = [
        "praat_pitch_mean", "praat_pitch_std",
        "praat_jitter_local", "praat_jitter_rap",
        "praat_shimmer_local", "praat_shimmer_apq3",
        "praat_hnr", "praat_f1", "praat_f2",
        "praat_voiced_fraction"
    ]

    # Cookie Theft
    ct_element_cols = [f"ct_{e}" for e in COOKIE_ELEMENTS.keys()]
    ct_other_cols   = [
        "ct_content_score", "ct_action_count", "ct_action_ratio",
        "ct_spatial_count", "ct_spatial_ratio",
        "ct_inference_count", "ct_inference_ratio",
        "ct_element_revisit", "ct_first_element_time",
        "ct_description_switches",
        "ct_main_elements", "ct_peripheral_elements"
    ]

    # Linguistic
    ling_cols = [
        "total_words", "unique_words", "lexical_diversity", "brunet_index",
        "repetitions", "filler_count", "filler_ratio",
        "SpeechErr", "NotRem", "verbTense",
        "mean_word_length", "sentence_count",
        "words_per_sentence", "words_per_minute"
    ]

    other = ["transcript"]

    ordered = (meta + frame_cols + praat_cols +
               ct_element_cols + ct_other_cols + ling_cols + other)

    # Only keep columns that actually exist in df
    return [c for c in ordered if c in df.columns]

# ═════════════════════════════════════════════
# MAIN (batch only)
# ═════════════════════════════════════════════
def run_batch_extraction():
    rows = []
    failed = []
    skipped = []
    already_done = set()

    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        df_existing = pd.read_csv(CSV_FILE)
        already_done = set(df_existing["sample"].tolist())
        rows = df_existing.to_dict("records")
        print(f"\n  CSV exists — {len(already_done)} files already done, skipping them.")

    print(f"\n{'═'*55}")
    print(f"  ULTIMATE FEATURE EXTRACTION")
    print(f"{'═'*55}")

    for folder_name, label in FOLDERS.items():
        folder_path = os.path.join(INPUT_FOLDER, folder_name)
        if not os.path.isdir(folder_path):
            print(f"\n⚠ Not found: {folder_path}")
            continue

        files = glob.glob(os.path.join(folder_path, "*.wav"))
        print(f"\n{'─'*55}")
        print(f"  {label} — {len(files)} files")
        print(f"{'─'*55}")

        for file_path in files:
            fname = os.path.basename(file_path)
            print(f"\n  ▶ {fname}")
            if fname in already_done:
                skipped.append(fname)
                print(f"  ⏭ Already extracted — skipped")
                continue
            try:
                row = extract_all_features(file_path, label)
                rows.append(row)
                already_done.add(fname)
                print(f"  ✓ Done")
            except Exception as e:
                failed.append((fname, str(e)))
                print(f"  ✗ Failed: {e}")
            if rows:
                df = pd.DataFrame(rows)
                final_cols = get_ordered_columns(df)
                df[final_cols].to_csv(CSV_FILE, index=False)

    print(f"\n{'═'*55}")
    print(f"  EXTRACTION COMPLETE")
    print(f"{'═'*55}")
    print(f"  Extracted : {len(rows) - len(skipped)}")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Failed    : {len(failed)}")
    print(f"  Total     : {len(rows)}")
    if rows:
        df_final = pd.read_csv(CSV_FILE)
        print(df_final["patient_type"].value_counts().to_string())
    print(f"\n  CSV saved → {CSV_FILE}")


if __name__ == "__main__":
    run_batch_extraction()