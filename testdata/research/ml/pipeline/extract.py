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
    print("✓ parselmouth available — Praat features enabled")
except ImportError:
    PRAAT_AVAILABLE = False
    print("⚠ parselmouth not installed — Praat features will be 0.0")
    print("  Install with: pip install praat-parselmouth")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
from ml.config import CLASS_FOLDERS, DATA_CROPPED, FEATURES_LEGACY_CSV, ensure_dirs

ensure_dirs()
CROPPED_FOLDER = str(DATA_CROPPED) + os.sep
CSV_FILE       = str(FEATURES_LEGACY_CSV)
FOLDERS        = CLASS_FOLDERS

# Filler words for linguistic features
FILLERS = {
    "uh", "um", "er", "ah", "like", "know", "mean",
    "basically", "literally", "actually", "right", "okay"
}

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
print("Loading Whisper...")
whisper_model = whisper.load_model("small")

# ═════════════════════════════════════════════
# 1 — MFCC FEATURES (39)
# mfcc1-13, mfcc_delta1-13, mfcc_delta2_1-13
# ═════════════════════════════════════════════
def extract_mfcc_features(y, sr):
    features = {}

    mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta     = librosa.feature.delta(mfccs)
    delta2    = librosa.feature.delta(mfccs, order=2)

    mfcc_mean   = np.mean(mfccs,  axis=1)
    delta_mean  = np.mean(delta,  axis=1)
    delta2_mean = np.mean(delta2, axis=1)

    for i, v in enumerate(mfcc_mean, 1):
        features[f"mfcc{i}"] = float(v)
    for i, v in enumerate(delta_mean, 1):
        features[f"mfcc_delta{i}"] = float(v)
    for i, v in enumerate(delta2_mean, 1):
        features[f"mfcc_delta2_{i}"] = float(v)

    return features

# ═════════════════════════════════════════════
# 2 — PROSODIC & VOICE QUALITY FEATURES (7)
# pitch_mean, pitch_std, jitter, shimmer,
# HNR, F1_mean, F2_mean
# All via Praat/parselmouth for clinical grade
# ═════════════════════════════════════════════
def extract_praat_features(file_path):
    features = {
        "pitch_mean": 0.0, "pitch_std": 0.0,
        "jitter":     0.0, "shimmer":   0.0,
        "HNR":        0.0, "F1_mean":   0.0,
        "F2_mean":    0.0
    }

    if not PRAAT_AVAILABLE:
        return features

    try:
        sound = parselmouth.Sound(file_path)

        # ── Pitch ──
        pitch      = call(sound, "To Pitch", 0.0, 75, 600)
        pitch_mean = call(pitch, "Get mean",               0, 0, "Hertz")
        pitch_std  = call(pitch, "Get standard deviation", 0, 0, "Hertz")

        features["pitch_mean"] = float(pitch_mean) if pitch_mean == pitch_mean else 0.0
        features["pitch_std"]  = float(pitch_std)  if pitch_std  == pitch_std  else 0.0

        # ── Jitter & Shimmer (clinical grade) ──
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

        jitter  = call(
            point_process,
            "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer = call(
            [sound, point_process],
            "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        features["jitter"]  = float(jitter)  if jitter  == jitter  else 0.0
        features["shimmer"] = float(shimmer) if shimmer == shimmer else 0.0

        # ── HNR ──
        harmonicity    = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr            = call(harmonicity, "Get mean", 0, 0)
        features["HNR"] = float(hnr) if hnr == hnr else 0.0

        # ── Formants F1, F2 ──
        formants       = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        f1             = call(formants, "Get mean", 1, 0, 0, "Hertz")
        f2             = call(formants, "Get mean", 2, 0, 0, "Hertz")
        features["F1_mean"] = float(f1) if f1 == f1 else 0.0
        features["F2_mean"] = float(f2) if f2 == f2 else 0.0

    except Exception as e:
        print(f"      ⚠ Praat error: {e}")

    return features

# ═════════════════════════════════════════════
# 3 — TEMPORAL FEATURES (3)
# speech_rate, pause_mean, pause_count
# ═════════════════════════════════════════════
def extract_temporal_features(y, sr):
    frame_length = 2048
    hop_length   = 512

    # Guard against very short audio
    if len(y) < frame_length:
        return {"speech_rate": 0.0, "pause_mean": 0.0, "pause_count": 0}

    frames    = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energy    = np.sum(frames ** 2, axis=0)
    threshold = np.mean(energy) * 0.5
    is_speech = energy > threshold

    speech_rate = float(np.mean(is_speech))
    pause_mean  = 1.0 - speech_rate
    pause_count = int(np.sum(np.diff(is_speech.astype(int)) != 0))

    return {
        "speech_rate": speech_rate,
        "pause_mean":  pause_mean,
        "pause_count": pause_count
    }

# ═════════════════════════════════════════════
# 4 — LINGUISTIC FEATURES (8)
# total_words, lexical_diversity, repetitions,
# filler_count, SpeechErr, NotRem,
# verbTense, unique_words
# ═════════════════════════════════════════════

# Verb tense markers
PAST_MARKERS    = r"\b(was|were|had|did|went|said|came|got|made|knew|thought|saw|told|felt|left|found|gave|took|brought|kept|began|showed|heard|stood|lost|met|ran|sat|set|cut|put|let|hit|cost|read|held|meant|sent|paid|led|built|fell|grew|drew|rose|drove|wrote|flew|swam|sung|rang|wore|chose|threw|knew|blew|grew)\b"
FUTURE_MARKERS  = r"\b(will|gonna|going to|shall|would|might|could|can)\b"
PRESENT_MARKERS = r"\b(is|are|am|have|has|do|does|see|look|think|know|feel|want|need|seem|appear|get|go|come|say|tell|show)\b"

def count_speech_errors(words):
    """
    Approximate speech error detection:
    - Very short words repeated consecutively (stuttering)
    - Single character words other than 'a' and 'i'
    """
    errors = 0
    for i in range(len(words) - 1):
        # Consecutive identical words = disfluency/repetition
        if words[i] == words[i+1] and len(words[i]) <= 3:
            errors += 1
    # Single char non-words
    errors += sum(1 for w in words if len(w) == 1 and w not in {"a", "i"})
    return errors

def extract_linguistic_features(file_path):
    features = {
        "total_words":       0,
        "unique_words":      0,
        "lexical_diversity": 0.0,
        "repetitions":       0,
        "filler_count":      0,
        "SpeechErr":         0,
        "NotRem":            0,
        "verbTense":         0.0,
        "transcript":        ""
    }

    try:
        result    = whisper_model.transcribe(file_path)
        text      = result["text"].strip()
        text_low  = text.lower()
        words     = re.findall(r"\b\w+\b", text_low)

        if not words:
            return features

        total_words  = len(words)
        unique_words = len(set(words))
        word_counts  = Counter(words)

        # Lexical diversity (Type-Token Ratio)
        lexical_diversity = unique_words / total_words if total_words > 0 else 0.0

        # Repetitions — words used more than once
        repetitions = sum(c - 1 for c in word_counts.values() if c > 1)

        # Filler count
        filler_count = sum(word_counts[f] for f in FILLERS if f in word_counts)

        # Speech errors (disfluencies)
        speech_err = count_speech_errors(words)

        # "I don't remember" or equivalent
        not_rem = len(re.findall(
            r"(i don'?t remember|i can'?t remember|i don'?t recall|i forgot|i'?m not sure)",
            text_low
        ))

        # Verb tense diversity score
        # 0.0 = only present, 1.0 = all three tenses used
        has_past    = 1 if re.search(PAST_MARKERS,    text_low) else 0
        has_future  = 1 if re.search(FUTURE_MARKERS,  text_low) else 0
        has_present = 1 if re.search(PRESENT_MARKERS, text_low) else 0
        verb_tense  = round((has_past + has_future + has_present) / 3.0, 4)

        features.update({
            "total_words":       total_words,
            "unique_words":      unique_words,
            "lexical_diversity": round(lexical_diversity, 4),
            "repetitions":       repetitions,
            "filler_count":      filler_count,
            "SpeechErr":         speech_err,
            "NotRem":            not_rem,
            "verbTense":         verb_tense,
            "transcript":        text
        })

    except Exception as e:
        print(f"      ⚠ Whisper error: {e}")

    return features

# ═════════════════════════════════════════════
# FULL PIPELINE PER FILE
# ═════════════════════════════════════════════
def extract_all_features(file_path, patient_type):
    print(f"    Extracting features...")

    y, sr = librosa.load(file_path, sr=None)

    # 1 — MFCCs (39 features)
    mfcc_feats = extract_mfcc_features(y, sr)

    # 2 — Praat prosodic (7 features)
    praat_feats = extract_praat_features(file_path)

    # 3 — Temporal (3 features)
    temporal_feats = extract_temporal_features(y, sr)

    # 4 — Linguistic (8 features + transcript)
    ling_feats = extract_linguistic_features(file_path)

    # Combine all
    row = {
        "sample":       os.path.basename(file_path),
        "patient_type": patient_type,
    }
    row.update(mfcc_feats)
    row.update(praat_feats)
    row.update(temporal_feats)
    row.update(ling_feats)

    return row

# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════
rows    = []
failed  = []
skipped = []

# Check existing CSV to skip already processed files
already_done = set()
if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
    df_existing   = pd.read_csv(CSV_FILE)
    already_done  = set(df_existing["sample"].tolist())
    rows          = df_existing.to_dict("records")
    print(f"\n  CSV exists — {len(already_done)} files already extracted, will skip them.")

print(f"\n{'═'*55}")
print(f"  FEATURE EXTRACTION")
print(f"{'═'*55}")

for folder_name, label in FOLDERS.items():
    folder_path = os.path.join(CROPPED_FOLDER, folder_name)

    if not os.path.isdir(folder_path):
        print(f"\n⚠ Folder not found: {folder_path}")
        continue

    files = glob.glob(os.path.join(folder_path, "*.wav"))
    print(f"\n{'─'*55}")
    print(f"  {label} — {len(files)} files")
    print(f"{'─'*55}")

    for file_path in files:
        fname = os.path.basename(file_path)
        print(f"\n  ▶ {fname}")

        # Skip if already in CSV
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

        # Save incrementally — every file is saved immediately
        # so if script crashes, progress is not lost
        if rows:
            df = pd.DataFrame(rows)

            # Enforce column order
            meta_cols     = ["sample", "patient_type"]
            mfcc_cols     = [f"mfcc{i}" for i in range(1, 14)]
            delta_cols    = [f"mfcc_delta{i}" for i in range(1, 14)]
            delta2_cols   = [f"mfcc_delta2_{i}" for i in range(1, 14)]
            praat_cols    = ["pitch_mean", "pitch_std", "jitter",
                             "shimmer", "HNR", "F1_mean", "F2_mean"]
            temporal_cols = ["speech_rate", "pause_mean", "pause_count"]
            ling_cols     = ["total_words", "unique_words", "lexical_diversity",
                             "repetitions", "filler_count", "SpeechErr",
                             "NotRem", "verbTense"]
            other_cols    = ["transcript"]

            ordered_cols = (meta_cols + mfcc_cols + delta_cols + delta2_cols +
                            praat_cols + temporal_cols + ling_cols + other_cols)

            # Only keep columns that exist
            final_cols = [c for c in ordered_cols if c in df.columns]
            df[final_cols].to_csv(CSV_FILE, index=False)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  EXTRACTION COMPLETE")
print(f"{'═'*55}")
print(f"  ✓ Extracted this run : {len(rows) - len(skipped)}")
print(f"  ⏭ Skipped (done)     : {len(skipped)}")
print(f"  ✗ Failed             : {len(failed)}")
print(f"  📁 Total in CSV      : {len(rows)}")
print(f"  📄 Saved to          : {CSV_FILE}")

if failed:
    print(f"\n  Failed files:")
    for fname, reason in failed:
        print(f"    - {fname}: {reason}")

# Final column count
if rows:
    df_final = pd.read_csv(CSV_FILE)
    feat_cols = [c for c in df_final.columns
                 if c not in ["sample", "patient_type", "transcript"]]
    print(f"\n  Feature columns : {len(feat_cols)}")
    print(f"  Total columns   : {len(df_final.columns)}")
    print(f"\n  Distribution:")
    print(df_final["patient_type"].value_counts().to_string())