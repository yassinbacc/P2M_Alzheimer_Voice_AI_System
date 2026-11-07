# Audio to Model Input Pipeline (Architecture + Models)

This document describes the full **audio ingestion to model-input** pipeline implemented in the project.
It focuses on **models and feature/decision architecture**, not on API/servers.

---

## 0) What the pipeline ultimately produces

The system turns a user recording into:

1. A **feature row** (tabular numeric representation) used by the clinical classifier.
2. A **transcript** (from Whisper) used by task adherence / quality gates and a subset of validation signals.
3. A final clinical output:
   - **Control / MCI / AD**, or
   - **Non_concluant** when quality gates reject the recording.

Architecturally this is a two-stage approach:

- **Stage A (Robustness / Task Compliance):**
  Validation logic decides whether the recording is usable.
- **Stage B (Clinical Inference):**
  A pretrained tabular classifier predicts the Alzheimer-related class.

---

## 1) Audio capture and ingestion

### Input sources

- User records audio in the Flutter app (web or native).
- User can also import an existing audio file.

### Goal of ingestion

Convert the captured file into a canonical representation suitable for downstream processing:

- **mono**
- **16 kHz**
- **PCM waveform** (or an equivalent decoded waveform representation)

Even if the capture happens in a different container/format (WAV/WebM/etc.), the ML pipeline assumes an ultimately decodable PCM stream.

---

## 2) Decode and resample normalizer (signal canonicalization)

Before any ML feature work:

1. **Decode** the audio container to raw waveform.
2. **Resample** to the canonical sampling rate.
3. **Downmix to mono** if needed.

Output:
- Waveform `y` with fixed sampling characteristics (`sr = 16000`, mono).

Why this stage matters:
- Feature extraction (prosody, pauses, speech-rate, energy statistics) becomes inconsistent if sample rate or channels differ.
- Resampling is a deterministic “ingestion contract” for the rest of the architecture.

---

## 3) Fast pre-Whisper quality gating (reject garbage early)

The pipeline applies quick sanity checks before spending compute on ASR / feature extraction:

- **Minimum duration gate** (in the project: `MIN_DURATION_SEC = 30`).
- **Non-empty / non-trivial audio** (avoid silence/failed captures).
- Basic constraints derived from the task context (Cookie Theft description should contain speech and task content).

This layer produces a *decision*:
- Pass to deeper extraction/inference, or
- Reject → ultimately yield **Non_concluant**.

Architectural role:
- Prevents the classifier from producing confident-but-meaningless clinical labels on unusable input.

---

## 4) Feature extraction core (tabular model input)

The classifier does not consume raw waveform directly.
Instead, the pipeline builds a **structured feature row** by extracting multiple acoustic/prosodic measures.

Conceptually:

- Input: canonical waveform `y`
- Output: feature vector/row `x` plus metadata (duration, transcript, etc.)

In code terms, this is done by `extract_all_features(...)` (production training/extraction) and then later fed to validation and/or the classifier.

Typical feature families your architecture uses:

- **Timing / pauses**: pause durations, pause statistics (mean/median-like measures).
- **Speech rate proxies**: rate-related measures derived from segments/speech activity.
- **Prosody / rhythm**: variability measures and pause/silence structure.
- **Word/talking pace signals (indirect)**:
  Many validation metrics depend on transcript-derived quantities, but the numeric model ultimately uses engineered acoustic/prosodic statistics.

Architectural role:
- This layer defines the “view” of the audio that the clinical model can reason about.

---

## 5) Whisper ASR (semantic + transcript-derived validation signals)

### Model

- **Whisper** (speech-to-text transformer).

### Role in your architecture

Whisper is used to transcribe the user’s Cookie Theft description.

What the transcript enables in your pipeline:

- **Word count**
- **Words per minute (WPM)**
- Task-adherence heuristics (“does the transcript include enough Cookie Theft content?”)
- Filler-related ratios / linguistic markers (depending on validation implementation)

Architectural role:
- Whisper is an upstream generator for validation.
- It increases robustness by checking whether the user actually performed the intended task (not only “did they speak”).

---

## 6) Validation / gating layer (Non_concluant decision system)

Your pipeline includes an explicit validation subsystem (`ml/inference/audio_validation.py`) that decides whether a recording is valid for clinical classification.

Inputs to validation:
- A `duration_sec` scalar
- Transcript-derived metrics (word count, WPM)
- Content scoring (Cookie Theft keyword/content heuristics)
- Speech rate / filler rules

Output:
- `valid = true/false`
- `code` such as:
  - `too_short`
  - invalid/noisy/off-task conditions (mapped to a “validationStatus: invalid” style response)

Architectural role:
- This layer prevents “garbage in → clinical label out”.
- Invalid inputs are routed to **Non_concluant** rather than forced Control/MCI/AD.

---

## 7) Clinical classifier inference (tabular ML model)

Once validation passes, the pipeline runs the **pretrained clinical model**.

### Model type

- A pretrained sklearn-like pipeline loaded via `CLASSIFIER_PKL` (using `joblib.load`).

### Expected input

- The numeric feature row `x` produced by the extraction layer.

### Output

- Predicted class: `Control / MCI / AD`
- Class probabilities (e.g., `prob_AD`, `prob_MCI`, `prob_Control`)
- A risk score and mapped risk level (Low/Moderate/High)

Architectural role:
- Stage B is “clinical inference” only.
- Stage A is what guarantees that the inference is based on task-appropriate, non-trivial recordings.

---

## 8) Post-processing (biomarkers/explanations packaging)

After inference, the pipeline packages outputs into:

- Top-level clinical label
- Confidence/probabilities
- Interpretable “markers” derived from:
  - predicted class
  - probability values
  - selected feature summaries (pause stats, speech rate stats, content score, filler ratio, etc.)

This does not change the model; it only structures outputs for the UI.

---

## 9) End-to-end chain summary (single diagram in words)

1. **Record/import** user audio
2. **Decode + canonicalize** waveform (mono, 16 kHz)
3. **Fast QC gate** (duration/emptiness checks)
4. **Feature extraction**: `audio waveform → engineered feature row x`
5. **Whisper ASR**: `audio waveform → transcript`
6. **Validation gating**: transcript + duration + features → `valid / invalid`
7. If invalid → **Non_concluant**
8. If valid → **classifier**: `x → probabilities + class`
9. Package results into UI-friendly biomarker explanations

---

## 10) Key architectural principles

- **Separation of concerns**:
  validation/gating is independent from clinical inference.
- **Task compliance via transcript**:
  Whisper enables transcript-derived adherence and WPM/content checks.
- **Engineered features over raw audio for clinical ML**:
  the classifier consumes a feature row `x` defined by the extraction architecture.
- **Deterministic ingestion contract**:
  decode/resample/mono canonicalization prevents feature drift.

