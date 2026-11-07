import os
import glob
import re
import warnings
import numpy as np
import librosa
import soundfile as sf
import whisper
import torch
# Suppress pyannote import-time warning noise on Windows.
warnings.filterwarnings("ignore")
from pyannote.audio import Pipeline
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
from ml.config import DATA_CROPPED, DATA_VERIFIED, ensure_dirs

ensure_dirs()
CROPPED_FOLDER  = str(DATA_CROPPED) + os.sep
VERIFIED_FOLDER = str(DATA_VERIFIED) + os.sep
FOLDERS         = ["control", "mci", "ad"]
HF_TOKEN        = "your_huggingface_token_here"   # optional fallback; prefer env var
MIN_DURATION    = 10.0
PADDING         = 0.1

# ─────────────────────────────────────────────
# KEYWORDS — for transcript double-check
# ─────────────────────────────────────────────
INVESTIGATOR_KEYWORDS = [
    r"tell me everything (that )?(you see|going on)",
    r"please tell me everything you see",
    r"what do you see going on",
    r"what('?s| is) going on in this picture",
    r"what do you see in this picture",
    r"tell me what you see",
    r"describe (everything|what) you see",
    r"here on your screen is a picture",
    r"here('?s| is) (a|the) picture",
    r"take a look at (this|the) picture",
    r"look at (this|the) picture and tell",
    r"i('?d like| want) you to (look|describe|tell)",
    r"we'?re going to just look at them",
    r"just look at (them|it) first",
    r"we'?re going to move on",
    r"move on to (this|the next|something|another)",
    r"here is another picture",
    r"let'?s move on",
    r"now i'?m going to ask",
    r"i'?m going to ask you",
    r"tell me a story about",
    r"that tells a story",
    r"all right[,.]?\s*(and\s*)?now",
    r"alright[,.]?\s*(and\s*)?now",
    r"so now i('?m| am) going to",
    r"okay[,.]?\s*(so\s*)?here is (a|the|another)",
]

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
print("Loading Whisper...")
whisper_model = whisper.load_model("small")

print("Loading diarization pipeline...")
def _resolve_hf_token():
    """
    Prefer environment variables so secrets are not hardcoded in source.
    Keep HF_TOKEN as a fallback for backwards compatibility.
    """
    for env_name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.getenv(env_name)
        if token:
            return token

    if HF_TOKEN and HF_TOKEN != "your_huggingface_token_here":
        return HF_TOKEN

    return None

def _load_diarization_pipeline():
    token = _resolve_hf_token()
    if not token:
        print("  WARN: No Hugging Face token found. Diarization disabled.")
        print("    Set HF_TOKEN or HUGGINGFACE_TOKEN in your environment.")
        return None

    # Try both kwarg styles to support multiple pyannote versions.
    last_error = None
    for kwargs in ({"token": token}, {"use_auth_token": token}):
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                **kwargs,
            )
            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
                print("  OK: Diarization loaded (GPU)")
            else:
                print("  OK: Diarization loaded (CPU)")
            return pipeline
        except Exception as e:
            last_error = e

    print(f"  WARN: Could not load diarization pipeline: {last_error}")
    print("    Make sure your token has accepted pyannote model terms.")
    return None

diarization_pipeline = _load_diarization_pipeline()

# ─────────────────────────────────────────────
# CREATE OUTPUT FOLDERS
# ─────────────────────────────────────────────
for f in FOLDERS:
    os.makedirs(os.path.join(VERIFIED_FOLDER, f), exist_ok=True)

# ═════════════════════════════════════════════
# STEP 1 — DIARIZATION
# Identify all speakers and their segments
# ═════════════════════════════════════════════
def run_diarization(file_path):
    """Load with librosa to bypass torchcodec, pass as tensor to pyannote."""
    if diarization_pipeline is None:
        raise RuntimeError("Diarization pipeline is not available.")

    y, sr = librosa.load(file_path, sr=16000, mono=True)
    waveform = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).unsqueeze(0)
    audio_input = {"waveform": waveform, "sample_rate": sr}
    diarization = diarization_pipeline(audio_input)

    # pyannote 3.x may return a DiarizeOutput object (not iterable) with an
    # `annotation` attribute, while older versions returned an Annotation-like
    # object that can be iterated/itertracks directly.
    annotation = getattr(diarization, "annotation", diarization)

    speaker_segments  = defaultdict(list)
    speaker_durations = defaultdict(float)

    if hasattr(annotation, "itertracks"):
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            dur = float(segment.end - segment.start)
            speaker_segments[speaker].append((float(segment.start), float(segment.end)))
            speaker_durations[speaker] += dur
    else:
        # Fallback for very old / unexpected objects
        for segment, _, speaker in annotation:
            dur = float(segment.end - segment.start)
            speaker_segments[speaker].append((float(segment.start), float(segment.end)))
            speaker_durations[speaker] += dur

    return dict(speaker_segments), dict(speaker_durations)
# ═════════════════════════════════════════════
# STEP 2 — IDENTIFY PATIENT
# Patient = speaker with most total speaking time
# ═════════════════════════════════════════════
def identify_patient(speaker_durations):
    if not speaker_durations:
        return None
    return max(speaker_durations, key=speaker_durations.get)

# ═════════════════════════════════════════════
# STEP 3 — FIND PATIENT SPEECH BOUNDARIES
# Instead of stitching, find the continuous
# block where the patient dominates
# (start of first patient segment to end of last)
# then trim any investigator bits at the edges
# ═════════════════════════════════════════════
def find_patient_boundaries(speaker_segments, speaker_durations, total_duration):
    """
    Strategy:
    - Find patient speaker
    - Get their first and last segment timestamps
    - These become our trim_start and trim_end
    - Any investigator speech BETWEEN patient segments is kept
      (it's short interjections like "okay", "yes")
    - Only trim investigator speech at the very START and END
    """
    patient = identify_patient(speaker_durations)
    if patient is None:
        return 0.0, total_duration, None

    patient_segs = sorted(speaker_segments[patient])

    if not patient_segs:
        return 0.0, total_duration, patient

    trim_start = patient_segs[0][0]   # start of first patient segment
    trim_end   = patient_segs[-1][1]  # end of last patient segment

    # Print speaker summary
    print(f"    Speakers detected: {len(speaker_durations)}")
    for spk, dur in sorted(speaker_durations.items(), key=lambda x: -x[1]):
        role = "<- PATIENT" if spk == patient else "<- investigator"
        print(f"      {spk}: {dur:.1f}s total  {role}")

    print(f"    Patient first segment: {trim_start:.1f}s")
    print(f"    Patient last segment:  {trim_end:.1f}s")

    return trim_start, trim_end, patient

# ═════════════════════════════════════════════
# STEP 4 — TRANSCRIPT DOUBLE CHECK
# Transcribe first and last 5 seconds of the
# diarization-trimmed audio to catch any
# remaining investigator phrases
# ═════════════════════════════════════════════
def is_investigator_text(text):
    text = text.lower().strip()
    return any(re.search(kw, text) for kw in INVESTIGATOR_KEYWORDS)

def transcript_check_edges(y, sr, trim_start, trim_end):
    """
    Transcribe the first and last 5 seconds of the
    proposed trimmed region and refine cut points.
    """
    check_dur = 5.0
    refined_start = trim_start
    refined_end   = trim_end

    # ── Check start region ──
    s = int(trim_start * sr)
    e = int(min((trim_start + check_dur) * sr, len(y)))
    y_start = y[s:e]

    if len(y_start) > 0:
        y_16k   = librosa.resample(y_start, orig_sr=sr, target_sr=16000)
        result  = whisper_model.transcribe(y_16k.astype(np.float32), fp16=False)
        text    = result["text"].lower().strip()
        segs    = result["segments"]
        print(f"    Transcript START: '{text[:90]}'")

        if is_investigator_text(text):
            for seg in segs:
                seg_text = seg["text"].lower().strip()
                if is_investigator_text(seg_text):
                    candidate = trim_start + seg["end"] + PADDING
                    if candidate > refined_start:
                        refined_start = candidate
                        print(f"      NOTE: Refine start -> {refined_start:.2f}s: '{seg_text[:50]}'")
                else:
                    break

    # ── Check end region ──
    e2 = int(trim_end * sr)
    s2 = int(max((trim_end - check_dur) * sr, 0))
    y_end = y[s2:e2]

    if len(y_end) > 0:
        y_16k   = librosa.resample(y_end, orig_sr=sr, target_sr=16000)
        result  = whisper_model.transcribe(y_16k.astype(np.float32), fp16=False)
        text    = result["text"].lower().strip()
        segs    = result["segments"]
        print(f"    Transcript END  : '{text[:90]}'")

        if is_investigator_text(text):
            for seg in segs:
                seg_text = seg["text"].lower().strip()
                if is_investigator_text(seg_text):
                    candidate = (trim_end - check_dur) + seg["start"] - PADDING
                    if candidate < refined_end:
                        refined_end = candidate
                        print(f"      NOTE: Refine end -> {refined_end:.2f}s: '{seg_text[:50]}'")
                    break

    return refined_start, refined_end

# ═════════════════════════════════════════════
# MAIN VERIFY FUNCTION
# ═════════════════════════════════════════════
def verify_file(file_path, out_path):
    # Load at original sr for saving
    y_orig, sr_orig = librosa.load(file_path, sr=None, mono=True)
    duration        = len(y_orig) / sr_orig

    print(f"    Duration: {duration:.1f}s")

    if duration < MIN_DURATION:
        return "too_short", f"{duration:.1f}s"

    # ── Step 1: Diarize ──
    print(f"    Running diarization...")
    try:
        speaker_segments, speaker_durations = run_diarization(file_path)
    except Exception as e:
        print(f"    WARN: Diarization failed: {e} - using transcript only")
        speaker_segments  = {}
        speaker_durations = {}

    # ── Step 2+3: Find patient boundaries ──
    if speaker_durations:
        trim_start, trim_end, patient = find_patient_boundaries(
            speaker_segments, speaker_durations, duration
        )
    else:
        trim_start = 0.0
        trim_end   = duration
        patient    = None
        print(f"    WARN: No speakers detected - using full audio")

    # ── Step 4: Transcript refinement ──
    print(f"    Running transcript edge check...")
    trim_start, trim_end = transcript_check_edges(
        y_orig, sr_orig, trim_start, trim_end
    )

    # ── Sanity checks ──
    trim_start = max(0.0, trim_start)
    trim_end   = min(duration, trim_end)

    if trim_end <= trim_start:
        print(f"    WARN: Invalid trim range - saving full audio")
        sf.write(out_path, y_orig, sr_orig)
        return "needs_review", f"trim conflict {trim_start:.1f}->{trim_end:.1f}s"

    # ── Apply trim ──
    s         = int(trim_start * sr_orig)
    e         = int(trim_end   * sr_orig)
    y_out     = y_orig[s:e]
    final_dur = len(y_out) / sr_orig

    if final_dur < MIN_DURATION:
        return "too_short", f"{final_dur:.1f}s after trimming"

    sf.write(out_path, y_out, sr_orig)

    removed = duration - final_dur
    changed = removed > 0.5

    print(f"    {'TRIM' if changed else 'OK'} {duration:.1f}s -> {final_dur:.1f}s "
          f"(removed {removed:.1f}s)")

    return "trimmed" if changed else "clean", None

# ═════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════
clean        = []
trimmed      = []
needs_review = []
too_short    = []
skipped      = []

print(f"\n{'='*55}")
print("  VERIFICATION - Diarization + Transcript")
print(f"  IN  : {CROPPED_FOLDER}")
print(f"  OUT : {VERIFIED_FOLDER}")
print(f"{'='*55}")

for folder_name in FOLDERS:
    in_folder  = os.path.join(CROPPED_FOLDER,  folder_name)
    out_folder = os.path.join(VERIFIED_FOLDER, folder_name)

    if not os.path.isdir(in_folder):
        print(f"\nWARN: Not found: {in_folder}")
        continue

    files = glob.glob(os.path.join(in_folder, "*.wav"))
    print(f"\n{'-'*55}")
    print(f"  {folder_name}/ - {len(files)} files")
    print(f"{'-'*55}")

    for file_path in files:
        fname    = os.path.basename(file_path)
        out_path = os.path.join(out_folder, fname)

        print(f"\n  >> {fname}")

        # Skip if already verified
        if os.path.exists(out_path):
            skipped.append(fname)
            print("  SKIP: Already verified - skipped")
            continue

        try:
            status, info = verify_file(file_path, out_path)

            if status == "clean":
                clean.append(fname)
                print("  OK: Clean -> saved")
            elif status == "trimmed":
                trimmed.append(fname)
                print("  OK: Trimmed -> saved")
            elif status == "too_short":
                too_short.append((fname, info))
                print(f"  DROP: Too short - {info}")
            elif status == "needs_review":
                needs_review.append((fname, info))
                print(f"  WARN: Needs review - {info}")

        except Exception as e:
            needs_review.append((fname, str(e)))
            print(f"  ERROR: Exception: {e}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  DONE")
print(f"{'='*55}")
print(f"  OK Clean         : {len(clean)}")
print(f"  TRIMMED          : {len(trimmed)}")
print(f"  SKIPPED          : {len(skipped)}")
print(f"  TOO SHORT        : {len(too_short)}")
print(f"  NEEDS REVIEW     : {len(needs_review)}")

if needs_review:
    print(f"\n  Manual review:")
    for f, r in needs_review:
        print(f"    - {f}: {r}")

print(f"\n  Output -> {VERIFIED_FOLDER}")