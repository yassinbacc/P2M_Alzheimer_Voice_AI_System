import os
import glob
import re
import hashlib
import warnings
import numpy as np
import librosa
import soundfile as sf
import whisper

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
from ml.config import CLASS_FOLDERS, DATA_CROPPED, DATA_RAW, ensure_dirs

ensure_dirs()
BASE_FOLDER    = str(DATA_RAW) + os.sep
CROPPED_FOLDER = str(DATA_CROPPED) + os.sep
FOLDERS        = CLASS_FOLDERS
MIN_DURATION   = 10.0
PADDING        = 0.15
CHECK_DURATION = 5.0

# ─────────────────────────────────────────────
# KEYWORDS
# ─────────────────────────────────────────────
INTRO_END_KEYWORDS = [
    r"please tell me everything you see",
    r"tell me everything (you see|going on)",
    r"everything (you see|going on) in this picture",
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
]

TRANSITION_KEYWORDS = [
    r"we'?re going to move on",
    r"we are going to move on",
    r"move on to (this|the next|something|another)",
    r"here is another picture",
    r"now here is another",
    r"and now here is",
    r"let'?s move on",
    r"so let'?s move on",
    r"now i'?m going to ask",
    r"i'?m going to ask you",
    r"tell me a story about",
    r"that tells a story",
    r"here is the next",
    r"next picture",
    r"look at this (one|picture|image)",
    r"okay[,.]?\s*(so\s*)?here is (a|the|another)",
    r"all right[,.]?\s*(and\s*)?now",
    r"alright[,.]?\s*(and\s*)?now",
    r"so now i('?m| am) going to",
]

FILLER_PATTERNS = [
    r"^okay\.?$", r"^ok\.?$", r"^alright\.?$", r"^all right\.?$",
    r"^good\.?$", r"^great\.?$", r"^yes\.?$", r"^sure\.?$",
    r"^mhm\.?$", r"^mm-?hmm\.?$", r"^uh-?huh\.?$", r"^right\.?$",
]

ALL_INVESTIGATOR = INTRO_END_KEYWORDS + TRANSITION_KEYWORDS

# ─────────────────────────────────────────────
# LOAD WHISPER
# ─────────────────────────────────────────────
print("Loading Whisper...")
whisper_model = whisper.load_model("small")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_md5(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def is_investigator_text(text):
    if any(re.search(kw, text) for kw in ALL_INVESTIGATOR):
        return True
    words = text.strip().split()
    if len(words) <= 3:
        if any(re.search(kw, text.strip()) for kw in FILLER_PATTERNS):
            return True
    return False

def transcribe_chunk(y_chunk, sr):
    y_16k  = librosa.resample(y_chunk, orig_sr=sr, target_sr=16000)
    result = whisper_model.transcribe(y_16k.astype(np.float32), fp16=False)
    return result["segments"], result["text"].lower().strip()

def find_start_time(segments):
    for seg in segments:
        text = seg["text"].lower().strip()
        if any(re.search(kw, text) for kw in INTRO_END_KEYWORDS):
            print(f"    Intro ends [{seg['start']:.1f}s]: '{text[:60]}'")
            return seg["end"]
    print(f"    ⚠ No intro detected — starting from 0.0s")
    return 0.0

def find_cut_time(segments, start_time):
    for seg in segments:
        if seg["start"] <= start_time:
            continue
        text = seg["text"].lower().strip()
        if any(re.search(kw, text) for kw in TRANSITION_KEYWORDS):
            print(f"    Transition [{seg['start']:.1f}s]: '{text[:60]}'")
            return seg["start"]
    print(f"    ⚠ No transition — keeping until end")
    return None

# ═════════════════════════════════════════════
# STAGE 0 — REMOVE DUPLICATES FROM INPUT
# Do this first so we never process same file twice
# ═════════════════════════════════════════════
def remove_input_duplicates():
    print(f"\n{'═'*55}")
    print(f"  STAGE 0 — REMOVING DUPLICATES FROM INPUT")
    print(f"{'═'*55}")

    total_removed = 0

    for folder_name in FOLDERS:
        in_folder = os.path.join(BASE_FOLDER, folder_name)
        if not os.path.isdir(in_folder):
            continue

        files    = glob.glob(os.path.join(in_folder, "*.wav"))
        hash_map = {}   # md5 → first file seen (keep this one)
        removed  = 0

        print(f"\n  {folder_name}/ — scanning {len(files)} files...")

        for f in sorted(files):   # sorted so we keep consistently named files
            h = get_md5(f)
            if h in hash_map:
                print(f"    🗑 REMOVE : {os.path.basename(f)}")
                print(f"       KEEP  : {os.path.basename(hash_map[h])}")
                try:
                    os.remove(f)
                    removed += 1
                except Exception as e:
                    print(f"    ✗ Could not remove: {e}")
            else:
                hash_map[h] = f

        if removed == 0:
            print(f"    ✓ No duplicates found")
        else:
            print(f"    → {removed} duplicates removed")

        total_removed += removed

    print(f"\n  Total duplicates removed from input: {total_removed}")

# ═════════════════════════════════════════════
# STAGE 1+2 — CROP + VERIFY (single pass)
# ═════════════════════════════════════════════
def process_file(file_path, out_path, is_ad=False):
    y, sr    = librosa.load(file_path, sr=None)
    duration = len(y) / sr

    if is_ad:
        y_cropped = y
    else:
        # ── Crop ──
        result     = whisper_model.transcribe(file_path)
        segments   = result["segments"]
        start_time = find_start_time(segments)
        cut_time   = find_cut_time(segments, start_time)

        start_sample = int(start_time * sr)
        end_sample   = int(cut_time * sr) if cut_time else len(y)
        y_cropped    = y[start_sample:end_sample]

        print(f"    Crop: {start_time:.1f}s → {end_sample/sr:.1f}s "
              f"= {len(y_cropped)/sr:.1f}s (was {duration:.1f}s)")

    # ── Verify start ──
    trim_start    = 0.0
    trim_end      = len(y_cropped) / sr
    check_samples = int(CHECK_DURATION * sr)

    segs_start, text_start = transcribe_chunk(y_cropped[:check_samples], sr)
    print(f"    START: '{text_start[:80]}'")

    if is_investigator_text(text_start):
        for seg in segs_start:
            seg_text = seg["text"].lower().strip()
            if is_investigator_text(seg_text):
                trim_start = seg["end"] + PADDING
                print(f"    → Trim start to {trim_start:.1f}s")
            else:
                break

    # ── Verify end ──
    end_offset             = max(0.0, trim_end - CHECK_DURATION)
    segs_end, text_end     = transcribe_chunk(
        y_cropped[int(end_offset * sr):], sr
    )
    print(f"    END  : '{text_end[:80]}'")

    if is_investigator_text(text_end):
        for seg in reversed(segs_end):
            seg_text = seg["text"].lower().strip()
            if is_investigator_text(seg_text):
                trim_end = end_offset + seg["start"] - PADDING
                print(f"    → Trim end to {trim_end:.1f}s")
            else:
                break

    # ── Final crop ──
    s     = int(max(0.0, trim_start) * sr)
    e     = int(min(len(y_cropped) / sr, trim_end) * sr)
    y_out = y_cropped[s:e]

    if len(y_out) / sr < MIN_DURATION:
        return False, f"too short ({len(y_out)/sr:.1f}s)"

    sf.write(out_path, y_out, sr)
    print(f"    ✓ Saved {len(y_out)/sr:.1f}s → {os.path.basename(out_path)}")
    return True, None

# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

# ── Stage 0: Clean input duplicates first ──
remove_input_duplicates()

# ── Build MD5 map of already existing output files ──
existing_hashes = {}
for folder_name in FOLDERS:
    out_dir = os.path.join(CROPPED_FOLDER, folder_name)
    if os.path.isdir(out_dir):
        for f in glob.glob(os.path.join(out_dir, "*.wav")):
            existing_hashes[get_md5(f)] = os.path.basename(f)

print(f"\n  data_cropped/ already contains "
      f"{len(existing_hashes)} files — these will be skipped.")

# ── Stage 1+2: Process ──
skipped   = []
processed = []
dupes_out = []
failed    = []

print(f"\n{'═'*55}")
print(f"  STAGE 1+2 — CROP + VERIFY")
print(f"{'═'*55}")

for folder_name, label in FOLDERS.items():
    in_folder  = os.path.join(BASE_FOLDER, folder_name)
    out_folder = os.path.join(CROPPED_FOLDER, folder_name)
    os.makedirs(out_folder, exist_ok=True)

    if not os.path.isdir(in_folder):
        print(f"\n⚠ Folder not found: {in_folder}")
        continue

    files = glob.glob(os.path.join(in_folder, "*.wav"))
    print(f"\n{'─'*55}")
    print(f"  {label} — {len(files)} files")
    print(f"{'─'*55}")

    for file_path in files:
        fname    = os.path.basename(file_path)
        out_path = os.path.join(out_folder, fname)

        print(f"\n  ▶ {fname}")

        # ── Skip: output already exists by filename ──
        if os.path.exists(out_path):
            skipped.append(fname)
            print(f"  ⏭ Already in data_cropped — skipped")
            continue

        # ── Skip: same content already in output under different name ──
        src_hash = get_md5(file_path)
        if src_hash in existing_hashes:
            dupes_out.append((fname, existing_hashes[src_hash]))
            print(f"  🗑 Content already saved as "
                  f"{existing_hashes[src_hash]} — skipped")
            continue

        # ── Process ──
        try:
            is_ad          = (folder_name == "ad")
            success, error = process_file(file_path, out_path, is_ad=is_ad)

            if success:
                processed.append(fname)
                existing_hashes[get_md5(out_path)] = fname
            else:
                failed.append((fname, error))
                print(f"  🗑 {error}")

        except Exception as e:
            failed.append((fname, str(e)))
            print(f"  ✗ Exception: {e}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  DONE")
print(f"{'═'*55}")
print(f"  ✓ Processed this run : {len(processed)}")
print(f"  ⏭ Skipped (done)     : {len(skipped)}")
print(f"  🗑 Dupes in output   : {len(dupes_out)}")
print(f"  ✗ Failed             : {len(failed)}")

if dupes_out:
    print(f"\n  Duplicate content skipped:")
    for fname, original in dupes_out:
        print(f"    - {fname} (same as {original})")

if failed:
    print(f"\n  Failed:")
    for fname, reason in failed:
        print(f"    - {fname}: {reason}")

print(f"\n  Clean files in: {CROPPED_FOLDER}")
"""```

---

**Order of operations:**
```
STAGE 0 → Scan data/control, data/mci, data/ad
           Find MD5 duplicates within each folder
           Delete the copies, keep the first one
           ↓
STAGE 1+2 → For each remaining file in data/:
             Check if already in data_cropped/ by filename → skip
             Check if same content already in data_cropped/ → skip
             Otherwise: crop + verify → save to data_cropped/"""