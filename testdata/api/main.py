"""
Alzheimer Voice Detection API — connects Flutter app to ML model.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.storage import get_assessment, init_db, list_assessments, save_assessment
from ml.config import WHISPER_MODEL_INFERENCE
from ml.inference.analyze_audio import analyze_wav_file, load_bundle

app = FastAPI(
    title="Alzheimer Voice Detection API",
    version="1.1.0",
    description="Analyze voice recordings for cognitive risk (Control / MCI / AD).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    try:
        load_bundle()
        print("Classifier bundle loaded.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
    print(f"Inference Whisper model: {WHISPER_MODEL_INFERENCE}")


@app.get("/api/health")
def health():
    model_ok = (ROOT / "artifacts" / "models" / "alzheimer_classifier.pkl").is_file()
    return {
        "status": "ok",
        "modelLoaded": model_ok,
        "whisperModel": WHISPER_MODEL_INFERENCE,
        "version": app.version,
    }


@app.post("/api/assessments/analyze")
async def analyze_assessment(
    patientId: str = Form(...),
    audio: UploadFile = File(...),
    preset: str | None = Form(default=None),
):
    if not patientId.strip():
        raise HTTPException(status_code=400, detail="patientId is required")

    filename = audio.filename or "recording.wav"
    suffix = Path(filename).suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Use WAV, MP3, M4A, OGG, or FLAC.",
        )

    content = await audio.read()
    if len(content) < 1000:
        raise HTTPException(status_code=400, detail="Audio file is empty or too small.")

    tmp_suffix = suffix if suffix else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    t0 = time.perf_counter()
    try:
        result = analyze_wav_file(
            tmp_path,
            patient_id=patientId.strip(),
            preset=preset or None,
        )
        result["processingTimeSec"] = round(time.perf_counter() - t0, 1)
        save_assessment(result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}") from e
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/api/assessments")
def list_assessments_endpoint(patientId: str | None = None):
    return list_assessments(patientId)


@app.get("/api/assessments/{assessment_id}")
def get_assessment_endpoint(assessment_id: str):
    item = get_assessment(assessment_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return item


@app.post("/api/assessments")
async def save_assessment_endpoint(assessment: dict):
    return save_assessment(assessment)
