# Alzheimer Voice API

FastAPI backend: upload WAV → feature extraction → ML prediction.

## Run

From project root:

```bash
pip install fastapi uvicorn python-multipart
python scripts/run_api.py
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Server + model status |
| POST | `/api/assessments/analyze` | `patientId` + `audio` file (multipart) |
| GET | `/api/assessments?patientId=` | List analyses this session |
| GET | `/api/assessments/{id}` | One assessment |

Optional form field: `preset` = `balanced` | `mci_priority` | `control_priority`

## Test with curl

```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/assessments/analyze ^
  -F patientId=test-001 ^
  -F audio=@path/to/recording.wav
```
