# Alzheimer Voice Detection (P2M)

Early cognitive screening from voice biomarkers (Cookie Theft picture description).  
Classes: **Control**, **MCI**, **AD**.

## Project layout (production)

```
testdata/
├── README.md
├── requirements.txt
│
├── data/                          # Datasets
│   ├── raw/                       # Original WAVs (Control/, MCI/, AD/)
│   ├── cropped/                   # After crop_audio
│   ├── verified/                  # After reverif
│   └── features/
│       └── features_ultimate.csv  # Training feature table
│
├── ml/                            # Production ML code
│   ├── config.py                  # All paths
│   ├── pipeline/
│   │   ├── crop.py
│   │   ├── reverif.py
│   │   └── extract_ultimate.py
│   ├── training/
│   │   ├── train_model.py
│   │   └── tune_mci_control.py
│   └── inference/
│       └── predict.py
│
├── scripts/                       # Run from project root
│   ├── crop_audio.py
│   ├── reverif_audio.py
│   ├── extract_ultimate.py
│   ├── train.py
│   ├── tune_mci.py
│   └── predict.py
│
├── artifacts/                     # Model + reports (regenerated)
│   ├── models/
│   │   └── alzheimer_classifier.pkl
│   └── reports/
│
├── mobile/app/                    # Flutter app
│
└── research/                      # Legacy / experiments (see research/README.md)
```

## Quick start

```bash
cd c:\Users\LOQ\Desktop\testdata
pip install -r requirements.txt

python scripts/predict.py --list-presets
python scripts/predict.py
```

## Full pipeline (new audio → model)

```bash
python scripts/crop_audio.py
python scripts/reverif_audio.py
python scripts/extract_ultimate.py
python scripts/train.py
python scripts/tune_mci.py
python scripts/predict.py --preset balanced
```

## Inference presets

| Preset | Use when |
|--------|----------|
| `balanced` | Default |
| `mci_priority` | Favor detecting MCI |
| `control_priority` | Favor Control recall |

## App enhancements (v1.1)

- **Fast inference** on API (no Praat, Whisper `base`, fewer frames)
- **SQLite** assessment history on server (`artifacts/api_assessments.db`)
- **Flutter**: real history + home stats from local DB, session restore, preset in settings, auto-save results, analysis elapsed timer

## Full stack (mobile + API + model)

**Terminal 1 — API server** (from project root):

```bash
pip install -r requirements.txt
python scripts/run_api.py
```

Check: http://localhost:8000/api/health

**Terminal 2 — Flutter app**:

```bash
cd mobile/app
flutter pub get
flutter run
```

- Record or **import** audio on the recording screen, then **Analyser**.
- First analysis loads Whisper **base** (~30s–2 min) then transcribes. Faster than before; set `WHISPER_MODEL_INFERENCE=tiny` for max speed (see below).
- Batch training pipeline still uses full extraction (`python scripts/extract_ultimate.py`).
- **Android emulator** uses `10.0.2.2:8000` automatically.
- **Physical phone**: `flutter run --dart-define=API_BASE_URL=http://YOUR_PC_IP:8000/api`

## Mobile app

```bash
cd mobile/app
flutter pub get
flutter run
```

## Faster API analysis

The app API uses a **fast inference** path (skips Praat, smaller Whisper, fewer frames). Optional env vars when starting the API:

```powershell
$env:WHISPER_MODEL_INFERENCE = "tiny"   # fastest; or base (default), small
python scripts/run_api.py
```

| Setting | Typical effect |
|---------|----------------|
| Default (`base`) | ~2–4× faster than old `small` + full pipeline |
| `tiny` | Fastest; slightly less accurate transcription |
| Full batch extract | Unchanged — still uses Praat + `small` + 5s frames |

## Disclaimer

Research / educational use only — not a certified medical device.
