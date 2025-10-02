# AlzDetect AI — Multimodal Alzheimer’s Detection Platform

Research-only platform that combines deep-learning analysis of MRI and PET images with an interactive cognitive assessment to produce a fused Alzheimer’s stage prediction and downloadable report.

Built from the research project “A Survey on Multi-Model Approach to Detect Alzheimer’s Diseases” by:
- Navautsav G (CSE, Sahyadri College of Engineering & Management)
- Pankaj Kumar Mahto (CSE, Sahyadri College of Engineering & Management)
- Pradeep M Lamani (CSE, Sahyadri College of Engineering & Management)
- Preeti Chandrahas Bisanalli (CSE, Sahyadri College of Engineering & Management)
- Under the guidance of Ms. Alakananda K (Assistant Professor, CSE)

This repository contains:
- MRI classifier (ConvNeXt), PET classifier (ViT), and an optional cognitive MLP
- A 5-class fusion head (CN, EMCI, LMCI, MCI, AD) that harmonizes MRI and PET taxonomies and incorporates cognitive assessment
- FastAPI backend with robust file handling, explainability outputs, random-image screening, and PDF report generation
- A modern React + Vite frontend for uploads, assessment, results visualization, and report download

Important: This tool is strictly for research and educational purposes. It is not intended for clinical diagnosis.

---

## Why (Motivation)
Early detection of Alzheimer’s is challenging. Single-modality systems (only MRI or only PET) miss complementary information. This platform:
- Integrates structural MRI, functional PET, and cognitive screening
- Harmonizes different label taxonomies into a shared 5-class fusion space
- Provides visual feedback (saliency/attention heatmaps) and a clean UI/UX to encourage experimentation

## What (Features)
- MRI (ConvNeXt) and PET (ViT-B/16) classifiers
- Cognitive assessment module (questionnaire) included in analysis (MLP if available; otherwise heuristic fallback)
- Late fusion to 5 classes: CN, EMCI, LMCI, MCI, AD
- File support: JPEG/PNG, DICOM (.dcm), and NIfTI (.nii/.nii.gz) with automatic conversion to PNG for inference
- Heuristic random image screening (rejects non–brain photos without running inference)
- Explainability artifacts saved to static/explain (e.g., mri_<patient>.png)
- API: /predict (multipart), /health, /report/{patient_id}
- Frontend: drag-and-drop uploads, assessment UI, progress, results charts, server and client-side PDF download

## Where (Data & Models)
- PET: ADNI (Alzheimer’s Disease Neuroimaging Initiative)
- MRI & Cognitive: Kaggle collections
- Trained weights are expected under experiments/:
  - experiments/mri_*/best_model.pth
  - experiments/pet_*/best_model.pth
  - (optional) experiments/cognitive_*/best_model.pth and scaler.joblib
- Class mappings and fusion mapping are under data/meta/:
  - mri_classes.json (e.g., NonDemented, VeryMildDemented, MildDemented, ModerateDemented)
  - pet_classes.json (CN, EMCI, LMCI, MCI, AD)
  - fusion_mapping.json (maps MRI and PET into 5-class fusion space)

## How (Architecture)
- Backend (FastAPI) — src/api
  - Routes in src/api/routes.py
  - Inference engine in src/inference/predict.py
  - Absolute paths are used to locate meta, weights, and static folders (stable on Windows)
  - DICOM/NIfTI → PNG slice conversion
  - Random-image screen (symmetry/aspect/center contrast) blocks non-brain uploads (threshold 0.45)
  - PDF reports streamed by FileResponse
- Frontend (React/Vite) — frontend/
  - Dev proxy to backend (/predict, /report, /health)
  - State management keeps images, cognitive answers, patient_id, and results
  - Displays per-modality and fused results; renders probability charts; supports PDF downloads

---

## Getting Started (Windows PowerShell)

1) Create environment and install dependencies

```pwsh path=null start=null
python -m venv venv
./venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Frontend dependencies

```pwsh path=null start=null
cd frontend
npm install
```

3) Run the backend

```pwsh path=null start=null
# Recommended dev run (Windows):
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

4) Run the frontend

```pwsh path=null start=null
# In the 'frontend' directory:
# Use the dev proxy by default to avoid CORS/ports issues on Windows
npm run dev
```

Frontend proxy is enabled by frontend/.env:
- VITE_USE_PROXY=true
- VITE_BACKEND_URL can be set if you prefer a fixed URL; when proxy is true it’s ignored.

5) Visit the app
- Frontend dev server: http://localhost:5173
- API docs: http://localhost:8000/docs

---

## File Formats & Preprocessing
- Supported: .jpg/.jpeg/.png (RGB), .dcm (DICOM), .nii/.nii.gz (NIfTI)
- DICOM and NIfTI are converted to PNG (central slice, normalized) for inference
- Saved uploads: data/uploads/<patient>_mri_<filename> / <patient>_pet_<filename>

## Random Image Screening
The backend rejects obviously non-brain images using a lightweight heuristic (left-right symmetry, aspect, center contrast). If an image fails, /predict returns status: "error" with guidance.
- Threshold: 0.45 in src/api/routes.py (increase to be stricter, decrease to be more permissive)

---

## Fusion Taxonomy (5 classes)
Fusion space reflects PET’s richer granularity:
- CN, EMCI, LMCI, MCI, AD
- MRI classes are mapped to fusion via data/meta/fusion_mapping.json
  - NonDemented → CN
  - VeryMildDemented → EMCI
  - MildDemented → MCI
  - ModerateDemented → AD

## Cognitive Assessment Integration
- If a trained cognitive MLP is available (experiments/cognitive_*), the backend uses it
- Otherwise a heuristic converts answers to a 5-class probability in the fusion space
- These cognitive probabilities are included in fusion alongside MRI/PET

---

## API

### POST /predict (multipart/form-data)
Fields:
- patient_id: string (required)
- mri_file: file (optional)
- pet_file: file (optional)
- cognitive: JSON string (optional), e.g. {"orientation_1": "2025", ...}

Example (PowerShell + curl.exe):

```pwsh path=null start=null
curl.exe -X POST http://localhost:8000/predict \
  -F "patient_id=demo" \
  -F "mri_file=@C:\\path\\to\\mri.jpg;type=image/jpeg" \
  -F "cognitive={\"orientation_1\":\"2025\"}"
```

Response (abridged):
```json path=null start=null
{
  "status": "ok",
  "result": {
    "patient_id": "demo",
    "inputs": { "mri_path": "data/uploads/...", "pet_path": null, "cognitive": { ... } },
    "predictions": {
      "mri": { "label": "MildDemented", "probabilities": [ ... ] },
      "pet": { ... },
      "cognitive": { "label": "LMCI", "probabilities": [pCN,pEMCI,pLMCI,pMCI,pAD] },
      "fusion": { "label": "MCI", "probabilities": [pCN,pEMCI,pLMCI,pMCI,pAD] }
    },
    "explainability": { "mri_heatmap": "static/explain/mri_<patient>.png" },
    "debug": { "mri": {"loaded": true}, "pet": {"loaded": true}, "cognitive": {"mode": "heuristic"} }
  }
}
```

### GET /report/{patient_id}
- Streams a PDF built on the server (experiments/reports/<patient>.pdf)

### GET /health and GET /predict
- Health returns {"status":"ok"}
- /predict (GET) returns usage instructions

---

## Frontend Usage
- Upload MRI and/or PET images (drag-and-drop)
- Fill the cognitive assessment (questions across orientation, memory, attention, language, visuospatial, and lifestyle)
- Submit assessment → Results page
  - Per-modality results with confidence
  - Fused prediction (CN/EMCI/LMCI/MCI/AD) with probability distribution
  - Basic overlay placeholder for heatmaps (backend writes real images to static/explain)
- Report page
  - Download PDF (Client) — generated via canvas
  - Download PDF (Server) — streams from backend

---

## Training (Optional)

### Cognitive MLP quick start (uses cognitive_dataset)

1) Ensure the dataset is present at one level up from the backend directory:
   - ../cognitive_dataset/cognitive_train.csv
   - ../cognitive_dataset/cognitive_val.csv
   - ../cognitive_dataset/cognitive_features.json

2) Train with the helper script:
```pwsh
python scripts/train_cognitive_from_dataset.py
```
This will create experiments/cognitive_YYYYMMDD-HHMMSS with:
- best_model.pth
- scaler.joblib
- epoch metrics and results.json

The API auto-detects this model and serves cognitive predictions.

### Full training suite
There are three models: MRI (ConvNeXt), PET (ViT), and Cognitive (MLP, optional). A late-fusion layer combines them to 5 classes (CN/EMCI/LMCI/MCI/AD).

A) Data format
- Create CSVs with at least: path,label. Example rows:
```csv path=null start=null
path,label
C:/data/mri/train/NonDemented/x1.png,NonDemented
C:/data/mri/train/MildDemented/x2.png,MildDemented
```
- Labels must exist in the corresponding classes JSON:
  - data/meta/mri_classes.json
  - data/meta/pet_classes.json
- For cognitive: CSV should contain a patient_id column and the numeric features used by your MLP. Provide a features JSON listing feature names in order.

B) Commands (adapt paths to your environment)
```pwsh path=null start=null
# MRI (ConvNeXt)
python src/training/train_mri.py \
  --train-csv data/meta/mri_train.csv \
  --val-csv   data/meta/mri_val.csv \
  --classes-json data/meta/mri_classes.json \
  --epochs 20 --batch-size 16 --lr 3e-4

# PET (ViT-B/16 by default; auto-detects shape hints from checkpoint)
python src/training/train_pet.py \
  --train-csv data/meta/pet_train.csv \
  --val-csv   data/meta/pet_val.csv \
  --classes-json data/meta/pet_classes.json \
  --epochs 20 --batch-size 16 --lr 3e-4

# Cognitive (optional; MLP)
python src/training/train_mlp.py \
  --train-csv data/meta/cognitive_train.csv \
  --val-csv   data/meta/cognitive_val.csv \
  --features-json data/meta/cognitive_features.json \
  --epochs 30 --batch-size 64 --lr 1e-3
```

C) Outputs
- MRI weights: experiments/mri_YYYYMMDD-HHMMSS/best_model.pth
- PET weights: experiments/pet_YYYYMMDD-HHMMSS/best_model.pth
- Cognitive weights: experiments/cognitive_YYYYMMDD-HHMMSS/best_model.pth and scaler.joblib
- The backend automatically discovers the most recent best_model.pth for each modality.

D) Hyperparameter tips
- Start with 20 epochs, batch size 16. Monitor validation metrics and early-stop if needed.
- Ensure train/val splits are patient-disjoint to avoid leakage.
- If PET ViT reports shape mismatches, it will auto-select a compatible ViT variant based on checkpoint tensor shapes.

---

## Project Structure (partial)
```
path=null start=null
src/
  api/
    main.py          # FastAPI app + CORS/static mounts
    routes.py        # /predict (multipart), /report, /health, GET help for /predict; random-image screening
  inference/
    predict.py       # InferenceEngine + cognitive integration + fusion + explainability
  models/           # ConvNeXt (MRI), ViT (PET), MLP (cognitive)
  preprocessing/    # Utilities to prep images/CSVs
  training/         # Training scripts (reference)
frontend/
  src/              # React app (pages, context, lib)
  vite.config.ts    # Dev proxy to backend
  .env              # VITE_USE_PROXY=true by default; VITE_BACKEND_URL for direct URL
static/explain/     # Heatmaps saved by backend
experiments/        # Model weights and reports (auto-discovered)
```

---

## Configuration
- Frontend
  - VITE_USE_PROXY=true (dev): routes /predict, /report, /health via Vite to http://localhost:8000
  - VITE_BACKEND_URL: explicit URL if you don’t use the proxy
- Backend
  - Random-image screening threshold and rules: src/api/routes.py (_brain_like_details and pass rules)
  - Fusion mapping: data/meta/fusion_mapping.json (5-class fusion space and modality mappings)

---

## Cognitive-Only Mode
- On the Upload page, click “Cognitive Only”. The app clears any uploaded images and proceeds to the assessment. Results will reflect only the cognitive model/heuristic.
- If you upload images, the client pre-screening labels each as VALID/INVALID; Continue is disabled until only valid images remain.

---

## Troubleshooting
- “No Results Available” after submit
  - Ensure the backend is running on 127.0.0.1:8000 and the frontend dev proxy is enabled (VITE_USE_PROXY=true)
  - Check the browser console: you should see “Backend predictions raw” with non-empty fields
- Random images still pass
  - Screening happens both on the client (pre-upload validation) and server (pre-inference). Tune rules in frontend/src/pages/UploadPage.tsx (screenImage) and src/api/routes.py (_brain_like_details / passes)
- DICOM/NIfTI failed
  - Confirm files aren’t corrupted; the backend converts to PNG automatically
- Paths differ between curl and UI
  - The backend uses absolute paths (BASE_DIR) — restart the server if you changed directory layouts

---

## System Requirements
- Python 3.10+
- Node.js 18+
- Windows 10/11 (tested); Linux/macOS should also work with equivalent commands
- GPU optional (reduces inference time; required for faster training). CUDA support depends on your local setup.

---

## Ethics, Security, and Privacy
- Research-only; never use the outputs for clinical diagnosis
- Do not upload identifiable PHI/PII; ensure test data complies with your IRB/data-use agreements
- Images are saved to data/uploads for the duration of the process, not permanently archived by default

## License
Specify the license appropriate for your project (e.g., MIT, Apache-2.0). If unsure, keep the code private or consult your institution’s policy.

## Acknowledgments
- ADNI for PET imaging datasets
- Kaggle for MRI and cognitive datasets
- The ViT and ConvNeXt authors and the PyTorch/timm communities
