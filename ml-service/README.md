# Cardio Fusion XAI — ML Service

## Notebook workflows

The notebook-first research workflow and execution order are documented in
[`NOTEBOOK_WORKFLOW_INDEX.md`](NOTEBOOK_WORKFLOW_INDEX.md). It includes separate
comparative experiments and calibration-ready notebooks for the selected
biomarker, CNN-BiLSTM, and PanEcho models. Existing experiment notebooks are
preserved; new variants write to their own output paths.

## Run the FastAPI service

From this directory, install the project dependencies into the project
environment, place credentials in `.env`, and start the service on port 3002:

```powershell
python -m pip install -e .
python -m api.main
```

The versioned API is:

- `GET /health` for the legacy liveness check
- `GET /v1/health` for device and loaded-artifact status
- `GET /v1/registry` for the configured disease registry
- `POST /v1/inference` with optional `biomarkers`, `ecg`, and `echo` HTTPS URL references; set `explain: true` to request per-modality explanations

Supported uploads are blood/serum PDF or JPG/JPEG/PNG, ECG DICOM/DAT/CSV/SCP/EDF,
and echo AVI/MP4. ECG inputs are normalized to the BiLSTM model shape `(12, 1000)`.

The 1D CNN, XResNet1D, and CNN+BiLSTM+Attention notebooks and model classes are
retained for research training and comparative evaluation. Only
CNN+BiLSTM+Attention is loaded by the application inference service.

- Blood/serum: PDF and image text are parsed into the existing canonical extraction schema.
- ECG: DICOM `WaveformSequence`, WFDB `.dat` with its matching `.hea`, numeric CSV/SCP, and EDF data are decoded and resampled to 100 Hz.
- Echo: AVI, MP4 and multi-frame DICOM are run through **PanEcho** when its
  checkpoint is present (16 consecutive frames, 224x224, ImageNet-normalised,
  4 clips averaged). Otherwise the R(2+1)D baseline input shape `(3, 32, 112, 112)`
  is used. See "Echo expert: PanEcho" below.

A DICOM file must contain usable modality data. ECG text without waveform data
is rejected because the BiLSTM model requires a 12-lead signal.

The service streams Cloudinary assets into request-local temporary files,
validates their content signatures, and removes them after processing. ECG text
is never treated as a waveform. Echo uses OpenCV and the checked-in
R(2+1)D-18 checkpoint when available. Biomarker reports are mapped to the
configured Zheen, UCI-HF, or Framingham feature order when every required field
is present. Incomplete reports return `partial_extraction` with missing fields;
the service never fills a clinical value with zero or an LLM guess. The
113-feature MI-Complications source is not reconstructed from ordinary reports.

Pinecone is an optional explanation/context layer. It can enrich reports when
`PINECONE_API_KEY`, `PINECONE_INDEX`, and `PINECONE_NAMESPACE` are configured,
but it never contributes to numerical disease probabilities.

Gemini recommendations are generated inside the ML service after model
outputs, calibration, and registry fusion. The service returns them under
`gemini_recommendations` with `source` set to `gemini` or `local_fallback`.
Gemini receives only whitelisted disease result fields: disease name, model
risk, confidence, status, primary modality, and detected modalities. It does
not receive raw reports, OCR text, ECG waveforms, Echo frames, or patient
identifiers. It cannot modify probabilities or create disease results.

## Echo expert: PanEcho

PanEcho (Holste et al., JAMA 2025) is a pretrained multi-task echo video model
(ConvNeXt-Tiny frame encoder + temporal Transformer, 40 task heads).

```bash
# 1. put the weights in place (accepts panecho.pt or panecho.pt.zip)
python scripts/prepare_panecho_checkpoint.py --source /path/to/panecho.pt.zip

# 2. zero-shot EchoNet-Dynamic evaluation of the PanEcho EF head
python -m src.evaluation.evaluate_echo_panecho --data-root data/raw/echo/echonet --tag zero_shot

# 3. optional: fine-tune on EchoNet-Dynamic (GPU recommended)
python -m src.training.train_echo_panecho --data-root data/raw/echo/echonet

# 4. start the API; /v1/health shows "echo": "loaded_panecho_pretrained" or "loaded_panecho_finetuned"
python -m api.main
```

Select the backend with `ECHO_MODEL_BACKEND=auto|panecho|r2plus1d` (default `auto`).
Only PanEcho's EF output enters the disease registry; the other heads are
returned under `extraction.echo.panecho_auxiliary` as non-validated research
outputs. PanEcho weights are licensed CC BY-NC-SA 4.0 (non-commercial).

---

PyTorch-based machine learning service for multimodal cardiovascular disease
detection and risk assessment using three complementary modalities:

- Blood / serum / clinical data
- 12-lead ECG
- Echocardiography

The system is designed as a progressive multimodal pipeline:

    Blood → ECG → Echo

The user can begin with one modality and progressively upload additional
evidence. The system updates the disease-specific assessment as additional
modalities become available.

---

## Project Scope

The ML service uses exactly six datasets:

### Blood / Clinical

1. Zheen Heart Attack Dataset
2. UCI Heart Failure Clinical Records
3. Myocardial Infarction Complications Dataset
4. Framingham Heart Study Dataset

### ECG

5. PTB-XL

### Echocardiography

6. EchoNet-Dynamic

No CCTA is used.

The Hugging Face heart-failure-prediction dataset is not part of the final
system.

---

## Important Architectural Constraint

The six datasets do not contain paired multimodal patients.

In other words, there is no training record containing:

    Blood + ECG + Echo

for the same patient across the complete dataset collection.

Therefore, this project does NOT use a falsely constructed end-to-end
trimodal neural network.

Instead, the architecture is:

    Blood/Clinical Expert
            |
            v
       ECG Expert
            |
            v
       Echo Expert
            |
            v
       Calibration
            |
            v
    Disease Orchestrator
            |
            v
     Decision-Level Fusion

Each modality has its own independently trained model. Because the cohorts are
not patient-linked, the service preserves the primary calibrated estimate and
reports other modalities as supporting evidence. It does not learn or claim a
patient-level multimodal probability.

---

# Dataset Groups

## 1. Zheen Heart Attack Dataset

Research endpoint:

- The dataset's binary heart-attack label, not a clinically validated acute-MI diagnosis.

Important features include:

- Age
- Gender
- Heart rate
- Systolic blood pressure
- Diastolic blood pressure
- Blood glucose
- CK-MB
- Troponin

The target is a binary heart-attack outcome.

---

## 2. UCI Heart Failure Clinical Records

Research endpoint:

- Death-event classification in this diagnosed-heart-failure cohort. This is not a general-population or fixed-horizon mortality-risk model.

Important clinical features include:

- Age
- Anaemia
- Creatinine phosphokinase
- Diabetes
- Ejection fraction
- High blood pressure
- Platelets
- Serum creatinine
- Serum sodium
- Sex
- Smoking

The `time` feature is excluded because it represents follow-up duration.
`DEATH_EVENT` is observed over differing follow-up periods, so the binary
classifier does not model event timing or censoring. The small UCI cohort is
intentionally left uncalibrated by the selected-model workflow.

---

## 3. Myocardial Infarction Complications

Used in offline research experiments for post-MI complications and outcomes including:

- Cardiogenic shock
- Acute LV failure / pulmonary edema
- Recurrent MI
- Post-MI arrhythmias
- Post-MI mortality
- Myocardial rupture
- Dressler syndrome
- Chronic heart failure
- Post-infarction angina

Important:

This dataset contains patients who already experienced myocardial infarction.

Its complication labels must not be interpreted as general-population
diagnosis labels. This dataset currently has no inference adapter in the API,
so these outcomes are returned as unsupported by the service.

---

## 4. Framingham Heart Study Dataset

Used for:

- 10-year CHD risk

Target:

    TenYearCHD

This is a fixed-horizon risk prediction endpoint, not a direct acute disease diagnosis.

The system must preserve the documented limitations of this dataset and must
not display artificially high-confidence predictions.

---

## 5. PTB-XL

Used for ECG-label research signals such as:

- MI superclass ECG evidence (not a direct acute-MI diagnosis)
- ST/T abnormalities
- Atrial fibrillation
- Bradyarrhythmia
- AV block
- LBBB / RBBB
- Other conduction abnormalities

PTB-XL contains 12-lead ECG recordings with diagnostic and rhythm annotations.

The ECG branch is multi-label because one ECG can have multiple annotations.
These annotations are not all clinical disease diagnoses.

---

## 6. EchoNet-Dynamic

Used for continuous ejection-fraction (EF) estimation. EF threshold outputs
are labeled as measurement flags only; EF alone does not establish
cardiomyopathy or heart failure. HFrEF is marked unsupported because a clinical
heart-failure syndrome is required.

The fusion registry consumes EF only. When a matching split-conformal artifact
exists, the service also reports an EF interval with nominal marginal coverage
under exchangeability; it is not patient-specific uncertainty.

---

# Model Architecture

## Blood / Clinical Model

The calibration-ready workflow selects a separate estimator for each source
and binary target. The service loads a matching source-specific joblib expert
when present, then uses the existing FT-Transformer only as a fallback. The
runtime adapter supports Zheen, UCI heart-failure records, and Framingham. The
MI-Complications cohort remains offline-only until an inference schema and
adapter are implemented.

---

## ECG Model

Runtime architecture:

    CNN + BiLSTM + Attention

Input:

    12-lead ECG waveform

Output:

    Multi-label disease/rhythm predictions

The ECG model must use sigmoid/BCE-style multi-label classification rather than
softmax because multiple findings can coexist on a single ECG.

---

## Echo Model

Runtime preference:

    Fine-tuned PanEcho when its checkpoint exists; otherwise the R(2+1)D baseline

Input:

    Echocardiography video clip

Output:

    Continuous LVEF percentage

The fusion registry consumes EF only. Other PanEcho heads remain report-only.

---

# Progressive Inference

The user does not need to upload all modalities simultaneously.

## Stage 1 — Blood

Example:

    Blood uploaded
         ↓
    BiomarkerNet
         ↓
    A calibrated estimate only when a matching calibration artifact is loaded
         ↓
    Display blood-supported findings
         ↓
    Recommend ECG where ECG is required/confirmatory

---

## Stage 2 — Blood + ECG

Example:

    Blood
       ↓
    Acute MI probability
       +
    ECG evidence
       ↓
    Disease orchestrator
       ↓
    Primary estimate plus separately displayed ECG evidence

ECG-only conditions such as atrial fibrillation or conduction abnormalities
can also become available at this stage.

---

## Stage 3 — Blood + ECG + Echo

Echo provides:

    LVEF
    Optional marginal EF interval

The engine reports EF and configured threshold flags. It does not infer HFrEF
or another clinical diagnosis from EF alone.

---

# Fusion Strategy

This project uses decision-level evidence fusion.

It does NOT blindly calculate:

    0.33 × Blood + 0.33 × ECG + 0.33 × Echo

and it does not invent a combined probability when no paired multimodal
training data exists.

Instead:

1. Each modality produces its own model output.
2. A matching holdout calibrator is applied when available; otherwise the output remains a model score.
3. The disease registry determines which modality is primary.
4. Confirmatory decisions are compared only when both outputs have matching calibration artifacts.
5. Agreement never increases confidence, which remains LOW because patient-level uncertainty is not estimated.
6. If the primary score is uncalibrated or its provenance is unknown, the service returns a model score and no patient-risk percentage.
7. Missing modalities can produce recommendations.

Example:

    Blood:
        Zheen heart-attack cohort label = 91%

    ECG:
        Supporting MI evidence = YES

    Final:
        Primary calibrated Zheen cohort-label probability: 91% (only if matching calibration exists)
        Confidence: LOW
        Supporting signal: PTB-XL MI superclass label (reported separately)
        No cross-modal agreement or combined probability is claimed.

An uncalibrated primary output is returned as a model score, not a patient-risk
percentage. The active registry treats PTB-XL MI as a separate ECG label signal
because it is not the same endpoint as the Zheen heart-attack label.
The Zheen endpoint and all other active registry endpoints are marked
`research_only`. A matching calibration artifact describes calibration to that
cohort label; it does not by itself establish clinical validity or performance
in a different hospital or patient population.

---

# Probability and Confidence

A neural network sigmoid output is not automatically a calibrated real-world
probability.

Therefore the project contains a calibration layer.

The selected-model notebook fits Platt scaling from reserved calibration
predictions and reports Brier score, log loss, ECE, and test-set bootstrap
intervals. Artifacts are accepted only when their SHA-256 matches the exact
model checkpoint. Under-supported targets remain uncalibrated, and the small
UCI cohort is always left uncalibrated.

Calibration models are stored under:

    checkpoints/calibration/

Calibration is not the same as patient-level confidence. The service does not
estimate patient-specific uncertainty, so confidence stays LOW and cross-modal
agreement cannot raise it. A 0.5 threshold is a configured decision rule, not a
clinical action threshold.

---

# Explainable AI

The optional `explain: true` request flag runs same-source SHAP for a selected
biomarker model, Integrated Gradients for the top ECG outputs, and temporal
occlusion for EF. Explanations are model sensitivity summaries, not causal or
clinical localization evidence. There is no trained paired-data fusion model,
so the service does not produce a fusion-level attribution.

## Blood / Clinical

Uses:

- SHAP
- Feature importance

Example explanation:

    Troponin      → strong contribution
    CK-MB         → strong contribution
    Blood glucose → moderate contribution

---

## ECG

Uses:

- Integrated Gradients

When requested, the service returns lead/time sensitivity summaries for the
top ECG model outputs. These are not clinical localization evidence.

---

## Echo

Uses:

- Temporal window occlusion for EF

The current explanation path does not provide LV segmentation or visual
overlays.

---

# Directory Structure

    ml-service/
    │
    ├── config/
    ├── data/
    ├── checkpoints/
    ├── artifacts/
    ├── notebooks/
    ├── scripts/
    │
    ├── src/
    │   ├── ingestion/
    │   ├── validation/
    │   ├── label_harmonization/
    │   ├── preprocessing/
    │   ├── datasets/
    │   ├── models/
    │   ├── losses/
    │   ├── training/
    │   ├── evaluation/
    │   ├── calibration/
    │   ├── fusion/
    │   ├── xai/
    │   ├── inference/
    │   └── utils/
    │
    ├── api/
    └── tests/

---

# Main Runtime Components

## Ingestion

Reads:

- Blood/clinical input
- ECG files
- Echo videos

## Validation

Checks whether uploaded data is valid for the requested modality.

## Preprocessing

Normalizes data according to its modality and dataset schema.

## Models

Contains the PyTorch implementations.

## Training

Contains independent training pipelines for:

- Biomarkers
- ECG
- Echo

## Calibration

Converts raw model outputs to calibrated probabilities.

## Fusion

Combines available modality evidence at the decision layer.

## Inference

Controls the real-time progressive workflow.

## XAI

Generates explanations for model predictions.

## API

Provides the interface consumed by the main backend server.

---

# Training Order

Recommended development order:

    1. Blood / Clinical
    2. ECG
    3. Echo
    4. Calibration
    5. Disease Registry
    6. Progressive Fusion
    7. XAI integration
    8. API integration

Start by validating each modality independently before implementing
cross-modality orchestration.

---

# Development

Create a virtual environment:

    python -m venv .venv

Activate on Linux/macOS:

    source .venv/bin/activate

Activate on Windows:

    .venv\Scripts\activate

Install the package:

    pip install -e .

Install development dependencies:

    pip install -e ".[dev]"

---

# Example Commands

Train the biomarker model:

    python scripts/train_biomarkers.py

Train ECG model:

    python scripts/train_ecg.py

Train Echo model:

    python scripts/train_echo.py

Calibrate trained models:

    python scripts/calibrate_models.py

Evaluate models:

    python scripts/evaluate_models.py

Run the ML API:

    uvicorn api.main:app --reload --host 0.0.0.0 --port 8001

---

# Testing

Run:

    pytest

For coverage:

    pytest --cov=src --cov=api

---

# GPU

The project is designed for PyTorch and supports CUDA when an appropriate
PyTorch/CUDA installation is available.

Check:

    python -c "import torch; print(torch.cuda.is_available())"

Check the selected device from Python:

    python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

---

# Important Limitations

This system is a machine-learning research/project implementation and is not
a clinical diagnostic device.

Important limitations include:

- The six datasets do not provide paired blood + ECG + echo patients.
- Some datasets represent specific populations rather than a general
  population.
- Post-MI complication predictions apply to patients represented in the
  MI-complications cohort.
- Model probabilities require calibration.
- Dataset shift can reduce real-world performance.
- Rare labels may have limited statistical power.
- Clinical diagnosis requires qualified medical interpretation and may require
  additional investigations.

The system must never present unsupported conditions as confidently detected.

---

# Data Integrity Principles

The project follows these rules:

1. Never invent labels that do not exist in the training data.
2. Never treat risk factors as direct diagnostic biomarkers.
3. Never use leakage-prone variables as predictive inputs.
4. Never fabricate multimodal pairing.
5. Never convert arbitrary neural network scores into misleading "clinical
   percentages".
6. Keep model validation metrics per disease.
7. Keep modality-specific evidence visible.
8. Explicitly report when a disease is unsupported or when additional evidence
   is required.

---

# License

Add the project license here.

Dataset licenses and usage conditions remain subject to the individual
dataset providers and must be reviewed before redistribution.
