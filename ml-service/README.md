# Cardio Fusion XAI — ML Service

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
- `POST /v1/inference` with optional `biomarkers`, `ecg`, and `echo` HTTPS URL references

DICOM is supported for all three modality references, and PNG/JPG/JPEG ECG
images are digitized through the standard 3x4 printed-lead layout:

- Blood/serum: DICOM Structured Report text and encapsulated PDF content are parsed into the existing canonical extraction schema.
- ECG: DICOM `WaveformSequence` data is decoded, validated for at least 12 leads and 10 seconds, and resampled to the XResNet1D shape `(12, 1000)`.
- ECG images: lead traces are detected and resampled to `(12, 1000)` when quality is sufficient; the response marks this as image digitization because it is not equivalent to an original digital waveform.
- Echo: DICOM multi-frame pixel data is decoded and sampled into the R(2+1)D input shape `(3, 32, 112, 112)`.

The DICOM file must contain usable modality data. A DICOM extension alone does
not make a file processable, and ECG diagnostic text without waveform data is
still rejected for the waveform model.

The service streams Cloudinary assets into request-local temporary files,
validates their content signatures, and removes them after processing. ECG text
is never treated as a waveform. Echo uses OpenCV and the checked-in
R(2+1)D-18 checkpoint when available. Biomarker reports are mapped to the
validated Zheen, UCI-HF, or Framingham feature order when every required field
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

Each modality has its own independently trained model.

The final system combines calibrated evidence at the decision/orchestration
layer.

---

# Dataset Groups

## 1. Zheen Heart Attack Dataset

Used for:

- Acute Myocardial Infarction / Heart Attack

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

Used for:

- Heart failure prognosis / mortality

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

The `time` feature is excluded from model training because it represents
follow-up duration and can introduce target leakage for `DEATH_EVENT`.

---

## 3. Myocardial Infarction Complications

Used for post-MI complications and outcomes including:

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

Therefore its complication labels must not be interpreted as general-population
diagnosis labels.

---

## 4. Framingham Heart Study Dataset

Used for:

- 10-year CHD risk

Target:

    TenYearCHD

This is a risk prediction task, not a direct acute disease diagnosis.

The system must preserve the documented limitations of this dataset and must
not display artificially high-confidence predictions.

---

## 5. PTB-XL

Used for ECG-based detection of:

- Acute MI-related ECG evidence
- ST/T abnormalities
- Atrial fibrillation
- Bradyarrhythmia
- AV block
- LBBB / RBBB
- Other conduction abnormalities

PTB-XL contains 12-lead ECG recordings with diagnostic and rhythm annotations.

The ECG branch is designed as a multi-label classification system because a
single ECG can contain multiple abnormalities.

---

## 6. EchoNet-Dynamic

Used for:

- Left ventricular ejection fraction
- Reduced LVEF
- LV systolic dysfunction
- Reduced-EF cardiomyopathy
- HFrEF / reduced LV function

The main Echo task is continuous LVEF regression.

Derived disease categories are calculated from the predicted EF rather than
requiring separate independent classifiers.

---

# Model Architecture

## Blood / Clinical Model

Primary model:

    FT-Transformer

Fallback / baseline:

    Residual MLP

The four clinical datasets have different feature schemas.

Therefore each dataset uses its own input adapter:

    Zheen        → Adapter ┐
    UCI-HF       → Adapter │
    MI-Comp      → Adapter ├→ Shared representation → Multi-task heads
    Framingham   → Adapter ┘

The model uses masked multi-task learning because not every dataset contains
labels for every disease.

Example:

    Zheen sample
        → contributes to Acute MI loss

    Framingham sample
        → contributes to TenYearCHD loss

    MI-Complications sample
        → contributes to relevant post-MI complication losses

---

## ECG Model

Primary architecture:

    XResNet1D

Input:

    12-lead ECG waveform

Output:

    Multi-label disease/rhythm predictions

The ECG model must use sigmoid/BCE-style multi-label classification rather than
softmax because multiple findings can coexist on a single ECG.

---

## Echo Model

Primary architecture:

    R(2+1)D-18

Input:

    Echocardiography video clip

Output:

    Continuous LVEF percentage

Derived outputs:

    LV dysfunction
    Reduced LVEF
    Reduced-EF cardiomyopathy
    HFrEF / reduced LV function

---

# Progressive Inference

The user does not need to upload all modalities simultaneously.

## Stage 1 — Blood

Example:

    Blood uploaded
         ↓
    BiomarkerNet
         ↓
    Disease-specific probabilities
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
    Updated confidence/evidence

ECG-only conditions such as atrial fibrillation or conduction abnormalities
can also become available at this stage.

---

## Stage 3 — Blood + ECG + Echo

EchoNet provides:

    LVEF
    LV-related evidence
    HFrEF-related evidence

The disease engine then performs another inference pass.

---

# Fusion Strategy

This project uses decision-level evidence fusion.

It does NOT blindly calculate:

    0.33 × Blood + 0.33 × ECG + 0.33 × Echo

and it does not invent a combined probability when no paired multimodal
training data exists.

Instead:

1. Each modality produces its own model output.
2. The output is calibrated.
3. The disease registry determines which modality is primary.
4. Confirmatory evidence is checked.
5. Agreement or disagreement changes the confidence/evidence state.
6. The displayed probability remains tied to an actual calibrated model.
7. Missing modalities can produce recommendations.

Example:

    Blood:
        Acute MI = 91%

    ECG:
        Supporting MI evidence = YES

    Final:
        Acute MI
        Risk: 91%
        Confidence: HIGH
        Evidence: Blood + ECG
        Status: ECG corroborates blood-based prediction

If evidence disagrees:

    Blood:
        Acute MI = 91%

    ECG:
        No supporting MI changes

The system does not silently average these values.

Instead:

    Blood evidence: 91%
    ECG evidence: not supportive
    Confidence: MODERATE
    Recommendation: clinical correlation / further evaluation

---

# Probability and Confidence

A neural network sigmoid output is not automatically a calibrated real-world
probability.

Therefore the project contains a calibration layer.

Supported methods include:

- Temperature scaling
- Platt scaling

Calibration models are stored under:

    checkpoints/calibration/

Confidence is based on validation performance and available corroborating
evidence rather than an arbitrary probability cutoff alone.

---

# Explainable AI

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

- Saliency
- Grad-CAM / gradient-based attribution

The system can identify ECG regions/leads that contributed to a prediction.

---

## Echo

Uses:

- LVEF explanation
- LV segmentation/visual overlays where available

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