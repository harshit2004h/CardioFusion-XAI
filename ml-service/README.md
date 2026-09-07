# CardioFusion-XAI — ML Service

The machine learning service for **CardioFusion-XAI**, a multimodal cardiovascular AI system combining ECG, echocardiography, and clinical biomarker data.

The ML service is implemented using **PyTorch** and is designed to support:

- ECG analysis
- Echocardiography analysis
- Clinical biomarker analysis
- Multimodal fusion
- Missing-modality handling
- Quality-aware modality weighting
- Explainable AI (XAI)
- Experiment tracking with Weights & Biases
- GPU training
- Cloud training through Modal
- Reproducible experiments

---

## 1. Project Architecture

The system contains three primary active modalities:

```text
ECG
 │
 └── 12-lead ECG encoder
        │
        └── 512-D ECG embedding


Echo
 │
 └── Frame CNN + Temporal Transformer
        │
        └── 512-D Echo embedding


Biomarkers
 │
 └── MLP
        │
        └── 512-D Biomarker embedding