# CardioFusion-XAI 🤖

CardioFusion-XAI is an end-to-end multimodal cardiovascular AI platform that combines ECG, Echocardiography, CCTA, and Serum Biomarkers to provide comprehensive and explainable cardiovascular disease assessment.

- **Client:** Web interface for patient data entry, medical file uploads, modality selection, and visualization of predictions, explanations, and results.
- **Server:** Handles authentication, patient records, file management, validation, request processing, and communication between the client and ML service.
- **ML Service:** FastAPI-based AI service responsible for preprocessing, modality-specific inference, feature extraction, multimodal fusion, and Explainable AI (XAI).
- **AI Pipeline:** Each modality is processed by a dedicated model, with their features and predictions combined through multimodal fusion to generate disease probability, risk, severity, and modality contribution.
- **Output:** Provides a unified cardiovascular assessment with predictions, confidence scores, contributing factors, visual explanations, and supporting clinical insights.
- **Architecture:** `Client → Server → ML Service → Modality Models → Multimodal Fusion → XAI → Server → Client`

### ❤️ Made with love, curiosity, and a passion for AI & healthcare.

### 🙏 Thanks for checking out CardioFusion-XAI!

> **Disclaimer:** This project is intended for research and educational purposes and is not a substitute for professional medical diagnosis or clinical decision-making.