from __future__ import annotations

import os
import uuid
from typing import Any

from src.echo_processing.video_reader import sample_video
from src.extraction.document import extract_document
from src.fusion.staged_fusion import StagedFusionEngine
from src.inference.model_manager import ModelManager
from src.ingestion.cloudinary import DownloadError, download_to_temp
from src.ingestion.dicom import read_echo_frames
from src.ingestion.ecg_waveform import ECGWaveformError, read_ecg_file
from src.normalization.biomarker_adapter import select_biomarker_input
from src.pinecone.client import KnowledgeRetriever
from src.recommendations.gemini import GeminiRecommendationService


class InferenceService:
    """Orchestrates modalities without making extraction or retrieval a classifier."""

    def __init__(self, engine: StagedFusionEngine, model_manager: ModelManager, version: str = "1.0.0"):
        self.engine = engine
        self.models = model_manager
        self.knowledge = KnowledgeRetriever()
        self.gemini = GeminiRecommendationService()
        self.version = version
        self.max_bytes = int(os.getenv("MAX_DOWNLOAD_BYTES", 50 * 1024 * 1024))

    @staticmethod
    def _request_id() -> str:
        return f"cf_{uuid.uuid4().hex}"

    def run(self, request: Any) -> dict[str, Any]:
        request_id = self._request_id()
        modalities = {}
        extraction = {}
        predictions: dict[str, dict[str, float]] = {}
        warnings: list[dict[str, str]] = []

        for modality in ("biomarkers", "ecg", "echo"):
            reference = getattr(request, modality, None)
            if reference is None:
                modalities[modality] = {"available": False, "status": "not_provided", "details": {}}
                continue
            try:
                with download_to_temp(str(reference.url), modality, max_bytes=self.max_bytes) as (path, file_format):
                    if modality == "biomarkers":
                        result = extract_document(path, file_format)
                        extraction[modality] = result.model_dump()
                        adapter_input = select_biomarker_input(result)
                        extraction[modality]["source_candidate"] = adapter_input.source if adapter_input else None
                        if adapter_input and not adapter_input.missing_features:
                            predictions[modality] = self.models.predict_biomarkers(adapter_input)
                            modalities[modality] = {"available": True, "status": "processed", "details": {"format": file_format, "dataset": adapter_input.source, "model": "biomarker-ft-v1", "calibrated": adapter_input.task in self.models.biomarker_calibrators}}
                        else:
                            missing = list(adapter_input.missing_features) if adapter_input else ["recognized biomarker features"]
                            modalities[modality] = {"available": True, "status": "partial_extraction", "details": {"format": file_format, "missing_required_features": missing}}
                            warnings.append({"code": "BIOMARKER_REQUIRED_FEATURES_MISSING", "message": f"The report does not contain enough configured features for a validated biomarker model: {', '.join(missing)}."})
                    elif modality == "ecg":
                        if file_format == "dat" and reference.header_url is None:
                            raise ECGWaveformError("WFDB .dat ECG requires its matching .hea header file.")
                        if file_format == "dat" and reference.header_url is not None:
                            header_path = path.with_suffix(".hea")
                            try:
                                with download_to_temp(str(reference.header_url), "ecg_header", max_bytes=self.max_bytes) as downloaded_header:
                                    header_path.write_bytes(downloaded_header.read_bytes())
                                    waveform, waveform_metadata = read_ecg_file(path, file_format)
                            finally:
                                header_path.unlink(missing_ok=True)
                        else:
                            waveform, waveform_metadata = read_ecg_file(path, file_format)
                        extraction[modality] = {"waveform": waveform_metadata}
                        predictions[modality] = self.models.predict_ecg(waveform)
                        modalities[modality] = {"available": True, "status": "processed", "details": {**waveform_metadata, "model": "cnn-bilstm-attention-v1"}}
                    else:
                        clip, metadata = read_echo_frames(path) if file_format == "dicom" else sample_video(path, format_label=file_format.upper())
                        extraction[modality] = {"video": metadata, "tensor_shape": list(clip.transpose(3, 0, 1, 2).shape)}
                        modalities[modality] = {"available": True, "status": "processed", "details": metadata}
                        try:
                            predictions[modality] = self.models.predict_echo(clip)
                            modalities[modality]["status"] = "processed"
                            modalities[modality]["details"]["model"] = "echonet-r2plus1d-v1"
                        except (RuntimeError, ValueError) as exc:
                            modalities[modality]["status"] = "model_unavailable"
                            warnings.append({"code": "ECHO_MODEL_UNAVAILABLE", "message": str(exc)})
            except ECGWaveformError as exc:
                modalities[modality] = {"available": False, "status": "waveform_unavailable", "details": {"error": str(exc)}}
                warnings.append({"code": "ECG_WAVEFORM_UNAVAILABLE", "message": str(exc)})
            except (DownloadError, ValueError, OSError) as exc:
                modalities[modality] = {"available": False, "status": "processing_failed", "details": {"error": str(exc)}}
                warnings.append({"code": f"{modality.upper()}_PROCESSING_FAILED", "message": str(exc)})

        disease_results = self.engine.evaluate_all(predictions)
        available_modalities = sorted(predictions)
        recommendations = sorted({recommendation for result in disease_results.values() for recommendation in result.get("recommendations", [])})
        gemini_recommendations = self.gemini.generate(list(disease_results.values()))
        return {
            "request_id": request_id,
            "service_version": self.version,
            "status": "completed" if not any(item["status"] == "processing_failed" for item in modalities.values()) else "partial",
            "modalities": modalities,
            "extraction": extraction,
            "predictions": predictions,
            "available_modalities": available_modalities,
            "result_scope": "ECG-only" if available_modalities == ["ecg"] else "multimodal" if len(available_modalities) > 1 else "partial",
            "disease_results": list(disease_results.values()),
            "recommendations": recommendations,
            "gemini_recommendations": gemini_recommendations,
            "explainability": {
                "status": "available_after_model_prediction",
                "knowledge_retrieval": "available" if self.knowledge.available else "fallback_local_templates",
                "gemini": gemini_recommendations["status"],
            },
            "warnings": warnings,
        }