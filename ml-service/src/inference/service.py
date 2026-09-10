from __future__ import annotations

import os
import uuid
from typing import Any

from src.ecg_processing.image_waveform import ECGImageError, digitize_ecg_image
from src.echo_processing.video_reader import sample_video
from src.extraction.document import extract_document
from src.fusion.staged_fusion import StagedFusionEngine
from src.inference.model_manager import ModelManager
from src.ingestion.cloudinary import DownloadError, download_to_temp
from src.ingestion.dicom import read_ecg_waveform, read_echo_frames
from src.pinecone.client import KnowledgeRetriever


class InferenceService:
    """Orchestrates modalities without making extraction or retrieval a classifier."""

    def __init__(self, engine: StagedFusionEngine, model_manager: ModelManager, version: str = "1.0.0"):
        self.engine = engine
        self.models = model_manager
        self.knowledge = KnowledgeRetriever()
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
                        modalities[modality] = {"available": True, "status": "processed", "details": {"format": file_format}}
                        warnings.append({"code": "BIOMARKER_FEATURE_ADAPTER_UNAVAILABLE", "message": "Biomarker extraction completed and the trained model is loaded, but the document-to-source feature adapter is not configured; prediction was not run."})
                    elif modality == "ecg":
                        result = extract_document(path, file_format)
                        extraction[modality] = {"machine_report": result.model_dump()}
                        if file_format == "dicom":
                            waveform, waveform_metadata = read_ecg_waveform(path)
                        elif file_format in {"png", "jpg", "jpeg"}:
                            waveform, waveform_metadata = digitize_ecg_image(path)
                        else:
                            waveform = None
                            waveform_metadata = None
                        if waveform is not None:
                            extraction[modality]["waveform"] = waveform_metadata
                            predictions[modality] = self.models.predict_ecg(waveform)
                            modalities[modality] = {"available": True, "status": "processed", "details": {"format": file_format, **waveform_metadata, "model": "ptbxl-xresnet1d-v1"}}
                            if file_format in {"png", "jpg", "jpeg"}:
                                warnings.append({"code": "ECG_IMAGE_DIGITIZATION", "message": "The ECG waveform was digitized from a printed image; it is not equivalent to an original digital ECG recording."})
                        else:
                            modalities[modality] = {"available": False, "status": "unsupported_for_current_model", "details": {"format": file_format, "waveform_available": False}}
                            warnings.append({"code": "ECG_WAVEFORM_UNAVAILABLE", "message": "ECG text is not a waveform; the current XResNet1D model was not run."})
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
            except ECGImageError as exc:
                modalities[modality] = {"available": False, "status": "waveform_quality_insufficient", "details": {"error": str(exc)}}
                warnings.append({"code": "ECG_WAVEFORM_QUALITY_INSUFFICIENT", "message": str(exc)})
            except (DownloadError, ValueError, OSError) as exc:
                modalities[modality] = {"available": False, "status": "processing_failed", "details": {"error": str(exc)}}
                warnings.append({"code": f"{modality.upper()}_PROCESSING_FAILED", "message": str(exc)})

        disease_results = self.engine.evaluate_all(predictions)
        recommendations = sorted({recommendation for result in disease_results.values() for recommendation in result.get("recommendations", [])})
        return {
            "request_id": request_id,
            "service_version": self.version,
            "status": "completed" if not any(item["status"] == "processing_failed" for item in modalities.values()) else "partial",
            "modalities": modalities,
            "extraction": extraction,
            "predictions": predictions,
            "disease_results": list(disease_results.values()),
            "recommendations": recommendations,
            "explainability": {
                "status": "available_after_model_prediction",
                "knowledge_retrieval": "available" if self.knowledge.available else "fallback_local_templates",
            },
            "warnings": warnings,
        }