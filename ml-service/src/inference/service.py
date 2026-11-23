from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from src.echo_processing.panecho_reader import read_echo_for_panecho
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
from src.xai.explanation_service import ExplanationService


class InferenceService:
    """Orchestrates modalities without making extraction or retrieval a classifier."""

    def __init__(self, engine: StagedFusionEngine, model_manager: ModelManager, version: str = "1.0.0"):
        self.engine = engine
        self.models = model_manager
        self.explanations = ExplanationService(model_manager)
        self.knowledge = KnowledgeRetriever()
        self.gemini = GeminiRecommendationService()
        self.version = version
        self.max_bytes = int(os.getenv("MAX_DOWNLOAD_BYTES", 50 * 1024 * 1024))

    @staticmethod
    def _request_id() -> str:
        return f"cf_{uuid.uuid4().hex}"

    def _run_panecho(self, path, file_format, modalities, extraction, predictions, warnings, explanation_inputs) -> None:
        """Echo branch for the PanEcho backend (16-frame, 224x224, multi-clip averaged)."""
        clips, metadata = read_echo_for_panecho(path, file_format, num_clips=self.models.echo_num_clips)
        explanation_inputs["echo"] = clips
        extraction["echo"] = {"video": metadata, "tensor_shape": list(clips.shape)}
        modalities["echo"] = {"available": True, "status": "processed", "details": dict(metadata)}
        try:
            predictions["echo"], auxiliary = self.models.predict_echo_panecho(clips)
        except (RuntimeError, ValueError) as exc:
            modalities["echo"]["status"] = "model_unavailable"
            warnings.append({"code": "ECHO_MODEL_UNAVAILABLE", "message": str(exc)})
            return
        modalities["echo"]["details"].update(predictions["echo"].get("_metadata", {}))
        modalities["echo"]["details"]["model"] = self.models.echo_model_name
        if "EF_interval_lower" in predictions["echo"]:
            modalities["echo"]["details"]["EF_interval"] = {
                "lower": predictions["echo"]["EF_interval_lower"],
                "upper": predictions["echo"]["EF_interval_upper"],
                "nominal_coverage": predictions["echo"]["_metadata"].get("ef_interval_nominal_coverage"),
            }
        if auxiliary:
            # PanEcho's other heads (valves, RV, LA, LVH, ...) were trained on Yale data and are
            # not validated on EchoNet-Dynamic here, so they are reported but never fused.
            extraction["echo"]["panecho_auxiliary"] = {
                "validated_on_echonet": False,
                "used_in_fusion": False,
                "note": "Research output from the pretrained PanEcho heads; not a diagnosis.",
                "outputs": auxiliary,
            }

    def run(self, request: Any) -> dict[str, Any]:
        request_id = self._request_id()
        modalities = {}
        extraction = {}
        predictions: dict[str, dict[str, Any]] = {}
        explanation_inputs: dict[str, Any] = {}
        explain_requested = bool(getattr(request, "explain", False))
        warnings: list[dict[str, str]] = []

        for modality in ("biomarkers", "ecg", "echo"):
            reference = getattr(request, modality, None)
            if reference is None:
                modalities[modality] = {"available": False, "status": "not_provided", "details": {}}
                continue
            try:
                with download_to_temp(
                    str(reference.url), modality, max_bytes=self.max_bytes, filename=reference.filename
                ) as (path, file_format):
                    if modality == "biomarkers":
                        result = extract_document(path, file_format)
                        extraction[modality] = result.model_dump()
                        adapter_input = select_biomarker_input(result)
                        extraction[modality]["source_candidate"] = adapter_input.source if adapter_input else None
                        if adapter_input and not adapter_input.missing_features:
                            if explain_requested:
                                explanation_inputs[modality] = adapter_input
                            predictions[modality] = self.models.predict_biomarkers(adapter_input)
                            modalities[modality] = {
                                "available": True,
                                "status": "processed",
                                "details": {
                                    "format": file_format,
                                    "dataset": adapter_input.source,
                                    "model": self.models.biomarker_model_name(adapter_input),
                                    "calibrated": self.models.biomarker_is_calibrated(adapter_input),
                                },
                            }
                        else:
                            missing = list(adapter_input.missing_features) if adapter_input else ["recognized biomarker features"]
                            modalities[modality] = {"available": True, "status": "partial_extraction", "details": {"format": file_format, "missing_required_features": missing}}
                            warnings.append({"code": "BIOMARKER_REQUIRED_FEATURES_MISSING", "message": f"The report does not contain enough configured features for a validated biomarker model: {', '.join(missing)}."})
                    elif modality == "ecg":
                        if file_format == "dat" and reference.header_url is None:
                            raise ECGWaveformError("WFDB .dat ECG requires its matching .hea header file.")
                        if file_format == "dat" and reference.header_url is not None:
                            # WFDB uses the .hea record base name and filenames inside
                            # that header to locate signal data. Stage the original pair
                            # together under its submitted basename before reading.
                            submitted_name = (reference.filename or "record.dat").replace("\\", "/").rsplit("/", 1)[-1]
                            record_stem = Path(submitted_name).stem or "record"
                            with tempfile.TemporaryDirectory(prefix="cardiofusion-wfdb-") as directory:
                                record_base = Path(directory) / record_stem
                                dat_path = Path(f"{record_base}.dat")
                                header_path = Path(f"{record_base}.hea")
                                dat_path.write_bytes(path.read_bytes())
                                with download_to_temp(
                                    str(reference.header_url), "ecg_header", max_bytes=self.max_bytes,
                                    filename=reference.header_filename,
                                ) as (downloaded_header, _header_format):
                                    header_path.write_bytes(downloaded_header.read_bytes())
                                waveform, waveform_metadata = read_ecg_file(dat_path, file_format)
                        else:
                            waveform, waveform_metadata = read_ecg_file(path, file_format)
                        extraction[modality] = {"waveform": waveform_metadata}
                        if explain_requested:
                            explanation_inputs[modality] = waveform
                        predictions[modality] = self.models.predict_ecg(waveform)
                        target_meta = predictions[modality].get("_metadata", {}).get("targets", {})
                        modalities[modality] = {
                            "available": True,
                            "status": "processed",
                            "details": {
                                **waveform_metadata,
                                "model": "cnn-bilstm-attention-v1",
                                "calibrated_targets": sorted(
                                    target for target, detail in target_meta.items() if detail.get("calibrated")
                                ),
                            },
                        }
                    elif self.models.echo_backend == "panecho":
                        self._run_panecho(
                            path, file_format, modalities, extraction, predictions, warnings,
                            explanation_inputs if explain_requested else {},
                        )
                    else:
                        clip, metadata = read_echo_frames(path) if file_format == "dicom" else sample_video(path, format_label=file_format.upper())
                        if explain_requested:
                            explanation_inputs[modality] = clip
                        extraction[modality] = {"video": metadata, "tensor_shape": list(clip.transpose(3, 0, 1, 2).shape)}
                        modalities[modality] = {"available": True, "status": "processed", "details": metadata}
                        try:
                            predictions[modality] = self.models.predict_echo(clip)
                            modalities[modality]["status"] = "processed"
                            modalities[modality]["details"].update(predictions[modality].get("_metadata", {}))
                            modalities[modality]["details"]["model"] = self.models.echo_model_name
                        except (RuntimeError, ValueError) as exc:
                            modalities[modality]["status"] = "model_unavailable"
                            warnings.append({"code": "ECHO_MODEL_UNAVAILABLE", "message": str(exc)})
            except ECGWaveformError as exc:
                modalities[modality] = {"available": False, "status": "waveform_unavailable", "details": {"error": str(exc)}}
                warnings.append({"code": "ECG_WAVEFORM_UNAVAILABLE", "message": str(exc)})
            except RuntimeError as exc:
                modalities[modality] = {"available": True, "status": "model_unavailable", "details": {"error": str(exc)}}
                warnings.append({"code": f"{modality.upper()}_MODEL_UNAVAILABLE", "message": str(exc)})
            except (DownloadError, ValueError, OSError) as exc:
                modalities[modality] = {"available": False, "status": "processing_failed", "details": {"error": str(exc)}}
                warnings.append({"code": f"{modality.upper()}_PROCESSING_FAILED", "message": str(exc)})

        disease_results = self.engine.evaluate_all(predictions)
        available_modalities = sorted(predictions)
        recommendations = sorted({recommendation for result in disease_results.values() for recommendation in result.get("recommendations", [])})
        gemini_recommendations = self.gemini.generate(list(disease_results.values()))
        explainability = (
            self.explanations.explain(explanation_inputs)
            if explain_requested
            else {"status": "not_requested", "modalities": {}, "fusion_explanation": "not_applicable_no_trained_paired_fusion_model"}
        )
        explainability["knowledge_retrieval"] = "available" if self.knowledge.available else "fallback_local_templates"
        explainability["gemini"] = gemini_recommendations["status"]
        failed_modalities = any(
            item["status"] not in {"processed", "not_provided"}
            for item in modalities.values()
        )
        return {
            "request_id": request_id,
            "service_version": self.version,
            "status": "partial" if failed_modalities else "completed",
            "modalities": modalities,
            "extraction": extraction,
            "predictions": predictions,
            "available_modalities": available_modalities,
            "result_scope": (
                "none" if not available_modalities else
                f"{available_modalities[0]}-only" if len(available_modalities) == 1 else
                "multimodal"
            ),
            "disease_results": list(disease_results.values()),
            "recommendations": recommendations,
            "gemini_recommendations": gemini_recommendations,
            "explainability": explainability,
            "warnings": warnings,
        }
