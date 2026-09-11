from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.normalization.biomarker_adapter import SOURCE_SCHEMAS, BiomarkerInput


class ModelManager:
    """Loads compatible trained models once and exposes plain Python outputs."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.device = "cpu"
        self.ecg_model: Any = None
        self.echo_model: Any = None
        self.biomarker_model: Any = None
        self.biomarker_payload: dict[str, Any] | None = None
        self.biomarker_calibrators: dict[str, Any] = {}
        self.ecg_diagnostic_targets: list[str] = []
        self.ecg_rhythm_targets: list[str] = []
        self.status: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            import torch

            requested = os.getenv("CARDIOFUSION_DEVICE", "auto")
            self.device = "cuda" if requested in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
            self._load_ecg(torch)
            self._load_echo(torch)
            self._load_biomarkers(torch)
        except Exception as exc:
            self.status["runtime_error"] = str(exc)

    def _checkpoint(self, env_name: str, default: str) -> Path:
        return Path(os.getenv(env_name, str(self.project_root / default)))

    def _load_ecg(self, torch: Any) -> None:
        path = self._checkpoint("ECG_CHECKPOINT", "checkpoints/ecg/ptbxl_xresnet1d101.pt")
        if not path.exists():
            self.status["ecg"] = "checkpoint_missing"
            return
        from src.models.ecg.xresnet1d import XResNet1D

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.biomarker_payload = payload
        self.ecg_diagnostic_targets = list(payload.get("diagnostic_targets", []))
        self.ecg_rhythm_targets = list(payload.get("rhythm_targets", []))
        model = XResNet1D(
            input_channels=int(payload.get("input_channels", 12)),
            num_diagnostic_classes=len(self.ecg_diagnostic_targets),
            num_rhythm_classes=len(self.ecg_rhythm_targets),
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        self.ecg_model = model.to(self.device).eval()
        self.status["ecg"] = "loaded"

    def _load_echo(self, torch: Any) -> None:
        path = self._checkpoint("ECHO_CHECKPOINT", "checkpoints/echo/echonet_r2plus1d18.pt")
        if not path.exists():
            self.status["echo"] = "checkpoint_missing"
            return
        from src.models.echo.r2plus1d import R2Plus1DRegressor

        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = R2Plus1DRegressor(pretrained=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        self.echo_model = model.to(self.device).eval()
        self.status["echo"] = "loaded"

    def _load_biomarkers(self, torch: Any) -> None:
        path = self._checkpoint("BIOMARKER_CHECKPOINT", "checkpoints/biomarkers/biomarker_ft_transformer.pt")
        preprocessing = self.project_root / "checkpoints" / "biomarkers" / "preprocessing"
        manifest_path = preprocessing / "preprocessing_manifest.json"
        if not path.exists():
            self.status["biomarkers"] = "checkpoint_missing"
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not manifest_path.exists():
            self.status["biomarkers"] = "preprocessing_artifact_missing"
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_dimensions = manifest["source_feature_dimensions"]
            if manifest.get("mode") != "identity" or expected_dimensions != payload["input_dimensions"]:
                self.status["biomarkers"] = "preprocessing_artifact_incompatible"
                return
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.status["biomarkers"] = "preprocessing_artifact_invalid"
            return
        from src.models.biomarkers.ft_transformer import BiomarkerFTTransformer

        model = BiomarkerFTTransformer(
            input_dimensions=payload["input_dimensions"],
            target_specs=payload["target_specs"],
            **payload.get("model_config", {}),
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        self.biomarker_model = model.to(self.device).eval()
        self._load_biomarker_calibrator()
        self.status["biomarkers"] = "loaded"

    def _load_biomarker_calibrator(self) -> None:
        from src.calibration.calibrator import PlattCalibrator

        calibration_root = self.project_root / "checkpoints" / "calibration" / "biomarker"
        for task in ("zheen_acute_mi", "uci_heart_failure_hf_mortality", "framingham_ten_year_chd_risk"):
            path = calibration_root / f"{task}.pkl"
            if path.exists():
                self.biomarker_calibrators[task] = PlattCalibrator.load(path)

    @property
    def models_loaded(self) -> bool:
        return bool(self.ecg_model or self.echo_model or self.biomarker_model)

    def predict_echo(self, clip: np.ndarray) -> dict[str, float]:
        if self.echo_model is None:
            raise RuntimeError("Echo model is unavailable.")
        import torch

        tensor = torch.from_numpy(clip.transpose(3, 0, 1, 2)).unsqueeze(0).float().to(self.device)
        with torch.inference_mode():
            ef = float(self.echo_model(tensor).reshape(-1)[0].item())
        if not np.isfinite(ef) or not 0.0 <= ef <= 100.0:
            raise ValueError("Echo model produced an invalid EF value.")
        return {"EF": ef}

    def predict_ecg(self, waveform: np.ndarray) -> dict[str, float]:
        if self.ecg_model is None:
            raise RuntimeError("ECG model is unavailable.")
        import torch

        if waveform.shape != (12, 1000):
            raise ValueError(f"ECG model requires (12, 1000), got {waveform.shape}.")
        tensor = torch.from_numpy(waveform).unsqueeze(0).float().to(self.device)
        with torch.inference_mode():
            output = self.ecg_model(tensor)
            diagnostic = torch.sigmoid(output["diagnostic"])[0].cpu().numpy()
            rhythm = torch.sigmoid(output["rhythm"])[0].cpu().numpy()
        result = dict(zip(self.ecg_diagnostic_targets, diagnostic.astype(float), strict=True))
        rhythm = dict(zip(self.ecg_rhythm_targets, rhythm.astype(float), strict=True))
        result.update({label.removeprefix("RHYTHM_"): value for label, value in rhythm.items()})
        return result

    def predict_biomarkers(self, inputs: BiomarkerInput) -> dict[str, float]:
        if self.biomarker_model is None or self.biomarker_payload is None:
            raise RuntimeError("Biomarker model is unavailable.")
        if inputs.missing_features:
            raise ValueError(f"Missing required {inputs.source} features: {', '.join(inputs.missing_features)}")
        import torch

        tensor = torch.from_numpy(inputs.values).unsqueeze(0).float().to(self.device)
        with torch.inference_mode():
            outputs = self.biomarker_model(tensor, inputs.source)
            raw = torch.sigmoid(outputs[inputs.task])[0].reshape(-1)[0].item()
        probability = float(raw)
        calibrator = self.biomarker_calibrators.get(SOURCE_SCHEMAS[inputs.source][2])
        if calibrator is not None:
            probability = float(calibrator.predict_proba([probability])[0])
        return {inputs.task: probability, f"{inputs.task}_raw": float(raw)}