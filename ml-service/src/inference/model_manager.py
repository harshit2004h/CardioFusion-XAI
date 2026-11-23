from __future__ import annotations

import hashlib
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
        # "panecho" or "r2plus1d" once a model is loaded; selects the matching reader in the service.
        self.echo_backend: str | None = None
        self.echo_model_name: str = "echonet-r2plus1d-v1"
        self.echo_num_clips: int = int(os.getenv("PANECHO_NUM_CLIPS", "4"))
        self.biomarker_model: Any = None
        self.biomarker_payload: dict[str, Any] | None = None
        self.ecg_payload: dict[str, Any] | None = None
        self.biomarker_calibrators: dict[str, Any] = {}
        self.biomarker_experts: dict[str, dict[str, Any]] = {}
        self.biomarker_expert_calibrators: dict[str, Any] = {}
        self.ecg_calibrators: dict[str, Any] = {}
        self.echo_ef_interval: dict[str, Any] | None = None
        self.ecg_checkpoint_path: Path | None = None
        self.biomarker_checkpoint_path: Path | None = None
        self.ecg_diagnostic_targets: list[str] = []
        self.ecg_rhythm_targets: list[str] = []
        self.status: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            import torch
        except Exception as exc:
            self.status["runtime_error"] = str(exc)
            return

        requested = os.getenv("CARDIOFUSION_DEVICE", "auto")
        self.device = "cuda" if requested in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
        for name, loader in (
            ("ecg", lambda: self._load_ecg(torch)),
            ("echo", lambda: self._load_echo(torch)),
            ("biomarkers", lambda: self._load_biomarkers(torch)),
        ):
            try:
                loader()
            except Exception as exc:
                # A broken optional checkpoint must not prevent other modalities loading.
                self.status[name] = f"load_error: {exc}"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _calibration_manifest(self) -> dict[str, Any]:
        paths = (
            self.project_root / "checkpoints/calibration/selected_models/v1/manifest.json",
            # Compatibility with artifacts created before the layout migration.
            self.project_root / "checkpoints/calibration/selected_models_manifest.json",
        )
        for path in paths:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and value.get("version") == 1:
                    return value
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return {}

    def _load_platt_if_matching(self, entry: Any, checkpoint: Path) -> Any | None:
        if not isinstance(entry, dict) or entry.get("status") != "platt":
            return None
        if entry.get("base_model_sha256") != self._sha256(checkpoint):
            return None
        artifact = entry.get("artifact")
        if not isinstance(artifact, str):
            return None
        path = self.project_root / "checkpoints" / "calibration" / artifact
        if not path.is_file():
            return None
        from src.calibration.calibrator import PlattCalibrator

        return PlattCalibrator.load(path)

    def _checkpoint(self, env_name: str, default: str) -> Path:
        return Path(os.getenv(env_name, str(self.project_root / default)))

    def _load_ecg(self, torch: Any) -> None:
        defaults = (
            self.project_root / "checkpoints/ecg/ptbxl/selected_cnn_bilstm/v1/model.pt",
            self.project_root / "checkpoints/ecg/ptbxl/ptbxl_cnn_bilstm_attention_100hz_record_zscore.pt",
        )
        configured = os.getenv("ECG_CHECKPOINT", "").strip()
        candidates = list(defaults)
        if configured:
            override = Path(configured).expanduser()
            override = override if override.is_absolute() else self.project_root / override
            candidates.insert(0, override)
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if not path.is_file():
            self.status["ecg"] = "checkpoint_missing"
            return
        if configured and path != candidates[0]:
            self.status["ecg_checkpoint_override"] = "configured path missing; loaded the selected CNN-BiLSTM fallback"
        self.ecg_checkpoint_path = path
        from src.models.ecg.cnn_bilstm_attention import CNNBiLSTMAttention

        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.ecg_payload = payload
        self.ecg_diagnostic_targets = list(payload.get("diagnostic_targets", []))
        self.ecg_rhythm_targets = list(payload.get("rhythm_targets", []))
        model_config = payload.get("model_config", {})
        model = CNNBiLSTMAttention(
            num_leads=int(payload.get("num_leads", 12)),
            num_diagnostic_classes=len(self.ecg_diagnostic_targets),
            num_rhythm_classes=len(self.ecg_rhythm_targets),
            cnn_channels=int(model_config.get("cnn_channels", 128)),
            lstm_hidden=int(model_config.get("lstm_hidden", 128)),
            lstm_layers=int(model_config.get("lstm_layers", 2)),
            dropout=float(model_config.get("dropout", 0.30)),
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        self.ecg_model = model.to(self.device).eval()
        self.status["ecg"] = "loaded_cnn_bilstm_attention"
        ecg_manifest = self._calibration_manifest().get("ecg", {})
        manifest = ecg_manifest.get("cnn_bilstm_attention", {}) if isinstance(ecg_manifest, dict) else {}
        for target, entry in manifest.items() if isinstance(manifest, dict) else ():
            try:
                calibrator = self._load_platt_if_matching(entry, path)
                if calibrator is not None:
                    self.ecg_calibrators[target] = calibrator
            except Exception as exc:
                self.status["ecg_calibration_error"] = f"{target}: {exc}"
        self.status["ecg_calibrated_targets"] = len(self.ecg_calibrators)

    def _panecho_checkpoint(self) -> Path | None:
        """Explicit PANECHO_CHECKPOINT, else the EchoNet fine-tuned file, else the released weights."""
        explicit = os.getenv("PANECHO_CHECKPOINT")
        if explicit:
            path = Path(explicit).expanduser()
            return path if path.is_absolute() else self.project_root / path
        try:
            import yaml

            config = yaml.safe_load(
                (self.project_root / "config" / "echo.yaml").read_text(encoding="utf-8")
            ) or {}
            panecho = config.get("panecho", {})
            checkpoints = panecho.get("checkpoints", {})
            candidates = (
                checkpoints.get("finetuned"),
                checkpoints.get("pretrained"),
            )
        except (OSError, AttributeError, ImportError):
            candidates = ()
        root = self.project_root / "checkpoints" / "echo" / "panecho"
        for candidate in ("checkpoints/echo/echonet/selected_panecho/v1/model.pt", *candidates,
                          "checkpoints/echo/panecho/panecho_echonet_finetuned.pt",
                          "checkpoints/echo/panecho.pt",
                          "checkpoints/echo/panecho/panecho.pt",
                          "checkpoints/echo/panecho/panecho.pt.zip"):
            if not candidate:
                continue
            path = Path(candidate)
            path = path if path.is_absolute() else self.project_root / path
            if path.exists():
                return path
        return None

    def _load_echo(self, torch: Any) -> None:
        backend = os.getenv("ECHO_MODEL_BACKEND")
        if not backend:
            try:
                import yaml

                config = yaml.safe_load(
                    (self.project_root / "config" / "echo.yaml").read_text(encoding="utf-8")
                ) or {}
                backend = str(config.get("backend", "auto"))
            except (OSError, ImportError):
                backend = "auto"
        backend = backend.strip().lower()
        if backend not in {"auto", "panecho", "r2plus1d"}:
            self.status["echo"] = f"invalid_backend:{backend}"
            return
        if backend in {"auto", "panecho"}:
            panecho_path = self._panecho_checkpoint()
            if panecho_path is not None and panecho_path.exists():
                self._load_panecho(torch, panecho_path)
                if self.echo_model is not None or backend == "panecho":
                    return
            if backend == "panecho":
                self.status["echo"] = "checkpoint_missing"
                return
        self._load_r2plus1d(torch)

    def _load_panecho(self, torch: Any, path: Path) -> None:
        from src.models.echo.panecho import load_panecho

        try:
            self.echo_model = load_panecho(path, tasks="auto", device=self.device)
        except Exception as exc:  # keep the other modalities loading
            self.status["echo"] = f"checkpoint_invalid: {exc}"
            return
        fine_tuned = bool(self.echo_model.checkpoint_metadata.get("fine_tuned"))
        self.echo_backend = "panecho"
        self.echo_model_name = "panecho-echonet-finetuned-v1" if fine_tuned else "panecho-v1-pretrained"
        self.status["echo"] = "loaded_panecho_finetuned" if fine_tuned else "loaded_panecho_pretrained"
        if fine_tuned:
            echo_manifest = self._calibration_manifest().get("echo", {})
            manifest = echo_manifest.get("panecho_echonet_finetuned", {}) if isinstance(echo_manifest, dict) else {}
            entry = manifest.get("EF") if isinstance(manifest, dict) else None
            if isinstance(entry, dict) and entry.get("status") == "split_conformal":
                try:
                    expected_hash = entry.get("base_model_sha256")
                    artifact = entry.get("artifact")
                    artifact_path = self.project_root / "checkpoints" / "calibration" / str(artifact)
                    if expected_hash == self._sha256(path) and artifact_path.is_file():
                        interval = json.loads(artifact_path.read_text(encoding="utf-8"))
                        quantile = float(interval["absolute_error_quantile"])
                        alpha = float(interval["alpha"])
                        if np.isfinite(quantile) and quantile >= 0 and 0 < alpha < 1:
                            self.echo_ef_interval = {**interval, "absolute_error_quantile": quantile}
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    self.echo_ef_interval = None
            self.status["echo_ef_interval"] = "loaded_split_conformal" if self.echo_ef_interval else "unavailable"

    def _load_r2plus1d(self, torch: Any) -> None:
        configured = os.getenv("ECHO_CHECKPOINT")
        if configured:
            path = Path(configured)
            path = path if path.is_absolute() else self.project_root / path
        else:
            candidates = (
                self.project_root / "checkpoints/echo/echonet/r2plus1d_baseline/v1/model.pt",
                self.project_root / "checkpoints/echo/echonet_r2plus1d18.pt",
            )
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if not path.exists():
            self.status["echo"] = "checkpoint_missing"
            return
        from src.models.echo.r2plus1d import R2Plus1DRegressor

        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = R2Plus1DRegressor(pretrained=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        self.echo_model = model.to(self.device).eval()
        self.echo_backend = "r2plus1d"
        self.echo_model_name = "echonet-r2plus1d-v1"
        self.status["echo"] = "loaded"

    def _load_biomarkers(self, torch: Any) -> None:
        import joblib

        manifest = self._calibration_manifest().get("biomarkers", {})
        sources = () if os.getenv("BIOMARKER_CHECKPOINT") else SOURCE_SCHEMAS.items()
        for source, (features, task, _) in sources:
            path = (
                self.project_root
                / "checkpoints"
                / "biomarkers"
                / source
                / task
                / "selected_gbdt_stacking"
                / "v1"
                / "model.joblib"
            )
            if not path.is_file():
                continue
            try:
                artifact = joblib.load(path)
                model = artifact["model"]
                feature_names = tuple(artifact["feature_names"])
                if artifact.get("target") != task or feature_names != features:
                    raise ValueError("selected model target or canonical feature order does not match the inference schema")
                if not callable(getattr(model, "predict_proba", None)):
                    raise ValueError("selected model does not expose predict_proba")
                classes = list(getattr(model, "classes_", []))
                if 1 not in classes:
                    raise ValueError("selected model has no positive class 1")
                self.biomarker_experts[source] = {
                    "model": model,
                    "feature_names": feature_names,
                    "selected_model": str(artifact.get("selected_model", type(model).__name__)),
                    "background_features": np.asarray(artifact.get("background_features", []), dtype=np.float32),
                    "path": path,
                    "sha256": self._sha256(path),
                }
                source_manifest = manifest.get(source, {}) if isinstance(manifest, dict) else {}
                entry = source_manifest.get(task) if isinstance(source_manifest, dict) else None
                calibrator = self._load_platt_if_matching(entry, path)
                if calibrator is not None:
                    self.biomarker_expert_calibrators[source] = calibrator
            except Exception as exc:
                self.status[f"biomarker_{source}"] = f"selected_model_invalid: {exc}"

        configured = os.getenv("BIOMARKER_CHECKPOINT")
        if configured:
            path = Path(configured)
            path = path if path.is_absolute() else self.project_root / path
        else:
            candidates = (
                self.project_root / "checkpoints/biomarkers/multisource/ft_transformer/v1/model.pt",
                self.project_root / "checkpoints/biomarkers/biomarker_ft_transformer.pt",
            )
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        preprocessing = self.project_root / "checkpoints" / "biomarkers" / "preprocessing"
        manifest_path = preprocessing / "preprocessing_manifest.json"
        if not path.exists():
            self.status["biomarkers"] = "loaded_selected_sources" if self.biomarker_experts else "checkpoint_missing"
            return
        self.biomarker_checkpoint_path = path
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.biomarker_payload = payload
            if not manifest_path.exists():
                raise FileNotFoundError("preprocessing artifact missing")
            preprocessing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_dimensions = preprocessing_manifest["source_feature_dimensions"]
            if preprocessing_manifest.get("mode") != "identity" or expected_dimensions != payload["input_dimensions"]:
                raise ValueError("preprocessing artifact incompatible")
            from src.models.biomarkers.ft_transformer import BiomarkerFTTransformer

            model = BiomarkerFTTransformer(
                input_dimensions=payload["input_dimensions"],
                target_specs=payload["target_specs"],
                **payload.get("model_config", {}),
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            self.biomarker_model = model.to(self.device).eval()
            self._load_biomarker_calibrator()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.status["biomarker_ft_transformer"] = f"unavailable: {exc}"
        if self.biomarker_experts:
            self.status["biomarkers"] = "loaded_selected_sources"
        elif self.biomarker_model is not None:
            self.status["biomarkers"] = "loaded_ft_transformer"
        else:
            self.status["biomarkers"] = self.status.get("biomarker_ft_transformer", "checkpoint_missing")

    def _load_biomarker_calibrator(self) -> None:
        calibration_root = self.project_root / "checkpoints" / "calibration" / "biomarker"
        legacy_artifacts = list(calibration_root.glob("*.pkl")) if calibration_root.is_dir() else []
        if any(path.is_file() for path in legacy_artifacts):
            # These older files have no base-checkpoint hash, so applying them
            # could silently calibrate a different model version.
            self.status["biomarker_legacy_calibration"] = "ignored_missing_checkpoint_provenance"
        else:
            self.status["biomarker_legacy_calibration"] = "not_available"

    @property
    def models_loaded(self) -> bool:
        return bool(self.ecg_model or self.echo_model or self.biomarker_model or self.biomarker_experts)

    def predict_echo_panecho(self, clips: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """PanEcho on [num_clips, 3, 16, 224, 224]; outputs are averaged over clips.

        Returns (fusion predictions, auxiliary outputs). Only EF goes to the fusion
        engine: it is the one PanEcho output validated against EchoNet labels here.
        """
        if self.echo_model is None or self.echo_backend != "panecho":
            raise RuntimeError("PanEcho echo model is unavailable.")
        import torch

        from src.models.echo.panecho import decode_outputs

        if clips.ndim != 5 or tuple(clips.shape[1:]) != (3, self.echo_model.clip_len, 224, 224):
            raise ValueError(f"PanEcho requires [clips, 3, {self.echo_model.clip_len}, 224, 224], got {tuple(clips.shape)}.")
        outputs = self.echo_model.predict_clips(clips.float().to(self.device))
        if not all(torch.isfinite(value).all() for value in outputs.values()):
            raise ValueError("Echo model produced non-finite outputs.")
        decoded = decode_outputs(outputs)
        ef = float(decoded["EF"]["value"])
        if not 0.0 <= ef <= 100.0:
            raise ValueError("Echo model produced an invalid EF value.")
        auxiliary = {name: value for name, value in decoded.items() if name != "EF"}
        fusion = {"EF": ef}
        if self.echo_ef_interval is not None:
            quantile = float(self.echo_ef_interval["absolute_error_quantile"])
            fusion["EF_interval_lower"] = float(np.clip(ef - quantile, 0.0, 100.0))
            fusion["EF_interval_upper"] = float(np.clip(ef + quantile, 0.0, 100.0))
        fusion["_metadata"] = {
            "model": self.echo_model_name,
            "ef_interval": "split_conformal" if self.echo_ef_interval is not None else "unavailable",
            "ef_interval_nominal_coverage": (
                float(self.echo_ef_interval["nominal_coverage"]) if self.echo_ef_interval is not None else None
            ),
        }
        return fusion, auxiliary

    def predict_echo(self, clip: np.ndarray) -> dict[str, Any]:
        if self.echo_model is None:
            raise RuntimeError("Echo model is unavailable.")
        if self.echo_backend == "panecho":
            raise RuntimeError("PanEcho is loaded; use predict_echo_panecho with PanEcho clips.")
        import torch

        tensor = torch.from_numpy(clip.transpose(3, 0, 1, 2)).unsqueeze(0).float().to(self.device)
        with torch.inference_mode():
            ef = float(self.echo_model(tensor).reshape(-1)[0].item())
        if not np.isfinite(ef) or not 0.0 <= ef <= 100.0:
            raise ValueError("Echo model produced an invalid EF value.")
        return {"EF": ef, "_metadata": {"model": self.echo_model_name, "ef_interval": "unavailable"}}

    def predict_ecg(self, waveform: np.ndarray) -> dict[str, Any]:
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
        if not np.isfinite(diagnostic).all() or not np.isfinite(rhythm).all():
            raise ValueError("ECG model produced non-finite probabilities.")
        result: dict[str, Any] = {}
        target_metadata: dict[str, dict[str, Any]] = {}
        for label, value in [*zip(self.ecg_diagnostic_targets, diagnostic, strict=True),
                             *zip(self.ecg_rhythm_targets, rhythm, strict=True)]:
            key = label.removeprefix("RHYTHM_")
            raw = float(value)
            calibrator = self.ecg_calibrators.get(label)
            calibrated = float(calibrator.predict_proba([raw])[0]) if calibrator is not None else raw
            if not np.isfinite(calibrated) or not 0.0 <= calibrated <= 1.0:
                raise ValueError(f"ECG calibrator produced an invalid probability for {label}.")
            result[key] = calibrated
            result[f"{key}_raw"] = raw
            target_metadata[key] = {
                "calibrated": calibrator is not None,
                "calibration_method": "platt" if calibrator is not None else None,
            }
        result["_metadata"] = {"model": "cnn-bilstm-attention-v1", "targets": target_metadata}
        return result

    def predict_biomarkers(self, inputs: BiomarkerInput) -> dict[str, Any]:
        if self.biomarker_model is None and inputs.source not in self.biomarker_experts:
            raise RuntimeError("Biomarker model is unavailable.")
        if inputs.missing_features:
            raise ValueError(f"Missing required {inputs.source} features: {', '.join(inputs.missing_features)}")
        if not np.isfinite(inputs.values).all():
            raise ValueError("Biomarker input contains non-finite values.")
        expert = self.biomarker_experts.get(inputs.source)
        model_name = "biomarker-ft-v1"
        calibrator = None
        if expert is not None:
            import pandas as pd

            if tuple(inputs.feature_names) != expert["feature_names"]:
                raise ValueError("Biomarker feature order does not match the selected model artifact.")
            frame = pd.DataFrame([inputs.values], columns=inputs.feature_names)
            model = expert["model"]
            classes = list(model.classes_)
            raw = float(model.predict_proba(frame)[0, classes.index(1)])
            calibrator = self.biomarker_expert_calibrators.get(inputs.source)
            model_name = expert["selected_model"]
        else:
            import torch

            tensor = torch.from_numpy(inputs.values).unsqueeze(0).float().to(self.device)
            with torch.inference_mode():
                outputs = self.biomarker_model(tensor, inputs.source)
                raw = float(torch.sigmoid(outputs[inputs.task])[0].reshape(-1)[0].item())
            calibrator = self.biomarker_calibrators.get(SOURCE_SCHEMAS[inputs.source][2])
        if not np.isfinite(raw) or not 0.0 <= raw <= 1.0:
            raise ValueError("Biomarker model produced an invalid probability.")
        probability = float(calibrator.predict_proba([raw])[0]) if calibrator is not None else raw
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Biomarker calibrator produced an invalid probability.")
        return {
            inputs.task: probability,
            f"{inputs.task}_raw": raw,
            "_metadata": {
                "model": model_name,
                "targets": {
                    inputs.task: {
                        "calibrated": calibrator is not None,
                        "calibration_method": "platt" if calibrator is not None else None,
                    }
                },
            },
        }

    def biomarker_model_name(self, inputs: BiomarkerInput) -> str:
        expert = self.biomarker_experts.get(inputs.source)
        return expert["selected_model"] if expert is not None else "biomarker-ft-v1"

    def biomarker_is_calibrated(self, inputs: BiomarkerInput) -> bool:
        return inputs.source in self.biomarker_expert_calibrators or (
            inputs.source in SOURCE_SCHEMAS
            and SOURCE_SCHEMAS[inputs.source][2] in self.biomarker_calibrators
        )
