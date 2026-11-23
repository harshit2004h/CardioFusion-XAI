from __future__ import annotations

from typing import Any

import numpy as np


class ExplanationService:
    """Run model-specific explanations when the caller explicitly enables XAI.

    This service deliberately returns per-modality explanations. It does not create
    a SHAP explanation for the deterministic evidence orchestrator because there is
    no trained fusion predictor in this unpaired-data project.
    """

    def __init__(self, model_manager: Any):
        self.models = model_manager

    def explain_biomarker(self, item: Any) -> dict[str, Any]:
        import pandas as pd
        from src.xai.biomarker_shap import explain_tabular_prediction

        source = item.source
        expert = self.models.biomarker_experts.get(source)
        if expert is None:
            raise RuntimeError(f"Selected biomarker model unavailable for {source}.")
        feature_names = expert["feature_names"]
        background = expert["background_features"]
        if background.ndim != 2 or background.shape[0] < 2:
            raise FileNotFoundError(f"Same-source training background is unavailable for {source}.")
        if tuple(item.feature_names) != feature_names or not np.isfinite(item.values).all():
            raise ValueError("Biomarker explanation input does not match the selected model schema.")
        model = expert["model"]
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError("Selected biomarker model does not contain positive class 1.")
        calibrator = self.models.biomarker_expert_calibrators.get(source)

        def predict_function(values: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(values, columns=feature_names)
            probabilities = model.predict_proba(frame)[:, classes.index(1)]
            return calibrator.predict_proba(probabilities) if calibrator is not None else probabilities

        return {
            "dataset": source,
            "target": item.task,
            "calibration_status": "platt" if calibrator is not None else "uncalibrated_model_score",
            **explain_tabular_prediction(predict_function, background, item.values, feature_names),
        }

    def explain_ecg(self, waveform: np.ndarray) -> dict[str, Any]:
        from src.xai.ecg_explainer import integrated_gradients_ecg

        if self.models.ecg_model is None:
            raise RuntimeError("ECG model is unavailable.")
        predictions = self.models.predict_ecg(waveform)
        output_names = [*self.models.ecg_diagnostic_targets, *self.models.ecg_rhythm_targets]
        output_keys = {name: name.removeprefix("RHYTHM_") for name in output_names}
        ordered = sorted(
            ((key, float(predictions[key])) for key in output_keys.values() if key in predictions),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        explanations = []
        diagnostic_lookup = {name: ("diagnostic", index) for index, name in enumerate(self.models.ecg_diagnostic_targets)}
        rhythm_lookup = {name.removeprefix("RHYTHM_"): ("rhythm", index) for index, name in enumerate(self.models.ecg_rhythm_targets)}
        lookup = {**diagnostic_lookup, **rhythm_lookup}
        for label, probability in ordered:
            if label not in lookup:
                continue
            head, index = lookup[label]
            explanations.append({"label": label, "model_probability": probability,
                                 **integrated_gradients_ecg(self.models.ecg_model, waveform, index, task=head, device=self.models.device)})
        return {"method": "IntegratedGradients", "explained_outputs": explanations}

    def explain_echo(self, clip: np.ndarray) -> dict[str, Any]:
        from src.xai.echo_explainer import temporal_occlusion_explanation

        if self.models.echo_model is None:
            raise RuntimeError("Echo model is unavailable.")
        if self.models.echo_backend == "panecho":
            import torch
            from src.models.echo.panecho import decode_outputs

            def predict_function(array: np.ndarray) -> float:
                tensor = torch.as_tensor(array, dtype=torch.float32, device=self.models.device)
                with torch.inference_mode():
                    output = self.models.echo_model.predict_clips(tensor)
                    decoded = decode_outputs(output)
                return float(decoded["EF"]["value"])
        else:
            def predict_function(array: np.ndarray) -> float:
                return float(self.models.predict_echo(array)["EF"])
        return temporal_occlusion_explanation(predict_function, clip, window_frames=4)

    def explain(self, inputs: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {"status": "partial_or_complete", "modalities": {}, "fusion_explanation": "not_applicable_no_trained_paired_fusion_model"}
        for modality, function in (
            ("biomarkers", lambda: self.explain_biomarker(inputs["biomarkers"])),
            ("ecg", lambda: self.explain_ecg(inputs["ecg"])),
            ("echo", lambda: self.explain_echo(inputs["echo"])),
        ):
            if modality not in inputs:
                continue
            try:
                result["modalities"][modality] = {"status": "generated", "explanation": function()}
            except Exception as exc:
                result["modalities"][modality] = {"status": "not_generated", "error": f"{type(exc).__name__}: {exc}"}
        if result["modalities"] and all(value["status"] == "generated" for value in result["modalities"].values()):
            result["status"] = "generated"
        elif result["modalities"]:
            result["status"] = "partial"
        else:
            result["status"] = "no_explanation_inputs"
        return result
