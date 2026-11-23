from __future__ import annotations

import json
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODELS = ("gemini-flash-latest", "gemini-flash-lite-latest")


def _is_probability(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0.0 <= value <= 1.0


def has_model_output(result: dict[str, Any]) -> bool:
    """True only when an uploaded, processed modality produced a value for this condition."""
    if _is_probability(result.get("risk")) or _is_probability(result.get("model_score")):
        return True
    if any(_is_probability(value) for value in (result.get("component_scores") or {}).values()):
        return True
    measurement = result.get("measurement") or {}
    value = measurement.get("value") if isinstance(measurement, dict) else None
    return isinstance(value, (int, float)) and math.isfinite(value)


class GeminiRecommendationService:
    """Generate cautious context after ML prediction; never calculate risk."""

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
        fallbacks = os.getenv("GEMINI_FALLBACK_MODELS")
        fallback_models = (
            [name.strip() for name in fallbacks.split(",") if name.strip()]
            if fallbacks is not None
            else list(DEFAULT_FALLBACK_MODELS)
        )
        self.models = list(dict.fromkeys([self.model, *fallback_models]))
        self._client: Any = None
        self._types: Any = None
        self.status = "not_configured"
        if self.api_key:
            try:
                from google import genai
                from google.genai import types

                self._client = genai.Client(api_key=self.api_key)
                self._types = types
                self.status = "available"
            except Exception as exc:
                logger.warning("Gemini SDK unavailable: %s", exc)
                self.status = "sdk_unavailable"

    @staticmethod
    def _safe_context(disease_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Whitelist prediction fields so documents and identifiers never reach Gemini.

        Only conditions with an actual model output from the uploaded evidence are sent;
        missing-modality and unsupported conditions are left out so Gemini cannot comment
        on results that were never produced.
        """
        fields = (
            "disease",
            "display_name",
            "risk",
            "model_score",
            "status",
            "support_level",
            "calibration_status",
            "population_scope",
            "component_scores",
            "measurement",
        )
        context = []
        for result in disease_results:
            if not has_model_output(result):
                continue
            safe_result = {field: result[field] for field in fields if result.get(field) is not None}
            finding = result.get("finding")
            if isinstance(finding, dict):
                safe_result["finding"] = {
                    key: finding[key] for key in ("present", "rule", "threshold_assessment") if key in finding
                }
            safe_result["detected_from"] = result.get("detected_from", [])
            context.append(safe_result)
        return context

    @staticmethod
    def _fallback(context: list[dict[str, Any]], reason: str) -> dict[str, Any]:
        high_signal = [
            item["display_name"]
            for item in context
            if (_is_probability(item.get("risk")) and item["risk"] >= 0.5)
            or (_is_probability(item.get("model_score")) and item["model_score"] >= 0.5)
            or (isinstance(item.get("finding"), dict) and item["finding"].get("present") is True)
        ]
        recommendations = [
            f"Discuss the model output for {name} with a qualified clinician."
            for name in high_signal[:4]
        ]
        if not recommendations:
            recommendations.append("Use these results as supporting information and discuss them with a qualified clinician.")
        return {
            "source": "local_fallback",
            "status": reason,
            "summary": "These are model outputs, not a diagnosis. A qualified clinician should interpret them with your history and symptoms.",
            "recommendations": recommendations,
        }

    def _prompt(self, context: list[dict[str, Any]]) -> str:
        return (
            "You are a cautious health information assistant. Based only on the "
            "following already-computed model results, provide general educational "
            "suggestions. Field meanings: 'risk' is a calibrated probability; "
            "'model_score' is an uncalibrated model score and must not be described "
            "as a patient's risk or chance; 'component_scores' are per-label model "
            "scores; 'measurement' is a measured value (for example ejection fraction "
            "in %), not a probability; 'finding' is a threshold flag, not a diagnosis. "
            "Treat research_only results as research signals. Only discuss the "
            "conditions listed below; do not mention any other condition. Do not "
            "diagnose, change or restate numbers as different values, infer symptoms, "
            "invent clinical facts, recommend prescription changes, or provide "
            "emergency triage. Mention that a qualified clinician should interpret "
            "results. Return JSON only with exactly: summary (string), "
            "recommendations (array of at most 4 short strings).\n\nRESULTS:\n"
            + json.dumps(context, separators=(",", ":"), allow_nan=False)
        )

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        text = (text or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(text)
        recommendations = parsed.get("recommendations") if isinstance(parsed, dict) else None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("summary"), str) or not isinstance(recommendations, list):
            raise ValueError("Gemini response did not match the recommendation schema.")
        return {
            "summary": parsed["summary"].strip(),
            "recommendations": [str(item).strip() for item in recommendations[:4] if str(item).strip()],
        }

    def generate(self, disease_results: list[dict[str, Any]]) -> dict[str, Any]:
        context = self._safe_context(disease_results)
        if not context:
            return self._fallback(context, "no_actionable_results")
        if self._client is None:
            return self._fallback(context, self.status)

        prompt = self._prompt(context)
        config = (
            self._types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
            if self._types is not None
            else None
        )
        for model in self.models:
            try:
                kwargs: dict[str, Any] = {"model": model, "contents": prompt}
                if config is not None:
                    kwargs["config"] = config
                response = self._client.models.generate_content(**kwargs)
                parsed = self._parse(getattr(response, "text", "") or "")
                return {"source": "gemini", "status": "generated", "model": model, **parsed}
            except Exception as exc:
                # Retired (404), overloaded (503), or rate-limited (429) models fall
                # through to the next configured model instead of failing silently.
                logger.warning("Gemini model %s failed: %s", model, str(exc)[:300])
        return self._fallback(context, "gemini_unavailable")
