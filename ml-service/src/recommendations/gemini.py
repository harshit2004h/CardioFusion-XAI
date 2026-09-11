from __future__ import annotations

import json
import os
from typing import Any


class GeminiRecommendationService:
    """Generate cautious context after ML prediction; never calculate risk."""

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._client: Any = None
        self.status = "not_configured"
        if self.api_key:
            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
                self.status = "available"
            except Exception:
                self.status = "sdk_unavailable"

    @staticmethod
    def _safe_context(disease_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Whitelist prediction fields so documents and identifiers never reach Gemini."""
        return [
            {
                "disease": result.get("disease"),
                "display_name": result.get("display_name"),
                "risk": result.get("risk"),
                "confidence": result.get("confidence"),
                "status": result.get("status"),
                "primary_modality": result.get("primary_modality"),
                "detected_from": result.get("detected_from", []),
            }
            for result in disease_results
            if result.get("risk") is not None or result.get("status") == "unsupported"
        ]

    @staticmethod
    def _fallback(context: list[dict[str, Any]], reason: str) -> dict[str, Any]:
        high_signal = [
            item["display_name"]
            for item in context
            if isinstance(item.get("risk"), (int, float)) and item["risk"] >= 0.5
        ]
        recommendations = [
            f"Discuss the model-estimated {name} result with a qualified clinician."
            for name in high_signal[:4]
        ]
        if not recommendations:
            recommendations.append("Use these results as supporting information and discuss them with a qualified clinician.")
        return {
            "source": "local_fallback",
            "status": reason,
            "summary": "These are model-estimated results, not a diagnosis. A qualified clinician should interpret them with your history and symptoms.",
            "recommendations": recommendations,
        }

    def generate(self, disease_results: list[dict[str, Any]]) -> dict[str, Any]:
        context = self._safe_context(disease_results)
        if not context:
            return self._fallback(context, "no_actionable_results")
        if self._client is None:
            return self._fallback(context, self.status)

        prompt = (
            "You are a cautious health information assistant. Based only on the "
            "following already-computed model results, provide general educational "
            "suggestions. Do not diagnose, change any risk, infer symptoms, invent "
            "clinical facts, recommend prescription changes, or provide emergency "
            "triage. Mention that a qualified clinician should interpret results. "
            "Return JSON only with exactly: summary (string), recommendations "
            "(array of at most 4 short strings).\n\nRESULTS:\n"
            + json.dumps(context, separators=(",", ":"), allow_nan=False)
        )
        try:
            response = self._client.models.generate_content(model=self.model, contents=prompt)
            text = (getattr(response, "text", "") or "").strip()
            text = text.removeprefix("```json").removesuffix("```").strip()
            parsed = json.loads(text)
            recommendations = parsed.get("recommendations")
            if not isinstance(parsed.get("summary"), str) or not isinstance(recommendations, list):
                raise ValueError("Gemini response did not match the recommendation schema.")
            return {
                "source": "gemini",
                "status": "generated",
                "summary": parsed["summary"],
                "recommendations": [str(item) for item in recommendations[:4]],
            }
        except Exception:
            return self._fallback(context, "gemini_unavailable")