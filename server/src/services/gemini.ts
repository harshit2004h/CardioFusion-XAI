import { GoogleGenAI } from "@google/genai";
import { config } from "../config.js";

type DiseaseResult = {
  disease: string;
  display_name?: string;
  risk?: number | null;
  model_score?: number | null;
  component_scores?: Record<string, number>;
  measurement?: { name?: string; value?: number; unit?: string | null };
  finding?: { present?: boolean | null; rule?: string };
  status?: string;
  support_level?: string;
  calibration_status?: string;
  detected_from?: string[];
};

const isProbability = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;

// Only conditions with an actual output from the uploaded evidence are summarized.
function hasModelOutput(item: DiseaseResult) {
  return isProbability(item.risk) || isProbability(item.model_score)
    || Object.values(item.component_scores ?? {}).some(isProbability)
    || (typeof item.measurement?.value === "number" && Number.isFinite(item.measurement.value));
}

function safeContext(results: DiseaseResult[]) {
  return results.filter(hasModelOutput).map((item) => ({
    disease: item.disease,
    display_name: item.display_name,
    risk: item.risk ?? undefined,
    model_score: item.model_score ?? undefined,
    component_scores: item.component_scores,
    measurement: item.measurement,
    finding: item.finding ? { present: item.finding.present, rule: item.finding.rule } : undefined,
    status: item.status,
    support_level: item.support_level,
    calibration_status: item.calibration_status,
    detected_from: item.detected_from ?? [],
  }));
}

export async function generateRecommendations(results: DiseaseResult[]) {
  const context = safeContext(results);
  const fallback = context
    .filter((item) => (isProbability(item.risk) && item.risk >= 0.5) || (isProbability(item.model_score) && item.model_score >= 0.5) || item.finding?.present === true)
    .slice(0, 4)
    .map((item) => `Discuss the model output for ${item.display_name ?? item.disease} with a qualified clinician.`);
  const localFallback = (summary: string) => ({
    source: "local_fallback",
    summary,
    recommendations: fallback.length ? fallback : ["Use these results as supporting information and discuss them with a qualified clinician."],
  });

  if (!config.geminiApiKey || context.length === 0) {
    return localFallback("These are model outputs, not a diagnosis. A qualified clinician should interpret them.");
  }

  const ai = new GoogleGenAI({ apiKey: config.geminiApiKey });
  const prompt = `You are a cautious health information assistant. Based only on these already-computed model results, write a short plain-language summary and up to 4 sensible next-step suggestions. 'risk' is a calibrated probability; 'model_score' is uncalibrated and must not be described as a patient's risk; 'measurement' is a measured value, not a probability; 'finding' is a threshold flag, not a diagnosis. Only discuss the listed conditions. Do not diagnose, invent symptoms, change numbers, or recommend medication changes. State that a clinician should interpret the results. Return JSON only with keys summary (string) and recommendations (array of strings). Results: ${JSON.stringify(context)}`;
  for (const model of [...new Set([config.geminiModel, ...config.geminiFallbackModels])]) {
    try {
      const response = await ai.models.generateContent({
        model,
        contents: prompt,
        config: { responseMimeType: "application/json", temperature: 0.2 },
      });
      const text = response.text?.replace(/^```(?:json)?\s*|\s*```$/g, "").trim() ?? "";
      const parsed = JSON.parse(text) as { summary?: unknown; recommendations?: unknown };
      if (typeof parsed.summary !== "string" || !Array.isArray(parsed.recommendations)) throw new Error("Gemini response did not match the schema.");
      return {
        source: "gemini",
        model,
        summary: parsed.summary,
        recommendations: parsed.recommendations.map(String).slice(0, 4),
      };
    } catch (error) {
      console.warn(`Gemini model ${model} failed:`, error instanceof Error ? error.message.slice(0, 300) : error);
    }
  }
  return localFallback("Gemini was unavailable, so only conservative local guidance is shown.");
}
