import { GoogleGenAI } from "@google/genai";
import { config } from "../config.js";

type DiseaseResult = {
  disease: string;
  display_name?: string;
  risk: number | null;
  confidence?: string;
  status?: string;
};

export async function generateRecommendations(results: DiseaseResult[]) {
  const meaningful = results.filter((item) => item.risk !== null || item.status === "unsupported");
  const fallback = meaningful
    .filter((item) => item.risk !== null && item.risk >= 0.5)
    .map((item) => `Review the ${item.display_name ?? item.disease} result with a qualified clinician.`);

  if (!config.geminiApiKey || meaningful.length === 0) {
    return {
      source: "local_fallback",
      summary: "Recommendations are based on the available modality evidence.",
      recommendations: fallback,
    };
  }

  try {
    const ai = new GoogleGenAI({ apiKey: config.geminiApiKey });
    const prompt = `You are a cautious health information assistant. Based only on these model-estimated results, write a short plain-language summary and up to 4 sensible next-step suggestions. Do not diagnose, invent symptoms, change probabilities, or recommend medication changes. State that a clinician should interpret the results. Return JSON only with keys summary and recommendations (array of strings). Results: ${JSON.stringify(meaningful)}`;
    const response = await ai.models.generateContent({ model: config.geminiModel, contents: prompt });
    const text = response.text?.replace(/^```json\s*|\s*```$/g, "").trim() ?? "";
    const parsed = JSON.parse(text) as { summary?: string; recommendations?: string[] };
    return {
      source: "gemini",
      summary: parsed.summary ?? "Review these model-estimated results with a qualified clinician.",
      recommendations: Array.isArray(parsed.recommendations) ? parsed.recommendations.slice(0, 4) : fallback,
    };
  } catch {
    return {
      source: "local_fallback",
      summary: "Gemini was unavailable, so only conservative local guidance is shown.",
      recommendations: fallback,
    };
  }
}
