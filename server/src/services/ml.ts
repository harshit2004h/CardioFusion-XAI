import { config } from "../config.js";

type InferenceAsset = string | { url: string; filename?: string; header_url?: string; header_filename?: string };

export async function runMlInference(urls: Record<string, InferenceAsset>) {
  const endpoint = `${config.mlServiceUrl.replace(/\/+$/, "")}/v1/inference`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(Object.fromEntries(Object.entries(urls).map(([modality, asset]) => [modality, typeof asset === "string" ? { url: asset } : asset]))),
    signal: AbortSignal.timeout(5 * 60 * 1000),
  });
  if (!response.ok) throw new Error(`ML service returned ${response.status}.`);
  return response.json() as Promise<{ disease_results?: Array<Record<string, unknown>>; [key: string]: unknown }>;
}
