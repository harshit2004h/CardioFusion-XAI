import { config } from "../config.js";

type InferenceAsset = string | { url: string; header_url: string };

export async function runMlInference(urls: Record<string, InferenceAsset>) {
  const response = await fetch(`${config.mlServiceUrl}/v1/inference`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(Object.fromEntries(Object.entries(urls).map(([modality, asset]) => [modality, typeof asset === "string" ? { url: asset } : asset]))),
  });
  if (!response.ok) throw new Error(`ML service returned ${response.status}.`);
  return response.json() as Promise<{ disease_results?: Array<Record<string, unknown>>; [key: string]: unknown }>;
}
