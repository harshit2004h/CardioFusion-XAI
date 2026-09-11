import { config } from "../config.js";

export async function runMlInference(urls: Record<string, string>) {
  const response = await fetch(`${config.mlServiceUrl}/v1/inference`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(Object.fromEntries(Object.entries(urls).map(([modality, url]) => [modality, { url }]))),
  });
  if (!response.ok) throw new Error(`ML service returned ${response.status}.`);
  return response.json() as Promise<{ disease_results?: Array<Record<string, unknown>>; [key: string]: unknown }>;
}
