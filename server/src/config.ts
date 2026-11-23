import "dotenv/config";

export const config = {
  port: Number(process.env.PORT ?? 4000),
  clientUrl: process.env.CLIENT_URL ?? "http://localhost:3000",
  mlServiceUrl: process.env.ML_SERVICE_URL ?? "http://localhost:3002",
  geminiApiKey: process.env.GEMINI_API_KEY ?? "",
  geminiModel: process.env.GEMINI_MODEL?.trim() || "gemini-3.8-flash",
  geminiFallbackModels: (process.env.GEMINI_FALLBACK_MODELS ?? "gemini-flash-latest,gemini-flash-lite-latest")
    .split(",").map((name) => name.trim()).filter(Boolean),
};

export const requiredEnv = ["DATABASE_URL", "CLERK_SECRET_KEY"] as const;
