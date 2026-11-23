import { afterEach, describe, expect, it, vi } from "vitest";
import { config } from "../config.js";
import { runMlInference } from "./ml.js";

afterEach(() => vi.unstubAllGlobals());

describe("ML service integration", () => {
  it("sends Cloudinary URL references in the FastAPI request schema", async () => {
    const result = { status: "completed", disease_results: [] };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(result), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const response = await runMlInference({
      biomarkers: "https://res.cloudinary.com/example/report.pdf",
      ecg: {
        url: "https://res.cloudinary.com/example/record.dat",
        header_url: "https://res.cloudinary.com/example/record.hea",
      },
    });

    expect(response).toEqual(result);
    expect(fetchMock).toHaveBeenCalledOnce();
    const [endpoint, options] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(endpoint).toBe(`${config.mlServiceUrl.replace(/\/+$/, "")}/v1/inference`);
    expect(options.method).toBe("POST");
    expect(JSON.parse(String(options.body))).toEqual({
      biomarkers: { url: "https://res.cloudinary.com/example/report.pdf" },
      ecg: {
        url: "https://res.cloudinary.com/example/record.dat",
        header_url: "https://res.cloudinary.com/example/record.hea",
      },
    });
    expect(options.signal).toBeInstanceOf(AbortSignal);
  });

  it("surfaces a failed ML service response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("unavailable", { status: 503 })));
    await expect(runMlInference({ echo: "https://res.cloudinary.com/example/study.mp4" }))
      .rejects.toThrow("ML service returned 503.");
  });
});

