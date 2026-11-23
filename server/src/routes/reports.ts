import { Router } from "express";
import multer from "multer";
import path from "node:path";
import type { Prisma } from "../../generated/prisma/client.js";
import { prisma } from "../db.js";
import { syncProfile } from "../middleware/profile.js";
import { uploadAsset } from "../services/cloudinary.js";
import { generateRecommendations } from "../services/gemini.js";
import { runMlInference } from "../services/ml.js";

const allowedExtensions: Record<string, Set<string>> = {
  biomarkers: new Set([".pdf", ".jpg", ".jpeg", ".png"]),
  ecg: new Set([".dcm", ".dicom", ".dat", ".hea", ".csv", ".scp", ".edf"]),
  echo: new Set([".avi", ".mp4"]),
};

class UploadValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UploadValidationError";
  }
}

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024, files: 4 },
  fileFilter: (_request, file, callback) => {
    const extension = path.extname(file.originalname).toLowerCase();
    if (allowedExtensions[file.fieldname]?.has(extension)) callback(null, true);
    else callback(new UploadValidationError(`Unsupported ${file.fieldname} file format: ${extension || "unknown"}.`));
  },
});

export const reportsRouter = Router();
reportsRouter.use(syncProfile);

reportsRouter.get("/", async (req, res, next) => {
  try {
    const reports = await prisma.report.findMany({
      where: { userId: req.profileUser!.id },
      orderBy: { createdAt: "desc" },
      take: 20,
    });
    res.json({ reports });
  } catch (error) {
    next(error);
  }
});

reportsRouter.post(
  "/",
  upload.fields([
    { name: "biomarkers", maxCount: 1 },
    { name: "ecg", maxCount: 2 },
    { name: "echo", maxCount: 1 },
  ]),
  async (req, res, next) => {
    const files = req.files as Record<string, Express.Multer.File[]> | undefined;
    const selected = Object.entries(files ?? {}).filter(([, values]) => values?.[0]);
    if (selected.length === 0) return res.status(400).json({ error: "Upload at least one modality file." });

    const ecgFiles = files?.ecg ?? [];
    const waveform = ecgFiles.find((file) => path.extname(file.originalname).toLowerCase() === ".dat");
    const header = ecgFiles.find((file) => path.extname(file.originalname).toLowerCase() === ".hea");
    if (waveform || header) {
      if (ecgFiles.length !== 2 || !waveform || !header) {
        return res.status(400).json({ error: "A WFDB ECG upload requires exactly one .dat and one .hea file." });
      }
      const waveformStem = path.basename(waveform.originalname, path.extname(waveform.originalname)).toLowerCase();
      const headerStem = path.basename(header.originalname, path.extname(header.originalname)).toLowerCase();
      if (waveformStem !== headerStem) {
        return res.status(400).json({ error: "The WFDB .dat and .hea files must have the same base filename." });
      }
    } else if (ecgFiles.length > 1) {
      return res.status(400).json({ error: "Upload one ECG file, or a matching .dat and .hea pair." });
    }

    let reportId: string | undefined;
    try {
      const report = await prisma.report.create({
        data: {
          userId: req.profileUser!.id,
          label: String(req.body.label || "New assessment"),
          modalities: selected.map(([modality]) => modality),
        },
      });
      reportId = report.id;
      const assets: Record<string, string | { url: string; filename?: string; header_url?: string; header_filename?: string }> = {};
      for (const [modality, values] of selected) {
        if (modality !== "ecg") {
          assets[modality] = { url: (await uploadAsset(values[0])).secure_url, filename: values[0].originalname };
          continue;
        }
        if (waveform && header) {
          assets.ecg = {
            url: (await uploadAsset(waveform)).secure_url,
            filename: waveform.originalname,
            header_url: (await uploadAsset(header)).secure_url,
            header_filename: header.originalname,
          };
          continue;
        }
        assets.ecg = { url: (await uploadAsset(values[0])).secure_url, filename: values[0].originalname };
      }
      const ml = await runMlInference(assets);
      const recommendations = ml.gemini_recommendations ?? await generateRecommendations((ml.disease_results ?? []) as never[]);
      const status = ml.status === "completed" ? "COMPLETED" : "PARTIAL";
      const updated = await prisma.report.update({
        where: { id: report.id },
        data: { status, result: ml as Prisma.InputJsonValue, recommendations: recommendations as Prisma.InputJsonValue },
      });
      res.status(201).json({ report: updated });
    } catch (error) {
      if (reportId) {
        try {
          await prisma.report.update({
            where: { id: reportId },
            data: {
              status: "FAILED",
              errorMessage: "Assessment processing failed. Check the files and service availability, then retry.",
            },
          });
        } catch (updateError) {
          console.error("Could not mark failed assessment:", updateError);
        }
      }
      next(error);
    }
  },
);

reportsRouter.get("/:id", async (req, res, next) => {
  try {
    const report = await prisma.report.findFirst({ where: { id: req.params.id, userId: req.profileUser!.id } });
    if (!report) return res.status(404).json({ error: "Report not found." });
    res.json({ report });
  } catch (error) {
    next(error);
  }
});
