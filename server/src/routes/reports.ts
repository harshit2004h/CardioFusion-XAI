import { Router } from "express";
import multer from "multer";
import type { Prisma } from "@prisma/client";
import { prisma } from "../db.js";
import { syncProfile } from "../middleware/profile.js";
import { uploadAsset } from "../services/cloudinary.js";
import { generateRecommendations } from "../services/gemini.js";
import { runMlInference } from "../services/ml.js";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024, files: 3 },
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
    { name: "ecg", maxCount: 1 },
    { name: "echo", maxCount: 1 },
  ]),
  async (req, res, next) => {
    const files = req.files as Record<string, Express.Multer.File[]> | undefined;
    const selected = Object.entries(files ?? {}).filter(([, values]) => values?.[0]);
    if (selected.length === 0) return res.status(400).json({ error: "Upload at least one modality file." });

    try {
      const report = await prisma.report.create({
        data: {
          userId: req.profileUser!.id,
          label: String(req.body.label || "New assessment"),
          modalities: selected.map(([modality]) => modality),
        },
      });
      const assets = await Promise.all(selected.map(async ([modality, values]) => [modality, (await uploadAsset(values[0])).secure_url] as const));
      const ml = await runMlInference(Object.fromEntries(assets));
      const recommendations = ml.gemini_recommendations ?? await generateRecommendations((ml.disease_results ?? []) as never[]);
      const status = ml.status === "completed" ? "COMPLETED" : "PARTIAL";
      const updated = await prisma.report.update({
        where: { id: report.id },
        data: { status, result: ml as Prisma.InputJsonValue, recommendations: recommendations as Prisma.InputJsonValue },
      });
      res.status(201).json({ report: updated });
    } catch (error) {
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
