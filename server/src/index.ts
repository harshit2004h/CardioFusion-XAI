import "dotenv/config";
import express from "express";
import cors from "cors";
import { clerkMiddleware } from "@clerk/express";
import { config } from "./config.js";
import { profileRouter } from "./routes/profile.js";
import { reportsRouter } from "./routes/reports.js";

const app = express();
app.use(cors({ origin: config.clientUrl, credentials: true }));
app.use(express.json({ limit: "2mb" }));

app.get("/health", (_req, res) => res.json({ status: "ok", service: "cardiofusion-server" }));
app.use(clerkMiddleware());
app.use("/api/profile", profileRouter);
app.use("/api/reports", reportsRouter);

app.use((error: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error(error.message);
  res.status(500).json({ error: "The request could not be completed." });
});

app.listen(config.port, () => console.log(`CardioFusion server listening on http://localhost:${config.port}`));
