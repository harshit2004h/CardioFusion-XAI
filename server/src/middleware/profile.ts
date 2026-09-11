import type { NextFunction, Request, Response } from "express";
import { getAuth } from "@clerk/express";
import { prisma } from "../db.js";

export async function syncProfile(req: Request, res: Response, next: NextFunction) {
  try {
    const { userId } = getAuth(req);
    if (!userId) return res.status(401).json({ error: "Authentication required." });
    const email = String(req.body?.email ?? req.headers["x-user-email"] ?? `${userId}@clerk.local`);
    req.profileUser = await prisma.user.upsert({
      where: { clerkId: userId },
      update: { email },
      create: { clerkId: userId, email },
    });
    next();
  } catch (error) {
    next(error);
  }
}
