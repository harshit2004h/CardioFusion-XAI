import { Router } from "express";
import { prisma } from "../db.js";
import { syncProfile } from "../middleware/profile.js";

export const profileRouter = Router();
profileRouter.use(syncProfile);

profileRouter.get("/", (req, res) => {
  res.json({ user: req.profileUser });
});

profileRouter.patch("/", async (req, res, next) => {
  try {
    const user = req.profileUser!;
    const updated = await prisma.user.update({
      where: { id: user.id },
      data: {
        username: req.body.username ?? user.username,
        firstName: req.body.firstName ?? user.firstName,
        lastName: req.body.lastName ?? user.lastName,
        phone: req.body.phone ?? user.phone,
        address: req.body.address ?? user.address,
      },
    });
    res.json({ user: updated });
  } catch (error) {
    next(error);
  }
});
