import { v2 as cloudinary } from "cloudinary";
import { Readable } from "node:stream";

const cloudName = process.env.CLOUDINARY_CLOUD_NAME;
const apiKey = process.env.CLOUDINARY_API_KEY;
const apiSecret = process.env.CLOUDINARY_API_SECRET;

if (cloudName && apiKey && apiSecret) {
  cloudinary.config({ cloud_name: cloudName, api_key: apiKey, api_secret: apiSecret, secure: true });
}

export function uploadAsset(file: Express.Multer.File, folder = "cardiofusion/reports") {
  if (!cloudName || !apiKey || !apiSecret) {
    throw new Error("Cloudinary credentials are not configured.");
  }

  return new Promise<{ secure_url: string }>((resolve, reject) => {
    const stream = cloudinary.uploader.upload_stream(
      { folder, resource_type: "auto" },
      (error, result) => {
        if (error || !result?.secure_url) reject(error ?? new Error("Cloudinary upload failed."));
        else resolve({ secure_url: result.secure_url });
      },
    );
    Readable.from(file.buffer).pipe(stream);
  });
}
