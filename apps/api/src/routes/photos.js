/**
 * Photo upload routes — accept user photos and persist to disk.
 */

// intent: photo upload endpoint matching frontend UploadResult interface
// status: done
// next: add image quality assessment (brightness, blur, dimensions)
// confidence: high

const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const { v4: uuidv4 } = require("uuid");

const router = express.Router();

const UPLOADS_DIR = path.resolve(__dirname, "../../uploads");

// Auto-create uploads directory if missing
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOADS_DIR),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname) || ".jpg";
    const name = `${uuidv4()}${ext}`;
    cb(null, name);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB
  fileFilter: (_req, file, cb) => {
    const allowed = /^image\/(jpeg|png|webp|gif)$/;
    if (allowed.test(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Only JPEG, PNG, WebP, and GIF images are accepted"));
    }
  },
});

/**
 * POST /api/photos/upload — accept a single image file.
 *
 * Returns UploadResult shape expected by the frontend:
 *   { photo_id, filename, url, width, height, quality }
 */
router.post("/upload", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ detail: "No file uploaded" });
  }

  const { filename, size } = req.file;
  const photoId = path.parse(filename).name; // uuid portion

  // Stub quality assessment — passes everything for now
  const quality = {
    brightness: { status: "pass", value: null },
    blur: { status: "pass", value: null },
    dimensions: { status: "pass", value: null },
    overall: "pass",
  };

  res.json({
    photo_id: photoId,
    filename,
    url: `/uploads/${filename}`,
    width: 0,  // would need sharp/image-size to read real dims
    height: 0,
    quality,
  });
});

module.exports = router;
