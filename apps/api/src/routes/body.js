/**
 * Body measurement routes — proxy to feature service.
 */

const express = require("express");
const multer = require("multer");
const router = express.Router();

const upload = multer({ storage: multer.memoryStorage() });

const FEATURE_SERVICE_URL =
  process.env.FEATURE_SERVICE_URL || "http://localhost:8001";

router.post("/", upload.single("photo"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ detail: "Photo is required" });
    }

    const form = new FormData();
    form.append(
      "image",
      new File([req.file.buffer], req.file.originalname || "photo.jpg", {
        type: req.file.mimetype || "image/jpeg",
      })
    );
    form.append("include_pose", "true");
    form.append("include_parsing", "false");
    form.append("include_embedding", "false");

    if (req.body.height_cm) form.append("height_cm", req.body.height_cm);
    if (req.body.weight_kg) form.append("weight_kg", req.body.weight_kg);

    const response = await fetch(`${FEATURE_SERVICE_URL}/extract`, {
      method: "POST",
      body: form,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res
        .status(response.status)
        .json({ detail: err.detail || "Feature extraction failed" });
    }

    const result = await response.json();
    res.json(result.body_measurements || result);
  } catch (err) {
    res.status(502).json({ detail: `Feature service error: ${err.message}` });
  }
});

router.post("/quick", async (req, res) => {
  res.status(501).json({
    detail: "Quick body measurement without photo is not implemented in feature-service yet",
  });
});

module.exports = router;
