/**
 * Feature extraction routes — proxy to feature service.
 */

const express = require("express");
const multer = require("multer");
const router = express.Router();

const upload = multer({ storage: multer.memoryStorage() });

const FEATURE_SERVICE_URL =
  process.env.FEATURE_SERVICE_URL || "http://localhost:8001";

router.post("/extract", upload.single("image"), async (req, res) => {
  try {
    const { Blob } = await import("node:buffer");
    const FormData = (await import("node-fetch")).FormData;

    const form = new FormData();
    form.append("image", new Blob([req.file.buffer]), "image.jpg");

    for (const key of [
      "include_pose",
      "include_parsing",
      "include_embedding",
      "height_cm",
      "weight_kg",
    ]) {
      if (req.body[key]) form.append(key, req.body[key]);
    }

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

    res.json(await response.json());
  } catch (err) {
    res.status(502).json({ detail: `Feature service error: ${err.message}` });
  }
});

router.post("/quality-check", upload.single("image"), async (req, res) => {
  try {
    const { Blob } = await import("node:buffer");
    const FormData = (await import("node-fetch")).FormData;

    const form = new FormData();
    form.append("image", new Blob([req.file.buffer]), "image.jpg");

    const response = await fetch(`${FEATURE_SERVICE_URL}/quality-check`, {
      method: "POST",
      body: form,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res
        .status(response.status)
        .json({ detail: err.detail || "Quality check failed" });
    }

    res.json(await response.json());
  } catch (err) {
    res.status(502).json({ detail: `Feature service error: ${err.message}` });
  }
});

module.exports = router;
