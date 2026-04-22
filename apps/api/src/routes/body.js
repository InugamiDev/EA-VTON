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
    const { Blob } = await import("node:buffer");
    const FormData = (await import("node-fetch")).FormData;

    const form = new FormData();
    form.append("image", new Blob([req.file.buffer]), "photo.jpg");
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
  try {
    const response = await fetch(`${FEATURE_SERVICE_URL}/extract`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        height_cm: req.body.height_cm,
        weight_kg: req.body.weight_kg,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res
        .status(response.status)
        .json({ detail: err.detail || "Quick measurement failed" });
    }

    res.json(await response.json());
  } catch (err) {
    res.status(502).json({ detail: `Feature service error: ${err.message}` });
  }
});

module.exports = router;
