// API gateway proxy for population-aware upper-body style recommendation.
// Forwards to recommendation service POST /recommend-style/upper.

const express = require("express");
const router = express.Router();

const RECOMMENDATION_URL =
  process.env.RECOMMENDATION_SERVICE_URL || "http://localhost:8003";

/**
 * POST /api/style/upper — Fit-Flatter-Match upper-body recommendation.
 *
 * Body:
 *   height_cm, predicted_size, population?, ratios {sh, wh, st?, tl?, at?},
 *   context {occasion, sliders {bold?, loose?, warm?, cover?}},
 *   user_palette_lab? [L, a, b], top_k?, weights?
 */
router.post("/upper", async (req, res) => {
  try {
    const response = await fetch(
      `${RECOMMENDATION_URL}/recommend-style/upper`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      },
    );

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res
        .status(response.status)
        .json({ detail: err.detail || "Style recommendation failed" });
    }

    res.json(await response.json());
  } catch (err) {
    res
      .status(502)
      .json({ detail: `Style recommendation error: ${err.message}` });
  }
});

/**
 * POST /api/style/upper/seed-items — return 18 stratified-diverse seed items
 * for cold-start preference elicitation. No body params required.
 */
router.post("/upper/seed-items", async (req, res) => {
  try {
    const response = await fetch(
      `${RECOMMENDATION_URL}/recommend-style/upper/seed-items`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body || {}),
      },
    );
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res
        .status(response.status)
        .json({ detail: err.detail || "Seed items request failed" });
    }
    res.json(await response.json());
  } catch (err) {
    res.status(502).json({ detail: `Seed items error: ${err.message}` });
  }
});

/**
 * POST /api/style/upper/calibrate — re-rank top-K via CLIP-centroid of liked items.
 * Body: same as /upper plus liked_garment_ids: string[] (3-20 items).
 */
router.post("/upper/calibrate", async (req, res) => {
  try {
    const response = await fetch(
      `${RECOMMENDATION_URL}/recommend-style/upper/calibrate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      },
    );
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res
        .status(response.status)
        .json({ detail: err.detail || "Calibration failed" });
    }
    res.json(await response.json());
  } catch (err) {
    res.status(502).json({ detail: `Calibration error: ${err.message}` });
  }
});

module.exports = router;
