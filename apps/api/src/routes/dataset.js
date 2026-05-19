// API gateway proxy for the dataset visualizer endpoint.
// Forwards to recommendation service GET /dataset/stats.

const express = require("express");
const router = express.Router();

const RECOMMENDATION_URL =
  process.env.RECOMMENDATION_SERVICE_URL || "http://localhost:8003";

router.get("/stats", async (_req, res) => {
  try {
    const response = await fetch(`${RECOMMENDATION_URL}/dataset/stats`);
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res
        .status(response.status)
        .json({ detail: err.detail || "Failed to fetch dataset stats" });
    }
    res.json(await response.json());
  } catch (err) {
    res.status(502).json({ detail: `Dataset stats error: ${err.message}` });
  }
});

module.exports = router;
