/**
 * Size recommendation routes — proxy to recommendation service.
 */

const express = require("express");
const router = express.Router();

const RECOMMENDATION_URL =
  process.env.RECOMMENDATION_SERVICE_URL || "http://localhost:8003";

router.post("/", async (req, res) => {
  try {
    const garments = require("../data/garments.json");
    const garment = garments.find((g) => g.id === req.body.garment_id);
    if (!garment) {
      return res.status(404).json({ detail: "Garment not found" });
    }

    const response = await fetch(`${RECOMMENDATION_URL}/recommend-size`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        height_cm: req.body.height_cm,
        weight_kg: req.body.weight_kg,
        fit_preference: req.body.fit_preference || "regular",
        population: req.body.population || "universal",
        size_chart: garment.sizeChart,
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res
        .status(response.status)
        .json({ detail: err.detail || "Recommendation failed" });
    }

    const result = await response.json();
    res.json({
      recommended_size: result.recommended_size,
      confidence: result.confidence,
      explanation: `Estimated chest: ${result.estimated_chest_cm}cm`,
      estimated_chest_cm: result.estimated_chest_cm,
      alternatives: result.alternatives,
    });
  } catch (err) {
    res.status(502).json({ detail: `Recommendation service error: ${err.message}` });
  }
});

router.post("/compare", async (req, res) => {
  try {
    const garments = require("../data/garments.json");
    const garment = garments.find((g) => g.id === req.body.garment_id);
    if (!garment) {
      return res.status(404).json({ detail: "Garment not found" });
    }

    // Call recommendation service for both populations
    const [universalRes, vnRes] = await Promise.all([
      fetch(`${RECOMMENDATION_URL}/recommend-size`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...req.body,
          population: "universal",
          size_chart: garment.sizeChart,
        }),
      }),
      fetch(`${RECOMMENDATION_URL}/recommend-size`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...req.body,
          population: "vietnamese",
          size_chart: garment.sizeChart,
        }),
      }),
    ]);

    const universal = await universalRes.json();
    const vn = await vnRes.json();

    res.json({
      input: {
        height_cm: req.body.height_cm,
        weight_kg: req.body.weight_kg,
        fit_preference: req.body.fit_preference || "regular",
      },
      baseline: universal,
      eavton: vn,
      comparison: {
        size_changed: universal.recommended_size !== vn.recommended_size,
        chest_difference_cm: +(
          universal.estimated_chest_cm - vn.estimated_chest_cm
        ).toFixed(1),
        explanation: `Universal estimates ${universal.estimated_chest_cm}cm chest vs Vietnamese ${vn.estimated_chest_cm}cm`,
      },
    });
  } catch (err) {
    res.status(502).json({ detail: `Comparison failed: ${err.message}` });
  }
});

/**
 * POST /api/size-recommendation/research — use trained ML models.
 * Calls recommendation service /recommend-size/research endpoint.
 */
router.post("/research", async (req, res) => {
  try {
    const response = await fetch(`${RECOMMENDATION_URL}/recommend-size/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        height_cm: req.body.height_cm,
        weight_kg: req.body.weight_kg,
        age: req.body.age || 30,
        body_type: req.body.body_type || "unknown",
        category: req.body.category || "dress",
        population: req.body.population || "vietnamese",
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res
        .status(response.status)
        .json({ detail: err.detail || "Research model failed" });
    }

    res.json(await response.json());
  } catch (err) {
    // Fall back to rule-based if research service unavailable
    try {
      const garments = require("../data/garments.json");
      const garment = garments[0]; // default garment for size chart
      const fallback = await fetch(`${RECOMMENDATION_URL}/recommend-size`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          height_cm: req.body.height_cm,
          weight_kg: req.body.weight_kg,
          fit_preference: "regular",
          population: req.body.population || "vietnamese",
          size_chart: garment.sizeChart,
        }),
      });
      const result = await fallback.json();
      res.json({ ...result, fallback: true, method: "rule_based" });
    } catch (fallbackErr) {
      res.status(502).json({ detail: `Research model error: ${err.message}` });
    }
  }
});

/**
 * POST /api/size-recommendation/all-models — compare ALL 6 trained variants.
 * For research/demo purposes.
 */
router.post("/all-models", async (req, res) => {
  try {
    const response = await fetch(`${RECOMMENDATION_URL}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        height_cm: req.body.height_cm,
        weight_kg: req.body.weight_kg,
        age: req.body.age || 30,
        body_type: req.body.body_type || "unknown",
        category: req.body.category || "dress",
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res
        .status(response.status)
        .json({ detail: err.detail || "Model comparison failed" });
    }

    res.json(await response.json());
  } catch (err) {
    res.status(502).json({ detail: `Model comparison error: ${err.message}` });
  }
});

module.exports = router;
