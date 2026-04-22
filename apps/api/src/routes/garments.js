/**
 * Garment catalog routes — serves garment data from JSON (no Python needed).
 */

const express = require("express");
const router = express.Router();
const garments = require("../data/garments.json");

router.get("/", (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const pageSize = parseInt(req.query.page_size) || 20;
  const start = (page - 1) * pageSize;
  const end = start + pageSize;

  res.json({
    garments: garments.slice(start, end),
    total: garments.length,
    page,
    pageSize,
  });
});

router.get("/:id", (req, res) => {
  const garment = garments.find((g) => g.id === req.params.id);
  if (!garment) {
    return res.status(404).json({ detail: "Garment not found" });
  }
  res.json(garment);
});

module.exports = router;
