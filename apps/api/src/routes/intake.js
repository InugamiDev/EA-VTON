/**
 * Participant intake routes — consented first-party dataset collection.
 */

// intent: collect consented participant images into restricted storage, not public uploads
// status: done
// next: run face redaction before records enter any training manifest
// blockers: production deployment still needs real auth, encrypted storage, and consent CRM integration
// confidence: medium

const express = require("express");
const fs = require("fs");
const path = require("path");
const multer = require("multer");
const { v4: uuidv4 } = require("uuid");

const router = express.Router();

const REPO_ROOT = path.resolve(__dirname, "../../../..");
const INTAKE_ROOT = path.resolve(
  process.env.INTAKE_STORAGE_DIR ||
    path.join(REPO_ROOT, "research/datasets/restricted/female_clothing_style_v1/intake")
);
const RAW_MANIFEST = path.join(INTAKE_ROOT, "raw_intake_manifest.jsonl");
const MAX_IMAGES = 20;
const MIN_IMAGES = 10;
const MAX_FILE_SIZE_BYTES = 8 * 1024 * 1024;
const CONSENT_VERSION = "female_clothing_style_v1_intake_2026-05-08";
const AGE_BANDS = new Set(["18_24", "25_34", "35_44", "45_54", "55_64", "65_plus", "prefer_not_to_say"]);

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_FILE_SIZE_BYTES, files: MAX_IMAGES },
  fileFilter: (_req, file, cb) => {
    if (/^image\/(jpeg|png|webp)$/.test(file.mimetype)) {
      cb(null, true);
      return;
    }
    cb(new Error("Only JPEG, PNG, and WebP images are accepted"));
  },
});

function ensureIntakeRoot() {
  fs.mkdirSync(INTAKE_ROOT, { recursive: true });
}

function boolField(value) {
  return value === true || value === "true" || value === "on" || value === "1";
}

function safeNote(value) {
  return String(value || "")
    .replace(/\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi, "[email_removed]")
    .replace(/https?:\/\/\S+|www\.\S+/gi, "[url_removed]")
    .replace(/(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)/g, "[phone_removed]")
    .split(/\s+/)
    .filter(Boolean)
    .join(" ")
    .slice(0, 500);
}

function extensionFor(mimetype) {
  if (mimetype === "image/png") return ".png";
  if (mimetype === "image/webp") return ".webp";
  return ".jpg";
}

function relativeFromRepo(filePath) {
  return path.relative(REPO_ROOT, filePath).split(path.sep).join("/");
}

function validateRequest(req, files) {
  if (!boolField(req.body.consent_training)) {
    return "Consent for style recommendation training is required";
  }
  if (!boolField(req.body.consent_research)) {
    return "Consent for internal research use is required";
  }
  if (req.body.gender_label !== "female") {
    return "This intake flow is currently scoped to self-identified female participants";
  }
  if (!AGE_BANDS.has(req.body.age_band)) {
    return "A valid age band is required";
  }
  if (!files || files.length < MIN_IMAGES || files.length > MAX_IMAGES) {
    return `Upload ${MIN_IMAGES}-${MAX_IMAGES} images for one participant`;
  }
  return null;
}

async function writeSubmission(req, files) {
  ensureIntakeRoot();
  const submissionId = uuidv4();
  const personId = `p_firstparty_${submissionId.replace(/-/g, "").slice(0, 12)}`;
  const submittedAt = new Date().toISOString();
  const personDir = path.join(INTAKE_ROOT, personId, "raw");
  await fs.promises.mkdir(personDir, { recursive: true });

  const lines = [];
  for (const [index, file] of files.entries()) {
    const slot = index + 1;
    const imageId = `${personId}_${String(slot).padStart(2, "0")}`;
    const destination = path.join(personDir, `${imageId}${extensionFor(file.mimetype)}`);
    await fs.promises.writeFile(destination, file.buffer);
    lines.push(
      JSON.stringify({
        person_id: personId,
        image_id: imageId,
        split: "unassigned",
        gender_label: {
          value: "female",
          source: "self_identified",
        },
        age_band: req.body.age_band,
        source_image_uri: relativeFromRepo(destination),
        face_boxes: [],
        face_redaction: {
          status: "pending",
          method: "none",
        },
        rights: {
          basis: "first_party_consent",
          status: "granted",
          scope: "style_rec_training_internal_research",
          consent_version: CONSENT_VERSION,
        },
        collection: {
          submission_id: submissionId,
          submitted_at: submittedAt,
          slot,
          original_mimetype: file.mimetype,
          original_size_bytes: file.size,
          participant_note: safeNote(req.body.participant_note),
          redaction_status: "pending",
          label_status: "pending",
        },
      })
    );
  }
  await fs.promises.appendFile(RAW_MANIFEST, `${lines.join("\n")}\n`, "utf-8");
  return { submissionId, personId, imageCount: files.length };
}

router.get("/status", (_req, res) => {
  ensureIntakeRoot();
  let records = 0;
  const people = new Set();
  if (fs.existsSync(RAW_MANIFEST)) {
    const lines = fs.readFileSync(RAW_MANIFEST, "utf-8").split("\n").filter(Boolean);
    records = lines.length;
    for (const line of lines) {
      try {
        const record = JSON.parse(line);
        if (record.person_id) people.add(record.person_id);
      } catch {
        // Ignore malformed local rows in status so intake remains inspectable.
      }
    }
  }
  res.json({
    storage: "restricted_local",
    records,
    people: people.size,
    min_images_per_submission: MIN_IMAGES,
    max_images_per_submission: MAX_IMAGES,
    raw_manifest: relativeFromRepo(RAW_MANIFEST),
  });
});

router.post("/participant", (req, res) => {
  upload.array("images", MAX_IMAGES)(req, res, async (uploadErr) => {
    if (uploadErr) {
      return res.status(400).json({ detail: uploadErr.message });
    }
    const validationError = validateRequest(req, req.files);
    if (validationError) {
      return res.status(400).json({ detail: validationError });
    }

    try {
      const submission = await writeSubmission(req, req.files);
      return res.status(201).json({
        submission_id: submission.submissionId,
        person_id: submission.personId,
        image_count: submission.imageCount,
        next_step: "Run face redaction before annotation or training.",
      });
    } catch (err) {
      return res.status(500).json({ detail: err.message || "Failed to save intake submission" });
    }
  });
});

module.exports = router;
