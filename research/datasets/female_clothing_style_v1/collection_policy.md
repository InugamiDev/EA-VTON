# Collection Policy

Face redaction does not make arbitrary image scraping acceptable. This dataset must not collect from private social media, dating apps, school/work pages, forums, or image search results unless the subject/source grants permission for ML training.

## Allowed Sources

- Direct participant uploads with explicit consent for style recommendation training.
- Public or research datasets whose terms allow the intended ML use.
- Product-only garment images for item metadata bootstrapping, after license review.

## Blocked Sources

- Random social-media profiles or posts.
- Screenshots of people from videos without consent.
- Dating apps, school pages, work pages, or private communities.
- Any source whose terms prohibit scraping, redistribution, or model training.

## Why Redaction Is Not Enough

- The target requires 10 images per same person, which creates person-level grouping even if faces are hidden.
- Faces are not the only identifiers: tattoos, locations, captions, usernames, EXIF, and rare outfits can identify people.
- Redacted images still need source rights and training permission.

## Practical Collection Plan

1. Use participant intake as the primary source for the 10,000 x 10 target.
2. Use public fashion datasets only for garment parsing and label pretraining.
3. Keep raw files restricted and encrypted.
4. Redact faces and strip metadata before annotation.
5. Validate manifests before any training job.
