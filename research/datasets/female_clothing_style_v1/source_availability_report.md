# Female Clothing Source Availability Report

## Target

- Requested person-level target: 10,000 people.
- Requested images per person: 10-20.
- Requested total image range: 100,000-200,000 redacted person images.

## Current Approved Source Audit

| Source | License/status | WOMEN-labeled images available locally/metadata | People/pids | People with >=10 images | Import status |
|---|---:|---:|---:|---:|---|
| `SaffalPoosh/deepFashion-with-masks` | Apache-2.0 HF metadata | 35,646 downloaded | 6,694 | 589 | balanced 10-20 subset imported |
| `huanngzh/DeepFashion-MultiModal-Parts2Whole` | Apache-2.0 HF metadata | 31,044 unique WOMEN target paths in metadata | 5,492 loose source people; strict same-outfit groups too sparse | 498 loose source people; 0 strict same-outfit groups | metadata only; full media needs ~53GB |
| `amazon_review_2018_fashion` | research data terms | n/a | n/a | n/a | product/review sidecar only; direct small-file download timed out locally |

## Candidate ReID Source Audit

These sources are large person-level ReID datasets. They are useful to track because several clear the raw identity-count threshold, but none is approved for the requested women-focused clothing/style training manifest yet.

| Source | Reported scale | Raw 10k identity threshold | Current blocker | Decision |
|---|---:|---:|---|---|
| `LUPerson` | 4M images, >200K identities | yes | research-only; commercial use forbidden; YouTube reconstruction; no verified female labels, consent basis, or fashion taxonomy | reject for target; possible generic ReID research only |
| `LUPerson-NL` | 10M images, >430K noisy identities | yes | noisy tracklet labels from street-view videos; no self-identified gender, fashion labels, or consent basis | reject for target; possible generic ReID research only |
| `LaST` | 10,862 identities, >228k images | yes | academic research only; movie-derived media; no verified female-label filter; storage/download audit not run | candidate blocked pending terms/gender/storage review |
| `ENTIRe-ID` | 13,540 identities, 4.45M images, 37 cameras | yes | public internet-camera source; no verified female-label filter or garment taxonomy; legal/privacy and storage review required | candidate blocked pending terms/privacy/gender/storage review |
| `awesome_reid_dataset_index` | index only | n/a | not a dataset; original source licenses vary | source discovery only |

### ReID Suitability Notes

- `LaST` is the closest paper fit for clothing-style pretraining because it reports clothing labels for the training set, but its academic-only license and movie-derived media block production use without a separate rights review.
- `ENTIRe-ID` is the closest raw-scale fit for person count and repeated images/person, and its paper discusses facial blurring. It is still not a women-focused fashion dataset: it lacks verified female labels, garment taxonomy labels, and consent/production-rights evidence.
- `LUPerson` and `LUPerson-NL` exceed the requested scale, but they are ReID pretraining corpora, not style/preference datasets. They should not be imported into this target manifest.
- ReID identity counts do not automatically satisfy this project. The target also requires approved rights, face redaction, female-label provenance, garment/outfit labels, and an auditable 10-20 images/person manifest.

## Imported Person-Image Subset

`manifest_deepfashion_with_masks_balanced_10_20.jsonl` is the largest strict 10-20 images/person subset currently available from downloaded approved media.

- Records: 8,194.
- People: 589.
- Min images/person: 10.
- Max images/person: 20.
- Gender label source: dataset label, not first-party self-identification.
- Rights basis: dataset license, research-only.
- Redaction: conservative black top-band heuristic; manual audit required before production training.

`manifest_deepfashion_with_masks_all_capped20.jsonl` preserves broader local coverage but does not enforce the 10-image minimum.

- Records: 34,142.
- People: 6,694.
- Max images/person: 20.
- Use case: general garment/style pretraining, not the requested balanced person-level training set.

## Gap To Target

The current approved/downloaded sources cannot produce 10,000 people with 10-20 images each.

- Shortfall by people: 9,411 people.
- Shortfall by minimum images: 91,806 images.
- The available DeepFashion-derived sources also overlap heavily by source pids, so simply combining them will not create 10,000 distinct people.
- Large ReID candidates may satisfy raw identity counts, but they are not approved women-focused clothing datasets and cannot be counted toward the target until terms, privacy, gender-label, redaction, fashion-label, storage, and manifest audits pass.
- Latest local disk check showed about 136GiB free. That is enough to consider the ~53GB Parts2Whole media import after an explicit approval/preflight, but still too small for careless multi-million-image ReID imports.

## Requirements To Reach Target Honestly

One of these must happen before the target can be met:

1. Run first-party participant intake for 10,000 consenting adult participants with 10-20 images each.
2. Provide access to another approved/licensed dataset that has at least 10,000 distinct female-labeled people with 10+ images each.
3. Provide enough storage for larger approved datasets, then audit whether they actually satisfy the person-count distribution before import.

Arbitrary web/social scraping is still excluded by the collection policy.

## Research Sources Checked

| Source | URL | Key finding |
|---|---|---|
| LUPerson GitHub | https://github.com/DengpanFu/LUPerson | Reports 4M images over 200K identities and states commercial usage is forbidden. |
| LUPerson paper | https://arxiv.org/abs/2012.03753 | Describes LUPerson as an unlabeled ReID pretraining dataset with 4M images over 200K identities. |
| LUPerson-NL GitHub | https://github.com/DengpanFu/LUPerson-NL | Reports 10M images over 430K noisy identities extracted from 21K street-view videos. |
| LUPerson-NL paper | https://arxiv.org/abs/2203.16533 | Describes noisy tracklet labels derived from LUPerson raw videos. |
| LaST GitHub | https://github.com/shuxjweb/last | Reports train/val/test splits totaling 10,862 identities and 228,156 images, with academic-research-only release terms. |
| LaST paper | https://arxiv.org/abs/2105.15076 | Describes 10,862 identities and more than 228k images with large spatial/temporal range. |
| ENTIRe-ID project | https://serdaryildiz.github.io/ENTIRe-ID/ | Reports 4.45M images from 37 cameras and links public data. |
| ENTIRe-ID paper | https://arxiv.org/html/2405.20465v1 | Reports 4.45M images, 13,540 IDs, 37 cameras, and discusses facial blurring/privacy concerns. |
| ReID dataset index | https://github.com/NEU-Gou/awesome-reid-dataset | Confirms ReID dataset counts and reminds users to refer to original licenses. |
