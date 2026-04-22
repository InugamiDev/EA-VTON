# EA-VTON Datasets

Downloaded and processed: 2026-03-19

## Processed (Ready to Use)

| File | Source | Rows | Size | Format |
|------|--------|------|------|--------|
| `processed/rtr_body_fit.parquet` | Rent The Runway | 192,311 | 2.7 MB | Parquet |
| `processed/rtr_full.parquet` | Rent The Runway | 192,544 | 40.7 MB | Parquet |
| `processed/bodym_measurements.parquet` | Amazon BodyM | 2,507 | 203 KB | Parquet |

## Dataset Details

### Rent The Runway (RTR)
- **Source**: McAuley Lab, UCSD
- **License**: CC BY 4.0
- **Columns (compact)**: `user_id`, `item_id`, `category`, `size`, `fit`, `height_cm`, `weight_kg`, `bmi`, `age`, `body_type_norm`, `bust_band`, `bust_cup`
- **Stats**: 105K users, 5,850 items, mean height 165.9cm, mean weight 62.3kg
- **Fit distribution**: 74% fit, 13% small, 13% large
- **Body types**: hourglass (31%), athletic (25%), pear (12%), petite (12%), full_bust (8%), rectangle (8%), apple (3%)
- **Note**: US-centric, no ethnicity data. Self-reported measurements.

### Amazon BodyM
- **Source**: Amazon Science Open Data (AWS S3)
- **License**: CC BY-NC 4.0
- **Columns**: `subject_id`, `gender`, `height_cm`, `weight_kg`, `split`, 14 body measurements (cm), `bmi`, `whr`, `shr`
- **14 measurements**: ankle, arm length, bicep, calf, chest, forearm, height, hip, leg length, shoulder breadth, shoulder-to-crotch, thigh, waist, wrist
- **Stats**: 2,507 subjects in our processed parquet (1,421 male, 1,086 female). The AWS Open Data registry page lists 2,505 real subjects paired with 8,978 frontal/lateral silhouettes; the small delta is because our processing step kept every row present in the source CSV metadata without deduplication.
- **Note**: Precise 3D-scan measurements + binary silhouettes (not raw RGB photos). No ethnicity metadata. Predominantly Western population per the dataset authors.

## Raw Data

| File | Size | Description |
|------|------|-------------|
| `raw/rtr_raw.json` | 118 MB | Original JSON lines |
| `raw/rtr_raw.json.gz` | 29 MB | Compressed original |
| `raw/bodym/` | ~1.3 MB | CSV metadata (masks not downloaded) |

## Datasets Not Yet Downloaded (Require Registration)

| Dataset | Subjects | Why Useful | Access |
|---------|----------|-----------|--------|
| SizeKorea | 14,200 | 119 measurements, closest Asian reference | sizekorea.kr (registration) |
| SizeUSA | 11,000 | 140 measurements + ethnicity | Commercial (TC2) |
| CLOTH3D | 2M frames | SMPL bodies + garment simulation | chalearnlap.cvc.uab.cat (registration) |
| DeepFashion2 | 491K images | Clothing segmentation/landmarks | GitHub form |

## Processing Scripts

```bash
source .venv/bin/activate
python datasets/scripts/process_rtr.py
python datasets/scripts/process_bodym.py
```
