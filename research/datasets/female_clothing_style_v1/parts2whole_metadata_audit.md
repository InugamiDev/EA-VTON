# Parts2Whole Metadata Audit

This audit uses JSONL metadata only and does not download or inspect media files.

Minimum images/person threshold: `10`

| Split | Rows | Unique targets | Source people | Source people >= threshold | WOMEN source people | WOMEN source people >= threshold | Strict outfit groups | WOMEN strict groups >= threshold | Parse failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train` | 40,098 | 33,847 | 5,825 | 582 | 5,308 | 482 | 9,050 | 0 | 0 |
| `test` | 1,380 | 1,159 | 205 | 21 | 184 | 16 | 307 | 0 | 0 |

## Decision Notes

- WOMEN-labeled loose source people across splits: `5,492`.
- WOMEN-labeled loose source people with at least `10` images: `498`.
- WOMEN-labeled strict same-outfit groups with at least `10` images: `0`.
- These are dataset labels, not self-identified gender labels.
- Loose source people may combine multiple outfits and depend on original DeepFashion IDs; strict groups are safer but too sparse.
- This source still cannot satisfy a 10,000-person WOMEN 10+ images/person target by itself.
- Media import still requires rights review, face-redaction audit, and disk preflight.

## Top Categories

### `train`
- `WOMEN/Tees_Tanks`: 9,747
- `WOMEN/Dresses`: 8,633
- `WOMEN/Blouses_Shirts`: 6,069
- `WOMEN/Sweaters`: 2,654
- `WOMEN/Rompers_Jumpsuits`: 2,405
- `MEN/Tees_Tanks`: 1,906
- `WOMEN/Jackets_Coats`: 1,898
- `WOMEN/Cardigans`: 1,548
- `WOMEN/Shorts`: 943
- `WOMEN/Graphic_Tees`: 917
- `WOMEN/Sweatshirts_Hoodies`: 648
- `WOMEN/Skirts`: 621

### `test`
- `WOMEN/Dresses`: 300
- `WOMEN/Tees_Tanks`: 277
- `WOMEN/Blouses_Shirts`: 216
- `WOMEN/Sweaters`: 98
- `MEN/Tees_Tanks`: 91
- `WOMEN/Rompers_Jumpsuits`: 76
- `WOMEN/Cardigans`: 61
- `WOMEN/Jackets_Coats`: 56
- `WOMEN/Graphic_Tees`: 33
- `WOMEN/Shorts`: 30
- `MEN/Sweatshirts_Hoodies`: 28
- `MEN/Jackets_Vests`: 19
