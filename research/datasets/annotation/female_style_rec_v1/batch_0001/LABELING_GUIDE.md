# Labeling Batch Guide

Batch size: `100` records.

Open `index.html` in a browser from this directory. The page uses redacted local images and stores progress in browser localStorage. Use `Export JSONL` or `Export CSV` when done.

Labeling rules:

- Verify visible clothing only.
- Keep labels non-identifying; do not add names, usernames, locations, or body-shape guesses.
- Correct the primary garment family/category when the suggestion is wrong.
- Use `needs_review` if the outfit is ambiguous or the image is hard to inspect.
- Use `reject` if garments are not visible, the image is corrupted, or face redaction is visibly insufficient.
- Aesthetic tags can be pipe-separated, for example `minimal|classic`.

Batch stats:

- Categories: `{'blouse': 8, 'cardigan': 8, 'coat': 8, 'denim_jacket': 8, 'hoodie': 8, 'leather_jacket': 8, 'midi_dress': 8, 'romper': 8, 'shorts': 8, 'skirt': 7, 'sweater': 7, 'tank': 7, 'trousers': 7}`
- Splits: `{'train': 100}`
