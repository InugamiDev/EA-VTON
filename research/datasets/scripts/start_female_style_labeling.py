"""Create a human labeling batch for the female style recommendation dataset.

The generated batch contains only redacted image references and prefilled
heuristic labels. Annotators verify or correct the labels in a local HTML page
and export JSONL/CSV review results.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("research/datasets/processed/female_style_rec_v1/images_train.jsonl")
DEFAULT_TAXONOMY = Path("research/datasets/female_clothing_style_v1/label_taxonomy.json")
DEFAULT_OUTPUT_DIR = Path("research/datasets/annotation/female_style_rec_v1/batch_0001")

TASK_FIELDS = [
    "task_id",
    "status",
    "person_id",
    "image_id",
    "split",
    "image_path",
    "suggested_family",
    "suggested_category",
    "suggested_occasion",
    "suggested_formality",
    "suggested_aesthetic_tags",
    "suggested_color_mood",
    "suggested_layering",
    "corrected_family",
    "corrected_category",
    "corrected_occasion",
    "corrected_formality",
    "corrected_aesthetic_tags",
    "corrected_color_mood",
    "corrected_layering",
    "quality_status",
    "notes",
]


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as records:
        for line_number, line in enumerate(records, 1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)


def primary_garment(record: dict[str, Any]) -> dict[str, Any]:
    garments = record.get("garments") or []
    for garment in garments:
        if garment.get("visibility") == "primary":
            return garment
    return garments[0] if garments else {}


def image_path_for(record: dict[str, Any], output_dir: Path) -> str:
    dataset_image = Path("research/datasets/female_clothing_style_v1") / record["redacted_image_uri"]
    return Path("../../..", dataset_image.relative_to("research/datasets")).as_posix()


def task_from_record(record: dict[str, Any], index: int, output_dir: Path) -> dict[str, Any]:
    garment = primary_garment(record)
    outfit = record.get("outfit") or {}
    return {
        "task_id": f"style_label_{index:05d}",
        "status": "pending",
        "person_id": record["person_id"],
        "image_id": record["image_id"],
        "split": record["split"],
        "image_path": image_path_for(record, output_dir),
        "suggested_family": garment.get("family", "unknown"),
        "suggested_category": garment.get("category", "unknown"),
        "suggested_occasion": outfit.get("occasion", "unknown"),
        "suggested_formality": outfit.get("formality", "unknown"),
        "suggested_aesthetic_tags": "|".join(outfit.get("aesthetic_tags") or ["unknown"]),
        "suggested_color_mood": outfit.get("color_mood", "unknown"),
        "suggested_layering": outfit.get("layering", "unknown"),
        "corrected_family": "",
        "corrected_category": "",
        "corrected_occasion": "",
        "corrected_formality": "",
        "corrected_aesthetic_tags": "",
        "corrected_color_mood": "",
        "corrected_layering": "",
        "quality_status": "usable",
        "notes": "",
    }


def select_diverse(records: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        category = primary_garment(record).get("category", "unknown")
        buckets[category].append(record)

    selected: list[dict[str, Any]] = []
    categories = sorted(buckets)
    while len(selected) < batch_size and categories:
        next_categories = []
        for category in categories:
            bucket = buckets[category]
            if bucket and len(selected) < batch_size:
                selected.append(bucket.pop(0))
            if bucket:
                next_categories.append(category)
        categories = next_categories
    return selected


def write_tasks_jsonl(path: Path, tasks: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for task in tasks:
            output.write(json.dumps(task, sort_keys=True) + "\n")


def write_tasks_csv(path: Path, tasks: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=TASK_FIELDS)
        writer.writeheader()
        for task in tasks:
            writer.writerow(task)


def html_options(values: list[str]) -> str:
    return "\n".join(f"<option value=\"{html.escape(value)}\"></option>" for value in values)


def write_labeling_html(path: Path, tasks: list[dict[str, Any]], taxonomy: dict[str, Any]) -> None:
    outfit = taxonomy["outfit"]
    families = taxonomy["garment_families"]
    categories = sorted(
        {
            category
            for values in taxonomy["categories_by_family"].values()
            for category in values
        }
    )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Female Style Rec Labeling Batch</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #151515; background: #f6f7f8; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 16px 20px; background: #101418; color: white; }}
    main {{ display: grid; grid-template-columns: minmax(360px, 48vw) 1fr; min-height: calc(100vh - 68px); }}
    .image-pane {{ display: grid; place-items: center; padding: 20px; background: #202428; }}
    .image-pane img {{ max-width: 100%; max-height: calc(100vh - 120px); object-fit: contain; background: #111; }}
    .form-pane {{ padding: 20px; overflow: auto; }}
    .row {{ display: grid; grid-template-columns: 180px 1fr; gap: 10px; align-items: center; margin-bottom: 10px; }}
    label {{ font-size: 13px; font-weight: 700; color: #333; }}
    input, select, textarea {{ width: 100%; box-sizing: border-box; padding: 9px 10px; border: 1px solid #cfd4da; border-radius: 6px; background: white; font: inherit; }}
    textarea {{ min-height: 70px; resize: vertical; }}
    .buttons {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }}
    button {{ border: 0; border-radius: 6px; padding: 10px 12px; font-weight: 700; cursor: pointer; }}
    .primary {{ background: #0f766e; color: white; }}
    .secondary {{ background: #dce3ea; color: #111; }}
    .danger {{ background: #b42318; color: white; }}
    .meta {{ font-size: 13px; color: #56616d; margin-bottom: 16px; }}
    .pill {{ display: inline-block; margin-right: 6px; padding: 3px 7px; border-radius: 999px; background: #e7ecef; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <div><strong>Style Rec Labeling</strong> <span id="position"></span></div>
    <div>
      <button class="secondary" onclick="prevTask()">Prev</button>
      <button class="secondary" onclick="nextTask()">Next</button>
      <button class="primary" onclick="exportJsonl()">Export JSONL</button>
      <button class="primary" onclick="exportCsv()">Export CSV</button>
    </div>
  </header>
  <main>
    <section class="image-pane">
      <img id="image" alt="redacted clothing record" />
    </section>
    <section class="form-pane">
      <div class="meta" id="meta"></div>
      <div id="suggestions"></div>
      <div class="row"><label>Family</label><input id="corrected_family" list="families" /></div>
      <div class="row"><label>Category</label><input id="corrected_category" list="categories" /></div>
      <div class="row"><label>Occasion</label><input id="corrected_occasion" list="occasions" /></div>
      <div class="row"><label>Formality</label><input id="corrected_formality" list="formalities" /></div>
      <div class="row"><label>Aesthetic tags</label><input id="corrected_aesthetic_tags" list="aesthetic_tags" placeholder="pipe-separated, e.g. minimal|classic" /></div>
      <div class="row"><label>Color mood</label><input id="corrected_color_mood" list="color_moods" /></div>
      <div class="row"><label>Layering</label><input id="corrected_layering" list="layering_values" /></div>
      <div class="row"><label>Quality</label><select id="quality_status"><option>usable</option><option>needs_review</option><option>reject</option></select></div>
      <div class="row"><label>Notes</label><textarea id="notes"></textarea></div>
      <div class="buttons">
        <button class="primary" onclick="saveTask('labeled')">Save Labeled</button>
        <button class="secondary" onclick="copySuggestions()">Copy Suggestions</button>
        <button class="secondary" onclick="saveTask('needs_review')">Needs Review</button>
        <button class="danger" onclick="saveTask('reject')">Reject</button>
      </div>
    </section>
  </main>

  <datalist id="families">{html_options(families)}</datalist>
  <datalist id="categories">{html_options(categories)}</datalist>
  <datalist id="occasions">{html_options(outfit["occasion"])}</datalist>
  <datalist id="formalities">{html_options(outfit["formality"])}</datalist>
  <datalist id="aesthetic_tags">{html_options(outfit["aesthetic_tags"])}</datalist>
  <datalist id="color_moods">{html_options(outfit["color_mood"])}</datalist>
  <datalist id="layering_values">{html_options(outfit["layering"])}</datalist>

  <script>
    const tasks = {json.dumps(tasks, sort_keys=True)};
    const storageKey = "female_style_rec_labeling_batch_0001";
    let annotations = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    let index = Number(localStorage.getItem(storageKey + "_index") || 0);
    const fields = ["corrected_family", "corrected_category", "corrected_occasion", "corrected_formality", "corrected_aesthetic_tags", "corrected_color_mood", "corrected_layering", "quality_status", "notes"];

    function currentTask() {{ return tasks[index]; }}
    function currentAnnotation() {{ return annotations[currentTask().task_id] || currentTask(); }}
    function setValue(id, value) {{ document.getElementById(id).value = value || ""; }}
    function getValue(id) {{ return document.getElementById(id).value.trim(); }}
    function render() {{
      const task = currentTask();
      const ann = currentAnnotation();
      document.getElementById("position").textContent = `${{index + 1}} / ${{tasks.length}}`;
      document.getElementById("image").src = task.image_path;
      document.getElementById("meta").innerHTML = `<strong>${{task.task_id}}</strong> | ${{task.image_id}} | ${{task.person_id}} | ${{task.split}}`;
      document.getElementById("suggestions").innerHTML = [
        ["family", task.suggested_family],
        ["category", task.suggested_category],
        ["occasion", task.suggested_occasion],
        ["formality", task.suggested_formality],
        ["aesthetic", task.suggested_aesthetic_tags],
        ["color", task.suggested_color_mood],
        ["layering", task.suggested_layering],
      ].map(([k, v]) => `<span class="pill">${{k}}: ${{v}}</span>`).join("");
      fields.forEach(field => setValue(field, ann[field] || task[field] || ""));
      if (!getValue("quality_status")) setValue("quality_status", "usable");
    }}
    function collect(status) {{
      const task = currentTask();
      const result = {{ ...task, status }};
      fields.forEach(field => result[field] = getValue(field));
      if (status === "reject") result.quality_status = "reject";
      if (status === "needs_review") result.quality_status = "needs_review";
      return result;
    }}
    function persist(result) {{
      annotations[result.task_id] = result;
      localStorage.setItem(storageKey, JSON.stringify(annotations));
      localStorage.setItem(storageKey + "_index", String(index));
    }}
    function saveTask(status) {{
      persist(collect(status));
      nextTask();
    }}
    function copySuggestions() {{
      const task = currentTask();
      setValue("corrected_family", task.suggested_family);
      setValue("corrected_category", task.suggested_category);
      setValue("corrected_occasion", task.suggested_occasion);
      setValue("corrected_formality", task.suggested_formality);
      setValue("corrected_aesthetic_tags", task.suggested_aesthetic_tags);
      setValue("corrected_color_mood", task.suggested_color_mood);
      setValue("corrected_layering", task.suggested_layering);
    }}
    function nextTask() {{
      index = Math.min(tasks.length - 1, index + 1);
      localStorage.setItem(storageKey + "_index", String(index));
      render();
    }}
    function prevTask() {{
      index = Math.max(0, index - 1);
      localStorage.setItem(storageKey + "_index", String(index));
      render();
    }}
    function reviewedRows() {{
      return tasks.map(task => annotations[task.task_id] || task);
    }}
    function download(name, text, type) {{
      const blob = new Blob([text], {{ type }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      URL.revokeObjectURL(url);
    }}
    function exportJsonl() {{
      download("batch_0001_labels.jsonl", reviewedRows().map(row => JSON.stringify(row)).join("\\n") + "\\n", "application/jsonl");
    }}
    function exportCsv() {{
      const keys = Object.keys(reviewedRows()[0]);
      const escape = value => `"${{String(value ?? "").replaceAll('"', '""')}}"`;
      const lines = [keys.join(","), ...reviewedRows().map(row => keys.map(key => escape(row[key])).join(","))];
      download("batch_0001_labels.csv", lines.join("\\n") + "\\n", "text/csv");
    }}
    document.addEventListener("keydown", event => {{
      if (event.key === "ArrowRight") nextTask();
      if (event.key === "ArrowLeft") prevTask();
    }});
    render();
  </script>
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def write_guide(path: Path, batch_size: int, stats: dict[str, Any]) -> None:
    path.write_text(
        "# Labeling Batch Guide\n\n"
        f"Batch size: `{batch_size}` records.\n\n"
        "Open `index.html` in a browser from this directory. The page uses redacted local images and stores progress in browser localStorage. Use `Export JSONL` or `Export CSV` when done.\n\n"
        "Labeling rules:\n\n"
        "- Verify visible clothing only.\n"
        "- Keep labels non-identifying; do not add names, usernames, locations, or body-shape guesses.\n"
        "- Correct the primary garment family/category when the suggestion is wrong.\n"
        "- Use `needs_review` if the outfit is ambiguous or the image is hard to inspect.\n"
        "- Use `reject` if garments are not visible, the image is corrupted, or face redaction is visibly insufficient.\n"
        "- Aesthetic tags can be pipe-separated, for example `minimal|classic`.\n\n"
        "Batch stats:\n\n"
        f"- Categories: `{stats['categories']}`\n"
        f"- Splits: `{stats['splits']}`\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--strategy", choices=["diverse", "first"], default="diverse")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    records = [record for _, record in iter_jsonl(args.source)]
    if args.strategy == "diverse":
        selected = select_diverse(records, args.batch_size)
    else:
        selected = records[: args.batch_size]

    if not selected:
        raise SystemExit(f"no records selected from {args.source}")

    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        task_from_record(record, index + 1, args.output_dir)
        for index, record in enumerate(selected)
    ]
    category_counts = Counter(task["suggested_category"] for task in tasks)
    split_counts = Counter(task["split"] for task in tasks)
    stats = {
        "source": str(args.source),
        "output_dir": str(args.output_dir),
        "batch_size": len(tasks),
        "strategy": args.strategy,
        "categories": dict(category_counts.most_common()),
        "splits": dict(sorted(split_counts.items())),
    }

    write_tasks_jsonl(args.output_dir / "tasks.jsonl", tasks)
    write_tasks_csv(args.output_dir / "labels_working.csv", tasks)
    write_labeling_html(args.output_dir / "index.html", tasks, taxonomy)
    write_guide(args.output_dir / "LABELING_GUIDE.md", len(tasks), stats)
    (args.output_dir / "batch_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"created labeling batch records={len(tasks):,} -> {args.output_dir}")
    print(f"categories={dict(category_counts.most_common(8))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
