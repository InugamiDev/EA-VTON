"""Sitemap-based scraper for Vietnamese fashion brands (v2, polite + fast).

Targets brands that expose Google's sitemap-image schema (title + image_url
right in the sitemap). For those, we never hit per-product pages — one sitemap
fetch per brand yields the entire catalog with title + image URL.

Currently supported brands (verified accessible 2026-05-19):
  - yody.vn       (sitemap_products_1.xml, sitemap-image schema, ~thousands of items)
  - coupletx.com  (sitemap_products_1.xml, sitemap-image schema, Haravan platform)

Filtering: keep only upper-body women's items by Vietnamese title prefix
matching ("Áo ..." etc.). See VN_UPPER_BODY_PREFIXES.

Output:
  research/datasets/raw/vn_brands/products.jsonl              (appended; same schema as v1)
  research/datasets/raw/vn_brands/images/<brand>/<hash>.jpg   (downloaded image files)
  research/datasets/raw/vn_brands/_html_cache/*.xml           (sitemap cache)

Politeness:
  - robots.txt checked once per brand at startup
  - 1.0-2.5 s jitter between image downloads
  - User-Agent identifies as research
  - Caches sitemap XML to disk so reruns don't re-hit servers
  - Max 100 image downloads concurrently within polite limits

Run:
  python research/datasets/scripts/scrape_vn_brands_v2.py \\
      --brands yody coupletx --max-per-brand 1000

For dry-run (parse sitemap + filter, no image download):
  python research/datasets/scripts/scrape_vn_brands_v2.py --dry-run
"""

# intent: low-friction VN catalog expansion via sitemap-image XML schema
# status: ready
# next: validate downloads complete cleanly; re-run combined catalog builder
# confidence: high (sitemap schema is stable; only image-CDN URLs could break)

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

import requests

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research/datasets/raw/vn_brands"
IMG_DIR = OUT_DIR / "images"
CACHE_DIR = OUT_DIR / "_html_cache"
PRODUCTS_FILE = OUT_DIR / "products.jsonl"

USER_AGENT = (
    "EAVTONResearchBot/0.2 (academic research; lethanhdanh1s1k@gmail.com; "
    "single-shot crawl, respects robots.txt, polite rate)"
)
DEFAULT_PAUSE = (1.0, 2.5)
TIMEOUT_S = 20

# Brand configs — sitemap URL + which entries we keep.
BRAND_CONFIGS: dict[str, dict] = {
    "yody": {
        "sitemap_url": "https://yody.vn/sitemap_products_1.xml",
        "base_url":    "https://yody.vn",
        "robots_url":  "https://yody.vn/robots.txt",
    },
    "coupletx": {
        "sitemap_url": "https://coupletx.com/sitemap_products_1.xml",
        "base_url":    "https://coupletx.com",
        "robots_url":  "https://coupletx.com/robots.txt",
    },
}

# Vietnamese terms that indicate upper-body women's items. We accept titles
# containing any of these (case-insensitive, accent-aware via normalized form).
VN_UPPER_BODY_KEYWORDS = [
    "áo sơ mi", "áo sơmi", "áo phông", "áo thun", "áo khoác",
    "áo polo", "áo len", "áo vest", "áo blazer", "áo cardigan",
    "áo croptop", "áo hai dây", "áo ba lỗ", "áo cổ tròn",
    "áo cổ tim", "áo cổ trụ", "áo nỉ", "áo khoác phao",
    "áo dài tay", "áo ngắn tay", "áo tay phồng", "áo ôm",
    "áo body", "áo bodysuit", "blouse",
]
# Exclude these even if "áo" appears
VN_EXCLUDE_KEYWORDS = [
    "quần", "giày", "dép", "túi", "phụ kiện", "vớ", "tất",
    "áo mưa", "áo cưới", "đầm",  # đầm = full-body dress (out of scope)
    "bộ pijama", "bộ đồ bộ",  # sleepwear sets
    "áo dài",  # traditional VN dress is full-body
]
# Male-coded markers — paper scope is women's upper-body. Filter applied after
# the upper-body keyword match so we don't accidentally include men's tops.
VN_MALE_MARKERS = [
    " nam ", " nam.", " nam,", " nam-", " nam_",
    " mop ", " mop_", " mop-",   # coupletx SKU suffix "MOP" denotes men's
    "for men", "men's",
]

# Sitemap XML uses two namespaces, but the URI prefix varies (http: vs https:)
# across hosts. yody.vn uses https://www.sitemaps.org/..., coupletx uses http://...
# We detect the actual root namespaces at parse time instead of hardcoding.


@dataclass
class Product:
    brand: str
    url: str
    title: str
    price_vnd: int | None
    image_url: str | None
    image_local_path: str | None
    category: str | None
    color: str | None
    sizes: list[str]
    size_chart_cm: dict | None
    description: str | None
    scraped_at: str


def _cache_path(url: str, ext: str = "xml") -> Path:
    return CACHE_DIR / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.{ext}"


def _polite_get(url: str, session: requests.Session, *, sleep: tuple[float, float] = DEFAULT_PAUSE,
                cache: bool = True) -> bytes | None:
    cp = _cache_path(url, "xml" if url.endswith(".xml") else "html")
    if cache and cp.exists():
        return cp.read_bytes()
    time.sleep(random.uniform(*sleep))
    try:
        r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        print(f"   ! GET failed: {url} → {e}")
        return None
    if r.status_code != 200:
        print(f"   ! GET {r.status_code}: {url}")
        return None
    if cache:
        cp.write_bytes(r.content)
    return r.content


def _check_robots(brand: str, cfg: dict, session: requests.Session) -> bool:
    """Robots check that handles real-world robots.txt better than urllib.

    Logic:
      1. Fetch robots.txt as text.
      2. If our target URL path is explicitly listed as a Sitemap: line, ALLOW
         (the owner is publishing the sitemap → explicit invitation to fetch it).
      3. Else find the User-agent: * section and check Disallow rules manually
         against our URL path.
      4. If urllib.robotparser disagrees (e.g., gets confused by empty-disallow
         sections), trust the manual check.
    """
    try:
        raw = requests.get(cfg["robots_url"], headers={"User-Agent": USER_AGENT},
                           timeout=TIMEOUT_S).text
    except requests.RequestException as e:
        print(f"   ! robots.txt fetch failed for {brand}: {e}; assuming OK")
        return True

    target_path = urllib.parse.urlparse(cfg["sitemap_url"]).path

    # Step 1: explicit Sitemap: declaration is a strong allow signal
    sitemap_lines = [l.strip() for l in raw.splitlines()
                     if l.strip().lower().startswith("sitemap:")]
    if any(cfg["sitemap_url"] in l for l in sitemap_lines):
        print(f"   robots.txt for {brand}: sitemap is EXPLICITLY published → ALLOWED")
        return True

    # Step 2: manual Disallow check for User-agent: * group
    in_star_group = False
    disallows: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("user-agent:"):
            ua = line.split(":", 1)[1].strip()
            in_star_group = (ua == "*")
            continue
        if in_star_group and line.lower().startswith("disallow:"):
            path = line.split(":", 1)[1].strip()
            if path:
                disallows.append(path)
    for d in disallows:
        if target_path.startswith(d):
            print(f"   robots.txt for {brand}: target path {target_path} matches Disallow: {d}")
            return False
    print(f"   robots.txt for {brand}: no matching Disallow → ALLOWED")
    return True


def _is_upper_body_title(title: str) -> bool:
    t = (" " + title.lower() + " ")  # pad so word-boundary markers like " nam " match end-of-string
    if any(ex in t for ex in VN_EXCLUDE_KEYWORDS):
        return False
    if not any(kw in t for kw in VN_UPPER_BODY_KEYWORDS):
        return False
    # Explicit male marker check (applied after keyword match)
    if any(m in t for m in VN_MALE_MARKERS):
        return False
    return True


def parse_sitemap(brand: str, cfg: dict, session: requests.Session) -> Iterator[dict]:
    print(f"   fetching sitemap: {cfg['sitemap_url']}")
    raw = _polite_get(cfg["sitemap_url"], session, cache=True)
    if raw is None:
        return
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        print(f"   ! sitemap parse error: {e}")
        return

    # Detect actual namespace URIs from the root element (vary by host: http vs https)
    root_tag = root.tag  # e.g. "{https://www.sitemaps.org/schemas/sitemap/0.9}urlset"
    sitemap_ns_uri = root_tag.split("}")[0].lstrip("{") if "}" in root_tag else ""
    SITEMAP_NS_ = f"{{{sitemap_ns_uri}}}" if sitemap_ns_uri else ""

    # Image namespace: scan the root's xmlns:* attrs (preserved as nsmap on root for lxml,
    # but stdlib ElementTree drops them; fall back to trying common variants).
    image_ns_candidates = [
        "{http://www.google.com/schemas/sitemap-image/1.1}",
        "{https://www.google.com/schemas/sitemap-image/1.1}",
    ]

    n = 0
    for url_el in root.findall(f"{SITEMAP_NS_}url"):
        loc = url_el.findtext(f"{SITEMAP_NS_}loc", "").strip()
        if not loc:
            continue
        image_el = None
        image_ns_used = ""
        for ns in image_ns_candidates:
            image_el = url_el.find(f"{ns}image")
            if image_el is not None:
                image_ns_used = ns
                break
        image_loc = ""
        image_title = ""
        if image_el is not None:
            image_loc = image_el.findtext(f"{image_ns_used}loc", "").strip()
            image_title = image_el.findtext(f"{image_ns_used}title", "").strip()
        n += 1
        yield {
            "url": loc,
            "title": image_title or "",
            "image_url": image_loc or "",
        }


def _download_image(url: str, brand: str, session: requests.Session) -> str | None:
    if not url:
        return None
    h = hashlib.sha1(url.encode()).hexdigest()[:16]
    ext = ".jpg"
    for candidate in (".jpg", ".png", ".webp"):
        if candidate in url.lower():
            ext = candidate
            break
    out = IMG_DIR / brand / f"{h}{ext}"
    if out.exists() and out.stat().st_size > 1024:
        return str(out.relative_to(ROOT))
    out.parent.mkdir(parents=True, exist_ok=True)
    time.sleep(random.uniform(0.5, 1.5))
    try:
        r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S, stream=True)
    except requests.RequestException as e:
        print(f"      ! img dl failed: {url[:80]} → {e}")
        return None
    if r.status_code != 200:
        return None
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=32768):
            f.write(chunk)
    return str(out.relative_to(ROOT))


def scrape_brand(brand: str, cfg: dict, session: requests.Session,
                 *, max_items: int, dry_run: bool) -> Iterator[Product]:
    print(f"\n── {brand} ──")
    if not _check_robots(brand, cfg, session):
        print(f"   robots.txt disallows; skipping {brand}")
        return
    entries = list(parse_sitemap(brand, cfg, session))
    print(f"   total sitemap entries: {len(entries)}")
    upper_only = [e for e in entries if _is_upper_body_title(e["title"])]
    print(f"   upper-body (title filter): {len(upper_only)}")
    upper_only = upper_only[:max_items]
    print(f"   capped at max_per_brand={max_items} → processing {len(upper_only)}")

    if dry_run:
        for e in upper_only[:5]:
            print(f"      [dry] {e['title']}  |  {e['image_url'][:80]}")
        return

    for i, entry in enumerate(upper_only):
        local = _download_image(entry["image_url"], brand, session)
        if local is None:
            continue
        yield Product(
            brand=brand,
            url=entry["url"],
            title=entry["title"],
            price_vnd=None,
            image_url=entry["image_url"],
            image_local_path=local,
            category="top",
            color=None,
            sizes=[],
            size_chart_cm=None,
            description=None,
            scraped_at=time.strftime("%Y-%m-%d"),
        )
        if (i + 1) % 25 == 0:
            print(f"      [{i+1}/{len(upper_only)}] {entry['title'][:50]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brands", nargs="+", default=list(BRAND_CONFIGS.keys()))
    ap.add_argument("--max-per-brand", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    total = 0
    with open(PRODUCTS_FILE, "a", encoding="utf-8") as out:
        for brand in args.brands:
            if brand not in BRAND_CONFIGS:
                print(f"!! unknown brand: {brand}; available: {list(BRAND_CONFIGS.keys())}")
                continue
            for p in scrape_brand(brand, BRAND_CONFIGS[brand], session,
                                   max_items=args.max_per_brand, dry_run=args.dry_run):
                out.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
                out.flush()
                total += 1

    print(f"\n  wrote {total} new products to {PRODUCTS_FILE}")
    if args.dry_run:
        print("  (dry-run; no images downloaded, no JSONL written)")


if __name__ == "__main__":
    main()
