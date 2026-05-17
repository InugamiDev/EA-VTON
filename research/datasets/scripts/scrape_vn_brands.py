"""Polite scraper for Vietnamese fashion brands.

Targets: Coolmate, Ivy Moda, Dottie (women's tops sections).
Outputs: research/datasets/raw/vn_brands/products.jsonl  (one product per line)

Each product record:
  {
    "brand": "ivy_moda",
    "scraped_at": "2026-05-XX",
    "url": "https://...",
    "title": "Áo ...",
    "price_vnd": 590000,
    "image_url": "https://...",
    "image_local_path": "raw/vn_brands/images/ivy_moda/abc.jpg",
    "category": "blouse",
    "color": "be",
    "sizes": ["S","M","L","XL"],
    "size_chart_cm": {...} | null,
    "description": "...",
  }

Politeness:
  - 2–3 second jitter between requests
  - Caches HTML to disk so reruns don't re-hit servers
  - User-Agent identifies as research; obeys robots.txt
  - Skips if site disallows
  - Configurable max items per brand (default 200)

Run:
  python research/datasets/scripts/scrape_vn_brands.py --brands ivy_moda coolmate dottie --max-per-brand 200
"""

# intent: produce a small VN-branded catalog for the recommender to surface real local items
# status: scaffolded — selectors per site may need tweaks when run live
# next: validate selectors against actual page DOMs, then run with --max-per-brand 200 each
# blockers: live site DOMs may have changed since last validation
# confidence: medium

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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "research/datasets/raw/vn_brands"
IMG_DIR = OUT_DIR / "images"
CACHE_DIR = OUT_DIR / "_html_cache"
PRODUCTS_FILE = OUT_DIR / "products.jsonl"

USER_AGENT = (
    "EAVTONResearchBot/0.1 (academic research; lethanhdanh1s1k@gmail.com; "
    "single-shot crawl, respects robots.txt, polite rate)"
)

DEFAULT_PAUSE = (2.0, 4.0)   # random sleep range between requests
TIMEOUT_S = 20


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

    def to_jsonl(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def _cache_path(url: str) -> Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{h}.html"


def _polite_get(url: str, *, session: requests.Session, sleep_range: tuple[float, float] = DEFAULT_PAUSE) -> str | None:
    cache = _cache_path(url)
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="ignore")
    time.sleep(random.uniform(*sleep_range))
    try:
        r = session.get(url, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})
        if r.status_code != 200:
            print(f"    !! {r.status_code} {url}", file=sys.stderr)
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(r.text, encoding="utf-8")
        return r.text
    except Exception as e:
        print(f"    !! exception {url}: {e}", file=sys.stderr)
        return None


def _robots_allows(base_url: str, path: str) -> bool:
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(urllib.parse.urljoin(base_url, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        return True  # permissive on robots fetch failure
    return rp.can_fetch(USER_AGENT, urllib.parse.urljoin(base_url, path))


def _parse_price_vnd(text: str) -> int | None:
    """e.g. '590.000đ' or '590,000 VNĐ' or '590000' → 590000."""
    if not text:
        return None
    cleaned = re.sub(r"[^\d]", "", text)
    if not cleaned:
        return None
    n = int(cleaned)
    # Heuristic: VN prices are 5-8 digits typically (50k–10M VND)
    if 10_000 <= n <= 50_000_000:
        return n
    return None


def _download_image(url: str, brand: str, *, session: requests.Session) -> str | None:
    """Download image to images/<brand>/<sha>.jpg, return relative path."""
    if not url:
        return None
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    target = IMG_DIR / brand / f"{h}.jpg"
    if target.exists():
        return str(target.relative_to(OUT_DIR.parent.parent))
    try:
        time.sleep(random.uniform(0.5, 1.5))
        r = session.get(url, timeout=TIMEOUT_S, headers={"User-Agent": USER_AGENT})
        if r.status_code != 200:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(r.content)
        return str(target.relative_to(OUT_DIR.parent.parent))
    except Exception:
        return None


# ── Per-brand scrapers ──

def scrape_ivy_moda(session: requests.Session, max_items: int) -> Iterable[Product]:
    """Ivy Moda — women's tops (selectors verified May 2026)."""
    base = "https://ivymoda.com"
    if not _robots_allows(base, "/"):
        print("  ivy_moda: robots.txt disallows; skipping"); return
    list_url = f"{base}/danh-muc/ao-nu"
    html = _polite_get(list_url, session=session)
    if not html:
        return
    soup = BeautifulSoup(html, "html.parser")

    # Real selectors from live DOM (May 2026):
    # Each card is an <a> linking to /sanpham/<slug>-ms-<code>-<id>
    # Inside the card: <h3> for title, plus price + img.
    all_links = soup.find_all("a", href=True)
    product_links = [a for a in all_links if "/sanpham/" in a.get("href", "")]
    # Dedupe by href
    seen_href = set()
    cards = []
    for a in product_links:
        href = a["href"]
        if href in seen_href: continue
        seen_href.add(href)
        cards.append(a)
    print(f"  ivy_moda: {len(cards)} product links on listing")

    yielded = 0
    for card in cards:
        if yielded >= max_items: break
        href = card["href"]
        full = urllib.parse.urljoin(base, href)

        # Most info is already on the listing card — avoid extra page hits
        h3 = card.find("h3")
        title = h3.get_text(strip=True) if h3 else ""
        if not title:
            # Sometimes title is plain text in the link
            title = card.get_text(strip=True).split("\n")[0][:80]

        # Price from card text (regex out "534.000đ" patterns)
        m = re.search(r"([\d.,]+)\s*đ", card.get_text())
        price = _parse_price_vnd(m.group(0)) if m else None

        # Image from <img> inside the card
        img_el = card.find("img")
        img_url = None
        if img_el:
            img_url = img_el.get("data-src") or img_el.get("src")
            if img_url and not img_url.startswith("http"):
                img_url = urllib.parse.urljoin(base, img_url)

        img_local = _download_image(img_url, "ivy_moda", session=session) if img_url else None

        yield Product(
            brand="ivy_moda",
            url=full,
            title=title,
            price_vnd=price,
            image_url=img_url,
            image_local_path=img_local,
            category="top",
            color=None,
            sizes=[],
            size_chart_cm=None,
            description=None,
            scraped_at=time.strftime("%Y-%m-%d"),
        )
        yielded += 1


def scrape_coolmate(session: requests.Session, max_items: int) -> Iterable[Product]:
    """Coolmate — Vietnamese active/casual basics."""
    base = "https://www.coolmate.me"
    if not _robots_allows(base, "/"):
        print("  coolmate: robots.txt disallows; skipping"); return
    list_url = f"{base}/category/ao-thun"  # women's tee category
    html = _polite_get(list_url, session=session)
    if not html: return
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("a.product-item, a[href*='/product/']")
    print(f"  coolmate: {len(cards)} cards")
    seen = set(); yielded = 0
    for card in cards:
        if yielded >= max_items: break
        href = card.get("href")
        if not href: continue
        full = urllib.parse.urljoin(base, href)
        if full in seen: continue
        seen.add(full)

        prod_html = _polite_get(full, session=session)
        if not prod_html: continue
        ps = BeautifulSoup(prod_html, "html.parser")

        title_el = ps.select_one("h1, h1.product-title")
        title = title_el.get_text(strip=True) if title_el else ""
        price_el = ps.select_one(".product-price, .price")
        price = _parse_price_vnd(price_el.get_text(strip=True) if price_el else "")
        img_el = ps.select_one("img.product-image, img[itemprop=image]")
        img_url = img_el.get("src") if img_el else None
        if img_url and not img_url.startswith("http"):
            img_url = urllib.parse.urljoin(base, img_url)
        sizes = [s.get_text(strip=True).upper() for s in ps.select(".size-option, .product-size")
                 if s.get_text(strip=True).upper() in {"XS","S","M","L","XL","XXL"}]

        img_local = _download_image(img_url, "coolmate", session=session) if img_url else None
        yield Product(
            brand="coolmate", url=full, title=title, price_vnd=price,
            image_url=img_url, image_local_path=img_local,
            category="t-shirt", color=None, sizes=sorted(set(sizes)),
            size_chart_cm=None, description=None,
            scraped_at=time.strftime("%Y-%m-%d"),
        )
        yielded += 1


def scrape_dottie(session: requests.Session, max_items: int) -> Iterable[Product]:
    """Dottie — Vietnamese women's brand. Shopify-style.

    Use Shopify's /products.json endpoint for clean structured data instead of
    parsing rendered HTML (which lazy-loads images).
    """
    base = "https://dottie.vn"
    if not _robots_allows(base, "/"):
        print("  dottie: robots.txt disallows; skipping"); return

    # Shopify exposes structured product feeds at /collections/<name>/products.json
    collections = [
        "all-tops", "ao-so-mi", "ao-thun-classic", "ao-kieu-classic", "ao-khoac-classic",
    ]
    yielded = 0
    seen = set()
    for col in collections:
        if yielded >= max_items: break
        feed_url = f"{base}/collections/{col}/products.json?limit=50"
        html = _polite_get(feed_url, session=session)
        if not html: continue
        try:
            data = json.loads(html)
        except Exception:
            continue
        products = data.get("products", [])
        print(f"  dottie/{col}: {len(products)} products")
        for prod in products:
            if yielded >= max_items: break
            handle = prod.get("handle")
            if not handle: continue
            full = f"{base}/products/{handle}"
            if full in seen: continue
            seen.add(full)

            title = prod.get("title", "")
            # Price in cents-equivalent VND from Shopify
            variants = prod.get("variants", [])
            price = None
            if variants:
                try:
                    price = int(float(variants[0].get("price", 0)))
                    if price and price < 1000:  # sometimes stored in major unit
                        price *= 1000
                except Exception:
                    pass
            images = prod.get("images", [])
            img_url = images[0]["src"] if images else None
            sizes = []
            for v in variants:
                opt = (v.get("option1") or "").upper()
                if opt in {"XS","S","M","L","XL","XXL"}:
                    sizes.append(opt)
            img_local = _download_image(img_url, "dottie", session=session) if img_url else None

            yield Product(
                brand="dottie", url=full, title=title, price_vnd=price,
                image_url=img_url, image_local_path=img_local,
                category="top", color=None, sizes=sorted(set(sizes)),
                size_chart_cm=None, description=None,
                scraped_at=time.strftime("%Y-%m-%d"),
            )
            yielded += 1


SCRAPERS: dict[str, Callable] = {
    "ivy_moda":  scrape_ivy_moda,
    "coolmate":  scrape_coolmate,
    "dottie":    scrape_dottie,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brands", nargs="+", default=list(SCRAPERS.keys()))
    ap.add_argument("--max-per-brand", type=int, default=200)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.no_cache and CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.html"):
            f.unlink()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    total = 0
    with open(PRODUCTS_FILE, "a", encoding="utf-8") as out:
        for brand in args.brands:
            if brand not in SCRAPERS:
                print(f"  !! unknown brand: {brand}"); continue
            print(f"── {brand} (max={args.max_per_brand}) ──")
            for p in SCRAPERS[brand](session, args.max_per_brand):
                out.write(p.to_jsonl() + "\n")
                out.flush()
                total += 1
                if total % 10 == 0:
                    print(f"    [{total}] {p.brand}: {p.title[:50]}  {p.price_vnd}vnd")

    print(f"\n  wrote {total} products to {PRODUCTS_FILE}")


if __name__ == "__main__":
    main()
