#!/usr/bin/env python3
"""
Build IKEA product embedding index for IKEACatalogProvider.

Usage:
    python scripts/build_ikea_index.py [--csv data/ikea_products.csv] [--out /app/models/ikea]

CSV format (comma-separated, with header):
    name,width_mm,height_mm,depth_mm,category,notes

If no CSV is provided, a small built-in sample dataset is used so the
index exists and the provider has something to match against immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SAMPLE_PRODUCTS = [
    {"name": "KALLAX Shelf unit 77x77", "width_mm": 770, "height_mm": 770, "depth_mm": 390,
     "category": "shelf", "notes": "KALLAX 2x2 cube shelf"},
    {"name": "KALLAX Shelf unit 147x77", "width_mm": 1470, "height_mm": 770, "depth_mm": 390,
     "category": "shelf", "notes": "KALLAX 4x2 cube shelf"},
    {"name": "KALLAX Shelf unit 77x147", "width_mm": 770, "height_mm": 1470, "depth_mm": 390,
     "category": "shelf", "notes": "KALLAX 2x4 cube shelf"},
    {"name": "BILLY Bookcase 80x202", "width_mm": 800, "height_mm": 2020, "depth_mm": 280,
     "category": "bookcase", "notes": "BILLY standard bookcase"},
    {"name": "BILLY Bookcase 40x202", "width_mm": 400, "height_mm": 2020, "depth_mm": 280,
     "category": "bookcase", "notes": "BILLY narrow bookcase"},
    {"name": "LACK Side table", "width_mm": 550, "height_mm": 450, "depth_mm": 550,
     "category": "table", "notes": "LACK side table 55x55cm"},
    {"name": "LACK TV unit", "width_mm": 1500, "height_mm": 450, "depth_mm": 350,
     "category": "table", "notes": "LACK TV bench 150x35cm"},
    {"name": "ALEX Drawer unit", "width_mm": 360, "height_mm": 760, "depth_mm": 580,
     "category": "storage", "notes": "ALEX with 9 drawers"},
    {"name": "LINNMON Table top 120x60", "width_mm": 1200, "height_mm": 750, "depth_mm": 600,
     "category": "desk", "notes": "LINNMON/ALEX desk combo"},
    {"name": "MICKE Desk 105x50", "width_mm": 1050, "height_mm": 750, "depth_mm": 500,
     "category": "desk", "notes": "MICKE desk with integrated storage"},
    {"name": "PAX Wardrobe 100x236", "width_mm": 1000, "height_mm": 2360, "depth_mm": 600,
     "category": "wardrobe", "notes": "PAX wardrobe frame 100cm"},
    {"name": "PAX Wardrobe 200x236", "width_mm": 2000, "height_mm": 2360, "depth_mm": 600,
     "category": "wardrobe", "notes": "PAX wardrobe frame 200cm"},
    {"name": "EKET Cabinet combination", "width_mm": 700, "height_mm": 700, "depth_mm": 350,
     "category": "shelf", "notes": "EKET 2x2 wall cabinet"},
    {"name": "HEMNES Bookcase 90x197", "width_mm": 900, "height_mm": 1970, "depth_mm": 220,
     "category": "bookcase", "notes": "HEMNES solid wood bookcase"},
    {"name": "BRIMNES Bookcase 60x190", "width_mm": 600, "height_mm": 1900, "depth_mm": 200,
     "category": "bookcase", "notes": "BRIMNES bookcase with door"},
    {"name": "MALM Desk 140x65", "width_mm": 1400, "height_mm": 730, "depth_mm": 650,
     "category": "desk", "notes": "MALM desk"},
    {"name": "BEKANT Desk 160x80", "width_mm": 1600, "height_mm": 730, "depth_mm": 800,
     "category": "desk", "notes": "BEKANT desk sit/stand"},
    {"name": "IVAR Shelf unit 89x124", "width_mm": 890, "height_mm": 1240, "depth_mm": 300,
     "category": "shelf", "notes": "IVAR pine shelf 89x30cm"},
]


def build_index(products: list[dict], out_dir: Path) -> None:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Encoding {len(products)} products…")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=str(out_dir.parent / "sentence_transformers"),
    )
    texts = [
        f"{p['name']} {p.get('category', '')} {p.get('notes', '')}".strip()
        for p in products
    ]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    embed_path = out_dir / "embeddings.npy"
    products_path = out_dir / "products.json"
    np.save(str(embed_path), embeddings.astype("float32"))
    products_path.write_text(json.dumps(products, ensure_ascii=False, indent=2))
    print(f"Saved {len(products)} products to {out_dir}")
    print(f"  embeddings: {embed_path}")
    print(f"  products:   {products_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="", help="Path to ikea_products.csv (optional)")
    ap.add_argument("--out", default="/app/models/ikea", help="Output directory")
    args = ap.parse_args()

    products = SAMPLE_PRODUCTS
    if args.csv:
        import csv
        with open(args.csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            products = [
                {
                    "name": row["name"],
                    "width_mm": float(row["width_mm"]),
                    "height_mm": float(row["height_mm"]),
                    "depth_mm": float(row.get("depth_mm", 0) or 0),
                    "category": row.get("category", ""),
                    "notes": row.get("notes", ""),
                }
                for row in reader
            ]
        print(f"Loaded {len(products)} products from {args.csv}")

    build_index(products, Path(args.out))


if __name__ == "__main__":
    main()
