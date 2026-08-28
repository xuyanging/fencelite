"""Verify that a rebuilt arrows cache is monotonic against a saved baseline."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_SLUGS = (
    "taylor_3_12",
    "drawings_volume_4_binder",
    "gladstone_dog_park",
    "rapid_city_2",
)


def _load(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stroke_token(stroke):
    points = tuple((round(float(point[0]), 2), round(float(point[1]), 2))
                   for point in stroke)
    return min(points, tuple(reversed(points)))


def _strokes(entry, field):
    return Counter(_stroke_token(row) for row in (entry.get(field) or [])
                   if isinstance(row, list) and len(row) >= 2)


def compare(baseline_root, data_root, slugs):
    problems = []
    old_count = new_count = 0
    image_only = []
    per_slug = {}
    for slug in slugs:
        old = _load(baseline_root / slug / "arrows.json")
        new = _load(data_root / slug / "arrows.json")
        slug_old = slug_new = 0
        image_only.extend(
            f"{slug}:P{page}" for page, entry in new.items()
            if isinstance(entry, dict) and entry.get("page_kind") == "image-only")
        for page, old_page in old.items():
            old_items = (old_page or {}).get("items")
            if not isinstance(old_items, dict):
                continue
            new_page = new.get(page)
            new_items = ((new_page or {}).get("items")
                         if isinstance(new_page, dict) else None)
            if not isinstance(new_items, dict):
                problems.append(f"{slug}:P{page} lost the complete page result")
                continue
            for key, old_entry in old_items.items():
                old_count += 1
                slug_old += 1
                new_entry = new_items.get(key)
                if not isinstance(new_entry, dict):
                    problems.append(f"{slug}:P{page}:{key} missing")
                    continue
                for field in ("leader_strokes", "arrow_strokes"):
                    old_strokes = _strokes(old_entry, field)
                    new_strokes = _strokes(new_entry, field)
                    missing = old_strokes - new_strokes
                    if missing:
                        problems.append(
                            f"{slug}:P{page}:{key} lost {sum(missing.values())} "
                            f"authored {field}")
                    old_points = sum(len(row) for row in old_entry.get(field) or [])
                    new_points = sum(len(row) for row in new_entry.get(field) or [])
                    if new_points < old_points:
                        problems.append(
                            f"{slug}:P{page}:{key} {field} points "
                            f"{old_points}->{new_points}")
                old_targets = len(old_entry.get("targets") or [])
                new_targets = len(new_entry.get("targets") or [])
                if new_targets < old_targets:
                    problems.append(
                        f"{slug}:P{page}:{key} targets {old_targets}->{new_targets}")
        for entry in new.values():
            items = (entry or {}).get("items") if isinstance(entry, dict) else None
            if isinstance(items, dict):
                new_count += len(items)
                slug_new += len(items)
        per_slug[slug] = {"old": slug_old, "new": slug_new,
                          "added": slug_new - slug_old}
    return {
        "ok": not problems,
        "old_items": old_count,
        "new_items": new_count,
        "added_items": new_count - old_count,
        "per_slug": per_slug,
        "image_only_pages": image_only,
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("slugs", nargs="*", default=list(DEFAULT_SLUGS))
    args = parser.parse_args()
    result = compare(args.baseline, args.data, args.slugs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
