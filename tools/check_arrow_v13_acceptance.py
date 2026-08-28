"""Acceptance checks for the unique-owner/shared-root arrow recovery release."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SLUGS = (
    "taylor_3_12",
    "drawings_volume_4_binder",
    "gladstone_dog_park",
    "rapid_city_2",
)
INTENTIONAL_OLD_FALSE_BRANCH = ("drawings_volume_4_binder", "6", "0")


def load(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stroke_token(stroke):
    points = tuple((round(float(point[0]), 2), round(float(point[1]), 2))
                   for point in stroke)
    return min(points, tuple(reversed(points)))


def tips(entry):
    return [tuple(round(float(value), 1) for value in target["tip"])
            for target in entry.get("targets") or []]


def close_tip(actual, expected, tolerance=1.0):
    return all(abs(float(left) - float(right)) <= tolerance
               for left, right in zip(actual, expected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()
    old_docs = {slug: load(args.baseline / slug / "arrows.json")
                for slug in SLUGS}
    docs = {slug: load(args.data / slug / "arrows.json") for slug in SLUGS}
    problems = []
    old_items = new_items = 0

    # Every result that existed before recovery remains byte-geometrically
    # present, except the independently verified NEW CONCRETE WALK branch.
    for slug, old_doc in old_docs.items():
        new_doc = docs[slug]
        for page, old_page in old_doc.items():
            old_entries = (old_page or {}).get("items")
            if not isinstance(old_entries, dict):
                continue
            new_entries = (new_doc.get(page) or {}).get("items") or {}
            for key, old_entry in old_entries.items():
                old_items += 1
                new_entry = new_entries.get(key)
                if not isinstance(new_entry, dict):
                    problems.append(f"{slug}:P{page}:{key} missing")
                    continue
                if (slug, page, key) == INTENTIONAL_OLD_FALSE_BRANCH:
                    continue
                for field in ("leader_strokes", "arrow_strokes"):
                    old_strokes = Counter(stroke_token(row)
                                          for row in old_entry.get(field) or [])
                    new_strokes = Counter(stroke_token(row)
                                          for row in new_entry.get(field) or [])
                    missing = old_strokes - new_strokes
                    if missing:
                        problems.append(
                            f"{slug}:P{page}:{key} lost {sum(missing.values())} "
                            f"original {field}")
                if len(new_entry.get("targets") or []) < len(
                        old_entry.get("targets") or []):
                    problems.append(f"{slug}:P{page}:{key} lost original target")
        new_items += sum(len((page or {}).get("items") or {})
                         for page in new_doc.values())

    def entries(slug, page):
        return (docs[slug][str(page)].get("items") or {})

    def exact_keys(slug, page, expected):
        actual = set(entries(slug, page))
        if actual != set(expected):
            problems.append(f"{slug}:P{page} keys {sorted(actual)}")

    exact_keys("gladstone_dog_park", 2, {"2", "3", "5"})
    g2 = entries("gladstone_dog_park", 2)
    if tips(g2["2"]) != [(461.0, 647.0), (342.0, 633.0)]:
        problems.append(f"gladstone:P2 key2 tips {tips(g2['2'])}")
    exact_keys("gladstone_dog_park", 7, {str(i) for i in range(6)})
    g7 = entries("gladstone_dog_park", 7)
    if any(len(entry.get("targets") or []) != 1 for entry in g7.values()):
        problems.append("gladstone:P7 must have exactly one target per callout")
    if tips(g7["3"]) != [(748.0, 266.0)]:
        problems.append(f"gladstone:P7 key3 tips {tips(g7['3'])}")

    exact_keys("drawings_volume_4_binder", 4,
               {"9", "10", "s4:23", "s7:0"})
    b4 = entries("drawings_volume_4_binder", 4)
    if tips(b4["10"]) != [(877.0, 609.0)]:
        problems.append(f"binder:P4 key10 tips {tips(b4['10'])}")
    exact_keys("drawings_volume_4_binder", 6, {"0"})
    b6 = entries("drawings_volume_4_binder", 6)["0"]
    if tips(b6) != [(65.0, 538.0)] or len(b6.get("leader_strokes") or []) != 1:
        problems.append(f"binder:P6 not uniquely owned: {tips(b6)}")

    t36 = entries("taylor_3_12", 36)
    required_36 = {"0", "2", "5", "6", "8", "10", "13", "14",
                   "15", "22", "37", "42", "43"}
    missing_36 = required_36 - set(t36)
    if missing_36:
        problems.append(f"taylor:P36 missing recovered {sorted(missing_36)}")
    for key, expected in {"4": (69.8, 408.5), "12": (177.9, 83.5),
                          "19": (261.9, 75.6)}.items():
        targets = t36[key].get("targets") or []
        if (len(targets) != 1
                or targets[0].get("terminal_kind") != "arrowhead"
                or not close_tip(targets[0].get("tip") or [], expected)):
            problems.append(f"taylor:P36 key{key} terminal {targets}")

    required = {
        ("taylor_3_12", 37): {"4"},
        ("taylor_3_12", 90): {str(i) for i in range(5)},
        ("taylor_3_12", 97): {str(i) for i in range(4)},
        ("taylor_3_12", 100): {str(i) for i in range(10)},
        ("taylor_3_12", 111): {str(i) for i in range(20)},
        ("drawings_volume_4_binder", 5): {"0", "2", "3"},
        ("drawings_volume_4_binder", 9): {"0", "1", "2", "10", "30", "37"},
    }
    for (slug, page), keys in required.items():
        missing = keys - set(entries(slug, page))
        if missing:
            problems.append(f"{slug}:P{page} missing positives {sorted(missing)}")

    image_pages = {page for page, entry in docs["taylor_3_12"].items()
                   if entry.get("page_kind") == "image-only"}
    expected_images = {"14", "31", "44", "45", "54", "55", "56",
                       "161", "163", "175"}
    if not expected_images <= image_pages:
        problems.append(f"image-only pages missing {sorted(expected_images-image_pages)}")

    stale = [f"{slug}:P{page}" for slug, doc in docs.items()
             for page, entry in doc.items()
             if isinstance(entry, dict) and entry.get("v") != 13]
    if stale:
        problems.append(f"non-v13 cache entries: {stale[:10]}")

    report = {
        "ok": not problems,
        "old_items": old_items,
        "new_items": new_items,
        "added_items": new_items - old_items,
        "image_only_pages": sorted(image_pages, key=int),
        "problems": problems,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
