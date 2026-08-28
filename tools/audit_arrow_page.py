"""Recompute selected arrow pages without changing their on-disk caches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from job import _placement_anchors
from steps import arrows, store
from steps.views import merge_view_types, plan_boxes


def recompute(slug, page):
    result_doc = store.load_results(slug)
    record = result_doc["pages"][str(page)]
    items = store.items_of(record)
    symbols = store.load_json(store.slug_dir(slug) / "symbols.json", {})
    views = store.load_json(store.slug_dir(slug) / "view_types.json", {})
    symbol_result = (symbols.get(str(page)) or {}).get("result") or {}
    regions = plan_boxes(merge_view_types(
        symbol_result.get("groups") or [], views.get(str(page))))
    extra = _placement_anchors(symbol_result)
    geometry = arrows.page_geometry_status(store.pdf_path(slug), page - 1)
    if geometry.get("state") == "image-only":
        return {"slug": slug, "page": page, "page_kind": "image-only"}
    found = arrows.find_page_arrows(
        store.pdf_path(slug), page - 1, items,
        plan_regions=regions, extra_anchors=extra)
    entries = {}
    for key, entry in found.items():
        item = items[key] if isinstance(key, int) else {}
        entries[str(key)] = {
            "text": item.get("text"),
            "label": item.get("label"),
            "leaders": len(entry.get("leader_strokes") or []),
            "heads": len(entry.get("arrow_strokes") or []),
            "targets": [
                {"tip": target.get("tip"),
                 "kind": target.get("terminal_kind")}
                for target in entry.get("targets") or []
            ],
        }
    return {"slug": slug, "page": page, "entries": entries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+", metavar="SLUG:PAGE")
    args = parser.parse_args()
    for case in args.cases:
        slug, raw_page = case.rsplit(":", 1)
        print(json.dumps(recompute(slug, int(raw_page)),
                         ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
