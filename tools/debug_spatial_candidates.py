"""Print spatial arrow-candidate distances for one cached project page."""
import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from steps import store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slug")
    parser.add_argument("page", type=int)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    results = store.load_results(args.slug)
    record = results["pages"][str(args.page)]
    items = store.items_of(record)
    with tempfile.TemporaryDirectory(prefix="arrow-debug-") as work:
        one = Path(work) / "page.pdf"
        src = fitz.open(store.pdf_path(args.slug))
        dst = fitz.open()
        try:
            dst.insert_pdf(src, from_page=args.page - 1, to_page=args.page - 1)
            dst.save(one)
        finally:
            dst.close()
            src.close()
        job = {
            "pdf": str(one),
            "page": 1,
            "boxes": [item["box_2d"] for item in items],
            "anchor_labels": [item.get("label", "") for item in items],
            "anchor_texts": [item.get("text", "") for item in items],
            "plan_regions": [],
            "debug_spatial_candidates": True,
            "debug_anchors": True,
        }
        sidecar = Path(__file__).parent / "arrow_sidecar" / "sidecar.mjs"
        proc = subprocess.run(
            [r"C:\Program Files\nodejs\node.exe", "--max-old-space-size=1536",
             str(sidecar)],
            input=json.dumps(job), capture_output=True, text=True, check=True,
        )
        payload = json.loads(proc.stdout)
        if not payload.get("ok"):
            print(json.dumps(payload, indent=2))
            return
        page_meta = payload.get("page") or {}
        page_diagonal = math.hypot(float(page_meta.get("width") or 0),
                                  float(page_meta.get("height") or 0)) or 1.0
        compact = []
        for candidate in payload.get("spatial_candidates", []):
            ranked = sorted(enumerate(candidate.pop("distances")),
                            key=lambda pair: pair[1])[:3]
            if not ranked or ranked[0][1] > page_diagonal * 0.025:
                continue
            candidate["nearest"] = ranked
            candidate["root_ratio"] = round(ranked[0][1] / page_diagonal, 5)
            compact.append(candidate)
        report = {
            "page": page_meta,
            "items": [{"index": i, "label": item.get("label"),
                       "text": item.get("text"), "box": item.get("box_2d")}
                      for i, item in enumerate(items)],
            "current": [{"index": row["index"], "has": row["has_leader"],
                         "leaders": row["leader_count"],
                         "source": row.get("source"),
                         "text": row.get("text"),
                         "debug": row.get("debug")}
                        for row in payload.get("results", [])],
            "spatial_recovered": payload.get("spatial_recovered", []),
            "candidates": compact,
        }
        if args.compact:
            report = {
                "slug": args.slug,
                "page": args.page,
                "missing": [row["index"] for row in report["current"]
                            if not row["has"]],
                "anchors": report["current"],
                "recovered": report["spatial_recovered"],
                "candidates": [{
                    "op": candidate["op_index"],
                    "kind": candidate["marker_kind"],
                    "root_end": candidate["marker_end"] ^ 1,
                    "nearest": candidate["nearest"],
                    "ratio": candidate["root_ratio"],
                    "paths": len(candidate["path_ops"]),
                    "arrows": len(candidate["arrow_ops"]),
                } for candidate in compact],
            }
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
