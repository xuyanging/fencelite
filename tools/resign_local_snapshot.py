"""Re-sign an already copied fence_lite snapshot for local, read-only UI QA.

Copying a PDF changes its mtime, while fence_lite deliberately includes mtime in
every publish signature.  This utility updates only cache identity metadata; it
does not call a model, rerun detection, or alter any detected geometry/content.

It refuses snapshots whose live PDF size differs from the size encoded in the
source revision.  Keep an untouched backup before running it.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from steps import arrows, store
from steps.placements import current_placement_context
from steps.symbols import has_current_symbols
from steps.views import (has_current_view_types, view_signature)


def _placement_anchors(result):
    anchors = []
    for si, symbol in enumerate(result.get("symbols") or []):
        for pi, box in enumerate(symbol.get("placements") or []):
            if isinstance(box, (list, tuple)) and len(box) == 4:
                anchors.append((f"s{si}:{pi}", list(box)))
    return anchors


def _rewrite_identity(records, revision):
    changed = 0
    for entry in (records or {}).values():
        identity = entry.get("vlm_identity") if isinstance(entry, dict) else None
        if isinstance(identity, dict):
            identity["pdf_revision"] = revision
            changed += 1
    return changed


def resign(slug, *, dry_run=False):
    slug = store.require_slug(slug)
    pdf = store.pdf_path(slug)
    data_dir = store.slug_dir(slug)
    results = store.load_json(data_dir / "results.json", None)
    if not pdf.is_file() or not isinstance(results, dict):
        raise FileNotFoundError(f"missing PDF/results for {slug}")

    old_revision = str(results.get("pdf_revision") or "")
    try:
        old_size = int(old_revision.split("-", 1)[0], 16)
    except (ValueError, IndexError):
        raise ValueError(f"invalid source revision: {old_revision!r}") from None
    if pdf.stat().st_size != old_size:
        raise ValueError(
            f"PDF byte size changed ({pdf.stat().st_size} != {old_size}); "
            "refusing to publish cached geometry over a different document")

    revision = store.pdf_revision(pdf)
    report = {"slug": slug, "old_revision": old_revision,
              "new_revision": revision, "dry_run": dry_run}
    if dry_run or revision == old_revision:
        return report

    pages = results.get("pages") or {}
    symbols_path = data_dir / "symbols.json"
    symbols = store.load_json(symbols_path, {}) or {}
    views_path = data_dir / "view_types.json"
    views = store.load_json(views_path, {}) or {}
    arrow_path = data_dir / "arrows.json"
    arrow_cache = store.load_json(arrow_path, {}) or {}

    # Validate every layer against the *old* identity before changing a single
    # signature.  Re-signing is metadata migration, never an algorithm-version
    # upgrade: stale symbol/view/placement/arrow results must remain stale.
    safe_symbols = set()
    safe_views = set()
    safe_arrows = {}
    for page, rec in pages.items():
        if not isinstance(rec, dict):
            continue
        items = store.items_of(rec)
        symbol_entry = symbols.get(str(page))
        if has_current_symbols(
                symbol_entry, store.sig_of(items, old_revision)):
            safe_symbols.add(str(page))
        result = (symbol_entry or {}).get("result") or {}
        groups = result.get("groups") or []
        view_entry = views.get(str(page))
        if has_current_view_types(
                view_entry, groups, old_revision,
                (view_entry or {}).get("model")):
            safe_views.add(str(page))
        context = current_placement_context(
            symbol_entry, store.sig_of(items, old_revision),
            view_entry, old_revision)
        if not context["placements_current"]:
            continue
        extra = _placement_anchors(context["placement_result"])
        old_arrow_sig = arrows.arrows_signature(
            items, old_revision, extra,
            plan_regions=context["plan_regions"])
        arrow_entry = arrow_cache.get(str(page))
        if arrows.has_current_arrows(arrow_entry, old_arrow_sig):
            safe_arrows[str(page)] = {
                "extra": extra,
                "plan_regions": context["plan_regions"],
            }

    results["pdf_revision"] = revision
    for rec in (results.get("pages") or {}).values():
        if not isinstance(rec, dict):
            continue
        for source in rec.get("vlm_sources") or []:
            identity = source.get("identity") if isinstance(source, dict) else None
            if isinstance(identity, dict):
                identity["pdf_revision"] = revision
    store.save_json(data_dir / "results.json", results)

    for name in ("vlm.json", "vlm_flash.json"):
        path = data_dir / name
        records = store.load_json(path, None)
        if isinstance(records, dict):
            report[name] = _rewrite_identity(records, revision)
            store.save_json(path, records)

    vec_path = data_dir / "vec.json"
    vec = store.load_json(vec_path, None)
    if isinstance(vec, dict):
        vec["pdf_mtime"] = pdf.stat().st_mtime
        store.save_json(vec_path, vec)

    sym_fixed = 0
    sym_skipped = 0
    for page, entry in symbols.items():
        rec = pages.get(str(page))
        if (str(page) in safe_symbols and isinstance(entry, dict)
                and isinstance(rec, dict)):
            entry["sig"] = store.sig_of(store.items_of(rec), revision)
            sym_fixed += 1
        elif isinstance(entry, dict):
            sym_skipped += 1
    if symbols:
        store.save_json(symbols_path, symbols)
    report["symbol_pages"] = sym_fixed
    report["symbol_pages_skipped_stale"] = sym_skipped

    view_fixed = 0
    view_skipped = 0
    for page, entry in views.items():
        sym_entry = symbols.get(str(page))
        if (str(page) in safe_views and isinstance(entry, dict)
                and isinstance(sym_entry, dict)):
            groups = ((sym_entry.get("result") or {}).get("groups") or [])
            entry["sig"] = view_signature(groups, revision, entry.get("model"))
            view_fixed += 1
        elif isinstance(entry, dict):
            view_skipped += 1
    if views:
        store.save_json(views_path, views)
    report["view_pages"] = view_fixed
    report["view_pages_skipped_stale"] = view_skipped

    arrow_fixed = 0
    arrow_skipped = 0
    for page, entry in arrow_cache.items():
        rec = pages.get(str(page))
        prerequisite = safe_arrows.get(str(page))
        if (prerequisite is not None and isinstance(entry, dict)
                and isinstance(rec, dict)):
            items = store.items_of(rec)
            entry["sig"] = arrows.arrows_signature(
                items, revision, prerequisite["extra"],
                plan_regions=prerequisite["plan_regions"])
            arrow_fixed += 1
        elif isinstance(entry, dict):
            arrow_skipped += 1
    if arrow_cache:
        store.save_json(arrow_path, arrow_cache)
    report["arrow_pages"] = arrow_fixed
    report["arrow_pages_skipped_stale_placements"] = arrow_skipped

    # The PDF bytes are unchanged, so old base images are safe to reuse under
    # the new metadata-only revision.  Copy, rather than move, for auditability.
    copied = 0
    pattern = re.compile(r"^base_P(\d+)_[0-9a-f]+-[0-9a-f]+\.jpg$")
    for source in data_dir.glob("base_P*.jpg"):
        match = pattern.match(source.name)
        if not match:
            continue
        target = data_dir / f"base_P{match.group(1)}_{revision}.jpg"
        if target != source and not target.exists():
            shutil.copy2(source, target)
            copied += 1
    report["base_images_copied"] = copied
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slugs", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for slug in args.slugs:
        print(resign(slug, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
