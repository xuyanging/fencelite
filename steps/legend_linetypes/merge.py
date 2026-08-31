"""Publication-time merge of arrow and supervised legend line types."""
from __future__ import annotations


class LegendMergeError(ValueError):
    """The two independently cached producers disagree structurally."""


def _rows(rows, source):
    result = {}
    for row in rows or ():
        if not isinstance(row, dict):
            raise LegendMergeError(f"{source} contains a non-object line type")
        number = row.get("line_type_number")
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise LegendMergeError(f"{source} contains an invalid line type number")
        if number in result:
            raise LegendMergeError(f"{source} contains duplicate type #{number}")
        result[number] = dict(row)
    return result


def _legend_numbers(row):
    """Validate and return the engine clusters supervised by one template."""
    primary = row["line_type_number"]
    raw = row.get("matched_line_type_numbers")
    if not isinstance(raw, list) or not raw:
        raise LegendMergeError(
            f"legend type #{primary} has no matched engine clusters")
    if any(isinstance(number, bool) or not isinstance(number, int)
           or number <= 0 for number in raw):
        raise LegendMergeError(
            f"legend type #{primary} has invalid matched engine clusters")
    numbers = sorted(set(raw))
    if len(numbers) != len(raw) or primary not in numbers:
        raise LegendMergeError(
            f"legend type #{primary} has inconsistent matched engine clusters")
    return numbers


def _validate_legend_type(supervised, engine_rows=None):
    """Fail closed if a supervised row cannot identify its engine result.

    Legend and arrow caches are produced independently.  Cluster numbers are
    deterministic, but a number alone is not an identity: when the ordinary
    cache exposes its complete audit list, also verify the primary UID/ID and
    every matched cluster's recognition source before geometry can be merged.
    """
    number = supervised["line_type_number"]
    if supervised.get("recognition_source") != "legend_template" \
            or supervised.get("base_line_type_number") != number:
        raise LegendMergeError(
            f"legend type #{number} does not identify its base cluster")
    numbers = _legend_numbers(supervised)
    sources = supervised.get("matched_cluster_sources")
    if not isinstance(sources, dict):
        raise LegendMergeError(
            f"legend type #{number} has no matched-cluster source audit")
    for matched in numbers:
        source = sources.get(str(matched), sources.get(matched))
        if source not in ("method1", "method2"):
            raise LegendMergeError(
                f"legend type #{number} has an invalid source for #{matched}")

    if not engine_rows:
        return
    for matched in numbers:
        ordinary = engine_rows.get(matched)
        if ordinary is None:
            raise LegendMergeError(
                f"legend type #{number} references missing engine type #{matched}")
        expected = sources.get(str(matched), sources.get(matched))
        if ordinary.get("recognition_source") != expected:
            raise LegendMergeError(
                f"legend type #{number} source differs for engine type #{matched}")

    base = engine_rows[number]
    for key in ("type_uid", "line_type_id"):
        legend_value = supervised.get(key)
        base_value = base.get(key)
        if legend_value and base_value \
                and legend_value != f"legend:{base_value}":
            raise LegendMergeError(
                f"legend type #{number} {key} differs from the engine audit")


def _merge_type(ordinary, supervised):
    number = supervised["line_type_number"]
    expected_source = supervised.get("base_recognition_source")
    actual_source = ordinary.get("recognition_source")
    if expected_source and actual_source and expected_source != actual_source:
        raise LegendMergeError(
            f"type #{number} recognition source differs between caches")

    # The supervised row is a semantic union and already contains every
    # compatible base/satellite run.  Keeping the ordinary row's base runs as
    # well would draw identical geometry twice (P3 Method2 #2 was 87 + 87
    # polylines) and make segment counts lie.  Preserve only an audit count.
    result = dict(ordinary)
    result.update(supervised)
    left = ordinary.get("by_run") or {}
    right = supervised.get("by_run") or {}
    if ((left and not isinstance(left, dict))
            or (right and not isinstance(right, dict))):
        raise LegendMergeError(f"type #{number} has invalid run geometry")
    result["by_run"] = dict(right)
    result["replaced_arrow_run_count"] = len(left)
    result["arrow_recognition_source"] = actual_source
    return result


def merge_entries(ordinary, legend):
    """Combine current independent caches without mutating either input.

    A legend semantic result deliberately reuses its primary engine cluster
    number so existing grouping/publication code can consume it.  On that one
    collision the supervised semantic geometry replaces the ordinary base
    geometry (it already contains the validated base plus compatible
    satellites).  Any structural or engine-identity disagreement fails
    closed.
    """
    ordinary = ordinary if isinstance(ordinary, dict) else {}
    legend = legend if isinstance(legend, dict) else {}
    normal_rows = _rows(ordinary.get("line_types"), "ordinary cache")
    legend_rows = _rows(legend.get("line_types"), "legend cache")
    engine_rows = _rows(ordinary.get("all_line_types"),
                        "ordinary engine audit")
    for number, row in legend_rows.items():
        _validate_legend_type(row, engine_rows)
        if number in normal_rows:
            normal_rows[number] = _merge_type(normal_rows[number], row)
        else:
            normal_rows[number] = row

    page = dict(ordinary.get("page") or {})
    legend_page = legend.get("page") or {}
    for key in ("legend_samples", "legend_matches", "legend_semantic_types"):
        if key in legend_page:
            page[key] = legend_page[key]
    if not page:
        page = dict(legend_page)

    bindings = [dict(row) for row in ordinary.get("bindings") or ()]
    bindings.extend(dict(row) for row in legend.get("bindings") or ())
    return {
        "engine": ordinary.get("engine") or legend.get("engine") or {},
        "page": page,
        "line_types": [normal_rows[number] for number in sorted(normal_rows)],
        "all_line_types": list(ordinary.get("all_line_types") or ()),
        "bindings": bindings,
        # regroup.resolve recomputes groups/visibility from bindings at read
        # time; cached groups from either producer are intentionally ignored.
        "groups": [],
        "used_all": [],
        "legend_samples": list(legend.get("samples") or ()),
    }


def debug_types(entry):
    """Flatten supervised run buckets for the All Line Types debug layer."""
    rows = _rows((entry or {}).get("line_types"), "legend cache")
    bound = {}
    for binding in (entry or {}).get("bindings") or ():
        nearest = binding.get("nearest_op") or {}
        number = nearest.get("owner")
        if isinstance(number, int) and not isinstance(number, bool):
            bound.setdefault(number, []).append({
                "key": binding.get("key"), "ti": binding.get("ti"),
                "distance": nearest.get("distance"),
            })
    out = []
    for number in sorted(rows):
        row = rows[number]
        buckets = row.pop("by_run", None) or {}
        if not isinstance(buckets, dict):
            raise LegendMergeError(
                f"legend type #{number} has invalid run geometry")
        row["polylines"] = [line for bucket in buckets.values()
                            for line in (bucket.get("polylines") or ())]
        row["runs"] = [{key: bucket.get(key) for key in (
            "run_id", "source_line_type_number", "source_run_id",
            "op_count", "segment_count", "bbox")}
            for bucket in buckets.values()]
        row["bound_by"] = bound.get(number) or []
        row["support_only"] = bool(
            row.get("op_count") and not row["polylines"]
            and not row.get("segment_count"))
        out.append(row)
    return out


__all__ = ["LegendMergeError", "debug_types", "merge_entries"]
