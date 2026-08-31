"""Supervised legend-line sidecar.

The input boxes are confirmed line samples from the symbol stage.  This
runner extracts their top-painted native PDF ink, associates it with the
ordinary full-page Method-1/Method-2 clusters, and publishes only supervised
matches.  It has an independent cache producer so changes here never stale the
expensive arrow-terminal line-type cache.

Input::

    {"pdf": str, "sheet": int, "samples": [{"symbol_index": int,
      "text_index": int, "box_2d": [y0,x0,y1,x1], ...}], "cpu_budget": 1}

Output is one JSON object.  Failures use the same explicit non-zero protocol
as ``run.py``; an empty successful match set is therefore distinguishable
from a crash or timeout.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


_HERE = Path(__file__).resolve().parent
_ENGINE = Path(os.environ.get("LINETYPE_ENGINE_PATH", str(_HERE / "engine")))
for _path in (_HERE, _ENGINE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run as _run                                              # noqa: E402

_fail = _run._fail
_bbox = _run._bbox
_connected_runs = _run._connected_runs
_method2_pattern_instances = _run._method2_pattern_instances


def _sha1(indices):
    return hashlib.sha1(
        ",".join(str(index) for index in sorted(set(indices))).encode()
    ).hexdigest()


def _dedupe_points(points):
    seen = set()
    out = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        key = round(float(point[0]), 2), round(float(point[1]), 2)
        if key in seen:
            continue
        seen.add(key)
        out.append([key[0], key[1]])
    return out


def main():
    try:
        job = json.load(sys.stdin)
    except Exception as error:                                  # noqa: BLE001
        _fail("BAD_JOB", f"cannot parse job JSON: {error}")

    pdf = str(job.get("pdf") or "")
    sheet = job.get("sheet")
    samples = job.get("samples")
    cpu_budget = job.get("cpu_budget", 1)
    if not pdf or not Path(pdf).is_file():
        _fail("BAD_JOB", f"pdf not found: {pdf!r}")
    if not isinstance(sheet, int) or isinstance(sheet, bool) or sheet < 1:
        _fail("BAD_JOB", f"sheet must be a 1-based int, got {sheet!r}")
    if not isinstance(samples, list) or not samples:
        _fail("BAD_JOB", "samples must be a non-empty list")
    # Method2 uses structurally different single- and multi-worker paths.  The
    # normal producer pins one worker because equivalence has not been proved;
    # this independently cached producer must make the same deterministic
    # choice rather than letting budget silently alter a cache value.
    if cpu_budget != 1:
        _fail("BAD_JOB", f"cpu_budget must be exactly 1, got {cpu_budget!r}")
    if not _ENGINE.is_dir():
        _fail("ENGINE_MISSING", f"engine source not found at {_ENGINE}")

    try:
        import pymupdf
        import pypdf
        import scipy
        from legend_supervised import (
            associate_template, extract_template, operation_lines,
            pattern_instances_outside_samples)
        from line_type_engine.clustering_api import (
            CLUSTERING_API_SCHEMA_VERSION, project_line_type_clusters)
        from line_type_engine.engine import recognize_page
        from line_type_engine.source_page_adapter import (
            SOURCE_PAGE_ADAPTER_VERSION, source_aligned_page_ir_from_pdf_path)
        from line_type_engine.versions import PAGE_IR_VERSION, PYTHON_ENGINE_VERSION
    except Exception as error:                                  # noqa: BLE001
        _fail("IMPORT_ERROR", f"{type(error).__name__}: {error}")

    try:
        started = time.time()
        aligned = source_aligned_page_ir_from_pdf_path(pdf, sheet)
        page = aligned.page
        seconds_ir = time.time() - started
    except Exception as error:                                  # noqa: BLE001
        _fail("PAGE_IR_ERROR", f"{type(error).__name__}: {error}")

    # Frame parameters come from a separate document.  Source extraction may
    # temporarily change /Rotate and must never leak that mutation here.
    try:
        document = pymupdf.open(pdf)
        try:
            fitz_page = document[sheet - 1]
            rotation_matrix = fitz_page.rotation_matrix
            rotated_width = float(fitz_page.rect.width)
            rotated_height = float(fitz_page.rect.height)
            fitz_rotation = int(fitz_page.rotation)
        finally:
            document.close()
    except Exception as error:                                  # noqa: BLE001
        _fail("PAGE_FRAME_ERROR", f"{type(error).__name__}: {error}")
    if rotated_width <= 0 or rotated_height <= 0:
        _fail("PAGE_FRAME_ERROR", "page rect has non-positive extent")
    if fitz_rotation != page.rotation_degrees:
        _fail("PAGE_FRAME_ERROR",
              f"rotation disagrees: fitz {fitz_rotation} vs IR "
              f"{page.rotation_degrees}")

    bounds = page.page_bounds
    unrotated_height = bounds.max_y - bounds.min_y
    unrotated_width = bounds.max_x - bounds.min_x
    expected = ((rotated_height, rotated_width)
                if page.rotation_degrees % 180
                else (rotated_width, rotated_height))
    if (abs(unrotated_width - expected[0]) > 1e-6
            or abs(unrotated_height - expected[1]) > 1e-6):
        _fail("PAGE_FRAME_ERROR",
              f"page_bounds {unrotated_width}x{unrotated_height} does not "
              f"match rotated rect {rotated_width}x{rotated_height}")

    def to_page_frame(x, y):
        point = pymupdf.Point(
            x + bounds.min_x,
            unrotated_height - y + bounds.min_y) * rotation_matrix
        return [round(point.y / rotated_height * 1000, 2),
                round(point.x / rotated_width * 1000, 2)]

    try:
        started = time.time()
        recognition = recognize_page(
            page, outputs=("method1", "method2", "fused"),
            method1_worker_count=1, parallel_methods=True)
        if recognition.fused is None:
            raise RuntimeError("fused recognition did not produce a result")
        projected = project_line_type_clusters(page, recognition.fused.result)
        seconds_cluster = time.time() - started
    except Exception as error:                                  # noqa: BLE001
        _fail("CLUSTER_ERROR", f"{type(error).__name__}: {error}")

    payload = projected.to_dict()
    operations = page.operations
    group_of = {}
    for group in payload["groups"]:
        group_id = str(group["group_id"])
        for local in group["line_types"]:
            for op_index in local["commands"]["op_indices"]:
                group_of[op_index] = group_id
        for op_index in group["residual_vector_commands"]["op_indices"]:
            group_of.setdefault(op_index, group_id)

    owner = {}
    cluster_ops = {}
    meta = {}
    for cluster in payload["global_line_types"]:
        number = int(cluster["line_type_number"])
        indices = list(cluster["commands"]["op_indices"])
        for op_index in indices:
            if not 0 <= op_index < len(operations):
                _fail("OP_INDEX_ERROR",
                      f"op_index {op_index} out of range (ops={len(operations)})")
            owner[op_index] = number
        cluster_ops[number] = indices
        meta[number] = {
            "line_type_number": number,
            "line_type_id": cluster["line_type_id"],
            "type_uid": cluster["type_uid"],
            "signature_family": cluster["signature_family"],
            "recognition_source": cluster["recognition_source"],
            "minimum_pair_similarity": cluster["minimum_pair_similarity"],
            "member_count": len(cluster["members"]),
        }

    # Recover Method-1 coverage lost only as a fusion side effect.  This is the
    # exact deterministic rule shared by run.py/run_all.py, so cluster numbers
    # and operation ownership remain comparable across all three producers.
    def global_types(envelope):
        if envelope is None:
            return ()
        return getattr(envelope.result, "global_types", ()) or ()

    method2_ops = set()
    for cluster in global_types(recognition.method2):
        method2_ops.update(cluster.op_indices)
    recovered = []
    next_number = max(cluster_ops or {0: []})
    for cluster in global_types(recognition.method1):
        keep = [op_index for op_index in cluster.op_indices
                if op_index not in owner and op_index not in method2_ops
                and 0 <= op_index < len(operations)]
        if not keep:
            continue
        next_number += 1
        for op_index in keep:
            owner[op_index] = next_number
        cluster_ops[next_number] = keep
        meta[next_number] = {
            "line_type_number": next_number,
            "line_type_id": cluster.global_type_id,
            "type_uid": "m1:" + cluster.global_type_id,
            "signature_family": cluster.signature_family,
            "recognition_source": "method1",
            "minimum_pair_similarity": cluster.minimum_pair_similarity,
            "member_count": len(cluster.members),
            "recovered_from_fusion": True,
            "op_count_in_method1": len(cluster.op_indices),
        }
        recovered.append(next_number)

    ir_geometry = {}
    for op_index, operation in enumerate(operations):
        lines = operation_lines(operation)
        if lines:
            ir_geometry[op_index] = lines
    run_of = {}
    for number, indices in cluster_ops.items():
        for op_index, run_id in _connected_runs(
                sorted(indices), ir_geometry).items():
            run_of[op_index] = str(run_id)

    try:
        pattern_instances = _method2_pattern_instances(
            recognition, payload, to_page_frame)
    except Exception as error:                                  # noqa: BLE001
        _fail("PATTERN_INSTANCE_ERROR", f"{type(error).__name__}: {error}")

    try:
        templates = [extract_template(
            page, sample, sample_index, to_page_frame)
            for sample_index, sample in enumerate(samples)]
    except Exception as error:                                  # noqa: BLE001
        _fail("TEMPLATE_ERROR", f"{type(error).__name__}: {error}")
    sample_boxes = [template.box_2d for template in templates]

    matches = []
    audits = []
    for template in templates:
        try:
            match = associate_template(
                template,
                pattern_instances=pattern_instances,
                cluster_ops=cluster_ops,
                owner=owner,
                operations=operations,
                run_of=run_of,
                sample_boxes=sample_boxes,
                to_page_frame=to_page_frame,
                group_of=group_of,
            )
        except Exception as error:                              # noqa: BLE001
            # One malformed swatch must not erase the other confirmed samples,
            # but it remains explicit in the successful page audit.
            match = {"status": "error",
                     "reason": f"{type(error).__name__}: {error}"}
        audit = template.audit()
        audit["match"] = match
        audits.append(audit)
        if match.get("status") == "matched":
            matches.append((template, match))

    # One semantic legend type may span several engine clusters (for example a
    # short compatible run split from a longer compound profile).  Publish a
    # supervised union under the primary cluster number, while prefixing run
    # ids with their source number so geometry can never collide.
    semantic = {}
    bindings = []
    for template, match in matches:
        primary = int(match["primary_line_type_number"])
        bucket = semantic.setdefault(primary, {
            "source_numbers": set(), "run_pairs": set(),
            "templates": [], "tips": [],
        })
        bucket["source_numbers"].update(
            int(number) for number in match["matched_line_type_numbers"])
        prefixed_runs = []
        for raw_number, run_ids in (
                match.get("matched_runs_by_line_type") or {}).items():
            number = int(raw_number)
            for run_id in run_ids:
                pair = number, str(run_id)
                bucket["run_pairs"].add(pair)
                prefixed_runs.append(f"lt{number}:r{run_id}")
        tips = _dedupe_points(match.get("tips") or ())
        bucket["tips"].extend(tips)
        bucket["templates"].append({
            "sample_index": template.sample_index,
            "symbol_index": template.symbol_index,
            "box_2d": list(template.box_2d),
            "match_kind": match.get("match_kind"),
            "matched_line_type_numbers": match["matched_line_type_numbers"],
        })
        first_tip = (tips[0] if tips else
                     [(template.box_2d[0] + template.box_2d[2]) / 2,
                      (template.box_2d[1] + template.box_2d[3]) / 2])
        first_run = prefixed_runs[0] if prefixed_runs else None
        nearest = {"distance": 0.0, "owner": primary,
                   "run_id": first_run, "nearest_point": first_tip}
        bindings.append({
            "source": "legend_template",
            "key": f"s{template.symbol_index}:0",
            "ti": 0,
            "tip": first_tip,
            "tips": tips,
            "sample_box": list(template.box_2d),
            "matched_runs": prefixed_runs,
            "matched_line_type_numbers": match["matched_line_type_numbers"],
            "nearest_op": dict(nearest),
            "nearest_owned_op": dict(nearest),
            "ranked": [{"line_type_number": primary, "distance": 0.0}],
        })

    line_types = []
    for primary, supervised in sorted(semantic.items()):
        source_numbers = sorted(supervised["source_numbers"])
        run_pairs = sorted(supervised["run_pairs"])
        by_run = {}
        used_ops = set()
        groups = set()
        all_page_lines = []
        source_pattern_instances = []
        for number, run_id in run_pairs:
            indices = [op_index for op_index in cluster_ops[number]
                       if str(run_of.get(op_index, "1")) == run_id]
            used_ops.update(indices)
            groups.update(group_of[op_index] for op_index in indices
                          if op_index in group_of)
            lines = [line for op_index in indices
                     for line in ir_geometry.get(op_index, ())]
            page_lines = [[to_page_frame(x, y) for x, y in line]
                          for line in lines]
            all_page_lines.extend(page_lines)
            prefixed = f"lt{number}:r{run_id}"
            by_run[prefixed] = {
                "run_id": prefixed,
                "source_line_type_number": number,
                "source_run_id": run_id,
                "op_count": len(indices),
                "segment_count": sum(max(0, len(line) - 1)
                                     for line in page_lines),
                "bbox": _bbox(page_lines),
                "polylines": page_lines,
            }
        for number in source_numbers:
            source_pattern_instances.extend(
                pattern_instances_outside_samples(
                    pattern_instances.get(number) or (), sample_boxes))
        base = meta[primary]
        sources = {str(number): meta[number]["recognition_source"]
                   for number in source_numbers}
        line_types.append({
            "line_type_number": primary,
            "line_type_id": "legend:" + str(base["line_type_id"]),
            "type_uid": "legend:" + str(base["type_uid"]),
            "signature_family": base["signature_family"],
            "recognition_source": "legend_template",
            "base_recognition_source": base["recognition_source"],
            "matched_cluster_sources": sources,
            "base_line_type_number": primary,
            "matched_line_type_numbers": source_numbers,
            "minimum_pair_similarity": min(
                float(meta[number].get("minimum_pair_similarity") or 0.0)
                for number in source_numbers),
            "member_count": sum(int(meta[number].get("member_count") or 0)
                                for number in source_numbers),
            "op_count": len(used_ops),
            "range_count": len(used_ops),
            "segment_count": sum(max(0, len(line) - 1)
                                 for line in all_page_lines),
            "ops_sha1": _sha1(used_ops),
            "bbox": _bbox(all_page_lines),
            "groups": sorted(groups),
            "pattern_instance_count": len(source_pattern_instances),
            "pattern_instances": source_pattern_instances,
            "legend_samples": supervised["templates"],
            "by_run": by_run,
        })

    engine_stamp = {
        "page_ir": PAGE_IR_VERSION,
        "source_page_adapter": SOURCE_PAGE_ADAPTER_VERSION,
        "clustering_schema": CLUSTERING_API_SCHEMA_VERSION,
        "engine": PYTHON_ENGINE_VERSION,
        "producer": page.producer,
        "pymupdf": pymupdf.__version__,
        "pypdf": pypdf.__version__,
        "scipy": scipy.__version__,
        "cpu_budget": 1,
    }
    all_line_types = [{key: value for key, value in row.items()
                       if key != "by_run"} for row in line_types]
    json.dump({
        "ok": True,
        "engine": engine_stamp,
        "page": {
            "sheet": sheet,
            "ops": payload["operation_count"],
            "path_ops": len(ir_geometry),
            "rotation": page.rotation_degrees,
            "groups": payload["group_count"],
            "base_line_types": len(payload["global_line_types"]) + len(recovered),
            "legend_samples": len(templates),
            "legend_matches": len(matches),
            "legend_semantic_types": len(line_types),
            "page_fingerprint": payload["page_fingerprint"],
            "tip_precision_pt": 0.0,
            "seconds_ir": round(seconds_ir, 2),
            "seconds_cluster": round(seconds_cluster, 2),
            "errors": payload["errors"],
        },
        "line_types": line_types,
        "all_line_types": all_line_types,
        "bindings": bindings,
        "samples": audits,
    }, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
