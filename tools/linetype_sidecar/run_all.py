"""调试用边车 —— 把这一页**全部**线型的几何吐出来，而不只是被指到的那几个.

为什么要单独一个入口，而不是给 run.py 加个开关：

  ``steps/linetypes/sidecar.py`` 的 ``engine_digest()`` 只哈希两样东西 ——
  vendored 的 ``engine/`` 源码树，和 ``run.py`` 的字节。这个摘要进缓存签名，
  所以**动一个字节的 run.py，盘上 54 页结果全部立刻变 stale**，整站空白等
  重跑。本文件不在摘要里，加它一行都不影响既有缓存的当期性。

  反过来说，这也意味着**本文件里的任何改动都不会自动作废任何缓存**。所以它
  只许产出「调试视图」这一种旁路产物，绝不许参与主结果的判定 —— 一旦让它
  影响主链路，就是一个静默的正确性缺口。

它算什么：和 run.py 完全同一条确定性流程（同一个提取器、同一次
``recognize_page``、同一套 fusion 连带补回规则），因此**线型编号天然一致**。
不过一致性不靠"我相信它一致"：输出里每个类型都带 ``ops_sha1``，调用方按
sha1 和主缓存的 ``all_line_types`` 对表认领编号，对不上的显式报出来。
``verify_all_geometry.py`` 就是逐行做这件事的。

为什么需要这个视图：一个 callout 没绑上线型时，正常视图里分不出两种完全不同
的原因 —— (a) 那条线**根本没被聚成线型**，(b) 聚出来了、但离末端更远因此
没被选中。实测 rapid_city_2 P11 的 callout ② 属于 (b)：方块 pattern 线是
``#48``（518 op / 1895 段），但末端落在 hatch 带里，最近的 ink 是 hatch 的短划
（0.26 pt），``#48`` 在 3.4 pt 之外。截图上这两种原因看起来一模一样。

所以输出里除了全部线型，还有 **residual**（不属于任何线型的 path ink）。有了
它，"这里根本没聚出线型"就能当场看见，而不是靠推断。

协议：stdin 一个 JSON job，stdout 一个 JSON。失败一律
``{"ok": false, "code": ..., "error": ...}`` + 非零退出码。

    job = {"pdf": str, "sheet": int(1-based), "cpu_budget": int,
           "residual": bool}          # residual 默认带上
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# 复用 run.py 的辅助函数 —— 走线切分、bbox、失败协议都必须和主入口同源。
# 只 import，不改它：它的字节进缓存签名。
import run as _run                                              # noqa: E402

_ENGINE = _HERE / "engine"
_fail = _run._fail
_bbox = _run._bbox
_connected_runs = _run._connected_runs
_method2_pattern_instances = _run._method2_pattern_instances


def main():
    try:
        job = json.load(sys.stdin)
    except Exception as error:                                  # noqa: BLE001
        _fail("BAD_JOB", f"cannot parse job JSON: {error}")

    pdf = str(job.get("pdf") or "")
    sheet = job.get("sheet")
    if not pdf or not Path(pdf).is_file():
        _fail("BAD_JOB", f"pdf not found: {pdf!r}")
    if not isinstance(sheet, int) or isinstance(sheet, bool) or sheet < 1:
        _fail("BAD_JOB", f"sheet must be a 1-based int, got {sheet!r}")
    cpu_budget = int(job.get("cpu_budget") or 1)
    want_residual = job.get("residual", True) is not False

    if not _ENGINE.is_dir():
        _fail("ENGINE_MISSING", f"engine source not found at {_ENGINE}")
    sys.path.insert(0, str(_ENGINE))

    try:
        import pymupdf
        import pypdf
        from line_type_engine.clustering_api import (
            CLUSTERING_API_SCHEMA_VERSION, project_line_type_clusters)
        from line_type_engine.engine import recognize_page
        from line_type_engine.source_page_adapter import (
            SOURCE_PAGE_ADAPTER_VERSION, source_aligned_page_ir_from_pdf_path)
        from line_type_engine.versions import (
            PAGE_IR_VERSION, PYTHON_ENGINE_VERSION)
    except Exception as error:                                  # noqa: BLE001
        _fail("IMPORT_ERROR", f"{type(error).__name__}: {error}")
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:                                           # noqa: BLE001
        scipy_version = None

    try:
        started = time.time()
        aligned = source_aligned_page_ir_from_pdf_path(pdf, sheet)
        page = aligned.page
        seconds_ir = time.time() - started
    except Exception as error:                                  # noqa: BLE001
        _fail("PAGE_IR_ERROR", f"{type(error).__name__}: {error}")

    engine_stamp = {
        "page_ir": PAGE_IR_VERSION,
        "source_page_adapter": SOURCE_PAGE_ADAPTER_VERSION,
        "clustering_schema": CLUSTERING_API_SCHEMA_VERSION,
        "engine": PYTHON_ENGINE_VERSION,
        "producer": page.producer,
        "pymupdf": pymupdf.__version__,
        "pypdf": pypdf.__version__,
        "scipy": scipy_version,
        "cpu_budget": cpu_budget,
    }

    # 转帧的量取自**另一个**新开的文档：引擎的抽取路径会临时改 /Rotate 且异常时
    # 不恢复（PyMuPDF 那几个 get_* 没有 try/finally），绝不能复用它碰过的 page。
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
              f"{page.rotation_degrees} — refusing to guess the frame")

    bounds = page.page_bounds
    unrotated_height = bounds.max_y - bounds.min_y
    unrotated_width = bounds.max_x - bounds.min_x
    expected = ((rotated_height, rotated_width)
                if page.rotation_degrees % 180 else (rotated_width, rotated_height))
    if (abs(unrotated_width - expected[0]) > 1e-6
            or abs(unrotated_height - expected[1]) > 1e-6):
        _fail("PAGE_FRAME_ERROR",
              f"page_bounds {unrotated_width}x{unrotated_height} does not match "
              f"rotated rect {rotated_width}x{rotated_height} "
              f"(rot={page.rotation_degrees}) — CropBox offset or UserUnit != 1")

    def to_page_frame(x, y):
        point = pymupdf.Point(x + bounds.min_x,
                              unrotated_height - y + bounds.min_y) * rotation_matrix
        return [round(point.y / rotated_height * 1000, 2),
                round(point.x / rotated_width * 1000, 2)]

    try:
        started = time.time()
        recognition = recognize_page(
            page, outputs=("method1", "method2", "fused"),
            method1_worker_count=cpu_budget, parallel_methods=True)
        if recognition.fused is None:
            raise RuntimeError("fused recognition did not produce a result")
        result = project_line_type_clusters(page, recognition.fused.result)
        seconds_cluster = time.time() - started
    except Exception as error:                                  # noqa: BLE001
        _fail("CLUSTER_ERROR", f"{type(error).__name__}: {error}")

    payload = result.to_dict()
    operations = page.operations

    def op_ir_lines(operation):
        """一条 op → IR 帧折线列表。与 run.py 逐字同源（曲线按控制点折线化）。"""
        lines, current = [], []
        for segment in getattr(operation, "segments", ()) or ():
            if segment.kind == "move":
                if len(current) > 1:
                    lines.append(current)
                current = [tuple(segment.end)]
            elif segment.kind == "line":
                current.append(tuple(segment.end))
            elif segment.kind == "curve":
                current.append(tuple(segment.control_1))
                current.append(tuple(segment.control_2))
                current.append(tuple(segment.end))
            elif segment.kind == "close" and len(current) > 1:
                current.append(current[0])
        if len(current) > 1:
            lines.append(current)
        return lines

    group_of = {}
    for grp in payload["groups"]:
        gid = str(grp["group_id"])
        for local in grp["line_types"]:
            for op_index in local["commands"]["op_indices"]:
                group_of[op_index] = gid
        for op_index in grp["residual_vector_commands"]["op_indices"]:
            group_of.setdefault(op_index, gid)

    owner = {}
    for cluster in payload["global_line_types"]:
        number = int(cluster["line_type_number"])
        for op_index in cluster["commands"]["op_indices"]:
            if not 0 <= op_index < len(operations):
                _fail("OP_INDEX_ERROR",
                      f"op_index {op_index} out of range (ops={len(operations)})")
            owner[op_index] = number

    # ---- 补回被 fusion 连带丢弃的 method1 覆盖（判据与 run.py 完全相同）----
    def _global_types(envelope):
        if envelope is None:
            return ()
        return getattr(envelope.result, "global_types", ()) or ()

    method2_ops = set()
    for cluster in _global_types(recognition.method2):
        method2_ops.update(cluster.op_indices)

    recovered = []
    next_number = max([int(c["line_type_number"])
                       for c in payload["global_line_types"]] or [0])
    for cluster in _global_types(recognition.method1):
        keep = [i for i in cluster.op_indices
                if i not in owner and i not in method2_ops
                and 0 <= i < len(operations)]
        if not keep:
            continue
        next_number += 1
        for op_index in keep:
            owner[op_index] = next_number
        recovered.append({
            "line_type_number": next_number,
            "source_global_type_id": cluster.global_type_id,
            "signature_family": cluster.signature_family,
            "minimum_pair_similarity": cluster.minimum_pair_similarity,
            "member_count": len(cluster.members),
            "op_indices": keep,
            "op_count_in_method1": len(cluster.op_indices),
        })

    ir_geometry = {}
    for op_index, operation in enumerate(operations):
        lines = op_ir_lines(operation)
        if lines:
            ir_geometry[op_index] = lines

    def clipped(op_index):
        operation = operations[op_index]
        box = getattr(operation, "bounds", None)
        lines = ir_geometry.get(op_index) or ()
        if box is None or not lines:
            return False
        slack = max(float(getattr(operation, "line_width", 0.0) or 0.0) / 2.0,
                    0.25) + 1e-6
        for line in lines:
            for x, y in line:
                if (x < box.min_x - slack or x > box.max_x + slack
                        or y < box.min_y - slack or y > box.max_y + slack):
                    return True
        return False

    ops_by_cluster = {}
    for op_index, number in owner.items():
        ops_by_cluster.setdefault(number, []).append(op_index)

    run_of = {}
    for number, ops in ops_by_cluster.items():
        for op_index, run_id in _connected_runs(sorted(ops), ir_geometry).items():
            run_of[op_index] = run_id

    ir_by_cluster = {}
    ir_by_cluster_run = {}
    clipped_by_cluster = {}
    for op_index, number in owner.items():
        lines = ir_geometry.get(op_index) or ()
        ir_by_cluster.setdefault(number, []).extend(lines)
        rid = str(run_of.get(op_index, 1))
        ir_by_cluster_run.setdefault(number, {}).setdefault(rid, []).extend(lines)
        if clipped(op_index):
            clipped_by_cluster[number] = clipped_by_cluster.get(number, 0) + 1

    try:
        pattern_instances_by_number = _method2_pattern_instances(
            recognition, payload, to_page_frame
        )
    except Exception as error:                                   # noqa: BLE001
        _fail("PATTERN_INSTANCE_ERROR", f"{type(error).__name__}: {error}")

    run_counts = {}
    for op_index, number in owner.items():
        key = (number, str(run_of.get(op_index, 1)))
        run_counts[key] = run_counts.get(key, 0) + 1

    def sha1_of(indices):
        return hashlib.sha1(
            ",".join(str(i) for i in sorted(indices)).encode()).hexdigest()

    def by_run_of(number):
        buckets = []
        for rid, lines in sorted((ir_by_cluster_run.get(number) or {}).items(),
                                 key=lambda kv: int(kv[0])):
            polylines = [[to_page_frame(x, y) for x, y in line] for line in lines]
            buckets.append({
                "run_id": rid,
                "op_count": run_counts.get((number, rid), 0),
                "segment_count": sum(max(0, len(line) - 1) for line in polylines),
                "bbox": _bbox(polylines),
                "polylines": polylines,
            })
        return buckets

    types = []
    for cluster in payload["global_line_types"]:
        number = int(cluster["line_type_number"])
        page_lines = [[to_page_frame(x, y) for x, y in line]
                      for line in ir_by_cluster.get(number) or ()]
        pattern_instances = pattern_instances_by_number.get(number) or []
        types.append({
            "line_type_number": number,
            "line_type_id": cluster["line_type_id"],
            "type_uid": cluster["type_uid"],
            "signature_family": cluster["signature_family"],
            "recognition_source": cluster["recognition_source"],
            "minimum_pair_similarity": cluster["minimum_pair_similarity"],
            "member_count": len(cluster["members"]),
            "op_count": len(cluster["commands"]["op_indices"]),
            "range_count": len(cluster["commands"]["ranges"]),
            "segment_count": sum(max(0, len(line) - 1) for line in page_lines),
            "ops_sha1": sha1_of(cluster["commands"]["op_indices"]),
            "bbox": _bbox(page_lines),
            "clipped_ops": clipped_by_cluster.get(number, 0),
            "groups": sorted({group_of[i] for i in
                              cluster["commands"]["op_indices"]
                              if i in group_of}),
            "pattern_instance_count": len(pattern_instances),
            "pattern_instances": pattern_instances,
            "by_run": by_run_of(number),
        })
    for row in recovered:
        number = row["line_type_number"]
        page_lines = [[to_page_frame(x, y) for x, y in line]
                      for line in ir_by_cluster.get(number) or ()]
        types.append({
            "line_type_number": number,
            "line_type_id": row["source_global_type_id"],
            "type_uid": "m1:" + row["source_global_type_id"],
            "signature_family": row["signature_family"],
            "recognition_source": "method1",
            "minimum_pair_similarity": row["minimum_pair_similarity"],
            "member_count": row["member_count"],
            "op_count": len(row["op_indices"]),
            "range_count": len(row["op_indices"]),
            "segment_count": sum(max(0, len(line) - 1) for line in page_lines),
            "ops_sha1": sha1_of(row["op_indices"]),
            "bbox": _bbox(page_lines),
            "clipped_ops": clipped_by_cluster.get(number, 0),
            "groups": sorted({group_of[i] for i in row["op_indices"]
                              if i in group_of}),
            "pattern_instance_count": 0,
            "pattern_instances": [],
            "recovered_from_fusion": True,
            "op_count_in_method1": row["op_count_in_method1"],
            "by_run": by_run_of(number),
        })

    # residual —— 有 ink 但不属于任何线型的 path op。这是"这里根本没聚出线型"
    # 的直接证据；没有它，(a) 没聚出来 和 (b) 聚出来了没被选中 在图上无法区分。
    residual = None
    if want_residual:
        indices = [i for i in sorted(ir_geometry) if i not in owner]
        polylines = [[to_page_frame(x, y) for x, y in line]
                     for i in indices for line in ir_geometry[i]]
        residual = {
            "op_count": len(indices),
            "segment_count": sum(max(0, len(line) - 1) for line in polylines),
            "bbox": _bbox(polylines),
            "polylines": polylines,
        }

    json.dump({
        "ok": True,
        "engine": engine_stamp,
        "page": {
            "sheet": sheet,
            "ops": payload["operation_count"],
            "path_ops": len(ir_geometry),
            "owned_path_ops": sum(1 for i in ir_geometry if i in owner),
            "rotation": page.rotation_degrees,
            "groups": payload["group_count"],
            "line_types": len(payload["global_line_types"]) + len(recovered),
            "fused_line_types": len(payload["global_line_types"]),
            "recovered_line_types": len(recovered),
            "page_fingerprint": payload["page_fingerprint"],
            "seconds_ir": round(seconds_ir, 2),
            "seconds_cluster": round(seconds_cluster, 2),
            "errors": payload["errors"],
            # 和主缓存 page.owned_ops_sha1 / fused_ops_sha1 逐位对得上，
            # 才能证明这份调试几何和正在显示的结果是同一次聚类的产物。
            "owned_ops_sha1": sha1_of(owner),
            "fused_ops_sha1": sha1_of(
                i for i, n in owner.items()
                if n <= len(payload["global_line_types"])),
        },
        "types": types,
        "residual": residual,
    }, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.flush()


if __name__ == "__main__":
    # 必须的 guard：cpu_budget>=2 时引擎开 spawn 池，子进程会重新 import
    # 本模块（作为 __mp_main__）。没有这一层就是 _check_not_importing_main 死锁。
    main()
