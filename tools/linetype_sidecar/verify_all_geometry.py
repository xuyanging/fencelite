"""通过运行中的 fence_lite 服务补全并验证 All line types 调试几何.

本脚本只负责枚举页面和调用 HTTP API。它**不会**直接启动 ``run_all.py``，也
**不会**直接写 ``.all.json``：生成必须经过服务里的 canonical 路径，由它统一
取得页锁和 heavy-sidecar 容量、逐类型核对 operation-set 指纹，再原子发布。
这样批量回填与网页点击、正常线型阶段并发时不会绕过同一套正确性和并发约束。

每页先 GET ``/api/linetypes_all/<slug>/<sheet>``。已经是 ``ok`` 就只读复用；
否则 POST 同一接口，请服务按需生成。POST 的默认超时为 3900 秒，覆盖 3600 秒
超密页边车上限和少量序列化余量；可用 ``FENCE_LITE_ALL_HTTP_TIMEOUT`` 调整。
服务基址默认 ``http://127.0.0.1:5051``，可用 ``FENCE_LITE_URL`` 覆盖。

    python verify_all_geometry.py <slug> <sheet> [<sheet> ...]
    python verify_all_geometry.py --all            # 所有已有主缓存的页
    python verify_all_geometry.py --missing        # 补服务判定 missing/stale 的页
    python verify_all_geometry.py --missing <slug> [<slug> ...]   # 限定项目

运行前必须先启动 fence_lite 服务。

退出码：0 = 每个枚举页均返回 ok；1 = 有页面仍非 ok；2 = 用法错误。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
FL = HERE.parent.parent
BASE_URL = os.environ.get(
    "FENCE_LITE_URL", "http://127.0.0.1:5051").rstrip("/")
HTTP_TIMEOUT = float(os.environ.get(
    "FENCE_LITE_ALL_HTTP_TIMEOUT", "3900"))


def api_url(slug, sheet):
    return (f"{BASE_URL}/api/linetypes_all/"
            f"{quote(str(slug), safe='')}/{int(sheet)}")


def request_json(slug, sheet, method="GET"):
    """Return ``(http_status, object, error)`` without leaking a traceback."""
    request = Request(
        api_url(slug, sheet), method=str(method).upper(),
        headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT) as response:
            status = int(response.status)
            raw = response.read()
    except HTTPError as error:
        status = int(error.code)
        raw = error.read()
    except (URLError, TimeoutError, OSError) as error:
        return 0, None, f"{type(error).__name__}: {error}"
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return status, None, f"invalid JSON response: {error}"
    if not isinstance(body, dict):
        return status, None, "JSON response is not an object"
    return status, body, None


def result_summary(body):
    types = body.get("types") or []
    residual = body.get("residual") or {}
    return (f"{len(types)} 型, residual {residual.get('op_count', 0)} op / "
            f"{residual.get('segment_count', 0)} 段")


def do_page(slug, sheet):
    started = time.time()
    status, body, error = request_json(slug, sheet, "GET")
    if error:
        print(f"  ERR  {slug} P{sheet}: GET {error}")
        return 1
    if status == 200 and body.get("state") == "ok":
        print(f"  OK   {slug} P{sheet}: 已验证缓存, {result_summary(body)}, "
              f"{time.time() - started:.1f}s")
        return 0

    prior = body.get("state") or body.get("error") or f"HTTP {status}"
    print(f"  BUILD {slug} P{sheet}: GET={prior}", flush=True)
    status, body, error = request_json(slug, sheet, "POST")
    wall = time.time() - started
    if error:
        print(f"  ERR  {slug} P{sheet}: POST {error} ({wall:.1f}s)")
        return 1
    if status != 200 or body.get("state") != "ok":
        detail = body.get("error") or body.get("detail") \
            or body.get("state") or f"HTTP {status}"
        print(f"  ERR  {slug} P{sheet}: POST {detail} ({wall:.1f}s)")
        return 1
    print(f"  OK   {slug} P{sheet}: 服务生成并验证, {result_summary(body)}, "
          f"{wall:.1f}s")
    return 0


def pages_with_cache(slugs=()):
    """有主缓存的页。slugs 非空时只看这些项目 —— 补全量时按项目切分成几个
    并行进程，taylor 一家就占累计耗时的 68%，不切的话它会拖着整批。"""
    out = []
    root = FL / "data"
    if not root.is_dir():
        return out
    wanted = {str(s) for s in slugs}
    for data_dir in sorted(root.iterdir()):
        if wanted and data_dir.name not in wanted:
            continue
        directory = data_dir / "linetypes"
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json"),
                           key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            if not path.stem.isdigit():
                continue
            sheet = int(path.stem)
            out.append((data_dir.name, sheet))
    return out


def main(argv):
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] in ("--all", "--missing"):
        # missing/stale 必须由服务的完整校验判定，不能以本地 .all.json 是否
        # 存在代替；旧生产者或损坏文件虽然存在，也必须 GET 后自动 POST 修复。
        pairs = pages_with_cache(slugs=args[1:])
    elif len(args) >= 2:
        pairs = [(args[0], int(s)) for s in args[1:]]
    else:
        print(__doc__)
        return 2
    if not pairs:
        print("  没有要处理的页")
        return 0
    print(f"  {len(pairs)} 页, service={BASE_URL}, "
          f"timeout={HTTP_TIMEOUT:g}s", flush=True)
    failures = 0
    for slug, sheet in pairs:
        failures += do_page(slug, sheet)
        sys.stdout.flush()
    verdict = "ALL GEOMETRY OK" if not failures else "ALL GEOMETRY FAILED"
    print(f"\n{verdict}: {len(pairs) - failures}/{len(pairs)} 页返回 ok")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
