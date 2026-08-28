"""Vendor / verify the line_type_engine source copied into this sidecar.

为什么要 vendor 而不是直接 import 上游那份：上游在 OneDrive 同步目录里
（``…\\OneDrive\\Desktop\\pdf_stream\\pdf-command-visualizer\\line_type_engine``），
路径会被同步客户端改动、也不该成为生产依赖。所以边车用自己目录下的 ``engine/``。

这个脚本同时是「改了算法要能验证」的**基线台账**：``manifest.json`` 记下每个
文件的 sha256。之后不论是从上游重新同步，还是我们自己动了 vendored 那份，
``verify`` 都能逐文件说出差异，而不是只给一句「不一样」。

    python vendor_engine.py sync     # 从上游重新拷一份并重写 manifest
    python vendor_engine.py stamp    # 只为当前 vendored 内容写 manifest
    python vendor_engine.py verify   # 对比 manifest；有差异退出码 1
    python vendor_engine.py diff     # 对比 vendored 与上游，列出文件级差异
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDORED = HERE / "engine" / "line_type_engine"
MANIFEST = HERE / "manifest.json"
UPSTREAM_DEFAULT = Path(
    r"C:\Users\Administrator\OneDrive\Desktop\pdf_stream"
    r"\pdf-command-visualizer\line_type_engine\line_type_engine"
)


def upstream() -> Path:
    return Path(os.environ.get("LINETYPE_ENGINE_UPSTREAM", str(UPSTREAM_DEFAULT)))


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _tree(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        out[path.relative_to(root).as_posix()] = _digest(path)
    return out


def _versions(root: Path) -> dict[str, str]:
    """Pull the engine's own version strings without importing it."""
    out: dict[str, str] = {}
    versions = root / "versions.py"
    if not versions.is_file():
        return out
    for line in versions.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name.isupper() and value and " " not in name:
            out[name] = value
    return out


def stamp() -> int:
    files = _tree(VENDORED)
    if not files:
        print(f"nothing vendored at {VENDORED}")
        return 1
    MANIFEST.write_text(json.dumps({
        "vendored_from": str(upstream()),
        "file_count": len(files),
        "tree_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "engine_versions": _versions(VENDORED),
        "files": files,
    }, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print(f"stamped {len(files)} files -> {MANIFEST.name}")
    return 0


def sync() -> int:
    source = upstream()
    if not source.is_dir():
        print(f"upstream not found: {source}")
        return 1
    if VENDORED.exists():
        shutil.rmtree(VENDORED)
    VENDORED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, VENDORED,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"synced from {source}")
    return stamp()


def _report(left: dict[str, str], right: dict[str, str],
            left_name: str, right_name: str) -> int:
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    changed = sorted(name for name in set(left) & set(right)
                     if left[name] != right[name])
    for name in only_left:
        print(f"  only in {left_name}: {name}")
    for name in only_right:
        print(f"  only in {right_name}: {name}")
    for name in changed:
        print(f"  differs: {name}")
    total = len(only_left) + len(only_right) + len(changed)
    print(f"{total} difference(s); {len(set(left) & set(right)) - len(changed)} identical")
    return 1 if total else 0


def verify() -> int:
    if not MANIFEST.is_file():
        print(f"no manifest at {MANIFEST}; run `stamp` first")
        return 1
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return _report(recorded.get("files") or {}, _tree(VENDORED),
                   "manifest", "vendored")


def diff() -> int:
    return _report(_tree(upstream()), _tree(VENDORED), "upstream", "vendored")


def main(argv: list[str]) -> int:
    action = (argv[1] if len(argv) > 1 else "verify").lower()
    if action == "sync":
        return sync()
    if action == "stamp":
        return stamp()
    if action == "verify":
        return verify()
    if action == "diff":
        return diff()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
