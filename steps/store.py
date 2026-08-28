"""data/<slug>/ 磁盘布局与读写 —— 全项目唯一的存储契约.

每个项目一个目录，源 PDF 在 projects/<slug>/input.pdf：

  vec.json         PDF 原生文字层（排版行级），schema 3，键 pdf_mtime
  textjudge.json   判词缓存 {v, model, verdicts: {NORM_STRING: bool}}
  vlm.json         整页图 VLM 原始响应 {p: {items, elapsed, usage, model,
                     vlm_identity: {pdf_revision, model, prompt_sha256}}}
  vlm_flash.json   无文字层扫描页的第二模型原始响应（同结构，独立文件；
                     绝不把双模型 union 伪装成一条 raw）
  results.json     步骤1 融合结果
                     {slug, fused_v, pdf_revision, page_count, no_text_layer,
                      judge_error, mode, target, generated,
                      pages: {p: rec}, llm_summary, wall_seconds}
                     rec = {vlm_items, vec_added, vec_covered,
                            has_text, vlm_error[, codes_stripped, debug]}
  symbols.json     步骤2 图例样例符号 + 步骤4 放置结果
                     {p: {sig, v, pv, model, raw, result}}
                     result = {symbols, groups[, plc_v, placement_note]}
  view_types.json  步骤3 视图投影分类 {p: {sig, v, model, views}}
  base_P<p>_<pdf_revision>.jpg   前端底图（600dpi → 长边 5000, JPEG q80）

两条缓存主键：
  pdf_revision = f"{size:x}-{mtime_ns:x}"  —— 换机/scp 会变，见 tools/import_project.py
  sig_of(items, revision)                  —— 只签 (text, box_2d)，
                                              label/tbl 变化不作废付费 raw
"""
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path

from core.config import BASE_DIR, PROJECTS_DIR

DATA_DIR = BASE_DIR / "data"
JOBS_DIR = BASE_DIR / "_jobs"
DATA_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def is_valid_slug(slug):
    """项目标识必须是单个安全路径段，永远不是路径。"""
    return isinstance(slug, str) and bool(_SLUG_RE.fullmatch(slug))


def require_slug(slug):
    if not is_valid_slug(slug):
        raise ValueError("invalid project slug")
    return slug


def slug_dir(slug):
    d = DATA_DIR / require_slug(slug)
    d.mkdir(exist_ok=True)
    return d


def pdf_path(slug):
    return PROJECTS_DIR / require_slug(slug) / "input.pdf"


def pdf_revision(path):
    """缓存键用的廉价文档身份（字节数 + 纳秒 mtime）。"""
    stat = Path(path).stat()
    return f"{stat.st_size:x}-{stat.st_mtime_ns:x}"


def load_json(path, default):
    path = Path(path)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return default


def save_json(path, obj):
    """单文件原子替换（同目录唯一临时文件 + replace）。

    只保证「这一次写入」的文件完整，不提供跨文件或读-改-写事务；
    {page: entry} 这种大字典的并发更新由调用方用单写者锁保证。
    """
    path = Path(path)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp",
                delete=False) as handle:
            tmp = Path(handle.name)
            json.dump(obj, handle, ensure_ascii=False, indent=1)
        for attempt in range(5):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                # Windows 上杀软可能短暂占锁；重试而不是失败。
                if attempt == 4:
                    raise
                time.sleep(0.3 * (attempt + 1))
    finally:
        # 成功时 replace 已经移走临时文件；失败路径尽力删除本次的临时文件，
        # 且清理失败绝不能掩盖原始异常。
        if tmp is not None:
            try:
                tmp.unlink()
            except OSError:
                pass


def results_path(slug):
    return slug_dir(slug) / "results.json"


def load_results(slug):
    return load_json(DATA_DIR / require_slug(slug) / "results.json", None)


def items_of(rec):
    """下游（符号步 / 未来的箭头步 / 前端）看到的统一 item 视图.

    顺序契约：vlm_items 在前、vec_added 紧随其后，下标即全栈公共 union index
    —— symbol 的 text_index、前端选中态、后续箭头模块的 idx 全锚在这里。
    调整拼接顺序或往中间插项 = 静默错位所有归属。
    没有 box_2d 的条目被丢弃（画不出来也没法当锚）。
    """
    return [{"text": it.get("text", ""), "box_2d": it["box_2d"],
             "label": it.get("label", ""), "tbl": bool(it.get("tbl"))}
            for it in rec.get("vlm_items", []) + rec.get("vec_added", [])
            if it.get("box_2d")]


def sig_of(items, revision=None):
    """付费缓存签名：只取 (text, box_2d)。

    label / tbl 这类元数据变化不得作废已付费的 VLM raw。
    """
    core = [{"text": it["text"], "box_2d": it["box_2d"]} for it in items]
    payload = core if revision is None else {"items": core, "pdf": revision}
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
