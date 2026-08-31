"""步骤②b：图例块裁剪补扫 —— 把图例块单独裁出来放大再问一次样例图形.

为什么要有这一步（步骤② 明明已经问过一次了）：
步骤② 是**一次全页图推理** —— 144 DPI、长边上限 5000px，42"×30" 的图纸实际
只有 ~119 DPI，还要在这张整页图里找几十像素高的图例样例。实测后果有三：
  * 漏检：ponderosa P2 整页那次 raw symbols = 0，两行落在 schedule 里的 fence
    文字一个符号都没配到（同一页更早的缓存曾找到过 shape '4.0' —— 说明是模型
    召回不稳，不是过滤误杀）；
  * 框不准：taylor P3 的 line 样例只框住一小段（宽 39/1000 页宽）；
  * 图例块越小、离页边越远，上面两件事越容易发生。
本步只做**补扫**：步骤② 的那次全页推理照旧（它负责切分区、也尽力找符号，
提示词一个字都不许动，否则每页重新付费），补扫只挑「组里还有 fence 文字没
配到符号」的 legend / schedule / note_cluster 块，把这一块裁出来、按高 DPI
重渲染、只问这一小块里的那几行。

为什么裁剪块上样例会大一个量级（两件事叠加，不只是 DPI）：
  1. 渲染更细：整页 144 DPI / 5000px → 本步 SWEEP_DPI(300) / SWEEP_PAGE_MAX_PX;
  2. **更关键**：模型的视觉编码器对一张大图会先降采样到固定 token 预算，
     整页 5000px 图进去只剩千把像素；而裁剪块本身就只有几百到几千像素，
     是**原样**送进去的 —— 同一个图例块在模型眼里放大了好几倍。
所以 SWEEP_MAX_PX 是**裁剪块**的长边上限（省 token），SWEEP_PAGE_MAX_PX 才是
整页渲染的上限。两者不是一个东西：如果拿 4000 去限整页，42" 图纸只有 ~95 DPI，
比步骤② 的整页图还糊，这一步就变成负收益了。

坐标契约（照 README 第 8 节第 2 条）：
  * 送给模型的行框、模型返回的框，都是**这张裁剪图**的 0-1000 帧；
  * 本模块返回的 symbol 框一律换回**页面帧** 0-1000（crop_box_to_page），
    调用方拿到的东西与步骤② 的 symbol 同帧、同结构
    {text_index, box_2d, category, value, group_index, type}，多一个溯源用的
    block_index（是哪一块补扫出来的）。

失败姿态：**sweep_page 不抛异常**。某一块三次全败、裁剪块太小、整页渲染失败，
都只是往返回值的 errors / skipped 里记一条 —— 步骤② 的结果还在，补扫是纯增益，
绝不能因为补扫失败把整页拖垮。同理，被 FL_SWEEP_MAX_BLOCKS 上限丢掉的块也一定
留痕，不许静默截断。

付费：每个待补扫的块一次调用（校验失败最多重试到 SWEEP_ATTEMPTS 次），
全部走 core.gemini.gen_json —— 费用自动进 RECORDER，不必改任何计量代码。
"""
import json
import os
import re
import time

from google.genai import types as _genai_types
from PIL import Image

from core.config import (HIRES_DPI, MAX_HIRES_PX, MODEL_NAME,
                         get_model_override, resolve_model)
from core.gemini import (_encode_image_for_gemini, gen_json,
                         should_retry_model_error, usage_from_response)
from core.parsing import _coerce_box, parse_json_value
from core.pdfio import render_pdf_page
from steps.prompts import SYMBOL_GROUP_KINDS


# ---------------------------------------------------------------- 版本 / 旋钮

# 语义 / 提示词 / schema 变化时 bump（本步的结果若被缓存，就靠它作废）。
#   2：只是作废了 v1 期间那一版短命的"窄框重问"启发式（见文件下方那段说明）
#      跑出来的结果；补扫本身的口径与 v1 相同 —— 只问「压根没配到样例」的行。
#   3：纯 marker 编码的文字行（"4CL" 这种漏进文字层的）不再被当成待配对的
#      图例描述行（_is_marker_code）—— 否则同框去重刚收敛掉重复，补扫又会
#      给它再配一次同一个 marker。
LEGEND_SWEEP_VERSION = 3


def _env_int(name, default, minimum=1):
    """env → 正整数；没设、设歪、太小都退回默认值（补扫不该因为环境变量崩）。"""
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _env_float(name, default, minimum=0.0):
    try:
        value = float(str(os.environ.get(name, "")).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


# 模型：走 resolve_model 是为了拒绝价目表之外的 id —— 未知 id 会让 RECORDER
# 算不出 USD（compute_cost 返回 None），费用统计静默变成 0。
SWEEP_MODEL = resolve_model(os.environ.get("FL_SWEEP_MODEL") or MODEL_NAME)
SWEEP_DPI = _env_int("FL_SWEEP_DPI", HIRES_DPI)            # 裁剪块的渲染 DPI
SWEEP_MAX_PX = _env_int("FL_SWEEP_MAX_PX", 4000)           # 裁剪块长边上限
SWEEP_PAGE_MAX_PX = _env_int("FL_SWEEP_PAGE_MAX_PX",       # 整页渲染长边上限
                             MAX_HIRES_PX)
SWEEP_PAD = _env_float("FL_SWEEP_PAD", 12.0)               # 组框外扩（0-1000）
SWEEP_MAX_BLOCKS = _env_int("FL_SWEEP_MAX_BLOCKS", 4)      # 每页最多补扫几块
SWEEP_MIN_CROP_PX = 40                                     # 太小的裁剪块无意义
SWEEP_TIMEOUT_MS = 180_000
SWEEP_ATTEMPTS = 3
SWEEP_BACKOFF_S = 1.5

# 归属判定容差：与 steps.symbols.symbol_in_allowed_group 的 ±2 同口径。
CENTER_TOL = 2.0
SWEEP_GROUP_KINDS = SYMBOL_GROUP_KINDS


# ---------------------------------------------------------------- 提示词

# 与步骤② 的 GROUP_SYMBOL_PROMPT **互相独立**：那一份是「整页 + 分区 + 找符号」，
# 改一个字就是每页重新付费；这一份只描述「一块图例的放大图」，可以自由演进
# （代价只是补扫这一步重跑）。两类定义（line / shape）刻意与步骤② 一致，
# 否则两条来源的 category 会打架。
SWEEP_PROMPT = """You are looking at a CROP cut out of ONE page of a construction / civil /
architecture drawing and shown enlarged.  This crop is a single LEGEND / KEY
block, SCHEDULE table, or KEYED-NOTE list — it is not a whole sheet.

Below is a JSON list of the TEXT ROWS inside this crop that still need their
sample graphic.  Each row has an idx, its text, and box_2d =
[ymin, xmin, ymax, xmax] as integers normalized to 0-1000 OF THIS CROPPED
IMAGE ([0,0] = top-left of the crop).

TASK — for each listed row, find the SAMPLE GRAPHIC that belongs to it.  The
sample is usually immediately LEFT of the caption on the same line;
occasionally it sits to the right of the caption, or directly above it.

Classify every sample into exactly one of TWO deliberately broad classes (do
NOT invent a third class):

  "line"  — any LINE-STYLE sample: a segment that demonstrates a line style —
            solid / dashed / dotted / dash-dot / double / ticked, a line
            carrying small squares, circles, X marks, hatch ticks, or letters
            riding on it (-SF-SF-).  If it reads as "what the line looks
            like", it is "line".
  "shape" — a CLOSED outline containing a short number / letter code.  The
            outline may be a circle, triangle, square, rectangle, hexagon,
            diamond or any polygon.  If it reads as a coded marker, it is
            "shape".

VALUE
  - "shape": read the short code inside the outline (e.g. "33", "F-04", "A").
    Beware OCR confusions: C/D/O/Q, E/F/P/R, 0/O, 5/S, 8/B.  If the rows of
    this block run in sequence, use row position as a sanity check.
  - "line": the letters riding on the line if any (e.g. "SF"), else "".

BOXES — tight around the sample graphic ITSELF:
  - "line": box the WHOLE sample segment, from its LEFT END to its RIGHT END.
    Do not box only a short middle piece of it.
  - "shape": box only the closed marker.  Do not swallow the caption text.
  - The given text box may ALREADY contain the marker (on many sheets the code
    is printed at the left end of the text line).  That does not change your
    job: still return that marker's own tight box.  Overlapping the given text
    box is expected and allowed.

NEVER box: plan bubbles, detail / view-title circles, north arrows, dimension
markers, or leader arrows.

If a row genuinely has NO sample graphic (it is plain descriptive text),
return NOTHING for that row.  Missing is better than invented.

OUTPUT: ONLY a JSON object, no prose, no markdown fences:
{"symbols": [{"idx": <one of the given idx values>,
              "box_2d": [ymin, xmin, ymax, xmax],
              "category": "line" | "shape",
              "value": "<the short code, or ''>"}]}
box_2d is in the 0-1000 frame of THIS CROPPED IMAGE.
If this crop has no sample graphic at all, output {"symbols": []}."""


SWEEP_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "symbols": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "box_2d": {"type": "array", "items": {"type": "number"}},
                    "category": {"type": "string",
                                 "enum": ["line", "shape"]},
                    "value": {"type": "string"},
                },
                "required": ["idx", "box_2d", "category", "value"],
            },
        },
    },
    "required": ["symbols"],
}


# ---------------------------------------------------------------- 几何小工具

def _valid_box(box):
    """页面帧 / 裁剪帧通用：4 个非 bool 数、正面积、落在 0-1000 内。"""
    return bool(
        isinstance(box, (list, tuple)) and len(box) == 4
        and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in box)
        and 0 <= box[0] < box[2] <= 1000
        and 0 <= box[1] < box[3] <= 1000
    )


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _covers_center(outer, box, tol=CENTER_TOL):
    cy, cx = _center(box)
    return (outer[0] - tol <= cy <= outer[2] + tol
            and outer[1] - tol <= cx <= outer[3] + tol)


def _grow(lo, hi):
    """保正面积：页面帧比裁剪帧粗得多，四舍五入可能把一个小 marker 压成一条线。"""
    if hi > lo:
        return lo, hi
    if hi < 1000:
        return lo, hi + 1
    return max(0, lo - 1), hi


def crop_window(box, width, height, pad=None):
    """图例组框（页面帧 0-1000）→ (crop_box 页面帧, (cx0, cy0, cw, ch) 像素窗).

    向外扩 SWEEP_PAD 的理由：模型给的组框常常紧贴表格边线，而 line 样例的
    左端、shape 的外框经常正好压在这条线外一两个单位 —— 不扩就把要找的东西
    裁掉了。像素窗是**整页渲染图**坐标系里的窗口，crop_box 是它对应的页面帧
    矩形（调试层直接画它）。
    """
    pad = SWEEP_PAD if pad is None else pad
    y0 = max(0.0, float(box[0]) - pad)
    x0 = max(0.0, float(box[1]) - pad)
    y1 = min(1000.0, float(box[2]) + pad)
    x1 = min(1000.0, float(box[3]) + pad)
    cx0 = max(0, min(width, int(round(x0 / 1000.0 * width))))
    cy0 = max(0, min(height, int(round(y0 / 1000.0 * height))))
    cx1 = max(0, min(width, int(round(x1 / 1000.0 * width))))
    cy1 = max(0, min(height, int(round(y1 / 1000.0 * height))))
    crop_box = [round(y0, 1), round(x0, 1), round(y1, 1), round(x1, 1)]
    return crop_box, (cx0, cy0, cx1 - cx0, cy1 - cy0)


def crop_box_to_page(box, cx0, cy0, cw, ch, width, height):
    """裁剪帧 0-1000 框 → 页面帧 0-1000 整数框（换不出合法框就 None）.

    (cx0, cy0, cw, ch) 是**整页渲染图**像素坐标系里的裁剪窗，width/height 是
    整页渲染图的尺寸。裁剪后为省 token 做的等比缩放**不参与**换算 —— 缩放不
    改变「这张裁剪图的 0-1000」这个相对坐标系。
    """
    if not (_valid_box(box) and cw > 0 and ch > 0 and width > 0 and height > 0):
        return None

    def _y(v):
        return (cy0 + float(v) / 1000.0 * ch) / height * 1000.0

    def _x(v):
        return (cx0 + float(v) / 1000.0 * cw) / width * 1000.0

    y0, y1 = _grow(int(round(_y(box[0]))), int(round(_y(box[2]))))
    x0, x1 = _grow(int(round(_x(box[1]))), int(round(_x(box[3]))))
    out = [y0, x0, y1, x1]
    return out if _valid_box(out) else None


def page_box_to_crop(box, cx0, cy0, cw, ch, width, height):
    """页面帧框 → 裁剪帧 0-1000 整数框（给模型看的文字行框）.

    这里**允许 clamp**：一行长文字可以伸出裁剪窗，截到窗边正是模型看到的样子。
    与 _coerce_box「越界就丢弃、绝不 clamp」不矛盾 —— 那条铁律管的是模型**输出**
    的框（clamp 会在无关的页边造出一个看似合理的框），这里是我们自己算的**输入**。
    """
    if not (_valid_box(box) and cw > 0 and ch > 0 and width > 0 and height > 0):
        return None

    def _y(v):
        return (float(v) / 1000.0 * height - cy0) / ch * 1000.0

    def _x(v):
        return (float(v) / 1000.0 * width - cx0) / cw * 1000.0

    def _clamp(v):
        return max(0, min(1000, int(round(v))))

    y0, y1 = _grow(_clamp(_y(box[0])), _clamp(_y(box[2])))
    x0, x1 = _grow(_clamp(_x(box[1])), _clamp(_x(box[3])))
    out = [y0, x0, y1, x1]
    return out if _valid_box(out) else None


# ---------------------------------------------------------------- 选块（免费）

def _covered_item_indices(symbols, item_count):
    """已经有已发布 symbol 的 item 下标 —— 这些行不用再花钱问一遍。"""
    covered = set()
    for symbol in symbols or []:
        if not isinstance(symbol, dict):
            continue
        text_index = symbol.get("text_index")
        if isinstance(text_index, bool) or not isinstance(text_index, int):
            continue
        if 0 <= text_index < item_count:
            covered.add(text_index)
    return covered


# 试过、并且被真实数据否掉的一条规则，留在这里免得有人再走一遍：
# 「line 样例框窄于『文字左边缘 − 图例块左边缘』的 40% 就算没框全，重问一次」。
# 反例 taylor_3_12 P3：那个图例块 x 690..927 里其实排了**两列**条目
# （EXISTING BUILDINGS/STRUCTURES 的文字在 x 722，NEW CHAIN LINK FENCE 在
# x 839.5），用块左沿当基准等于把隔壁那一列的宽度也算进了"样例区"，于是把一个
# 本来正确的 37 宽的框判成"没框全"。补扫重问后模型给的框几乎一模一样（宽 37 vs
# 39），矢量层也证实样例线本身就那么长。要做这类判定，基准必须是**同一行上文字
# 左侧最近的内容边界**，而不是整块的左沿。


# 纯 marker 编码的"文字行"：图纸上编码本来就压在图例样例里，步骤1 的
# strip_marker_codes 只在能拿到图形证据（编码落在小闭合图形内 / 打断线条）时才
# 剥它，拿不到证据时它就以独立文字项的身份漏进来（实测 drawings_volume_4_binder
# P5 的 "4CL" / "6CL"）。这种行不是图例的**描述行**，它自己就是那个 marker，
# 样例的归属应该落在旁边那条完整描述上 —— 不排除掉的话，同框去重刚把重复收敛
# 掉，补扫又会把它当成"还没配到样例"的行、再配一次同一个 marker。
# 形状口径与 steps/text/clean.py 的 _CODE_RE/_TOKEN_RE 一致，另加"数字在前"的
# 写法（4CL / 6DMP），那正是那两条正则漏掉的形态。
_MARKER_CODE_RE = re.compile(
    r"^(?:[A-Za-z]{0,3}[-.]?\d{1,4}[A-Za-z]?|[A-Za-z]{1,3}|\d{1,4}[A-Za-z]{1,3})$")


def _is_marker_code(text):
    # 刻意**不**折叠内部空白：真正的裸编码是印在标记里的一个词（4CL / F-04），
    # 中间不会有空格。折叠了的话 "ROW 2" 这种正经标签会被当成编码整行丢掉。
    # 注意这比 steps/text/clean.py 的同名判据严：那边正则只负责**提名**候选，
    # 真要删还得拿到图形证据；这边匹配上就直接不问了，所以宁严勿宽。
    word = str(text or "").strip()
    return bool(word) and not any(c.isspace() for c in word) \
        and bool(_MARKER_CODE_RE.match(word))


def sweep_needed(items, groups, symbols, *, max_blocks=None):
    """挑出需要补扫的图例块（纯本地，零成本）.

    返回 [{group_index, kind, box_2d, missing: [item_idx, ...], skipped}]：
      * 只看 kind ∈ legend/schedule/note_cluster 且框合法的组；
      * missing = 几何落在该组内（中心 ±2）、且还没有任何已发布 symbol 的 item；
      * missing 为空的组不返回（省钱）；
      * 一个 item 同时落在多个组里时归给**最小面积**的组 —— 与 fenceline 早年
        `_owner_view` 同口径：大组套小组时，小组更具体，不能让外层大组把内层
        小块的行吃掉；
      * 超出 max_blocks（默认 FL_SWEEP_MAX_BLOCKS=4，防病态页面把钱烧光）的块
        **仍然返回**，只是带上 skipped="max_blocks" —— 不许静默截断。
    列表按 (missing 数降序, group_index 升序) 排，前 max_blocks 个是要跑的。
    调用方判「这页值不值得花钱」：any(not b["skipped"] for b in sweep_needed(...))。
    """
    items = items or []
    groups = groups or []
    limit = SWEEP_MAX_BLOCKS if max_blocks is None else int(max_blocks)
    covered = _covered_item_indices(symbols, len(items))

    legend_groups = [
        (index, group) for index, group in enumerate(groups)
        if isinstance(group, dict) and group.get("kind") in SWEEP_GROUP_KINDS
        and _valid_box(group.get("box_2d"))
    ]
    missing_by_group = {}
    for item_index, item in enumerate(items):
        if item_index in covered:
            continue
        box = (item or {}).get("box_2d")
        if not _valid_box(box):
            continue
        if _is_marker_code((item or {}).get("text")):
            continue
        owners = [(_area(group["box_2d"]), index)
                  for index, group in legend_groups
                  if _covers_center(group["box_2d"], box)]
        if not owners:
            continue
        _owner_area, owner_index = min(owners)
        missing_by_group.setdefault(owner_index, []).append(item_index)

    blocks = [{"group_index": index,
               "kind": groups[index].get("kind"),
               # 拷一份：调用方不该能通过返回值改到 groups 里的框
               "box_2d": list(groups[index]["box_2d"]),
               "missing": indices,
               "skipped": None}
              for index, indices in missing_by_group.items()]
    blocks.sort(key=lambda block: (-len(block["missing"]),
                                   block["group_index"]))
    for block in blocks[max(0, limit):]:
        block["skipped"] = "max_blocks"
    return blocks


# ---------------------------------------------------------------- 严校验

def parse_sweep_payload(text, asked):
    """一次裁剪推理的响应 → [{idx, box_2d(裁剪帧), category, value}]，严校验.

    口径照抄步骤② parse_group_symbol_payload：宁可抛出去重试，也不把
    「半懂的答案」当成「这一块什么都没有」缓存下来。任何一条不合规都会 raise
    （不是丢掉那一条）—— 一条坏行说明这次回答整体不可信。
    """
    payload = parse_json_value(text or "")
    if not isinstance(payload, dict):
        raise RuntimeError("legend sweep response must be a JSON object")
    if "symbols" not in payload:
        raise RuntimeError("legend sweep response missing required key: symbols")
    rows = payload["symbols"]
    if not isinstance(rows, list):
        raise RuntimeError("legend sweep response symbols must be an array")

    allowed = set(asked or ())
    out, seen = [], set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"legend sweep row {row_index} must be an object")
        missing = [key for key in ("idx", "box_2d", "category", "value")
                   if key not in row]
        if missing:
            raise RuntimeError(
                f"legend sweep row {row_index} missing required key(s): "
                + ", ".join(missing))
        idx = row["idx"]
        if isinstance(idx, bool) or not isinstance(idx, int) \
                or idx not in allowed:
            raise RuntimeError(
                f"legend sweep row {row_index} has an invalid idx")
        box = _coerce_box(row["box_2d"])
        if box is None:
            raise RuntimeError(
                f"legend sweep row {row_index} has an invalid box_2d")
        if not isinstance(row["category"], str):
            raise RuntimeError(
                f"legend sweep row {row_index} has an invalid category")
        category = row["category"].strip().lower()
        if category not in ("line", "shape"):
            raise RuntimeError(
                f"legend sweep row {row_index} has an invalid category")
        if not isinstance(row["value"], str):
            raise RuntimeError(
                f"legend sweep row {row_index} has an invalid value")
        value = row["value"].strip()
        # 逐字重复的行是纯噪声（同 idx 同框同类同码），确定性地只留第一条；
        # 同一行真有两个样例（一个 line 一个 shape）时框/类不同，两条都留。
        key = (idx, tuple(box), category, value)
        if key in seen:
            continue
        seen.add(key)
        out.append({"idx": idx, "box_2d": box, "category": category,
                    "value": value})
    return out


# ---------------------------------------------------------------- 付费推理

def _merge_usage(base, extra):
    merged = dict(base or {})
    for key, value in (extra or {}).items():
        try:
            merged[key] = int(merged.get(key, 0)) + int(value)
        except (TypeError, ValueError):
            continue
    return merged


def _retry_suffix(attempt):
    """第 2/3 次调用追加的尾巴.

    nonce 不只是给人看的：Gemini 服务端对**逐字节相同**的请求会返回同一份
    响应（隐式缓存），不改字节的话「重试」拿回来的还是那份坏答案。
    """
    return (
        "\n\nRETRY VALIDATION ATTEMPT " + str(attempt)
        + f" (nonce=sweep-retry-{attempt}). The previous response failed "
        "strict validation. Return ONLY the required JSON object; use only "
        "the listed idx values; category must be exactly \"line\" or "
        "\"shape\"; every box_2d must be [ymin, xmin, ymax, xmax] integers "
        "inside 0-1000 of this crop with ymin < ymax and xmin < xmax."
    )


def _ask_block(crop_image, rows, model, timeout_ms):
    """一块裁剪图 → {rows|None, elapsed, usage, calls, error}（不抛异常）.

    最多 SWEEP_ATTEMPTS 次；失败的那几次同样花了钱，所以 usage / elapsed /
    calls 一律累加进去（RECORDER 那边本来就已经计了，这里是给页级汇总看的）。
    """
    data, mime = _encode_image_for_gemini(crop_image)
    part = _genai_types.Part.from_bytes(data=data, mime_type=mime)
    asked = [row["idx"] for row in rows]
    base_prompt = (SWEEP_PROMPT + "\n\nTEXT ROWS IN THIS CROP:\n"
                   + json.dumps(rows, ensure_ascii=False))
    elapsed, usage, calls, last_error = 0.0, {}, 0, None
    for attempt in range(SWEEP_ATTEMPTS):
        prompt = base_prompt + (_retry_suffix(attempt + 1) if attempt else "")
        started = time.perf_counter()
        try:
            calls += 1
            response = gen_json(model, [part, prompt], timeout_ms=timeout_ms,
                                response_json_schema=SWEEP_RESPONSE_SCHEMA)
            elapsed += time.perf_counter() - started
            usage = _merge_usage(usage, usage_from_response(response))
            parsed = parse_sweep_payload(response.text or "", asked)
            return {"rows": parsed, "elapsed": elapsed, "usage": usage,
                    "calls": calls, "error": None}
        except Exception as exc:                               # noqa: BLE001
            elapsed += time.perf_counter() - started
            last_error = f"{type(exc).__name__}: {exc}"
            if not should_retry_model_error(
                    exc, attempt, SWEEP_ATTEMPTS):
                break
            if attempt + 1 < SWEEP_ATTEMPTS:
                time.sleep(SWEEP_BACKOFF_S * (attempt + 1))
    return {"rows": None, "elapsed": elapsed, "usage": usage, "calls": calls,
            "error": last_error}


def _crop_for_block(page_image, block):
    """裁 + 必要时等比缩 → (crop_image, crop_box, (cx0, cy0, cw, ch)) 或 skip 原因."""
    width, height = page_image.size
    crop_box, window = crop_window(block["box_2d"], width, height)
    cx0, cy0, cw, ch = window
    if min(cw, ch) < SWEEP_MIN_CROP_PX:
        return None, crop_box, window, "crop_too_small"
    crop = page_image.crop((cx0, cy0, cx0 + cw, cy0 + ch))
    long_side = max(crop.size)
    if SWEEP_MAX_PX and long_side > SWEEP_MAX_PX:
        # 只为省 token —— 缩放不改变裁剪帧 0-1000，换算仍用原始像素窗。
        scale = SWEEP_MAX_PX / float(long_side)
        crop = crop.resize((max(1, int(round(crop.width * scale))),
                            max(1, int(round(crop.height * scale)))),
                           Image.LANCZOS)
    return crop, crop_box, window, None


def sweep_page(pdf_path, page_index, items, groups, symbols, *, dbg=None,
               model=None, timeout_ms=SWEEP_TIMEOUT_MS, max_blocks=None):
    """对每个待补扫的图例块各发一次裁剪推理.

    返回（**从不抛异常**）：
      {"version", "model", "calls", "elapsed", "usage",
       "blocks": [{group_index, kind, box_2d, crop_box, crop_px, crop_size,
                   asked: [item_idx...], found: [symbol...], elapsed, usage}],
       "added":   [symbol...],          # 全部块的 found 拍平 + 确定性去重
       "skipped": [{group_index, kind, reason, ...}],
       "errors":  [{group_index, error}]}

    symbol 结构与步骤② 一致：{box_2d, category, value, type, text_index,
    group_index}，坐标是**页面帧** 0-1000，另多一个 block_index（溯源用：
    blocks 里的下标）。这些 symbol 天然满足步骤② 的 owner 闸（text_index 就是
    items 的下标）与组内闸（group_index 是我们自己挑的图例组），但注意：裁剪窗
    是组框**外扩 SWEEP_PAD** 后的框，所以样例可能落在原组框外最多 SWEEP_PAD 个
    单位 —— 发布闸若按「中心落在组框 ±2」判，请对补扫来的 symbol 用
    blocks[].crop_box 当取景框（那才是它真正的几何来源）。
    """
    # Precedence: explicit argument > job-scoped override > SWEEP_MODEL.
    # SWEEP_MODEL is evaluated at import time, so falling back to it directly
    # would pin this stage to the process default and silently keep the sweep
    # on Gemini while the rest of the run switched providers — a mixed-provider
    # run whose results mean nothing. A run-wide override therefore outranks
    # FL_SWEEP_MODEL: one comparison run stays on one model.
    resolved_model = resolve_model(model or get_model_override() or SWEEP_MODEL)
    out = {"version": LEGEND_SWEEP_VERSION, "model": resolved_model,
           "blocks": [], "added": [], "skipped": [], "errors": [],
           "calls": 0, "elapsed": 0.0, "usage": {}}

    def _skip(block, reason, **extra):
        row = {"group_index": block["group_index"], "kind": block.get("kind"),
               "reason": reason, "missing": list(block["missing"])}
        row.update(extra)
        out["skipped"].append(row)
        if dbg is not None:
            dbg.add("legend_sweep",
                    {"group_index": block["group_index"],
                     "kind": block.get("kind"),
                     "crop_box": extra.get("crop_box"),
                     "asked": list(block["missing"]), "found": 0,
                     "skipped": reason, "elapsed": 0.0})

    candidates = sweep_needed(items, groups, symbols, max_blocks=max_blocks)
    todo = []
    for block in candidates:
        if block["skipped"]:
            _skip(block, block["skipped"])
        else:
            todo.append(block)
    if not todo:
        return out

    try:
        page_image = render_pdf_page(pdf_path, page_index, dpi=SWEEP_DPI,
                                     max_px=SWEEP_PAGE_MAX_PX)
    except Exception as exc:                                   # noqa: BLE001
        # 渲染挂了不该把整页的步骤② 结果拖下水，但也绝不能静默 —— 记一条页级
        # error，调用方照常拿到「补扫什么都没加」。
        out["errors"].append({"group_index": None,
                              "error": f"{type(exc).__name__}: {exc}"})
        if dbg is not None:
            dbg.add("legend_sweep", {"group_index": None, "kind": None,
                                     "crop_box": None, "asked": [], "found": 0,
                                     "skipped": "render_failed",
                                     "elapsed": 0.0})
        return out

    width, height = page_image.size
    seen_added = set()
    for block in todo:
        crop, crop_box, window, skip_reason = _crop_for_block(page_image,
                                                              block)
        if skip_reason:
            _skip(block, skip_reason, crop_box=crop_box,
                  crop_px=list(window))
            continue
        cx0, cy0, cw, ch = window
        rows = []
        for item_index in block["missing"]:
            item = items[item_index] or {}
            row_box = page_box_to_crop(item.get("box_2d"), cx0, cy0, cw, ch,
                                       width, height)
            if row_box is None:
                continue
            rows.append({"idx": item_index,
                         "text": str(item.get("text") or ""),
                         "box_2d": row_box})
        if not rows:
            _skip(block, "no_rows_in_crop", crop_box=crop_box,
                  crop_px=list(window))
            continue

        answer = _ask_block(crop, rows, resolved_model, timeout_ms)
        out["calls"] += answer["calls"]
        out["elapsed"] += answer["elapsed"]
        out["usage"] = _merge_usage(out["usage"], answer["usage"])
        asked = [row["idx"] for row in rows]
        if answer["error"] is not None:
            # 三次全败：这一块作废，其它块与步骤② 的结果照旧。
            out["errors"].append({"group_index": block["group_index"],
                                  "error": answer["error"]})
        found = []
        for row in answer["rows"] or []:
            page_box = crop_box_to_page(row["box_2d"], cx0, cy0, cw, ch,
                                        width, height)
            if page_box is None or not _covers_center(crop_box, page_box):
                out["errors"].append(
                    {"group_index": block["group_index"],
                     "error": f"unmappable box from crop: {row['box_2d']}"})
                continue
            value = row["value"]
            symbol = {"box_2d": page_box,
                      "category": row["category"],
                      "value": value,
                      "type": row["category"] + (f" {value}" if value else ""),
                      "text_index": row["idx"],
                      "group_index": block["group_index"],
                      # 溯源：这条是哪一块补扫出来的（blocks 里的下标）——
                      # 发布闸给补扫符号取景用的就是那一块的 crop_box。
                      "block_index": len(out["blocks"])}
            found.append(symbol)
            key = (row["idx"], tuple(page_box), row["category"], value)
            if key in seen_added:
                continue
            seen_added.add(key)
            out["added"].append(symbol)

        out["blocks"].append({"group_index": block["group_index"],
                              "kind": block.get("kind"),
                              "box_2d": list(block["box_2d"]),
                              "crop_box": crop_box,
                              "crop_px": list(window),
                              "crop_size": list(crop.size),
                              "asked": asked,
                              "found": found,
                              "elapsed": round(answer["elapsed"], 1),
                              "usage": answer["usage"]})
        if dbg is not None:
            dbg.add("legend_sweep",
                    {"group_index": block["group_index"],
                     "kind": block.get("kind"),
                     "crop_box": crop_box,
                     "asked": asked,
                     "found": len(found),
                     "skipped": None,
                     "error": answer["error"],
                     "elapsed": round(answer["elapsed"], 1)})

    out["elapsed"] = round(out["elapsed"], 1)
    return out
