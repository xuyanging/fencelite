"""步骤2（图例样例符号）与步骤3（视图投影分类）的提示词与枚举 —— 全项目唯一一处.

两个提示词都是付费调用的输入，两套缓存各自独立：
  * 改 GROUP_SYMBOL_PROMPT / GROUP_SYMBOL_RESPONSE_SCHEMA / GROUP_KINDS
    必须 bump versions.SYMBOL_PROMPT_V（每页重新付费推理）；
  * 改 VIEW_CLASSIFIER_PROMPT / VIEW_CLASSIFIER_SCHEMA / VIEW_TYPES
    必须 bump versions.VIEW_VERSION（只重付分类，不动 symbol raw）。
提示词正文与生产 5051 逐字一致，唯一例外见 VIEW_CLASSIFIER_PROMPT 上方注释。
"""

GROUP_SYMBOL_PROMPT = """You are looking at ONE page of a construction / civil / architecture drawing.
Below is a JSON list of fence-related TEXT items already located on this page.
Each has an index, its text, and box_2d = [ymin, xmin, ymax, xmax] in integers
normalized to 0-1000 ([0,0] = top-left of the image).

Perform TWO tasks in ONE pass:

TASK A — PAGE STRUCTURE (groups)
Construction sheets are organized into logical GROUPS. Identify EVERY group
visible on this page:
  - "view":        a single drawing (plan / elevation / section / detail)
                   with its title, scale, dimensions and callouts
  - "legend":      a legend / key block — title plus rows of sample
                   symbols with captions
  - "schedule":    a table — title plus rows (incl. REFERENCE NOTES /
                   keynote schedules with code bubbles per row)
  - "note_cluster": a self-contained block of stacked notes (incl. keyed
                   note lists with a marker bubble per note)
  - "title_block": the sheet's project / firm / number block
  - "other":       anything that fits none of the above
Each group gets a group box that fully encloses ALL of its elements but
does NOT bleed into neighbouring groups.

TASK B — LEGEND SAMPLE SYMBOLS (exactly TWO broad classes)
ONLY INSIDE groups of kind "legend", "schedule" or "note_cluster": for each
given fence TEXT that is a key/legend/keynote entry, find the SAMPLE GRAPHIC
paired with it — usually immediately LEFT of the caption, sometimes right
or above — and classify it into exactly one of TWO deliberately broad
classes (do NOT invent finer sub-types):

  "line"  — any LINE-STYLE sample: a short segment that demonstrates a
            line style — solid / dashed / dotted / double / ticked, or a
            line combined with small squares, circles, X marks, hatch
            ticks, letters riding on it (-SF-SF-), or any other linear
            decoration.  If it reads as a line sample, it is "line".
  "shape" — a closed outline containing (or paired with) a short number /
            letter code: the outline may be a circle, triangle, square,
            rectangle, hexagon, diamond, any polygon or anything similar.
            If it reads as a coded marker, it is "shape".

For "shape" samples also READ the short code ("value", e.g. "33", "F-04",
"A"); beware OCR confusions (C/D/O/Q, E/F/P/R, 0/O/D, 5/S, 8/B, 8/6, F/P,
M/N) — read every character of the code instead of settling for a
familiar-looking one, and if the block's siblings run in sequence, use row
position as a sanity check.  For "line" samples set value to the riding
letters if any (e.g. "SF"), else "".

BOX EACH SAMPLE INDIVIDUALLY AND TIGHTLY — this is as important as finding
it.  A legend's samples sit in one column, which tempts you to reuse one
column-shaped box for every row; that is wrong and the result is unusable:
- Every "shape" box MUST enclose the marker outline together with the code
  characters printed inside it.  Sanity-check each box: the code you just
  read for "value" has to be INSIDE the box you emit.  A box that stops
  just short of the code, or sits in the blank part of the column beside
  the marker, is a failure.
- Every "line" box MUST span the whole sample segment, left end to right
  end, and be tight vertically.
- Give each row its OWN box.  Do not emit the same rectangle twice and do
  not copy one row's box to its neighbours: consecutive rows are only a few
  units apart, so a box shifted by one row lands on the wrong marker.
- Work the block ROW BY ROW, top to bottom, over EVERY given text that sits
  in it.  Small marks are easy to skip; a silently skipped row is the single
  most common failure here.  If a row genuinely has no sample graphic,
  simply emit nothing for it.

HARD SCOPE RULE — legend regions only: every symbol box MUST sit inside a
legend / key / schedule / keyed-note block.  NEVER box a marker out in a
"view" area — no plan bubbles, no detail/view-title circles, no north
arrows, no dimension or leader markers.  If an input text is itself a keyed
CALLOUT in a view (a marker + key pointing at drawn work), do NOT box the
callout's own marker — find the row with the SAME key in the legend /
schedule block and box THAT row's sample instead; if no such block exists
on this page, emit nothing for that text.  A text with no adjacent sample
gets nothing.

OUTPUT: ONLY a JSON object, no prose, no markdown fences:
{"groups":  [{"box_2d": [ymin, xmin, ymax, xmax],
              "kind": "view" | "legend" | "schedule" | "note_cluster" | "title_block" | "other"}],
 "symbols": [{"text_index": <index of the owning fence text>,
              "box_2d": [ymin, xmin, ymax, xmax],   // ONE tight box around the sample graphic only
              "category": "line" | "shape",
              "value": "<the short code, or '' >",
              "group_index": <index into groups of the block it sits in>}]}
If the page has no groups at all, output {"groups": [], "symbols": []}."""

GROUP_KINDS = ("view", "legend", "schedule", "note_cluster", "title_block",
               "other")
VIEW_TYPES = ("plan", "elevation", "section", "detail", "other")
SYMBOL_GROUP_KINDS = ("legend", "schedule", "note_cluster")


# Transport-level JSON constraint for GROUP_SYMBOL_PROMPT.  It does not alter
# the semantic cache contract; it prevents otherwise-correct long responses
# from arriving with a duplicated/truncated field near the final symbol.
GROUP_SYMBOL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {"type": "array", "items": {"type": "number"}},
                    "kind": {"type": "string", "enum": list(GROUP_KINDS)},
                    "view_type": {"type": "string", "enum": list(VIEW_TYPES)},
                },
                "required": ["box_2d", "kind"],
            },
        },
        "symbols": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text_index": {"type": "integer"},
                    "box_2d": {"type": "array", "items": {"type": "number"}},
                    "category": {"type": "string", "enum": ["line", "shape"]},
                    "value": {"type": "string"},
                    "group_index": {"type": "integer"},
                },
                "required": ["text_index", "box_2d", "category", "value",
                             "group_index"],
            },
        },
    },
    "required": ["groups", "symbols"],
}


# 与生产 5051 唯一的一处正文差异（倒数第二段最后一句）：原句是
# "Later processing searches repeating fenceline geometry only inside plan
#  views" —— 整线追踪已下线，本项目 plan 视图的下游用途是「匹配图例符号的
# 全部放置」，故改写。语义（拿不准就别猜 plan）完全不变。
VIEW_CLASSIFIER_PROMPT = """You are classifying drawing VIEW regions on ONE
construction / civil / architecture sheet.  The image contains one or more
existing view boxes.  Return one classification for every supplied
group_index.

Use exactly one view_type:
- plan: a TOP-DOWN horizontal projection, including site, civil, terrain,
  topographic, grading, erosion-control, landscape, floor, roof, fence-layout
  or enlarged plan.  Contours do not make a drawing a section.
- elevation: an orthographic SIDE/FRONT vertical projection without a cut.
- section: a CUT-THROUGH view showing layers, profiles, structural sections,
  section hatching, or a title such as SECTION / CROSS SECTION.
- detail: a local construction/detail drawing that is not top-down plan,
  elevation, or section.
- other: perspective, isometric, schematic, diagram, or genuinely unclear.

Classify the actual projection, not merely the title.  In particular, a
top-down enlarged detail is still plan.  Fence-related words do not turn a
section/elevation into a plan.  Later processing matches every placement of
a legend symbol only inside plan views, so do not guess plan when the
evidence is uncertain.

INPUT VIEW GROUPS:
{groups_json}

Return ONLY JSON:
{{"views":[{{"group_index":0,"view_type":"plan",
           "reason":"short visual reason"}}]}}
"""

VIEW_CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "views": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "group_index": {"type": "integer"},
                    "view_type": {"type": "string",
                                  "enum": list(VIEW_TYPES)},
                    "reason": {"type": "string"},
                },
                "required": ["group_index", "view_type", "reason"],
            },
        },
    },
    "required": ["views"],
}
