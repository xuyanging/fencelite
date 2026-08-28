"""One editable detection TARGET, two fixed task wrappers.

Product requirement: the user may only edit *what to detect* (the semantic
target), never the *output format* — editing the format would break JSON
parsing.  And the same target must drive BOTH first-step model tasks, which
have the identical goal but different inputs:

  * the image VLM scan (reads a rendered page, returns boxed text records)
  * the text judge     (reads native vector text strings, flags the matches)

So the target lives here once; ``build_vlm_prompt`` / ``build_judge_prompt``
wrap it in their respective fixed scaffolding (which carries the immutable
output-format contract).  The default target reproduces the original
fence-detection semantics.
"""

# ---- the ONLY user-editable part (shown, verbatim, in the upload dialog) ----
TARGET_DEFAULT = """Fences and fencing in ANY form. This is a SEMANTIC target, not keyword \
matching — treat as a match any text a construction professional would read \
as being about a fence, for example:
- the word FENCE / FENCES / FENCED / FENCING in any form or capitalization
- fence systems and materials, even when the word "fence" is absent:
  CHAIN LINK / CHAINLINK / CHAIN-LINK, barbed wire, woven or welded wire
  mesh, split rail, picket, privacy slats, ornamental / wrought iron,
  wood / vinyl / PVC fence systems, silt fence, snow fence, temporary
  construction fencing, tree-protection fencing
- fence components mentioned in a fencing context: fence post, gate post,
  fence fabric, tension wire, top / bottom rail of a fence, fence footing
- gates that are part of a fence line (chain link gate, maintenance gate,
  double swing gate, access gate in a fence)
- abbreviations that clearly mean a fence (C.L. FENCE, CLF, CL FENCE)

Do NOT match: handrails, guardrails, balustrades or stair railings (those are
railings, not fences), and generic structural terms (post, rail, mesh) when
the context is clearly NOT a fence."""


# ---- fixed scaffolding — NOT user-editable (output-format contract) ---------
_VLM_HEAD = """You are looking at ONE page of a construction / architecture / civil site drawing.

TASK — you have exactly ONE job: find EVERY piece of TEXT on this page that
matches the following detection target.

--- DETECTION TARGET ---
"""

_VLM_TAIL = """
--- END DETECTION TARGET ---

Scan the ENTIRE page systematically, region by region: every callout and
leader-line label, every note paragraph, every legend row, every schedule /
table cell, every keynote, every dimension annotation, every view title,
every title-block line. Text may be tiny, rotated (vertical / diagonal),
partially overlapped by linework, or inside a table — include it all.

For EACH occurrence output one record:
- "text":   the exact visible text of the note / label / row that matches the
            target (the whole line or note, not just one word)
- "box_2d": [ymin, xmin, ymax, xmax] — a TIGHT box around that text,
            integers normalized to 0-1000, [0,0] = top-left of the image
- "label":  one of "callout", "note", "legend entry", "schedule row",
            "keynote", "dimension", "view title", "title block", "other"

Rules:
- One record per printed occurrence. The same wording appearing in 5 places
  on the page = 5 records, each with its own box.
- A note spanning multiple lines gets ONE box covering all its lines.
- Do NOT include purely graphical marks / symbols that have no text.
- Do NOT include text that does not match the target.
- Favor recall: when a text plausibly matches the target, include it rather
  than skip it.

CODE MARKERS ARE SYMBOLS, NOT TEXT — this matters as much as recall:
  A short alphanumeric code — a handful of letters and/or digits, sometimes
  with a hyphen or period — printed INSIDE a small closed marker (hexagon,
  bubble, circle, flag, diamond, box) or standing alone in a legend's SYMBOL
  column is a SYMBOL, not a piece of matching text.
- Where such a marker is stamped on the drawing itself, at each place the
  keyed item is installed, do NOT emit a record for it. One sheet can carry
  dozens of identical markers; each is a symbol instance and a separate step
  detects them, so emitting them buries the real callouts.
- In a legend / schedule / keynote row, the row's own leading code marker is
  likewise a symbol: report ONLY the description text — exclude the code from
  BOTH "text" AND "box_2d", so the box starts at the first character of the
  description and does not reach back over the marker.
- This applies to codes inside markers only. Ordinary text that merely begins
  with a number, a dimension, a list number or a common abbreviation is
  normal text: keep it and box it in full.

OUTPUT: ONLY a JSON array, no prose, no markdown fences:
[{"text": "...", "box_2d": [ymin, xmin, ymax, xmax], "label": "..."}, ...]
If the page has no text matching the target at all, output []."""

_JUDGE_HEAD = """You are reviewing text strings extracted from the pages of ONE construction /
architecture / civil engineering drawing set.

TASK: decide for EACH string whether it matches the following detection
target. This is a SEMANTIC judgment.

--- DETECTION TARGET ---
"""

_JUDGE_TAIL = """
--- END DETECTION TARGET ---

Favor recall on genuine ambiguity, but never flag text whose subject is
clearly something else (company names, sheet numbers, unrelated notes).

A string that is JUST a short alphanumeric code with no other words is a
symbol/marker label, not a description of anything: do NOT flag it, even when
you can infer what the code abbreviates. Such codes are stamped at every
install location on the drawing, so flagging one flags dozens of duplicates
and buries the real callouts; a separate step detects the markers themselves.
The same code appearing together with words is judged on those words as
usual.

INPUT: a JSON array of {"id": <int>, "t": "<string>"}.
OUTPUT: ONLY a JSON array of the ids that match the target, e.g. [2,17,45].
No prose, no markdown. If none qualify, output []."""


def build_vlm_prompt(target=None):
    """Full image-scan prompt for a given target (fixed format scaffolding)."""
    return _VLM_HEAD + (target or TARGET_DEFAULT).strip() + _VLM_TAIL


def build_judge_prompt(target=None):
    """Full text-judge prompt for a given target (fixed format scaffolding)."""
    return _JUDGE_HEAD + (target or TARGET_DEFAULT).strip() + _JUDGE_TAIL


def is_default_target(target):
    """True when the target is blank or the built-in fence definition."""
    t = (target or "").strip()
    return (not t) or t == TARGET_DEFAULT.strip()
