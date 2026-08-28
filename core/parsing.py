"""Pure parsing / geometry helpers — no I/O, no model calls."""
import json
import math
import re


def parse_pages_spec(spec: str, total: int):
    s = (spec or "").strip().lower()
    if not s or s == "all":
        return list(range(1, total + 1))
    out = set()
    for part in re.split(r"[,\s]+", s):
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a, b = int(a_str), int(b_str)
            for p in range(min(a, b), max(a, b) + 1):
                if 1 <= p <= total:
                    out.add(p)
        else:
            p = int(part)
            if 1 <= p <= total:
                out.add(p)
    return sorted(out)


def parse_json_array(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    start = raw.find("[")
    if start == -1:
        return []
    end = raw.rfind("]")
    if end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass
    # A long response cut off mid-array (output token cap) would otherwise
    # lose EVERYTHING — recover every complete element up to the truncation.
    dec = json.JSONDecoder()
    out = []
    i = start + 1
    n = len(raw)
    while i < n:
        while i < n and raw[i] in " \t\r\n,":
            i += 1
        if i >= n or raw[i] == "]":
            break
        try:
            obj, i = dec.raw_decode(raw, i)
        except json.JSONDecodeError:
            break
        out.append(obj)
    return out


def parse_json_value(raw: str):
    """Parse a Gemini JSON response that may be either an object or an array.
    Strips markdown fences, returns the parsed Python value or None on failure."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    # Try parsing the whole string first.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Gemini's JSON mode sometimes emits trailing garbage (e.g. a stray
    # extra closing brace). raw_decode parses the FIRST complete value and
    # ignores whatever follows.
    for opener in ("{", "["):
        s = raw.find(opener)
        if s != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(raw[s:])
                return obj
            except json.JSONDecodeError:
                pass
    # Fallback: extract the outermost object or array.
    for opener, closer in (("{", "}"), ("[", "]")):
        s = raw.find(opener)
        e = raw.rfind(closer)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s:e + 1])
            except json.JSONDecodeError:
                continue
    return None


def is_normalized_box(raw):
    """Return whether ``raw`` is a drawable normalized model/UI box.

    Model-facing ``box_2d`` values use the inclusive 0..1000 coordinate
    frame, but the rectangle itself must have positive area.  Do not clamp a
    protocol-violating response: doing so can manufacture a plausible box at
    an unrelated page edge.  PDF-native source geometry has separate
    semantics and deliberately does not pass through this predicate.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return False
    # ``bool`` is a subclass of ``int`` in Python, so float(True) would
    # otherwise silently turn a malformed model coordinate into 1.0.
    if any(isinstance(v, bool) for v in raw):
        return False
    try:
        box = [float(v) for v in raw]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in box):
        return False
    y0, x0, y1, x1 = box
    return (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000)


def _coerce_box(raw):
    """Coerce a model box to integers, rejecting invalid normalized boxes."""
    if not is_normalized_box(raw):
        return None
    try:
        box = [int(v) for v in raw]
    except (OverflowError, TypeError, ValueError):
        return None
    return box if is_normalized_box(box) else None


def _bbox_overlap_ratio(a, b):
    """Intersection / smaller-area for two [ymin, xmin, ymax, xmax] boxes."""
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    bb = max(0, bx1 - bx0) * max(0, by1 - by0)
    smaller = min(aa, bb)
    return inter / smaller if smaller > 0 else 0.0


def _normalize_value(v):
    """Lenient normalization for value comparison.

    Lowercases, strips whitespace, drops non-alphanumeric, and maps a
    small set of well-known OCR-confusable singletons to a canonical
    form so that for example "O" and "0" compare equal, "I" / "l" / "1"
    compare equal. Multi-character values keep their content (we don't
    want to merge "12" with "I2")."""
    if not v:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    if len(s) == 1:
        if s == "o":
            s = "0"
        elif s in ("i", "l"):
            s = "1"
    return s
