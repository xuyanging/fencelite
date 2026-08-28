"""剥符号码用的矢量上下文 —— 小闭合图形 + 直线段.

Vector context used by step 1 to strip marker codes and line tokens.

只服务 clean.strip_marker_codes：它需要「哪些小闭合图形里坐着一个短码」
（六边形 F-04 / 圆圈 33 → 那是第 2 步符号的领域，不是 fence 文字）以及
整页的直线段（判断短 token 是否被线条从两个相对方向夹住，-SF-SF-）。
和第 2 步的图例符号检测无关，不要合并。
"""
import re

from core.vecgeom import _extract_page


def _closed_loops(geometry_items):
    """Return small vertex loops embedded in a polyline path."""
    vertices = []
    for item in geometry_items:
        if item[0] == "l":
            if not vertices:
                vertices.append((item[1], item[2]))
            vertices.append((item[3], item[4]))
        elif item[0] == "c":
            if not vertices:
                vertices.append((item[1], item[2]))
            vertices.append((item[7], item[8]))
    loops = []
    indices = {}
    for index, vertex in enumerate(vertices):
        key = (round(vertex[0]), round(vertex[1]))
        start = indices.get(key)
        if start is not None and 3 <= index - start <= 12:
            loop = vertices[start:index]
            xs = [value[0] for value in loop]
            ys = [value[1] for value in loop]
            loops.append((loop, min(xs), min(ys), max(xs), max(ys)))
        indices[key] = index
    return loops


def _classify_poly(vertices, x0, y0, x1, y1):
    """Classify only enough closed geometry to build marker-code boxes."""
    count = len(vertices)
    width, height = x1 - x0, y1 - y0
    if count == 3:
        by_y = sorted(vertices, key=lambda value: value[1])
        top_spread = abs(by_y[0][1] - by_y[1][1])
        bottom_spread = abs(by_y[2][1] - by_y[1][1])
        return "triangle_down" if bottom_spread < top_spread else "triangle"
    if count == 4:
        scale = max(width, height)
        axis_aligned = all(
            abs(vertices[index][0] - vertices[(index + 1) % 4][0])
            < 0.15 * scale
            or abs(vertices[index][1] - vertices[(index + 1) % 4][1])
            < 0.15 * scale
            for index in range(4))
        if axis_aligned:
            return ("square" if 0.75 <= width / max(height, 1e-6) <= 1.33
                    else "rect")
        return "diamond"
    if count == 6:
        return "hexagon"
    return "poly"


def _harvest(data):
    """Collect small closed shapes and straight segments in 0-1000 space."""
    width, height = data["w"], data["h"]
    zoom = data["zoom"]
    normalize_x = lambda value: value / width * 1000          # noqa: E731
    normalize_y = lambda value: value / height * 1000         # noqa: E731
    shapes, segments = [], []

    def add_shape(kind, x0, y0, x1, y1):
        size = ((x1 - x0) + (y1 - y0)) / 4 / zoom
        if not 0.4 <= size <= 25:
            return
        shapes.append({
            "kind": kind,
            "box": [normalize_y(y0), normalize_x(x0),
                    normalize_y(y1), normalize_x(x1)],
            "cx": normalize_x((x0 + x1) / 2),
            "cy": normalize_y((y0 + y1) / 2),
            "h": normalize_y(y1) - normalize_y(y0),
        })

    for unit in data["units"]:
        source_index = unit[0]
        geometry = data["geom"][source_index]
        if not geometry:
            continue
        line_count, curve_count = unit[8], unit[9]
        for item in geometry:
            if item[0] == "re":
                x0, x1 = sorted((item[1], item[3]))
                y0, y1 = sorted((item[2], item[4]))
                box_width, box_height = x1 - x0, y1 - y0
                kind = ("square" if box_height and
                        0.75 <= box_width / max(box_height, 1e-6) <= 1.33
                        else "rect")
                add_shape(kind, x0, y0, x1, y1)
            elif item[0] == "qu":
                xs = (item[1], item[3], item[5], item[7])
                ys = (item[2], item[4], item[6], item[8])
                vertices = list(zip(xs, ys))
                add_shape(_classify_poly(
                    vertices, min(xs), min(ys), max(xs), max(ys)),
                    min(xs), min(ys), max(xs), max(ys))
            elif item[0] == "l" \
                    and abs(item[3] - item[1]) + abs(item[4] - item[2]) > 0.1:
                segments.append({
                    "ax": normalize_x(item[1]), "ay": normalize_y(item[2]),
                    "bx": normalize_x(item[3]), "by": normalize_y(item[4]),
                    "dash": data["dashes"][source_index] or "",
                    "src": source_index,
                })
        if curve_count >= 2 and line_count == 0:
            x0, y0, x1, y1 = unit[1], unit[2], unit[3], unit[4]
            box_width, box_height = x1 - x0, y1 - y0
            if box_height > 0:
                kind = ("circle" if
                        0.72 <= box_width / max(box_height, 1e-6) <= 1.38
                        else "ellipse")
                add_shape(kind, x0, y0, x1, y1)
        elif line_count > 0:
            for vertices, x0, y0, x1, y1 in _closed_loops(geometry):
                add_shape(_classify_poly(vertices, x0, y0, x1, y1),
                          x0, y0, x1, y1)
    return shapes, segments


_CODE_TEXT = re.compile(r"^[A-Za-z]{0,3}[-.]?\d{1,4}[A-Za-z]?$")


def _code_labels(data):
    """Return short marker-code words in normalized page coordinates."""
    width, height = data["w"], data["h"]
    labels = []
    for text in data["texts"]:
        value = text["text"].strip()
        if 1 <= len(value) <= 6 and _CODE_TEXT.match(value):
            labels.append({
                "text": value,
                "cx": (text["x0"] + text["x1"]) / 2 / width * 1000,
                "cy": (text["y0"] + text["y1"]) / 2 / height * 1000,
            })
    return labels


def strip_context(pdf_path, page_index):
    """Return marker boxes and straight segments needed by text cleanup."""
    data = _extract_page(str(pdf_path), page_index)
    shapes, segments = _harvest(data)
    labels = _code_labels(data)
    marker_boxes = []
    for shape in shapes:
        half_width = max(shape["box"][3] - shape["box"][1], 1e-3)
        for label in labels:
            if abs(label["cx"] - shape["cx"]) <= 0.45 * half_width \
                    and abs(label["cy"] - shape["cy"]) \
                    <= 0.45 * max(shape["h"], 1e-3):
                marker_boxes.append(shape["box"])
                break
    return {"mboxes": marker_boxes, "segs": segments}
