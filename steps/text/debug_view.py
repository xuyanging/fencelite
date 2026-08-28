"""文字步的调试视图 —— 纯本地、可缓存、零模型调用.

Build the cacheable, model-free debug view for the text-union step.
"""

from steps.text.judge import KW, norm_text


def attach_text_debug(rec, debug_sink, vector_items, judged_new=None,
                      use_kw_floor=True):
    """Attach provenance for every selected vector candidate to ``rec``.

    ``strip_marker_codes`` records rejected/stripped items in ``debug_sink``.
    This function adds the accepted vector candidates and whether each one was
    covered by VLM or entered the union as a supplement.  It is deterministic
    and makes no model calls, so batch and single-page paths can share it.

    ``use_kw_floor`` must mirror the flag ``select_instances`` ran under: the
    fence keyword floor is off for a custom target, so a candidate matching
    the fence regex still got in through the judge and must be reported as
    such — never as ``keyword_floor``.
    """
    if debug_sink is None:
        return rec

    judged_new = judged_new or {}
    data = debug_sink.data
    covered = {(item.get("text"), tuple(item.get("box_2d") or ()))
               for item in rec.get("vec_covered", [])}
    data["vector_candidates"] = [
        {
            "text": item.get("text", ""),
            "box_2d": item.get("box_2d"),
            "judge": (
                "keyword_floor" if use_kw_floor and KW.search(item.get("text", ""))
                else ("judge_new" if norm_text(item.get("text", ""))
                      in judged_new else "judge_cache")
            ),
            "outcome": (
                "covered_by_vlm"
                if (item.get("text"), tuple(item.get("box_2d") or ()))
                in covered else "vector_added"
            ),
        }
        for item in vector_items
    ]
    if judged_new:
        data["judged_new"] = judged_new
    if data:
        rec["debug"] = data
    return rec
