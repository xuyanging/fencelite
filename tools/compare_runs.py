"""Diff two runs of the same PDF — typically the same project under two models.

    venv/bin/python tools/compare_runs.py gladstone_dog_park gladstone_dog_park__claude-sonnet-5

Reads only what is already on disk (data/<slug>/results.json + symbols.json), so
it costs nothing and can be re-run freely.

What it reports, and why each column is there:

  * text recall  — items each run found per page. A model that finds fewer
    strings is missing content; one that finds more may be finding real extra
    content or hallucinating, which is why the text overlap column matters.
  * text overlap — items whose normalized text appears in BOTH runs. This is
    the like-for-like set; everything else is one-sided.
  * localization — for the overlapping items only, IoU and centre distance of
    the two boxes. Same text in both runs but disjoint boxes means one of them
    is drawing the overlay in the wrong place, which is invisible in a
    count-only comparison and is exactly the failure this tool exists to
    surface. Treating either run as ground truth is up to the reader: the tool
    reports disagreement, not correctness.
  * cost / time   — from llm_summary, as the recorder billed it.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from steps.store import DATA_DIR, load_json  # noqa: E402


def norm(s):
    """Compare text the way the pipeline's own judge cache does."""
    return " ".join(str(s or "").split()).upper()


def iou(a, b):
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def centre_dist(a, b):
    return (((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) ** 2
            + ((a[1] + a[3]) / 2 - (b[1] + b[3]) / 2) ** 2) ** 0.5


def items_of(rec):
    """Every text item the run published for one page, keyed by normalized text."""
    out = {}
    for bucket in ("vlm_items", "vec_added"):
        for it in (rec or {}).get(bucket, []) or []:
            box = it.get("box_2d")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                out.setdefault(norm(it.get("text")), [float(v) for v in box])
    return out


def load(slug):
    res = load_json(DATA_DIR / slug / "results.json", None)
    if not res:
        sys.exit(f"no results.json for {slug} — has that run finished?")
    syms = load_json(DATA_DIR / slug / "symbols.json", {}) or {}
    return res, syms


def sym_count(syms, page):
    entry = syms.get(str(page)) or {}
    result = entry.get("result") or {}
    s = result.get("symbols") or []
    plc = sum(len(x.get("placements") or []) for x in s if isinstance(x, dict))
    return len(s), plc


def main(slug_a, slug_b):
    res_a, sym_a = load(slug_a)
    res_b, sym_b = load(slug_b)
    pages = max(int(res_a.get("page_count") or 0), int(res_b.get("page_count") or 0))

    print(f"A = {slug_a}")
    print(f"B = {slug_b}")
    print()
    hdr = (f"{'pg':>3} {'A txt':>6} {'B txt':>6} {'both':>5} "
           f"{'A only':>7} {'B only':>7} {'IoU':>6} {'ctr err':>8} "
           f"{'disjoint':>9} {'A sym':>6} {'B sym':>6}")
    print(hdr)
    print("-" * len(hdr))

    tot = dict(a=0, b=0, both=0, aonly=0, bonly=0, disjoint=0, ious=[], errs=[])
    for p in range(1, pages + 1):
        ra = (res_a.get("pages") or {}).get(str(p)) or {}
        rb = (res_b.get("pages") or {}).get(str(p)) or {}
        ia, ib = items_of(ra), items_of(rb)
        both = set(ia) & set(ib)
        ious = [iou(ia[k], ib[k]) for k in both]
        errs = [centre_dist(ia[k], ib[k]) for k in both]
        disjoint = sum(1 for v in ious if v == 0.0)
        sa, pa = sym_count(sym_a, p)
        sb, pb = sym_count(sym_b, p)

        tot["a"] += len(ia); tot["b"] += len(ib); tot["both"] += len(both)
        tot["aonly"] += len(set(ia) - set(ib)); tot["bonly"] += len(set(ib) - set(ia))
        tot["disjoint"] += disjoint
        tot["ious"] += ious; tot["errs"] += errs

        mi = f"{sum(ious)/len(ious):6.3f}" if ious else "     -"
        me = f"{sum(errs)/len(errs):8.1f}" if errs else "       -"
        print(f"{p:>3} {len(ia):>6} {len(ib):>6} {len(both):>5} "
              f"{len(set(ia)-set(ib)):>7} {len(set(ib)-set(ia)):>7} {mi} {me} "
              f"{disjoint:>9} {sa:>6} {sb:>6}")

    print("-" * len(hdr))
    mi = (sum(tot["ious"]) / len(tot["ious"])) if tot["ious"] else 0.0
    me = (sum(tot["errs"]) / len(tot["errs"])) if tot["errs"] else 0.0
    print(f"{'ALL':>3} {tot['a']:>6} {tot['b']:>6} {tot['both']:>5} "
          f"{tot['aonly']:>7} {tot['bonly']:>7} {mi:6.3f} {me:8.1f} "
          f"{tot['disjoint']:>9}")

    print()
    print("Localization (over the "
          f"{tot['both']} items whose text both runs found):")
    if tot["ious"]:
        tight = sum(1 for v in tot["ious"] if v >= 0.5)
        partial = sum(1 for v in tot["ious"] if 0.0 < v < 0.5)
        print(f"  tight  (IoU >= 0.5) : {tight:>4}  ({tight/len(tot['ious'])*100:.0f}%)")
        print(f"  partial(0 < IoU <.5): {partial:>4}  ({partial/len(tot['ious'])*100:.0f}%)")
        print(f"  disjoint (IoU == 0) : {tot['disjoint']:>4}  "
              f"({tot['disjoint']/len(tot['ious'])*100:.0f}%)")
        print(f"  mean centre error   : {me:.1f} units (0-1000 page frame)")

    print()
    for tag, res in (("A", res_a), ("B", res_b)):
        s = res.get("llm_summary") or {}
        print(f"{tag}: model={','.join((s.get('by_model') or {}).keys()) or '?':28} "
              f"calls={s.get('calls')} cost=${s.get('cost_usd')} "
              f"model_s={s.get('model_seconds')} peak={s.get('peak_concurrency')} "
              f"in={s.get('input_tokens')} out={s.get('output_tokens')} "
              f"think={s.get('thoughts_tokens')} wall={res.get('wall_seconds')}s")
        errs = [f"P{p}" for p, r in sorted((res.get("pages") or {}).items())
                if (r or {}).get("vlm_error")]
        if errs:
            print(f"    pages with vlm_error: {', '.join(errs)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
