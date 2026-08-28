"""Global configuration: env-driven settings, model pricing, directories."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def _load_env_file(path):
    """Load the project-local .env without adding an import-time dependency."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key.isidentifier() or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


_load_env_file(BASE_DIR / ".env")

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required (environment or project .env)")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
RENDER_DPI = 144
HIRES_DPI = 300
# Hard caps on rendered pixel dimensions. Big PDFs (architectural sheets at 200 DPI
# would be 50+ MP per page) blow up memory, blow past Gemini's inline image limit,
# and choke the browser canvas. Cap the long side so a 42"×30" sheet still renders
# at usable detail without exploding everything else.
MAX_RENDER_PX = 5000   # for display / detection
MAX_HIRES_PX  = 7000   # for legend symbol detection / line extraction — caps the rendered image's long side to keep encoding + transmission fast
# Byte budget for a single inline image part; PNGs above this are re-encoded
# as JPEG q92. (Docs once said ~20 MB for inline images, but 90 MB has worked
# in practice — the value is deliberately generous.)
GEMINI_INLINE_BYTES_LIMIT = 90 * 1024 * 1024

# USD per 1M tokens. Source: https://ai.google.dev/gemini-api/docs/pricing
# Pro models are prompt-size-tiered at 200K prompt tokens; thinking tokens billed as output.
# Order matters: this is also the dropdown order in the UI.
PRICING = {
    "gemini-3.1-pro-preview":      {"display": "Gemini 3.1 Pro (Preview)",   "note": "视觉最强",   "thr": 200_000, "in_low": 2.00, "in_high": 4.00, "out_low": 12.00, "out_high": 18.00},
    "gemini-3-pro-preview":        {"display": "Gemini 3 Pro (Preview)",     "note": "旧版本",     "thr": 200_000, "in_low": 2.00, "in_high": 4.00, "out_low": 12.00, "out_high": 18.00},
    "gemini-2.5-pro":              {"display": "Gemini 2.5 Pro",             "note": "上一代 Pro", "thr": 200_000, "in_low": 1.25, "in_high": 2.50, "out_low": 10.00, "out_high": 15.00},
    "gemini-3.5-flash":            {"display": "Gemini 3.5 Flash",           "note": "快 / 中价",                  "flat": True, "in_low": 1.50, "out_low":  9.00},
    "gemini-3-flash-preview":      {"display": "Gemini 3 Flash (Preview)",   "note": "快 / 价格未定",              "flat": True, "in_low": 1.50, "out_low":  9.00},
    "gemini-2.5-flash":            {"display": "Gemini 2.5 Flash",           "note": "便宜 / 快",                  "flat": True, "in_low": 0.30, "out_low":  2.50},
    "gemini-3.1-flash-lite":       {"display": "Gemini 3.1 Flash-Lite",      "note": "便宜 / 3.x",                 "flat": True, "in_low": 0.25, "out_low":  1.50},
    "gemini-3.1-flash-lite-preview": {"display": "Gemini 3.1 Flash-Lite (Preview)", "note": "便宜 / 3.x preview", "flat": True, "in_low": 0.25, "out_low":  1.50},
    "gemini-2.5-flash-lite":       {"display": "Gemini 2.5 Flash-Lite",      "note": "最便宜",                     "flat": True, "in_low": 0.10, "out_low":  0.40},
    "gemini-2.0-flash":            {"display": "Gemini 2.0 Flash",           "note": "老一代",                     "flat": True, "in_low": 0.10, "out_low":  0.40},
    "gemini-2.0-flash-lite":       {"display": "Gemini 2.0 Flash-Lite",      "note": "最老",                       "flat": True, "in_low": 0.075,"out_low":  0.30},
}

# Anthropic Claude models, served through core/llm.py instead of google-genai.
# Same schema as the Gemini block plus "provider"; flat (no prompt-size tier).
# USD per 1M tokens, verified against platform.claude.com/docs/en/about-claude/pricing.
# Appended after the Gemini literal so Gemini 3.1 Pro stays first in the UI order
# and remains the default — selecting one of these is an explicit opt-in.
PRICING.update({
    "claude-sonnet-5": {"display": "Claude Sonnet 5", "note": "Anthropic / 对比", "provider": "anthropic", "flat": True, "in_low": 2.00, "out_low": 10.00},
    "claude-opus-5":   {"display": "Claude Opus 5",   "note": "Anthropic / 最强", "provider": "anthropic", "flat": True, "in_low": 5.00, "out_low": 25.00},
})


def provider_of(model: str) -> str:
    """Which backend serves this model id."""
    return (PRICING.get(model) or {}).get("provider", "gemini")


# ── Job-scoped model override ───────────────────────────────────────────────
# The pipeline picks its model through resolve_model(), and almost every step
# passes model=None, so a single override switches the whole run. This mirrors
# the CallRecorder pattern in core/gemini.py: a global session toggled by the
# orchestrator around one job. Safe for the same reason — the service is pinned
# to gunicorn -w 1 and job.py serializes runs behind _PROC_LOCK, so two runs
# never overlap. An explicit per-call model argument still wins.
_MODEL_OVERRIDE = None


def set_model_override(model):
    """Pin the model for the current job. Pass None to clear."""
    global _MODEL_OVERRIDE
    _MODEL_OVERRIDE = model if (model and model in PRICING) else None
    return _MODEL_OVERRIDE


def get_model_override():
    return _MODEL_OVERRIDE


def resolve_model(name: str) -> str:
    """Validate a user-supplied model id, fall back to the job override, then
    to the process default."""
    if name and name in PRICING:
        return name
    if _MODEL_OVERRIDE:
        return _MODEL_OVERRIDE
    return MODEL_NAME


def compute_cost(model: str, usage: dict):
    p = PRICING.get(model)
    if not p or not usage:
        return None
    in_tok = int(usage.get("input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0) + int(usage.get("thoughts_tokens") or 0)
    if p.get("flat"):
        r_in, r_out = p["in_low"], p["out_low"]
    else:
        r_in, r_out = (p["in_low"], p["out_low"]) if in_tok <= p["thr"] else (p["in_high"], p["out_high"])
    cost_in = in_tok * r_in / 1_000_000
    cost_out = out_tok * r_out / 1_000_000
    return {
        "input_usd":  round(cost_in,  6),
        "output_usd": round(cost_out, 6),
        "total_usd":  round(cost_in + cost_out, 6),
        "rate_in_per_mtok":  r_in,
        "rate_out_per_mtok": r_out,
    }


PROJECTS_DIR = BASE_DIR / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)
