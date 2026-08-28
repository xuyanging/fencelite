"""VLM 原始响应的缓存身份契约 —— 付费产物只在身份完全一致时才复用.

Raw page responses are paid artifacts.  Reusing one is safe only when the
input PDF revision, the *resolved* model id, and the exact prompt bytes all
match the current call.  The prompt digest is derived from its content so a
prompt edit cannot be forgotten behind a hand-maintained version number.

两种角色，各自独立落盘（绝不把 union 结果伪装成一条 raw）：
  primary         —— 配置主模型的整页扫描，写 vlm.json
  secondary_union —— 无文字层扫描页的第二模型（Flash）扫描，写 vlm_flash.json

没有 ``vlm_identity`` 的记录（外部导入的旧缓存）故意不通过
:func:`is_current_vlm_record`；要复用就由 tools/ 下的导入脚本显式改写
``vlm_identity.pdf_revision``，把「这份 raw 属于哪个 PDF」的断言交给操作者。
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from core.config import resolve_model
from steps.store import pdf_revision

IDENTITY_FIELD = "vlm_identity"
ROLE_FIELD = "vlm_role"
PRIMARY_ROLE = "primary"
SECONDARY_UNION_ROLE = "secondary_union"


def prompt_sha256(prompt: str) -> str:
    """Return the digest of the exact UTF-8 prompt content."""
    if not isinstance(prompt, str):
        raise TypeError("VLM prompt must be a string")
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def vlm_identity_for_revision(revision: str, model, prompt: str) -> dict:
    """Build an identity when the caller already captured the PDF revision."""
    return {
        "pdf_revision": revision,
        "model": resolve_model(model),
        "prompt_sha256": prompt_sha256(prompt),
    }


def vlm_identity(pdf_path, model, prompt: str) -> dict:
    """Build the identity of the call that ``scan_page`` will execute."""
    return vlm_identity_for_revision(
        pdf_revision(Path(pdf_path)), model, prompt)


def valid_vlm_items(items) -> bool:
    """Validate the durable raw response shape accepted by the text scanner."""
    if not isinstance(items, list):
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        text = item.get("text")
        box = item.get("box_2d")
        if not isinstance(text, str) or not text.strip():
            return False
        if not isinstance(box, list) or len(box) != 4:
            return False
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in box):
            return False
        y0, x0, y1, x1 = box
        if not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
            return False
        label = item.get("label")
        if label is not None and not isinstance(label, str):
            return False
    return True


def valid_vlm_record(record) -> bool:
    """A reusable raw response must be successful and complete."""
    return (isinstance(record, dict)
            and not record.get("error")
            and valid_vlm_items(record.get("items")))


def is_current_vlm_record(record, expected_identity: dict) -> bool:
    """Return true only for a structurally valid, exact-identity cache hit."""
    return (valid_vlm_record(record)
            and record.get(IDENTITY_FIELD) == expected_identity
            and record.get("model") == expected_identity.get("model"))


def is_current_primary_record(record, expected_identity: dict) -> bool:
    """Validate one record of the primary store (vlm.json).

    Store placement is part of the durable contract: a second-model
    (``secondary_union``) response copied into the primary store must not
    become a cache hit merely because its own identity is self-consistent.
    """
    if not isinstance(record, dict):
        return False
    if record.get(ROLE_FIELD) not in (None, PRIMARY_ROLE):
        return False
    return is_current_vlm_record(record, expected_identity)


def is_current_secondary_record(record, expected_identity: dict) -> bool:
    """Validate the separately stored no-text-layer union source."""
    return (isinstance(record, dict)
            and record.get(ROLE_FIELD) == SECONDARY_UNION_ROLE
            and is_current_vlm_record(record, expected_identity))


def make_vlm_record(*, identity: dict, items=None, elapsed=None, usage=None,
                    error=None, role=PRIMARY_ROLE) -> dict:
    """Create one stamped success or failure record.

    Failure records are stamped as well, but never become cache hits because
    :func:`valid_vlm_record` rejects their ``error`` field.
    """
    record = {
        "items": list(items or []),
        IDENTITY_FIELD: dict(identity),
        # Retain the historical convenience field, now with the resolved id.
        "model": identity["model"],
        ROLE_FIELD: role,
    }
    if error:
        record["error"] = str(error)
        return record
    if elapsed is not None:
        record["elapsed"] = round(float(elapsed), 1)
    if usage is not None:
        record["usage"] = usage
    return record
