"""Bounded pypdf content-stream parsing for unusually large authored pages.

pypdf deliberately caps Flate output at 75 MB.  Some trusted corpus pages
contain a single valid content stream larger than that (Grand Island P73 is
126,639,407 decoded bytes).  Keep the library default for ordinary streams and
retry its explicit :class:`LimitReachedError` once with a project-owned finite
ceiling. Other pypdf limit failures may reach the same retry path, but remain
fail-closed if the finite retry cannot parse them. The pypdf limit is
process-global, so every helper call is serialized and the prior value is
restored on every exit path.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from threading import RLock
from typing import Any

from pypdf import filters
from pypdf._utils import read_non_whitespace, read_until_regex
from pypdf.errors import LimitReachedError
from pypdf.generic import ContentStream, NameObject, read_object


CONTENT_STREAM_DECODE_LIMIT_BYTES = 256 * 1024 * 1024
_CONTENT_STREAM_LIMIT_LOCK = RLock()


def _bounded_content_stream(
    stream_reference: Any,
    reader: Any,
) -> ContentStream:
    """Decode one stream with one bounded retry for pypdf's decode limit.

    All non-limit failures propagate unchanged.  Setting pypdf's limit to
    ``0`` (unbounded) is intentionally forbidden here.
    """

    with _CONTENT_STREAM_LIMIT_LOCK:
        original_limit = filters.ZLIB_MAX_OUTPUT_LENGTH
        try:
            try:
                return ContentStream(stream_reference, reader)
            except LimitReachedError:
                if (
                    not isinstance(original_limit, int)
                    or isinstance(original_limit, bool)
                    or original_limit <= 0
                    or original_limit >= CONTENT_STREAM_DECODE_LIMIT_BYTES
                ):
                    raise
                filters.ZLIB_MAX_OUTPUT_LENGTH = CONTENT_STREAM_DECODE_LIMIT_BYTES
                return ContentStream(stream_reference, reader)
        finally:
            filters.ZLIB_MAX_OUTPUT_LENGTH = original_limit


def iter_content_stream_operations(
    stream_reference: Any,
    reader: Any,
) -> Iterator[tuple[Any, bytes]]:
    """Yield pypdf-equivalent operations without retaining its eager list.

    ``ContentStream.operations`` calls pypdf's private parser, which appends
    every operation to one Python list before returning. A valid engineering
    page can contain more than thirteen million operators, making that list
    alone consume tens of gigabytes. The loop below is a deliberately literal
    generator form of pypdf 6.14.2's ``ContentStream._parse_content_stream``.
    It uses the same tokenizer, delimiter, object reader, forced encoding and
    inline-image reader, but releases each operands list after its consumer
    advances. The exact pypdf dependency pin and parity tests make any
    upstream parser change explicit rather than silently approximating it.
    """

    content = _bounded_content_stream(stream_reference, reader)
    stream = BytesIO(content.get_data())
    operands: list[Any] = []
    while True:
        peek = read_non_whitespace(stream)
        if peek in (b"", 0):
            return
        stream.seek(-1, 1)
        if peek.isalpha() or peek in (b"'", b'"'):
            operator = read_until_regex(
                stream=stream,
                regex=NameObject.delimiter_pattern,
                length=ContentStream._OPERATOR_LENGTH_LIMIT,
            )
            if operator == b"BI":
                # Match pypdf's invariant: BI may not inherit operands from a
                # preceding malformed command.
                assert operands == []
                yield content._read_inline_image(stream), b"INLINE IMAGE"
            else:
                yield operands, operator
                operands = []
        elif peek == b"%":
            while peek not in (b"\r", b"\n", b""):
                peek = stream.read(1)
        else:
            operands.append(read_object(stream, None, content.forced_encoding))


def content_stream_operations(
    stream_reference: Any,
    reader: Any,
) -> list[tuple[Any, bytes]]:
    """Compatibility materialization of :func:`iter_content_stream_operations`.

    New source parsing must consume the iterator. This list-returning wrapper
    remains for small diagnostic callers and exact parity tests only.
    """

    return list(iter_content_stream_operations(stream_reference, reader))


__all__ = [
    "CONTENT_STREAM_DECODE_LIMIT_BYTES",
    "content_stream_operations",
    "iter_content_stream_operations",
]
