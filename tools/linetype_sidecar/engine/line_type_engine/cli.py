"""Headless command entry for the Python line-type engine.

The ``recognize`` command is the maintained whole-document backend entry.  It
does not start or contact the React viewer, HTTP service, or port 3001.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence

from .document_runner import DocumentRunOptions, run_document
from .runtime import assert_supported_pymupdf_runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pdf-line-types-python")
    subparsers = parser.add_subparsers(dest="command", required=True)
    method1 = subparsers.add_parser(
        "method1-page",
        help="run the candidate Python Method1 pipeline on one one-based PDF page",
    )
    method1.add_argument("pdf")
    method1.add_argument("--page", type=int, default=1)
    method1.add_argument("--workers", type=int, default=None)
    method1.add_argument(
        "--full-result",
        action="store_true",
        help="write the full validated result JSON to stdout",
    )
    method1.add_argument(
        "--output",
        help="atomically write the JSON payload to this path instead of stdout",
    )
    recognize = subparsers.add_parser(
        "recognize",
        help="run the candidate Python engine over a whole PDF or page selection",
    )
    recognize.add_argument("pdf")
    recognize.add_argument(
        "--output-dir",
        required=True,
        help=(
            "dedicated candidate result directory; do not point this at a "
            "production viewer cache"
        ),
    )
    recognize.add_argument(
        "--pages",
        default="all",
        help="all, a page number, or comma-separated ranges such as 1,3-5",
    )
    recognize.add_argument(
        "--outputs",
        default="fused",
        help=(
            "comma-separated input,method1,method2,fused; input writes only "
            "PageIR/Grouping/source-alignment identities and the serialized "
            "Method1 input hash (fused computes both recognizers)"
        ),
    )
    recognize.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Method1 Group worker count; defaults to available CPUs",
    )
    recognize.add_argument(
        "--no-resume",
        action="store_true",
        help="recompute selected pages even when their durable payload validates",
    )
    return parser


def _method1_page(arguments: argparse.Namespace) -> dict:
    if arguments.page < 1:
        raise ValueError("--page must be one-based")
    if arguments.workers is not None and arguments.workers < 1:
        raise ValueError("--workers must be positive")
    from .grouping import group_page_sequentially
    from .method1 import recognize_method1_candidate
    from .source_page_adapter import source_aligned_page_ir_from_pdf_path

    page = source_aligned_page_ir_from_pdf_path(arguments.pdf, arguments.page).page
    grouping = group_page_sequentially(page)
    recognized = recognize_method1_candidate(
        page,
        grouping,
        worker_count=arguments.workers,
    )
    output = {
        "status": "candidate",
        "engine_version": recognized.audit.engine_version,
        "source": page.source_name,
        "page_number": page.page_number,
        "operation_count": len(page.operations),
        "group_count": len(grouping.groups),
        "summary": recognized.result.summary.to_dict(),
        "elapsed_ms": recognized.audit.elapsed_ms,
        "stage_ms": {
            stage.stage: stage.elapsed_ms
            for stage in recognized.audit.stages
        },
    }
    if arguments.full_result:
        output["result"] = recognized.result.to_dict()
    return output


def _document(arguments: argparse.Namespace):  # type: ignore[no-untyped-def]
    raw_outputs = arguments.outputs.split(",")
    if any(not item.strip() for item in raw_outputs):
        raise ValueError("--outputs must not contain empty items")
    outputs = tuple(item.strip().lower() for item in raw_outputs)
    return run_document(DocumentRunOptions(
        input_pdf=Path(arguments.pdf),
        output_directory=Path(arguments.output_dir),
        pages=arguments.pages,
        outputs=outputs,
        method1_worker_count=arguments.workers,
        resume=not arguments.no_resume,
    ))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        arguments = parser.parse_args(argv)
        assert_supported_pymupdf_runtime()
        if arguments.command == "recognize":
            run = _document(arguments)
            print(json.dumps(run.to_dict(), ensure_ascii=False, separators=(",", ":")))
            return run.exit_code
        if arguments.command != "method1-page":  # pragma: no cover - argparse owns choices.
            parser.error(f"unsupported command {arguments.command!r}")
            return 2
        if arguments.output:
            source = Path(arguments.pdf).resolve()
            destination = Path(arguments.output).resolve()
            same_target = destination == source
            if not same_target and source.exists() and destination.exists():
                same_target = os.path.samefile(source, destination)
            if same_target:
                raise ValueError("--output must not overwrite the input PDF")
        output = _method1_page(arguments)
        encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
        if arguments.output:
            destination = Path(arguments.output).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_name = temporary.name
                    temporary.write(encoded)
                    temporary.write("\n")
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_name, destination)
                temporary_name = None
            finally:
                if temporary_name is not None:
                    try:
                        os.unlink(temporary_name)
                    except FileNotFoundError:
                        pass
            print(json.dumps({
                "status": output["status"],
                "output": str(destination),
                "summary": output["summary"],
            }, ensure_ascii=False, separators=(",", ":")))
        else:
            print(encoded)
        return 0
    except KeyboardInterrupt:
        print("pdf-line-types-python: interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"pdf-line-types-python: {error}", file=sys.stderr)
        return 1


__all__ = ["main"]
