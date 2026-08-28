"""Fail-closed runtime policy for the standalone Python engine.

PyMuPDF issue #5042 is a native reference-count defect in
``Page.get_texttrace()`` in releases 1.27.2.3 and 1.28.0.  It can terminate a
long-running interpreter instead of raising a Python exception.  PyMuPDF
1.28.2 is the first released version whose official release notes include the
fix, so every production entry point validates the loaded binary before PDF
work starts.

The source-content iterator deliberately mirrors pypdf 6.14.2's private
tokenizer primitives to avoid materializing multi-million-operation streams.
That parser contract is exact, not a compatible-version range: an unreviewed
pypdf update must fail before source parsing rather than silently change PDF
operator semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib import metadata as importlib_metadata
import platform
import re
from typing import Mapping


PYMUPDF_DISTRIBUTION = "PyMuPDF"
PYMUPDF_MIN_VERSION = "1.28.2"
PYMUPDF_MAX_VERSION = "2"
PYMUPDF_REQUIREMENT = (
    f"{PYMUPDF_DISTRIBUTION}>={PYMUPDF_MIN_VERSION},<{PYMUPDF_MAX_VERSION}"
)
PYMUPDF_ISSUE_5042 = "https://github.com/pymupdf/PyMuPDF/issues/5042"
PYMUPDF_FIXED_RELEASE = "https://github.com/pymupdf/PyMuPDF/releases/tag/1.28.2"
PYPDF_DISTRIBUTION = "pypdf"
PYPDF_PINNED_VERSION = "6.14.2"
PYPDF_REQUIREMENT = f"{PYPDF_DISTRIBUTION}=={PYPDF_PINNED_VERSION}"
RUNTIME_VERSIONS_SCHEMA_VERSION = 1

_MIN_RELEASE = (1, 28, 2, 0)
_MAX_RELEASE = (2, 0, 0, 0)
_FINAL_RELEASE = re.compile(r"^(?P<release>\d+(?:\.\d+){1,3})$")


class UnsupportedRuntimeError(RuntimeError):
    """Raised before PDF work when the native runtime is unsafe or ambiguous."""


def _unsupported(message: str) -> UnsupportedRuntimeError:
    return UnsupportedRuntimeError(
        f"{message}; this engine requires {PYMUPDF_REQUIREMENT} because "
        f"PyMuPDF issue #5042 can terminate long get_texttrace() batch runs. "
        f"Install it with: python -m pip install --upgrade "
        f"\"{PYMUPDF_REQUIREMENT}\""
    )


def _unsupported_pypdf(message: str) -> UnsupportedRuntimeError:
    return UnsupportedRuntimeError(
        f"{message}; this engine requires {PYPDF_REQUIREMENT} because its "
        "streaming source parser is token-for-token aligned with pypdf "
        f"{PYPDF_PINNED_VERSION}'s private ContentStream parser. Install it "
        f"with: python -m pip install --upgrade \"{PYPDF_REQUIREMENT}\""
    )


def _release_tuple(version: str, *, label: str) -> tuple[int, int, int, int]:
    if not isinstance(version, str) or not version or len(version) > 128:
        raise _unsupported(f"{label} does not expose a bounded version string")
    matched = _FINAL_RELEASE.fullmatch(version)
    if matched is None:
        raise _unsupported(
            f"{label} version {version!r} is not an unambiguous final release"
        )
    components = tuple(int(value) for value in matched.group("release").split("."))
    return (*components, *(0 for _ in range(4 - len(components))))


def validate_pymupdf_runtime_versions(
    *,
    distribution_version: str,
    module_version: str,
    binding_version: str,
) -> "PyMuPDFRuntime":
    """Validate distribution metadata and the actually loaded native binding."""

    versions = {
        "distribution": (
            distribution_version,
            _release_tuple(distribution_version, label="PyMuPDF distribution"),
        ),
        "module": (
            module_version,
            _release_tuple(module_version, label="pymupdf module"),
        ),
        "binding": (
            binding_version,
            _release_tuple(binding_version, label="pymupdf native binding"),
        ),
    }
    for label, (raw, release) in versions.items():
        if release < _MIN_RELEASE or release >= _MAX_RELEASE:
            raise _unsupported(f"{label} version {raw!r} is unsupported")
    releases = {release for _raw, release in versions.values()}
    if len(releases) != 1:
        rendered = ", ".join(
            f"{label}={raw}" for label, (raw, _release) in versions.items()
        )
        raise _unsupported(f"PyMuPDF distribution/module/binding disagree ({rendered})")
    return PyMuPDFRuntime(
        distribution_version=distribution_version,
        module_version=module_version,
        binding_version=binding_version,
    )


@dataclass(frozen=True, slots=True)
class PyMuPDFRuntime:
    distribution_version: str
    module_version: str
    binding_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement": PYMUPDF_REQUIREMENT,
            "distribution_version": self.distribution_version,
            "module_version": self.module_version,
            "binding_version": self.binding_version,
            "issue_5042": PYMUPDF_ISSUE_5042,
            "fixed_release": PYMUPDF_FIXED_RELEASE,
            "supported": True,
        }


@dataclass(frozen=True, slots=True)
class PypdfRuntime:
    distribution_version: str
    module_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement": PYPDF_REQUIREMENT,
            "distribution_version": self.distribution_version,
            "module_version": self.module_version,
            "private_content_stream_contract": PYPDF_PINNED_VERSION,
            "supported": True,
        }


def assert_supported_pymupdf_runtime() -> PyMuPDFRuntime:
    """Load and validate the exact PyMuPDF distribution and native module."""

    try:
        distribution_version = importlib_metadata.version(PYMUPDF_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError as error:
        raise _unsupported("PyMuPDF is not installed") from error
    try:
        pymupdf = import_module("pymupdf")
    except ImportError as error:
        raise _unsupported("the pymupdf module cannot be imported") from error
    module_version = str(getattr(pymupdf, "__version__", ""))
    binding_version = str(getattr(pymupdf, "VersionBind", ""))
    return validate_pymupdf_runtime_versions(
        distribution_version=distribution_version,
        module_version=module_version,
        binding_version=binding_version,
    )


def validate_pypdf_runtime_versions(
    *,
    distribution_version: str,
    module_version: str,
) -> PypdfRuntime:
    """Require the exact pypdf version whose private tokenizer is mirrored."""

    versions = {
        "distribution": distribution_version,
        "module": module_version,
    }
    for label, value in versions.items():
        if not isinstance(value, str) or not value or len(value) > 128:
            raise _unsupported_pypdf(
                f"pypdf {label} does not expose a bounded version string"
            )
    if distribution_version != module_version:
        raise _unsupported_pypdf(
            "pypdf distribution/module disagree "
            f"(distribution={distribution_version}, module={module_version})"
        )
    if distribution_version != PYPDF_PINNED_VERSION:
        raise _unsupported_pypdf(
            f"pypdf version {distribution_version!r} is unsupported"
        )
    return PypdfRuntime(
        distribution_version=distribution_version,
        module_version=module_version,
    )


def assert_supported_pypdf_runtime() -> PypdfRuntime:
    """Load and validate the exact pypdf source-parser implementation."""

    try:
        distribution_version = importlib_metadata.version(PYPDF_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError as error:
        raise _unsupported_pypdf("pypdf is not installed") from error
    try:
        pypdf = import_module("pypdf")
    except ImportError as error:
        raise _unsupported_pypdf("the pypdf module cannot be imported") from error
    module_version = str(getattr(pypdf, "__version__", ""))
    return validate_pypdf_runtime_versions(
        distribution_version=distribution_version,
        module_version=module_version,
    )


def _installed_distribution_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def describe_runtime_versions() -> dict[str, object]:
    """Return the canonical guarded runtime identity used by API and corpus."""

    pymupdf = assert_supported_pymupdf_runtime()
    pypdf = assert_supported_pypdf_runtime()
    packages: Mapping[str, str] = {
        "pymupdf": pymupdf.distribution_version,
        "pypdf": pypdf.distribution_version,
        "numpy": _installed_distribution_version("numpy"),
        "scipy": _installed_distribution_version("scipy"),
    }
    return {
        "schema_version": RUNTIME_VERSIONS_SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": dict(packages),
        "requirements": {
            "pymupdf": PYMUPDF_REQUIREMENT,
            "pypdf": PYPDF_REQUIREMENT,
        },
        "pymupdf_runtime": pymupdf.to_dict(),
        "pypdf_runtime": pypdf.to_dict(),
    }


__all__ = [
    "PYMUPDF_FIXED_RELEASE",
    "PYMUPDF_ISSUE_5042",
    "PYMUPDF_MAX_VERSION",
    "PYMUPDF_MIN_VERSION",
    "PYMUPDF_REQUIREMENT",
    "PYPDF_DISTRIBUTION",
    "PYPDF_PINNED_VERSION",
    "PYPDF_REQUIREMENT",
    "PypdfRuntime",
    "PyMuPDFRuntime",
    "RUNTIME_VERSIONS_SCHEMA_VERSION",
    "UnsupportedRuntimeError",
    "assert_supported_pymupdf_runtime",
    "assert_supported_pypdf_runtime",
    "describe_runtime_versions",
    "validate_pymupdf_runtime_versions",
    "validate_pypdf_runtime_versions",
]
