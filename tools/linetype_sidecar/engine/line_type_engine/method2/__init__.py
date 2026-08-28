"""Method2 recognition building blocks.

Modules in this package form the browser-neutral Python candidate pipeline
that targets the frozen r46 specification.  The implementation is wired into
the headless candidate runner, while its persisted identity remains distinct
from the frozen TypeScript oracle until migration parity is accepted.
"""

from .multi_path_carrier import (
    CarrierDelimitedMultiPathDetectionInput,
    CarrierDelimitedMultiPathEvidence,
    CarrierDelimitedMultiPathRegion,
    MultiPathCarrierDescriptor,
    MultiPathCarrierEndpoint,
    MultiPathCarrierPoint,
    detect_carrier_delimited_multi_path_regions,
)
from .contract import (
    LINE_TYPE_METHOD2_CONFIG_HASH,
    LINE_TYPE_METHOD2_FEATURES,
    METHOD2_ENGINE_VERSION,
    METHOD2_LOCAL_PROJECTION_VERSION,
    METHOD2_TARGET_SPEC_VERSION,
    LineTypeMethod2Audit,
    LineTypeMethod2Envelope,
    validate_line_type_method2_envelope,
)
from .recognizer import (
    LineTypeMethod2Result,
    line_type_method2_input_hash,
    recognize_line_types_method2,
    recognize_line_types_method2_page,
)

__all__ = [
    "CarrierDelimitedMultiPathDetectionInput",
    "CarrierDelimitedMultiPathEvidence",
    "CarrierDelimitedMultiPathRegion",
    "LINE_TYPE_METHOD2_CONFIG_HASH",
    "LINE_TYPE_METHOD2_FEATURES",
    "LineTypeMethod2Audit",
    "LineTypeMethod2Envelope",
    "LineTypeMethod2Result",
    "METHOD2_ENGINE_VERSION",
    "METHOD2_LOCAL_PROJECTION_VERSION",
    "METHOD2_TARGET_SPEC_VERSION",
    "MultiPathCarrierDescriptor",
    "MultiPathCarrierEndpoint",
    "MultiPathCarrierPoint",
    "detect_carrier_delimited_multi_path_regions",
    "line_type_method2_input_hash",
    "recognize_line_types_method2",
    "recognize_line_types_method2_page",
    "validate_line_type_method2_envelope",
]
