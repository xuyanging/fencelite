"""Single source of Python migration identities.

Candidate versions deliberately do not impersonate the frozen TypeScript
r10/r46 cache identities.  They become production identities only after the
operation-level migration gates pass.
"""

PYTHON_ENGINE_VERSION = (
    "python-line-type-engine-p10-source-text-image-style-aligned-2026-08-24"
)
PYTHON_METHOD1_ENGINE_VERSION = (
    f"{PYTHON_ENGINE_VERSION}/method1-r10-source-text-contract-candidate"
)
PYTHON_METHOD2_LOCAL_PROJECTION_VERSION = (
    "method2-local-group-projection-v1-source-group-owned-2026-08-24"
)
PYTHON_METHOD2_ENGINE_VERSION = (
    f"{PYTHON_ENGINE_VERSION}/method2-r46-source-native-frame-candidate"
)
PYTHON_FUSION_ENGINE_VERSION = (
    f"{PYTHON_ENGINE_VERSION}/fusion-policy-v1-local-projection-candidate"
)
PAGE_IR_VERSION = "source-aligned-page-ir-v10"
GROUPING_IR_VERSION = (
    "python-sequential-grouping-v3-source-text-image-style-candidate"
)
RESULT_SCHEMA_VERSION = 1
PAGE_ANALYSIS_SCHEMA_VERSION = 4
DOCUMENT_RUN_SCHEMA_VERSION = 3

FROZEN_TS_METHOD1_ENGINE_VERSION = "method1-alternating-crop-r10-2026-08-21"
FROZEN_TS_METHOD2_ENGINE_VERSION = "method2-sequential-multipath-r46-2026-08-24"
FROZEN_TS_FUSION_POLICY_VERSION = "method2-owns-overlap-v1"
