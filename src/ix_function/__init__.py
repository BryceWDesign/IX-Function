"""IX-Function package root.

IX-Function is a source-available research runtime for governed cross-domain
causal transfer evaluation. It is not an AGI claim and does not provide
operational autonomy by itself.
"""

from __future__ import annotations

__all__ = [
    "DONOR_REPOSITORIES",
    "PROJECT_NAME",
    "PROJECT_SLUG",
    "__version__",
]

PROJECT_NAME = "IX-Function"
PROJECT_SLUG = "ix-function"
__version__ = "0.1.0"

DONOR_REPOSITORIES: tuple[str, ...] = (
    "IX-CognitionKernel",
    "IX-IntentRealityLoop",
    "IX-BlackFox-WorldTwin",
    "IX-BlackFox-Cognition",
    "IX-BlackFox",
    "IX-Autonomy-Assurance-Case-Runtime",
    "IX",
)
