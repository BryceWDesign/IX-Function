from __future__ import annotations

from ix_function import DONOR_REPOSITORIES, PROJECT_NAME, PROJECT_SLUG, __version__


def test_project_identity_is_stable() -> None:
    assert PROJECT_NAME == "IX-Function"
    assert PROJECT_SLUG == "ix-function"
    assert __version__ == "0.1.0"


def test_declared_donor_repositories_are_explicit() -> None:
    assert DONOR_REPOSITORIES == (
        "IX-CognitionKernel",
        "IX-IntentRealityLoop",
        "IX-BlackFox-WorldTwin",
        "IX-BlackFox-Cognition",
        "IX-BlackFox",
        "IX-Autonomy-Assurance-Case-Runtime",
        "IX",
    )
