"""Opt-in acceptance marker for the physical browser microphone pipeline."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("JASH005_LIVE_MIC_ACCEPTANCE") != "1",
    reason="set JASH005_LIVE_MIC_ACCEPTANCE=1 and run the physical browser harness",
)
def test_physical_browser_microphone_acceptance_requires_recorded_evidence() -> None:
    """Prevents CI from claiming physical microphone evidence without the operator run."""
    evidence_path = os.getenv("JASH005_MIC_EVIDENCE_FILE")
    if not evidence_path:
        pytest.fail("JASH005_MIC_EVIDENCE_FILE must point to the operator acceptance telemetry")
    assert os.path.isfile(evidence_path), evidence_path
