"""Opt-in process-level ML pipeline acceptance checks."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("VOXSENTINEL_RUN_REAL_INTEGRATION") != "1", reason="set VOXSENTINEL_RUN_REAL_INTEGRATION=1 with ML and backend services running")
def test_real_pipeline_acceptance_is_run_by_the_controlled_harness() -> None:
    """Requires the process-level harness instead of faking a real model run."""
    pytest.fail("Run scripts/run_ml_acceptance.py against the verified real services and record its evidence")
