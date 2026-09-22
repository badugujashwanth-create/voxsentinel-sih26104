from app.services.acceptance_telemetry import AcceptanceTelemetry, AcceptanceTelemetryRegistry


def test_snapshot_is_bounded_and_contains_no_audio_payload() -> None:
    telemetry = AcceptanceTelemetry("call-1", max_inferences=2)
    telemetry.record_inference(
        audio_window_sequence=1,
        raw_spoof_score=0.2,
        aggregate_score=0.2,
        policy_state="NORMAL",
        overall_risk_score=20,
        risk_level="LOW",
        recommended_action="MONITOR",
        ml_inference_latency_ms=10.0,
        steady_state_latency_ms=None,
    )
    telemetry.record_inference(
        audio_window_sequence=2,
        raw_spoof_score=0.8,
        aggregate_score=0.5,
        policy_state="ELEVATED_AUTHENTICITY_REVIEW",
        overall_risk_score=70,
        risk_level="HIGH",
        recommended_action="REQUIRE_CALLBACK",
        ml_inference_latency_ms=11.0,
        steady_state_latency_ms=42.0,
    )
    telemetry.record_inference(
        audio_window_sequence=3,
        raw_spoof_score=0.7,
        aggregate_score=0.7,
        policy_state="ELEVATED_AUTHENTICITY_REVIEW",
        overall_risk_score=70,
        risk_level="HIGH",
        recommended_action="REQUIRE_CALLBACK",
        ml_inference_latency_ms=12.0,
        steady_state_latency_ms=43.0,
    )

    snapshot = telemetry.snapshot()

    assert len(snapshot["inferences"]) == 2
    assert snapshot["inferences"][0]["audio_window_sequence"] == 2
    assert snapshot["inferences"][1]["raw_spoof_score"] == 0.7
    assert snapshot["inferences"][1]["score_semantics"] == "uncalibrated"
    assert "samples" not in snapshot["inferences"][0]
    assert "pcm" not in snapshot["inferences"][0]


def test_snapshot_records_warmup_and_cleanup_without_changing_policy() -> None:
    telemetry = AcceptanceTelemetry("call-2")
    telemetry.mark_stream_started()
    telemetry.microphone_state = "STREAMING"
    telemetry.mark_first_event()
    telemetry.record_inference(
        audio_window_sequence=1,
        raw_spoof_score=0.1,
        aggregate_score=0.1,
        policy_state="NORMAL",
        overall_risk_score=20,
        risk_level="LOW",
        recommended_action="MONITOR",
        ml_inference_latency_ms=5.0,
        steady_state_latency_ms=None,
    )
    telemetry.mark_cleanup()

    snapshot = telemetry.snapshot()

    assert snapshot["microphone_state"] == "IDLE"
    assert snapshot["warmup_latency_ms"] is not None
    assert snapshot["cleanup"] == {
        "producer_active": False,
        "canonicalizer_active": False,
        "session_active": False,
    }
    assert snapshot["inferences"][0]["overall_risk_score"] == 20


def test_registry_prunes_completed_snapshots_to_a_bounded_history() -> None:
    """Completed acceptance snapshots do not grow the process forever."""
    registry = AcceptanceTelemetryRegistry(max_snapshots=2)
    first = registry.create("call-1")
    first.mark_cleanup()
    second = registry.create("call-2")
    second.mark_cleanup()
    registry.create("call-3")

    assert registry.get("call-1") is None
    assert registry.get("call-2") is not None
    assert registry.get("call-3") is not None

def test_acceptance_registry_dependency_is_process_scoped():
    """The API telemetry dependency must be shared across start/read/stop calls."""
    from app.api.calls import get_acceptance_telemetry_registry
    assert get_acceptance_telemetry_registry() is get_acceptance_telemetry_registry()
