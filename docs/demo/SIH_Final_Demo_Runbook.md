# SIH Final Demo Runbook

## Presenter wording

LIVE: "This is LIVE MODEL OUTPUT. Browser microphone audio travels through the AudioWorklet, backend canonicalizer, AASIST, ECAPA, deterministic fusion, and the risk WebSocket. Replay, prosody, and context are N/A because those models are not implemented."

DEMO: "This is a SIMULATED POLICY DEMONSTRATION. The values are deterministic fixtures for the policy and UI flow; they are not model evidence."

## Primary sequence

1. Run `scripts/pre_demo_check.ps1` and confirm both models are ready.
2. Open the deployed or local application.
3. Select LIVE, select the high-value transfer scenario, and press Start.
4. Grant microphone permission and wait for `MIC STREAMING`, `LIVE CALL`, `audio_ready`, AASIST evidence, ECAPA similarity when a profile is configured, fusion state, and action.
5. Explain that the first AASIST decision waits for approximately 64,600 canonical samples at 16 kHz; the interim state is baseline/analyzing, not invented evidence.
6. Press Stop, then Reset. Start a second session and show a new call ID and clean state.

## Fallbacks

- Fallback A: verified local controlled LIVE model output.
- Fallback B: DEMO with the exact simulated-policy wording above.

Do not claim a physical microphone, hosted model output, Indian-language validation, telecom validation, blockchain, or unsupported signal.
