# VoxSentinel

## Project

VoxSentinel

## Problem

Real-time detection and prevention of AI voice-cloning impersonation attacks.

## Prototype objective

Build a judge-ready SIH prototype capable of:

1. receiving live/near-live voice audio;
2. detecting synthetic/cloned speech;
3. verifying claimed speaker identity;
4. calculating a continuously updating impersonation risk;
5. explaining the risk;
6. triggering secondary verification for high-risk actions.

## Current Status

Core development is complete. The local controlled LIVE path loads AASIST and ECAPA, and the deterministic fallback is labeled `SIMULATED POLICY DEMONSTRATION`. Vercel/Render configuration is present in `vercel.json` and `render.yaml`; public deployment remains pending provider authentication and account setup.

Use `docs/demo/SIH_Final_Demo_Runbook.md` for the judge sequence and `scripts/pre_demo_check.ps1` before a local LIVE demonstration.
