# Architecture

This document records only the agreed top-level boundaries. Detailed architecture will be added through reviewed tasks.

```text
Audio Source
    ↓
Streaming Backend
    ↓
Voice Analysis / ML
    ↓
Risk Fusion
    ↓
API / WebSocket
    ↓
Judge-Facing UI
    ↓
Verification / Prevention Action
```

## Ownership

```text
backend/ → Rohan
ml/      → Rohan

frontend/ → Person 2
demo/     → Person 2

docs/    → shared, reviewed
scripts/ → shared, reviewed
```
