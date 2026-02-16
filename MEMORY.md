# MEMORY.md — Adaptive Question Selector

## Current State

- Standalone IRT adaptive testing engine — 2PL model implementation
- Two source files: `src/irt.py` (core engine) and `src/server.py` (FastAPI API)
- Fully functional with session management, ability estimation, and question selection
- Test suite: `tests/test_irt.py` (engine) and `tests/test_api.py` (API)
- Security-hardened: input validation on all endpoints, session limits, TTL eviction

## Relationship to adaptivetest-platform

- This repo may be the **earlier standalone prototype**
- The platform version in `adaptivetest-platform/src/adaptive/` is the **canonical production version**
- This repo serves as the **public portfolio showcase** of the core algorithm
- Changes here should be synced to the platform if algorithm logic diverges

## Security Hardening Applied

- **Input validation:** All endpoints use Pydantic models (including `/estimate` via `ResponseInput`)
- **NaN guards:** Overflow protection in `probability_correct` (exponent clamped at ±700); `math.isfinite()` check on MLE output
- **SE caps:** Fisher information returns 0 when P or Q < 0.0001
- **Parameter validation:** Pydantic models enforce ranges (difficulty: -3 to 3, discrimination: 0.1 to 3)
- **Selection method:** Validated as `Literal["max_info", "target_50"]` — rejects invalid values at 422
- **Edge cases:** All-correct and all-incorrect response patterns handled with Bayesian fallback
- **Epsilon clamping:** Log-likelihood uses `max(min(p, 0.9999), 0.0001)` to prevent log(0)
- **Session security:** 128-bit entropy IDs (`secrets.token_hex(16)`), max 10,000 sessions, 1-hour TTL
- **DoS protection:** Question pool capped at 1,000; `/simulate` params bounded (pool_size ≤ 1000, num_questions ≤ 200)
- **CORS:** Wildcard origins without `allow_credentials` (safe configuration)
- **Error handling:** Global exception handler prevents stack trace leakage
- **Duplicate rejection:** Duplicate question IDs in a pool return 400

## Key Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| θ (theta) | -3 to +3 | 0.0 | Student ability |
| b (difficulty) | -3 to +3 | — | Item difficulty |
| a (discrimination) | 0.1 to 3.0 | 1.0 | Item discrimination |
| stopping_se | 0.1 to 1.0 | 0.3 | SE threshold to stop |
| max_questions | 1 to 100 | 20 | Max items administered |
