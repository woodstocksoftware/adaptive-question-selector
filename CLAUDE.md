# CLAUDE.md — Adaptive Question Selector

## Overview

IRT-based adaptive question selection engine using the 2-Parameter Logistic (2PL) model. Selects optimal questions based on real-time student ability estimation.

## Tech Stack

- **Language:** Python 3.12
- **Framework:** FastAPI + Uvicorn
- **Math:** SciPy (optimization + statistics)
- **Validation:** Pydantic v2
- **Testing:** pytest + httpx (async test client)
- **Linting:** Ruff

## Architecture

```
src/
├── __init__.py        # Package init
├── irt.py             # Core IRT engine (2PL model, MLE, Fisher information)
└── server.py          # FastAPI REST API (sessions, answers, simulation)
tests/
├── test_irt.py        # IRT engine unit tests
└── test_api.py        # API integration tests
```

- **No database** — in-memory session storage (dict-based, with TTL eviction)
- **Stateless engine** — `IRTEngine` is instantiated per session
- **Session lifecycle:** create → answer questions → auto-complete on stopping rule
- **Session limits:** max 10,000 concurrent sessions, 1-hour TTL

## IRT Model Details

**2-Parameter Logistic (2PL):**
- `θ` (theta) — student ability, range [-3, +3]
- `b` — item difficulty, range [-3, +3]
- `a` — item discrimination, range [0.1, 3.0]
- Formula: `P(X=1|θ) = 1 / (1 + e^(-a(θ - b)))`

**Key Algorithms:**

1. **MLE Ability Estimation** (`estimate_ability`)
   - Maximizes log-likelihood via `scipy.optimize.minimize_scalar`
   - Weak Bayesian prior (N(0, σ²)) for regularization
   - Edge case handling: all-correct → `max(difficulty) + 1`, all-incorrect → `min(difficulty) - 1`
   - Standard error from inverse Fisher information
   - NaN guard on optimization output

2. **Fisher Information Question Selection** (`select_next_question`)
   - `max_info`: selects question maximizing `I(θ) = a² × P(θ) × Q(θ)`
   - `target_50`: selects question closest to 50% correct probability

3. **Stopping Rules:**
   - SE below threshold (default 0.3)
   - Maximum questions reached (default 20)
   - No remaining questions in pool

## Commands

```bash
# Setup
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run server
python -m uvicorn src.server:app --reload --port 8002

# Run tests
python -m pytest tests/ -v

# Lint
ruff check src/ tests/
ruff format --check src/ tests/
```

## API Endpoints

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| GET | `/health` | 200 | Health check + active session count |
| POST | `/sessions` | 201 | Create adaptive test session with question pool |
| GET | `/sessions/{id}` | 200 | Get session status + current ability estimate |
| POST | `/sessions/{id}/answer` | 200 | Submit answer → updates ability → returns next question |
| DELETE | `/sessions/{id}` | 200 | Delete session |
| POST | `/estimate` | 200 | Standalone ability estimation from response data |
| GET | `/simulate` | 200 | Simulate full adaptive test with known true theta |

## Security Hardening

- **Input validation:** All endpoints use Pydantic models with range constraints
- **`/estimate` endpoint:** Uses typed `ResponseInput` model (not raw dicts)
- **`/simulate` bounds:** `pool_size` ≤ 1000, `num_questions` ≤ 200, `true_theta` ∈ [-4, 4]
- **CORS:** Wildcard origins allowed, but `allow_credentials` disabled
- **Session IDs:** 128-bit entropy via `secrets.token_hex(16)`
- **Session limits:** Max 10,000 sessions, 1-hour TTL with automatic eviction
- **Question pool cap:** Max 1,000 questions per session
- **Duplicate ID rejection:** Duplicate question IDs in a pool return 400
- **Selection method validation:** `Literal["max_info", "target_50"]` — invalid methods return 422
- **Global exception handler:** Unhandled errors return generic 500 (no stack traces)
- **Overflow protection:** Exponent clamped at ±700 in probability calculation
- **Numeric stability:** Fisher information returns 0 at extreme probabilities; NaN guard on MLE output
