# Adaptive Question Selector

Production-grade adaptive testing engine powered by Item Response Theory (IRT). Dynamically selects optimal questions based on real-time student ability estimation, delivering precise assessments in fewer questions than traditional fixed-length tests.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![IRT 2PL Model](https://img.shields.io/badge/IRT-2PL%20Model-purple.svg)](#mathematical-foundation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is Adaptive Testing?

Traditional tests give every student the same questions. Adaptive tests are smarter — they adjust in real time:

1. **Start** with a medium-difficulty question
2. **Student answers** correctly → next question is harder
3. **Student answers** incorrectly → next question is easier
4. **Converge** on the student's true ability in fewer questions

This is the same approach used by the GRE, GMAT, and many standardized assessments. The underlying math is **Item Response Theory (IRT)** — a psychometric framework that models the relationship between student ability and question difficulty.

```
Student answers → Update ability estimate → Select optimal next question → Repeat
        ↑                                                                      |
        └──────────────────────────────────────────────────────────────────────┘
```

## Features

- **2-Parameter Logistic (2PL) IRT model** — models both difficulty and discrimination
- **Maximum Likelihood Estimation (MLE)** — precise ability estimation with Bayesian regularization
- **Fisher Information question selection** — picks the most informative question at each step
- **Adaptive stopping rules** — ends when measurement precision is sufficient
- **Real-time REST API** — create sessions, submit answers, get next question
- **Simulation endpoint** — validate algorithm behavior with known true abilities
- **Security hardened** — input validation, session limits, TTL eviction, overflow protection
- **Health monitoring** — `/health` endpoint for orchestration and load balancers

## Quick Start

```bash
# Clone and setup
git clone https://github.com/woodstocksoftware/adaptive-question-selector.git
cd adaptive-question-selector
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the server
python -m uvicorn src.server:app --reload --port 8002
```

The API is now running at `http://localhost:8002`. Interactive docs at `http://localhost:8002/docs`.

## How It Works

### 1. Create a Session with Your Question Pool

```bash
curl -X POST http://localhost:8002/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "question_pool": [
      {"id": "q1", "difficulty": -2.0, "discrimination": 1.2, "content": "What is 2+2?"},
      {"id": "q2", "difficulty": -1.0, "discrimination": 1.0, "content": "Solve: 3x = 12"},
      {"id": "q3", "difficulty": 0.0,  "discrimination": 1.5, "content": "Factor: x² - 4"},
      {"id": "q4", "difficulty": 1.0,  "discrimination": 0.8, "content": "Derivative of sin(x)"},
      {"id": "q5", "difficulty": 2.0,  "discrimination": 1.3, "content": "Evaluate: ∫ e^x dx"}
    ],
    "selection_method": "max_info",
    "max_questions": 10,
    "stopping_se": 0.4
  }'
```

The response includes the first selected question and the initial ability estimate (θ = 0.0).

### 2. Submit Answers

```bash
curl -X POST http://localhost:8002/sessions/{session_id}/answer \
  -H "Content-Type: application/json" \
  -d '{"question_id": "q3", "correct": true}'
```

Each answer returns:
- Updated ability estimate (θ) with standard error
- The next optimal question (or session completion)
- 95% confidence interval on ability

### 3. Get Results

When the session completes (SE threshold met or max questions reached), the response includes a full summary:

```json
{
  "theta": 0.85,
  "standard_error": 0.38,
  "confidence_interval": [-0.105, 1.805],
  "percentile": 80.2,
  "performance_level": "Proficient",
  "questions_answered": 7,
  "correct": 5,
  "accuracy": 71.4
}
```

### 4. Simulate to Validate

Test the algorithm against a known true ability:

```bash
curl "http://localhost:8002/simulate?true_theta=1.5&num_questions=20&pool_size=100"
```

Returns step-by-step convergence history showing how the estimate approaches the true value.

## Mathematical Foundation

### The 2PL Model

The probability of a correct response is:

$$P(X=1|\theta) = \frac{1}{1 + e^{-a(\theta - b)}}$$

Where:
- **θ** (theta) — student ability, typically in [-3, +3]
- **b** — item difficulty, same scale as θ
- **a** — item discrimination, how well the item differentiates ability levels (0.1 to 3.0)

### Fisher Information

The information a question provides about ability:

$$I(\theta) = a^2 \cdot P(\theta) \cdot Q(\theta)$$

Where Q(θ) = 1 - P(θ). Information is **maximized** when P(θ) = 0.5 — when the question difficulty matches the student's ability. Higher discrimination (a) yields more information.

### Maximum Likelihood Estimation

Student ability is estimated by finding the θ that maximizes the log-likelihood:

$$\hat{\theta} = \arg\max_\theta \sum_{i} \left[ x_i \ln P_i(\theta) + (1-x_i) \ln Q_i(\theta) \right]$$

A weak N(0, σ²) prior provides regularization, preventing extreme estimates with sparse data. Standard error is derived from the inverse of total Fisher information.

## API Reference

### Endpoints

| Method | Path | Status | Description |
|--------|------|--------|-------------|
| `GET` | `/health` | 200 | Health check and session count |
| `POST` | `/sessions` | 201 | Create adaptive test session |
| `GET` | `/sessions/{id}` | 200 | Get session status and ability estimate |
| `POST` | `/sessions/{id}/answer` | 200 | Submit answer, receive next question |
| `DELETE` | `/sessions/{id}` | 200 | Delete session |
| `POST` | `/estimate` | 200 | Standalone ability estimation |
| `GET` | `/simulate` | 200 | Simulate adaptive test |

### GET /health

Returns server status, version, and active session count. Use for health checks and monitoring.

### POST /sessions → 201

Create a new adaptive testing session. Question pool is capped at 1,000 items; duplicate IDs are rejected.

**Request body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question_pool` | `QuestionCreate[]` | required | Questions with IRT parameters (max 1,000) |
| `selection_method` | `"max_info" \| "target_50"` | `"max_info"` | Selection strategy |
| `max_questions` | `int` | `20` | Maximum questions to administer (1-100) |
| `stopping_se` | `float` | `0.3` | Stop when SE falls below this (0.1-1.0) |

**Question parameters:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `id` | `string` | — | Unique identifier (auto-generated if omitted) |
| `difficulty` | `float` | [-3, 3] | Item difficulty (b parameter) |
| `discrimination` | `float` | [0.1, 3] | Item discrimination (a parameter) |
| `content` | `string` | — | Question text |
| `topic_id` | `string` | — | Optional topic grouping |

### POST /sessions/{id}/answer

Submit an answer and receive the next question.

**Request body:**

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | `string` | ID of the question being answered |
| `correct` | `bool` | Whether the answer was correct |

### POST /estimate

Standalone ability estimation from a batch of responses. Each response is validated via Pydantic.

**Request body:** Array of `ResponseInput` objects:

```json
[
  {"difficulty": -1.0, "discrimination": 1.0, "correct": true},
  {"difficulty": 0.5, "discrimination": 1.2, "correct": false},
  {"difficulty": -0.5, "discrimination": 0.8, "correct": true}
]
```

| Field | Type | Range | Default | Description |
|-------|------|-------|---------|-------------|
| `difficulty` | `float` | [-3, 3] | required | Item difficulty (b) |
| `discrimination` | `float` | [0.1, 3] | `1.0` | Item discrimination (a) |
| `correct` | `bool` | — | required | Whether the response was correct |

### GET /simulate

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `true_theta` | `float` | [-4, 4] | `0.0` | Simulated student ability |
| `num_questions` | `int` | [1, 200] | `20` | Questions to administer |
| `pool_size` | `int` | [1, 1000] | `100` | Random question pool size |

## Selection Methods

| Method | Strategy | Best For |
|--------|----------|----------|
| `max_info` | Maximize Fisher Information at current θ | Fastest convergence, most precise |
| `target_50` | Select question closest to 50% probability | Balanced student experience |

## Performance Levels

| θ Range | Level | Percentile |
|---------|-------|------------|
| ≥ 1.5 | Advanced | ~93rd+ |
| 0.5 to 1.5 | Proficient | ~69th - 93rd |
| -0.5 to 0.5 | Basic | ~31st - 69th |
| -1.5 to -0.5 | Below Basic | ~7th - 31st |
| < -1.5 | Needs Support | Below ~7th |

## Security

The API is hardened for production use:

- **Input validation** — all endpoints use Pydantic models with enforced parameter ranges
- **Session limits** — max 10,000 concurrent sessions with 1-hour TTL eviction
- **Session IDs** — 128-bit entropy via `secrets.token_hex(16)`
- **DoS protection** — question pool capped at 1,000; simulation params bounded
- **Overflow guards** — exponent clamping in probability calculation; `math.isfinite()` on MLE output
- **Error handling** — global exception handler prevents stack trace leakage
- **CORS** — wildcard origins without credentials (safe default)
- **Duplicate rejection** — duplicate question IDs in a pool return 400

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

69 tests with 98% coverage:
- IRT probability calculations and overflow protection
- MLE ability estimation (normal, all-correct, all-incorrect, empty)
- Fisher information computation
- Question selection (max_info and target_50 methods)
- API session lifecycle (create → answer → complete → delete)
- Stopping rules (SE threshold, max questions, pool exhaustion)
- Input validation (parameter ranges, duplicate IDs, invalid methods)
- Session security (capacity limits, TTL eviction)
- Standalone endpoints (/estimate validation, /simulate bounds)

## Research References

- Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques*. Marcel Dekker.
- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Lawrence Erlbaum.
- van der Linden, W. J., & Glas, C. A. W. (2010). *Elements of Adaptive Testing*. Springer.
- Weiss, D. J. (1982). Improving measurement quality and efficiency with adaptive testing. *Applied Psychological Measurement*, 6(4), 473-492.
- Hambleton, R. K., Swaminathan, H., & Rogers, H. J. (1991). *Fundamentals of Item Response Theory*. Sage.

## Part of the Ed-Tech Suite

| Component | Description |
|-----------|-------------|
| **Adaptive Question Selector** | IRT-based adaptive testing (this repo) |
| [Question Bank MCP](https://github.com/woodstocksoftware/question-bank-mcp) | Question management |
| [Student Progress Tracker](https://github.com/woodstocksoftware/student-progress-tracker) | Performance analytics |
| [Simple Quiz Engine](https://github.com/woodstocksoftware/simple-quiz-engine) | Real-time quizzes |
| [Learning Curriculum Builder](https://github.com/woodstocksoftware/learning-curriculum-builder) | Curriculum design |
| [Real-Time Event Pipeline](https://github.com/woodstocksoftware/realtime-event-pipeline) | Event routing |

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Ensure tests pass (`python -m pytest tests/ -v`)
4. Ensure linting passes (`ruff check src/ tests/`)
5. Submit a pull request

## License

MIT

---

Built by [Jim Williams](https://linkedin.com/in/woodstocksoftware) | [GitHub](https://github.com/woodstocksoftware)
