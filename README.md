# Adaptive Question Selector (IRT)

Item Response Theory-based adaptive testing engine that selects optimal questions based on student ability.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![IRT](https://img.shields.io/badge/IRT-2PL%20Model-purple)

## Features

- **2-Parameter Logistic (2PL) IRT Model**
- **Maximum Likelihood Estimation** for ability
- **Maximum Fisher Information** question selection
- **Real-time adaptation** after each response
- **Stopping rules** (SE threshold, max questions)
- **Simulation endpoint** for testing

## How It Works
```
P(correct) = 1 / (1 + e^(-a(θ - b)))

θ = student ability (-3 to +3)
b = question difficulty (-3 to +3)
a = discrimination (0.5 to 2.5)
```

The algorithm selects questions that maximize information at the student's current estimated ability level.

## Quick Start
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.server:app --reload --port 8002
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sessions` | POST | Create adaptive test session |
| `/sessions/{id}` | GET | Get session status |
| `/sessions/{id}/answer` | POST | Submit answer, get next question |
| `/simulate` | GET | Run simulation with true theta |
| `/estimate` | POST | Standalone ability estimation |

## Usage Example

### Create Session
```bash
curl -X POST http://localhost:8002/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "question_pool": [
      {"id": "q1", "difficulty": -1.5, "content": "Easy"},
      {"id": "q2", "difficulty": 0.0, "content": "Medium"},
      {"id": "q3", "difficulty": 1.5, "content": "Hard"}
    ],
    "max_questions": 10,
    "stopping_se": 0.4
  }'
```

### Submit Answer
```bash
curl -X POST http://localhost:8002/sessions/{session_id}/answer \
  -H "Content-Type: application/json" \
  -d '{"question_id": "q2", "correct": true}'
```

### Simulate Test
```bash
curl "http://localhost:8002/simulate?true_theta=1.0&num_questions=20"
```

## Selection Methods

| Method | Description |
|--------|-------------|
| `max_info` | Maximum Fisher Information (most precise) |
| `target_50` | Target 50% probability (balanced challenge) |

## Performance Levels

| Theta Range | Level |
|-------------|-------|
| ≥ 1.5 | Advanced |
| 0.5 to 1.5 | Proficient |
| -0.5 to 0.5 | Basic |
| -1.5 to -0.5 | Below Basic |
| < -1.5 | Needs Support |

## Part of Ed-Tech Suite

| Component | Repository |
|-----------|------------|
| [Question Bank MCP](https://github.com/woodstocksoftware/question-bank-mcp) | Question management |
| [Student Progress Tracker](https://github.com/woodstocksoftware/student-progress-tracker) | Performance analytics |
| [Simple Quiz Engine](https://github.com/woodstocksoftware/simple-quiz-engine) | Real-time quizzes |
| [Learning Curriculum Builder](https://github.com/woodstocksoftware/learning-curriculum-builder) | Curriculum design |
| [Real-Time Event Pipeline](https://github.com/woodstocksoftware/realtime-event-pipeline) | Event routing |
| **Adaptive Question Selector** | IRT-based adaptation |

## License

MIT

---

Built by [Jim Williams](https://linkedin.com/in/woodstocksoftware) | [GitHub](https://github.com/woodstocksoftware)
