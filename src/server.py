"""
Adaptive Question Selector API

REST API for adaptive testing using Item Response Theory (IRT).
Manages test sessions, processes answers, and selects optimal questions.
"""

import logging
import random
import secrets
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .irt import AbilityEstimate, IRTEngine, Question, Response

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

MAX_SESSIONS = 10_000
SESSION_TTL_SECONDS = 3600  # 1 hour
MAX_QUESTION_POOL_SIZE = 1000

# ============================================================
# REQUEST MODELS
# ============================================================


class QuestionCreate(BaseModel):
    """Request model for adding a question to a session pool."""

    id: str | None = None
    difficulty: float = Field(..., ge=-3, le=3, description="Difficulty parameter b (-3 to 3)")
    discrimination: float = Field(
        1.0, ge=0.1, le=3, description="Discrimination parameter a (0.1 to 3)"
    )
    content: str = ""
    topic_id: str | None = None


class SessionCreate(BaseModel):
    """Request model for creating an adaptive test session."""

    question_pool: list[QuestionCreate] = Field(..., max_length=MAX_QUESTION_POOL_SIZE)
    selection_method: Literal["max_info", "target_50"] = "max_info"
    max_questions: int = Field(20, ge=1, le=100)
    stopping_se: float = Field(
        0.3, ge=0.1, le=1.0, description="Stop when SE falls below this threshold"
    )


class AnswerSubmit(BaseModel):
    """Request model for submitting a student's answer."""

    question_id: str
    correct: bool


class ResponseInput(BaseModel):
    """Request model for a single response in standalone estimation."""

    difficulty: float = Field(..., ge=-3, le=3, description="Difficulty parameter b (-3 to 3)")
    discrimination: float = Field(
        1.0, ge=0.1, le=3, description="Discrimination parameter a (0.1 to 3)"
    )
    correct: bool


# ============================================================
# RESPONSE MODELS
# ============================================================


class QuestionResponse(BaseModel):
    """Response model for a selected question."""

    id: str
    difficulty: float
    discrimination: float
    content: str
    topic_id: str | None
    probability_correct: float


class AbilityResponse(BaseModel):
    """Response model for a student's current ability estimate."""

    theta: float
    standard_error: float
    confidence_interval: list[float]
    questions_answered: int


class TestSummaryResponse(BaseModel):
    """Response model for test summary statistics."""

    theta: float
    standard_error: float
    confidence_interval: list[float]
    percentile: float
    performance_level: str
    questions_answered: int
    correct: int
    accuracy: float


class SessionStatus(BaseModel):
    """Response model for the full session state."""

    session_id: str
    status: str  # "active" or "completed"
    current_ability: AbilityResponse
    questions_remaining: int
    next_question: QuestionResponse | None
    summary: TestSummaryResponse | None


class SimulationStep(BaseModel):
    """A single step in a simulation history."""

    step: int
    question_difficulty: float
    correct: bool
    estimated_theta: float
    standard_error: float


class SimulationResponse(BaseModel):
    """Response model for simulation results."""

    true_theta: float
    final_estimate: float
    estimation_error: float
    standard_error: float
    questions_used: int
    history: list[SimulationStep]


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    active_sessions: int


class DeleteResponse(BaseModel):
    """Response model for session deletion."""

    status: str


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Adaptive Question Selector",
    description="IRT-based adaptive testing engine for education platforms",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return a safe 500 response."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# In-memory session storage
sessions: dict[str, dict] = {}


def _evict_expired_sessions() -> None:
    """Remove sessions older than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [sid for sid, s in sessions.items() if now - s["created_at"] > SESSION_TTL_SECONDS]
    for sid in expired:
        del sessions[sid]


# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring and orchestration."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        active_sessions=len(sessions),
    )


@app.post("/sessions", response_model=SessionStatus, status_code=201)
async def create_session(request: SessionCreate) -> SessionStatus:
    """Create a new adaptive testing session.

    Provide a pool of questions with IRT parameters. The engine will
    select the optimal first question based on the chosen method.
    """
    _evict_expired_sessions()

    if len(sessions) >= MAX_SESSIONS:
        raise HTTPException(
            status_code=503,
            detail=f"Server at capacity ({MAX_SESSIONS} active sessions). Try again later.",
        )

    # Reject duplicate question IDs
    ids = [q.id for q in request.question_pool if q.id is not None]
    if len(ids) != len(set(ids)):
        raise HTTPException(
            status_code=400,
            detail="Duplicate question IDs in pool",
        )

    session_id = f"irt_{secrets.token_hex(16)}"

    engine = IRTEngine()
    questions: list[Question] = []

    for q in request.question_pool:
        question = Question(
            id=q.id or f"q_{uuid.uuid4().hex[:6]}",
            difficulty=q.difficulty,
            discrimination=q.discrimination,
            content=q.content,
            topic_id=q.topic_id,
        )
        engine.add_question(question)
        questions.append(question)

    ability = AbilityEstimate()
    next_q = engine.select_next_question(ability, questions, request.selection_method)

    sessions[session_id] = {
        "engine": engine,
        "questions": questions,
        "ability": ability,
        "method": request.selection_method,
        "max_questions": request.max_questions,
        "stopping_se": request.stopping_se,
        "answered": 0,
        "status": "active",
        "created_at": time.time(),
    }

    return _build_status(session_id, next_q)


@app.get("/sessions/{session_id}", response_model=SessionStatus)
async def get_session(session_id: str) -> SessionStatus:
    """Get current session status including ability estimate and next question."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    engine: IRTEngine = session["engine"]
    ability: AbilityEstimate = session["ability"]

    next_q = None
    if session["status"] == "active":
        available = [q for q in session["questions"] if not q.used]
        next_q = engine.select_next_question(ability, available, session["method"])

    return _build_status(session_id, next_q)


@app.post("/sessions/{session_id}/answer", response_model=SessionStatus)
async def submit_answer(session_id: str, answer: AnswerSubmit) -> SessionStatus:
    """Submit an answer and get the next optimal question.

    Updates the ability estimate based on the response and selects
    the next question using Fisher Information or target probability.
    The session completes when a stopping rule is met.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Session already completed")

    engine: IRTEngine = session["engine"]
    questions: list[Question] = session["questions"]
    ability: AbilityEstimate = session["ability"]

    # Find the question
    question = next((q for q in questions if q.id == answer.question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    if question.used:
        raise HTTPException(status_code=400, detail="Question already answered")

    # Mark as used and update
    question.used = True
    session["answered"] += 1

    ability = engine.update_ability(ability, question, answer.correct)
    session["ability"] = ability

    # Check stopping conditions
    available = [q for q in questions if not q.used]
    should_stop = (
        session["answered"] >= session["max_questions"]
        or ability.standard_error <= session["stopping_se"]
        or len(available) == 0
    )

    if should_stop:
        session["status"] = "completed"
        return _build_status(session_id, None)

    next_q = engine.select_next_question(ability, available, session["method"])
    return _build_status(session_id, next_q)


@app.delete("/sessions/{session_id}", response_model=DeleteResponse)
async def delete_session(session_id: str) -> DeleteResponse:
    """Delete a session and free its resources."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    return DeleteResponse(status="deleted")


@app.post("/estimate", response_model=TestSummaryResponse)
async def estimate_ability_standalone(
    responses: list[ResponseInput],
) -> TestSummaryResponse:
    """Standalone ability estimation from response data.

    Useful for batch processing or integration with other systems.
    """
    if not responses:
        raise HTTPException(
            status_code=400,
            detail="At least one response is required",
        )

    engine = IRTEngine()

    response_objects = [
        Response(
            question_id=f"q{i}",
            correct=r.correct,
            difficulty=r.difficulty,
            discrimination=r.discrimination,
        )
        for i, r in enumerate(responses)
    ]

    ability = engine.estimate_ability(response_objects)
    summary = engine.get_test_summary(ability)
    return TestSummaryResponse(**summary)


@app.get("/simulate", response_model=SimulationResponse)
async def simulate_test(
    true_theta: float = Query(0.0, ge=-4, le=4),
    num_questions: int = Query(20, ge=1, le=200),
    pool_size: int = Query(100, ge=1, le=1000),
) -> SimulationResponse:
    """Simulate an adaptive test to demonstrate algorithm convergence.

    Creates a random question pool and simulates responses based on
    a known true_theta. Returns step-by-step convergence history.
    """
    engine = IRTEngine()
    questions: list[Question] = []

    for i in range(pool_size):
        q = Question(
            id=f"sim_q{i}",
            difficulty=random.uniform(-2.5, 2.5),
            discrimination=random.uniform(0.5, 2.0),
        )
        engine.add_question(q)
        questions.append(q)

    ability = AbilityEstimate()
    history: list[SimulationStep] = []

    for step in range(num_questions):
        available = [q for q in questions if not q.used]
        if not available:
            break

        next_q = engine.select_next_question(ability, available, "max_info")
        if not next_q:
            break

        p_correct = engine.probability_correct(true_theta, next_q)
        correct = random.random() < p_correct

        next_q.used = True
        ability = engine.update_ability(ability, next_q, correct)

        history.append(
            SimulationStep(
                step=step + 1,
                question_difficulty=round(next_q.difficulty, 2),
                correct=correct,
                estimated_theta=round(ability.theta, 3),
                standard_error=round(ability.standard_error, 3),
            )
        )

    return SimulationResponse(
        true_theta=true_theta,
        final_estimate=round(ability.theta, 3),
        estimation_error=round(abs(ability.theta - true_theta), 3),
        standard_error=round(ability.standard_error, 3),
        questions_used=len(history),
        history=history,
    )


# ============================================================
# HELPERS
# ============================================================


def _build_status(session_id: str, next_question: Question | None) -> SessionStatus:
    """Build a SessionStatus response from current session state."""
    session = sessions[session_id]
    engine: IRTEngine = session["engine"]
    ability: AbilityEstimate = session["ability"]

    available = [q for q in session["questions"] if not q.used]

    ability_response = AbilityResponse(
        theta=round(ability.theta, 3),
        standard_error=round(ability.standard_error, 3),
        confidence_interval=[round(x, 3) for x in ability.confidence_interval],
        questions_answered=session["answered"],
    )

    next_q_response = None
    if next_question:
        p = engine.probability_correct(ability.theta, next_question)
        next_q_response = QuestionResponse(
            id=next_question.id,
            difficulty=round(next_question.difficulty, 3),
            discrimination=round(next_question.discrimination, 3),
            content=next_question.content,
            topic_id=next_question.topic_id,
            probability_correct=round(p, 3),
        )

    summary = None
    if session["status"] == "completed":
        raw_summary = engine.get_test_summary(ability)
        summary = TestSummaryResponse(**raw_summary)

    return SessionStatus(
        session_id=session_id,
        status=session["status"],
        current_ability=ability_response,
        questions_remaining=len(available),
        next_question=next_q_response,
        summary=summary,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)
