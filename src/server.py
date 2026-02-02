"""
Adaptive Question Selector API

REST API for adaptive testing using IRT.
"""

import uuid
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .irt import IRTEngine, Question, AbilityEstimate


# ============================================================
# MODELS
# ============================================================

class QuestionCreate(BaseModel):
    id: Optional[str] = None
    difficulty: float = Field(..., ge=-3, le=3, description="Difficulty parameter (-3 to 3)")
    discrimination: float = Field(1.0, ge=0.1, le=3, description="Discrimination parameter")
    content: str = ""
    topic_id: Optional[str] = None


class SessionCreate(BaseModel):
    question_pool: List[QuestionCreate]
    selection_method: str = Field("max_info", description="max_info or target_50")
    max_questions: int = Field(20, ge=1, le=100)
    stopping_se: float = Field(0.3, ge=0.1, le=1.0, description="Stop when SE below this")


class AnswerSubmit(BaseModel):
    question_id: str
    correct: bool


class QuestionResponse(BaseModel):
    id: str
    difficulty: float
    discrimination: float
    content: str
    topic_id: Optional[str]
    probability_correct: float


class AbilityResponse(BaseModel):
    theta: float
    standard_error: float
    confidence_interval: List[float]
    questions_answered: int


class SessionStatus(BaseModel):
    session_id: str
    status: str  # active, completed
    current_ability: AbilityResponse
    questions_remaining: int
    next_question: Optional[QuestionResponse]
    summary: Optional[dict]


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Adaptive Question Selector",
    description="IRT-based adaptive testing engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory session storage
sessions: dict[str, dict] = {}


# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/sessions", response_model=SessionStatus)
async def create_session(request: SessionCreate):
    """
    Create a new adaptive testing session.
    
    Provide a pool of questions with IRT parameters.
    """
    session_id = f"irt_{uuid.uuid4().hex[:8]}"
    
    # Initialize IRT engine
    engine = IRTEngine()
    questions = []
    
    for q in request.question_pool:
        question = Question(
            id=q.id or f"q_{uuid.uuid4().hex[:6]}",
            difficulty=q.difficulty,
            discrimination=q.discrimination,
            content=q.content,
            topic_id=q.topic_id
        )
        engine.add_question(question)
        questions.append(question)
    
    # Initialize ability estimate
    ability = AbilityEstimate()
    
    # Select first question
    next_q = engine.select_next_question(ability, questions, request.selection_method)
    
    # Store session
    sessions[session_id] = {
        "engine": engine,
        "questions": questions,
        "ability": ability,
        "method": request.selection_method,
        "max_questions": request.max_questions,
        "stopping_se": request.stopping_se,
        "answered": 0,
        "status": "active"
    }
    
    return _build_status(session_id, next_q)


@app.get("/sessions/{session_id}", response_model=SessionStatus)
async def get_session(session_id: str):
    """Get current session status."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    engine = session["engine"]
    ability = session["ability"]
    
    # Get next question if session still active
    next_q = None
    if session["status"] == "active":
        available = [q for q in session["questions"] if not q.used]
        next_q = engine.select_next_question(ability, available, session["method"])
    
    return _build_status(session_id, next_q)


@app.post("/sessions/{session_id}/answer", response_model=SessionStatus)
async def submit_answer(session_id: str, answer: AnswerSubmit):
    """
    Submit an answer and get the next question.
    
    The system will update ability estimate and select the optimal next question.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if session["status"] != "active":
        raise HTTPException(status_code=400, detail="Session already completed")
    
    engine: IRTEngine = session["engine"]
    questions: List[Question] = session["questions"]
    ability: AbilityEstimate = session["ability"]
    
    # Find the question
    question = next((q for q in questions if q.id == answer.question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    if question.used:
        raise HTTPException(status_code=400, detail="Question already answered")
    
    # Mark as used
    question.used = True
    session["answered"] += 1
    
    # Update ability estimate
    ability = engine.update_ability(ability, question, answer.correct)
    session["ability"] = ability
    
    # Check stopping conditions
    available = [q for q in questions if not q.used]
    should_stop = (
        session["answered"] >= session["max_questions"] or
        ability.standard_error <= session["stopping_se"] or
        len(available) == 0
    )
    
    if should_stop:
        session["status"] = "completed"
        return _build_status(session_id, None)
    
    # Select next question
    next_q = engine.select_next_question(ability, available, session["method"])
    
    return _build_status(session_id, next_q)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"status": "deleted"}


@app.post("/estimate")
async def estimate_ability_standalone(responses: List[dict]):
    """
    Standalone ability estimation from response data.
    
    Useful for batch processing or integration with other systems.
    
    Each response: {"difficulty": float, "discrimination": float, "correct": bool}
    """
    from .irt import Response
    
    engine = IRTEngine()
    
    response_objects = [
        Response(
            question_id=f"q{i}",
            correct=r["correct"],
            difficulty=r["difficulty"],
            discrimination=r.get("discrimination", 1.0)
        )
        for i, r in enumerate(responses)
    ]
    
    ability = engine.estimate_ability(response_objects)
    summary = engine.get_test_summary(ability)
    
    return summary


@app.get("/simulate")
async def simulate_test(
    true_theta: float = 0.0,
    num_questions: int = 20,
    pool_size: int = 100
):
    """
    Simulate an adaptive test to demonstrate the algorithm.
    
    Creates a random question pool and simulates responses based on true_theta.
    """
    import random
    
    engine = IRTEngine()
    questions = []
    
    # Generate random question pool
    for i in range(pool_size):
        q = Question(
            id=f"sim_q{i}",
            difficulty=random.uniform(-2.5, 2.5),
            discrimination=random.uniform(0.5, 2.0)
        )
        engine.add_question(q)
        questions.append(q)
    
    ability = AbilityEstimate()
    history = []
    
    for step in range(num_questions):
        available = [q for q in questions if not q.used]
        if not available:
            break
        
        # Select next question
        next_q = engine.select_next_question(ability, available, "max_info")
        if not next_q:
            break
        
        # Simulate response based on true theta
        p_correct = engine.probability_correct(true_theta, next_q)
        correct = random.random() < p_correct
        
        next_q.used = True
        
        # Update ability
        ability = engine.update_ability(ability, next_q, correct)
        
        history.append({
            "step": step + 1,
            "question_difficulty": round(next_q.difficulty, 2),
            "correct": correct,
            "estimated_theta": round(ability.theta, 3),
            "standard_error": round(ability.standard_error, 3)
        })
    
    return {
        "true_theta": true_theta,
        "final_estimate": round(ability.theta, 3),
        "estimation_error": round(abs(ability.theta - true_theta), 3),
        "standard_error": round(ability.standard_error, 3),
        "questions_used": len(history),
        "history": history
    }


# ============================================================
# HELPERS
# ============================================================

def _build_status(session_id: str, next_question: Optional[Question]) -> SessionStatus:
    """Build session status response."""
    session = sessions[session_id]
    engine: IRTEngine = session["engine"]
    ability: AbilityEstimate = session["ability"]
    
    available = [q for q in session["questions"] if not q.used]
    
    ability_response = AbilityResponse(
        theta=round(ability.theta, 3),
        standard_error=round(ability.standard_error, 3),
        confidence_interval=[round(x, 3) for x in ability.confidence_interval],
        questions_answered=session["answered"]
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
            probability_correct=round(p, 3)
        )
    
    summary = None
    if session["status"] == "completed":
        summary = engine.get_test_summary(ability)
    
    return SessionStatus(
        session_id=session_id,
        status=session["status"],
        current_ability=ability_response,
        questions_remaining=len(available),
        next_question=next_q_response,
        summary=summary
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
