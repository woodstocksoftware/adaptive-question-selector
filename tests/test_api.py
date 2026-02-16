"""Tests for the FastAPI adaptive testing API."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.server import MAX_SESSIONS, app, sessions


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear session storage before each test."""
    sessions.clear()
    yield
    sessions.clear()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


SAMPLE_POOL = [
    {"id": "q1", "difficulty": -1.5, "discrimination": 1.0, "content": "Easy"},
    {"id": "q2", "difficulty": -0.5, "discrimination": 1.2, "content": "Below avg"},
    {"id": "q3", "difficulty": 0.0, "discrimination": 1.5, "content": "Medium"},
    {"id": "q4", "difficulty": 0.5, "discrimination": 1.0, "content": "Above avg"},
    {"id": "q5", "difficulty": 1.5, "discrimination": 1.3, "content": "Hard"},
]


# ---------------------------------------------------------------
# Health check
# ---------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient):
        """GET /health should return ok status."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert data["active_sessions"] == 0

    @pytest.mark.asyncio
    async def test_health_check_counts_sessions(self, client: AsyncClient):
        """Health check should report the number of active sessions."""
        await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL},
        )
        resp = await client.get("/health")
        assert resp.json()["active_sessions"] == 1


# ---------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_session(self, client: AsyncClient):
        """Creating a session should return 201 with active status."""
        resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL, "max_questions": 5},
        )
        assert resp.status_code == 201
        data = resp.json()

        assert data["status"] == "active"
        assert data["session_id"].startswith("irt_")
        assert len(data["session_id"]) > 20  # Full entropy session ID
        assert data["current_ability"]["theta"] == 0.0
        assert data["current_ability"]["questions_answered"] == 0
        assert data["questions_remaining"] == 5
        assert data["next_question"] is not None

    @pytest.mark.asyncio
    async def test_create_session_auto_ids(self, client: AsyncClient):
        """Questions without IDs should get auto-generated IDs."""
        resp = await client.post(
            "/sessions",
            json={"question_pool": [{"difficulty": 0.0}, {"difficulty": 1.0}]},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["next_question"]["id"].startswith("q_")

    @pytest.mark.asyncio
    async def test_create_session_custom_method(self, client: AsyncClient):
        """Session should accept target_50 selection method."""
        resp = await client.post(
            "/sessions",
            json={
                "question_pool": SAMPLE_POOL,
                "selection_method": "target_50",
            },
        )
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_reject_invalid_selection_method(self, client: AsyncClient):
        """Invalid selection_method should return 422."""
        resp = await client.post(
            "/sessions",
            json={
                "question_pool": SAMPLE_POOL,
                "selection_method": "random",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_reject_duplicate_question_ids(self, client: AsyncClient):
        """Duplicate question IDs should return 400."""
        pool = [
            {"id": "q1", "difficulty": 0.0},
            {"id": "q1", "difficulty": 1.0},
        ]
        resp = await client.post(
            "/sessions",
            json={"question_pool": pool},
        )
        assert resp.status_code == 400
        assert "Duplicate" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_reject_oversized_question_pool(self, client: AsyncClient):
        """Question pool exceeding max_length should return 422."""
        pool = [{"difficulty": 0.0}] * 1001
        resp = await client.post(
            "/sessions",
            json={"question_pool": pool},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------
# Submit answers
# ---------------------------------------------------------------


class TestSubmitAnswer:
    @pytest.mark.asyncio
    async def test_submit_answer_updates_ability(self, client: AsyncClient):
        """Submitting an answer should update the ability estimate."""
        create_resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL, "max_questions": 10},
        )
        session_id = create_resp.json()["session_id"]
        first_q_id = create_resp.json()["next_question"]["id"]

        resp = await client.post(
            f"/sessions/{session_id}/answer",
            json={"question_id": first_q_id, "correct": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["current_ability"]["questions_answered"] == 1
        assert data["next_question"] is not None

    @pytest.mark.asyncio
    async def test_submit_to_nonexistent_session(self, client: AsyncClient):
        """Submitting to a missing session should return 404."""
        resp = await client.post(
            "/sessions/fake_id/answer",
            json={"question_id": "q1", "correct": True},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_submit_nonexistent_question(self, client: AsyncClient):
        """Submitting an answer for a missing question should return 404."""
        create_resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL},
        )
        session_id = create_resp.json()["session_id"]

        resp = await client.post(
            f"/sessions/{session_id}/answer",
            json={"question_id": "nonexistent", "correct": True},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_submit_already_answered_question(self, client: AsyncClient):
        """Answering the same question twice should return 400."""
        create_resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL, "max_questions": 10},
        )
        session_id = create_resp.json()["session_id"]
        first_q_id = create_resp.json()["next_question"]["id"]

        await client.post(
            f"/sessions/{session_id}/answer",
            json={"question_id": first_q_id, "correct": True},
        )

        resp = await client.post(
            f"/sessions/{session_id}/answer",
            json={"question_id": first_q_id, "correct": False},
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_session_completes_on_max_questions(self, client: AsyncClient):
        """Session should complete when max_questions is reached."""
        create_resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL, "max_questions": 2},
        )
        session_id = create_resp.json()["session_id"]
        q_id = create_resp.json()["next_question"]["id"]

        resp = await client.post(
            f"/sessions/{session_id}/answer",
            json={"question_id": q_id, "correct": True},
        )
        data = resp.json()
        q_id = data["next_question"]["id"]

        resp = await client.post(
            f"/sessions/{session_id}/answer",
            json={"question_id": q_id, "correct": False},
        )
        data = resp.json()
        assert data["status"] == "completed"
        assert data["summary"] is not None
        assert "theta" in data["summary"]
        assert "performance_level" in data["summary"]

    @pytest.mark.asyncio
    async def test_cannot_answer_after_completion(self, client: AsyncClient):
        """Answering after session completion should return 400."""
        create_resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL, "max_questions": 1},
        )
        session_id = create_resp.json()["session_id"]
        q_id = create_resp.json()["next_question"]["id"]

        await client.post(
            f"/sessions/{session_id}/answer",
            json={"question_id": q_id, "correct": True},
        )

        resp = await client.post(
            f"/sessions/{session_id}/answer",
            json={"question_id": "q2", "correct": True},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_get_session(self, client: AsyncClient):
        """GET session should return current status."""
        create_resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL},
        )
        session_id = create_resp.json()["session_id"]

        resp = await client.get(f"/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, client: AsyncClient):
        """GET on missing session should return 404."""
        resp = await client.get("/sessions/fake_id")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session(self, client: AsyncClient):
        """DELETE should remove the session and return typed response."""
        create_resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL},
        )
        session_id = create_resp.json()["session_id"]

        resp = await client.delete(f"/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        resp = await client.get(f"/sessions/{session_id}")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, client: AsyncClient):
        """DELETE on missing session should return 404."""
        resp = await client.delete("/sessions/fake_id")
        assert resp.status_code == 404


# ---------------------------------------------------------------
# Standalone endpoints
# ---------------------------------------------------------------


class TestEstimateEndpoint:
    @pytest.mark.asyncio
    async def test_estimate_with_typed_input(self, client: AsyncClient):
        """POST /estimate should accept typed ResponseInput objects."""
        responses = [
            {"difficulty": -1.0, "discrimination": 1.0, "correct": True},
            {"difficulty": 0.0, "discrimination": 1.0, "correct": True},
            {"difficulty": 0.5, "discrimination": 1.0, "correct": False},
        ]
        resp = await client.post("/estimate", json=responses)
        assert resp.status_code == 200
        data = resp.json()
        assert "theta" in data
        assert "performance_level" in data
        assert "percentile" in data
        assert "accuracy" in data

    @pytest.mark.asyncio
    async def test_estimate_validates_difficulty(self, client: AsyncClient):
        """POST /estimate should reject difficulty outside [-3, 3]."""
        resp = await client.post(
            "/estimate",
            json=[{"difficulty": 10.0, "correct": True}],
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_estimate_validates_discrimination(self, client: AsyncClient):
        """POST /estimate should reject discrimination outside [0.1, 3]."""
        resp = await client.post(
            "/estimate",
            json=[{"difficulty": 0.0, "discrimination": 0.01, "correct": True}],
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_estimate_rejects_missing_fields(self, client: AsyncClient):
        """POST /estimate should reject items missing required fields."""
        resp = await client.post(
            "/estimate",
            json=[{"correct": True}],  # Missing difficulty
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_estimate_empty_list(self, client: AsyncClient):
        """POST /estimate with empty list should return 400."""
        resp = await client.post("/estimate", json=[])
        assert resp.status_code == 400
        assert "At least one response" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_estimate_default_discrimination(self, client: AsyncClient):
        """POST /estimate should default discrimination to 1.0."""
        resp = await client.post(
            "/estimate",
            json=[
                {"difficulty": -1.0, "correct": True},
                {"difficulty": 1.0, "correct": False},
            ],
        )
        assert resp.status_code == 200


class TestSimulateEndpoint:
    @pytest.mark.asyncio
    async def test_simulate_endpoint(self, client: AsyncClient):
        """GET /simulate should return typed simulation results."""
        resp = await client.get("/simulate?true_theta=1.0&num_questions=10&pool_size=50")
        assert resp.status_code == 200
        data = resp.json()
        assert data["true_theta"] == 1.0
        assert data["questions_used"] == 10
        assert len(data["history"]) == 10
        assert "final_estimate" in data
        assert "estimation_error" in data
        # Verify typed history steps
        step = data["history"][0]
        assert "step" in step
        assert "question_difficulty" in step
        assert "correct" in step
        assert "estimated_theta" in step
        assert "standard_error" in step

    @pytest.mark.asyncio
    async def test_simulate_rejects_extreme_pool_size(self, client: AsyncClient):
        """GET /simulate should reject pool_size > 1000."""
        resp = await client.get("/simulate?pool_size=5000")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_simulate_rejects_extreme_num_questions(self, client: AsyncClient):
        """GET /simulate should reject num_questions > 200."""
        resp = await client.get("/simulate?num_questions=500")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_simulate_rejects_extreme_theta(self, client: AsyncClient):
        """GET /simulate should reject true_theta outside [-4, 4]."""
        resp = await client.get("/simulate?true_theta=10.0")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_simulate_pool_exhaustion(self, client: AsyncClient):
        """Simulation should stop early when pool is smaller than num_questions."""
        resp = await client.get("/simulate?num_questions=10&pool_size=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["questions_used"] == 3
        assert len(data["history"]) == 3


# ---------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.asyncio
    async def test_difficulty_out_of_range(self, client: AsyncClient):
        """Difficulty outside [-3, 3] should be rejected."""
        resp = await client.post(
            "/sessions",
            json={"question_pool": [{"difficulty": 5.0}]},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_discrimination_too_low(self, client: AsyncClient):
        """Discrimination below 0.1 should be rejected."""
        resp = await client.post(
            "/sessions",
            json={"question_pool": [{"difficulty": 0.0, "discrimination": 0.01}]},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_max_questions_out_of_range(self, client: AsyncClient):
        """max_questions outside [1, 100] should be rejected."""
        resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL, "max_questions": 0},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_stopping_se_out_of_range(self, client: AsyncClient):
        """stopping_se outside [0.1, 1.0] should be rejected."""
        resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL, "stopping_se": 0.01},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------
# Session capacity and TTL
# ---------------------------------------------------------------


class TestSessionLimits:
    @pytest.mark.asyncio
    async def test_session_capacity_limit(self, client: AsyncClient):
        """Server should reject new sessions when at capacity."""
        # Fill sessions dict directly to avoid making 10k HTTP requests
        import time

        for i in range(MAX_SESSIONS):
            sessions[f"irt_fake_{i}"] = {
                "status": "active",
                "created_at": time.time(),
            }

        resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL},
        )
        assert resp.status_code == 503
        assert "capacity" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_expired_sessions_evicted(self, client: AsyncClient):
        """Expired sessions should be cleaned up on new session creation."""
        import time

        # Add an expired session
        sessions["irt_expired"] = {
            "status": "active",
            "created_at": time.time() - 7200,  # 2 hours ago
        }

        resp = await client.post(
            "/sessions",
            json={"question_pool": SAMPLE_POOL},
        )
        assert resp.status_code == 201
        assert "irt_expired" not in sessions


# ---------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------


class TestExceptionHandler:
    @pytest.mark.asyncio
    async def test_unhandled_exception_returns_500(self):
        """Unhandled exceptions should return 500 without stack traces."""
        from unittest.mock import patch

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            with patch("src.server._evict_expired_sessions", side_effect=RuntimeError("boom")):
                resp = await c.post(
                    "/sessions",
                    json={"question_pool": SAMPLE_POOL},
                )
        assert resp.status_code == 500
        assert resp.json() == {"detail": "Internal server error"}
