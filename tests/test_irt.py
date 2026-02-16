"""Tests for the IRT engine — probability, information, estimation, and selection."""

import pytest

from src.irt import AbilityEstimate, IRTEngine, Question, Response


@pytest.fixture
def engine() -> IRTEngine:
    return IRTEngine()


@pytest.fixture
def sample_questions() -> list[Question]:
    return [
        Question(id="easy", difficulty=-1.5, discrimination=1.0),
        Question(id="medium", difficulty=0.0, discrimination=1.5),
        Question(id="hard", difficulty=1.5, discrimination=1.2),
        Question(id="very_hard", difficulty=2.5, discrimination=0.8),
        Question(id="very_easy", difficulty=-2.0, discrimination=1.0),
    ]


# ---------------------------------------------------------------
# Probability calculation
# ---------------------------------------------------------------


class TestProbabilityCorrect:
    def test_matching_difficulty_gives_50_percent(self, engine: IRTEngine):
        """When theta == difficulty, P(correct) should be 0.5."""
        q = Question(id="q1", difficulty=1.0, discrimination=1.0)
        p = engine.probability_correct(1.0, q)
        assert abs(p - 0.5) < 0.001

    def test_high_ability_high_probability(self, engine: IRTEngine):
        """High ability relative to difficulty -> high probability."""
        q = Question(id="q1", difficulty=-1.0, discrimination=1.0)
        p = engine.probability_correct(2.0, q)
        assert p > 0.9

    def test_low_ability_low_probability(self, engine: IRTEngine):
        """Low ability relative to difficulty -> low probability."""
        q = Question(id="q1", difficulty=2.0, discrimination=1.0)
        p = engine.probability_correct(-1.0, q)
        assert p < 0.1

    def test_higher_discrimination_steeper_curve(self, engine: IRTEngine):
        """Higher discrimination makes the curve steeper around difficulty."""
        q_low_a = Question(id="q1", difficulty=0.0, discrimination=0.5)
        q_high_a = Question(id="q2", difficulty=0.0, discrimination=2.5)

        p_low = engine.probability_correct(1.0, q_low_a)
        p_high = engine.probability_correct(1.0, q_high_a)
        assert p_high > p_low

    def test_probability_bounded_0_to_1(self, engine: IRTEngine):
        """Probability must always be in [0, 1]."""
        q = Question(id="q1", difficulty=0.0, discrimination=2.0)
        for theta in [-10, -3, -1, 0, 1, 3, 10]:
            p = engine.probability_correct(theta, q)
            assert 0.0 <= p <= 1.0

    def test_overflow_protection_extreme_positive(self, engine: IRTEngine):
        """Extremely low ability relative to difficulty should not overflow."""
        q = Question(id="q1", difficulty=100.0, discrimination=10.0)
        p = engine.probability_correct(-100.0, q)
        assert p == 0.0

    def test_overflow_protection_extreme_negative(self, engine: IRTEngine):
        """Extremely high ability relative to difficulty should not overflow."""
        q = Question(id="q1", difficulty=-100.0, discrimination=10.0)
        p = engine.probability_correct(100.0, q)
        assert p == 1.0


# ---------------------------------------------------------------
# Fisher information
# ---------------------------------------------------------------


class TestFisherInformation:
    def test_max_info_at_matching_difficulty(self, engine: IRTEngine):
        """Information is maximized when theta matches question difficulty."""
        q = Question(id="q1", difficulty=1.0, discrimination=1.0)
        info_at_match = engine.information(1.0, q)
        info_away = engine.information(3.0, q)
        assert info_at_match > info_away

    def test_higher_discrimination_more_info(self, engine: IRTEngine):
        """Higher discrimination yields more information."""
        q_low = Question(id="q1", difficulty=0.0, discrimination=0.5)
        q_high = Question(id="q2", difficulty=0.0, discrimination=2.0)
        assert engine.information(0.0, q_high) > engine.information(0.0, q_low)

    def test_info_nonnegative(self, engine: IRTEngine):
        """Fisher information is always >= 0."""
        q = Question(id="q1", difficulty=0.0, discrimination=1.5)
        for theta in [-3, -1, 0, 1, 3]:
            assert engine.information(theta, q) >= 0.0

    def test_info_zero_at_extreme_theta(self, engine: IRTEngine):
        """At extreme theta, P is near 0 or 1, so information should be ~0."""
        q = Question(id="q1", difficulty=0.0, discrimination=1.0)
        assert engine.information(100.0, q) == 0.0
        assert engine.information(-100.0, q) == 0.0

    def test_info_formula_manual(self, engine: IRTEngine):
        """Verify I(theta) = a^2 * P * Q manually."""
        q = Question(id="q1", difficulty=0.0, discrimination=1.5)
        theta = 0.5
        p = engine.probability_correct(theta, q)
        expected = (1.5**2) * p * (1 - p)
        assert abs(engine.information(theta, q) - expected) < 0.0001


# ---------------------------------------------------------------
# Ability estimation
# ---------------------------------------------------------------


class TestAbilityEstimation:
    def test_no_responses_returns_prior(self, engine: IRTEngine):
        """No responses -> theta=0, SE=1 (prior)."""
        est = engine.estimate_ability([])
        assert est.theta == 0.0
        assert est.standard_error == 1.0

    def test_all_correct_high_theta(self, engine: IRTEngine):
        """All correct responses -> positive theta estimate."""
        responses = [
            Response(question_id="q1", correct=True, difficulty=0.0, discrimination=1.0),
            Response(question_id="q2", correct=True, difficulty=0.5, discrimination=1.0),
            Response(question_id="q3", correct=True, difficulty=1.0, discrimination=1.0),
        ]
        est = engine.estimate_ability(responses)
        assert est.theta > 1.0

    def test_all_incorrect_low_theta(self, engine: IRTEngine):
        """All incorrect responses -> negative theta estimate."""
        responses = [
            Response(question_id="q1", correct=False, difficulty=0.0, discrimination=1.0),
            Response(question_id="q2", correct=False, difficulty=-0.5, discrimination=1.0),
            Response(question_id="q3", correct=False, difficulty=-1.0, discrimination=1.0),
        ]
        est = engine.estimate_ability(responses)
        assert est.theta < -1.0

    def test_mixed_responses_reasonable_theta(self, engine: IRTEngine):
        """Mixed correct/incorrect -> theta between extremes."""
        responses = [
            Response(question_id="q1", correct=True, difficulty=-1.0, discrimination=1.0),
            Response(question_id="q2", correct=True, difficulty=0.0, discrimination=1.0),
            Response(question_id="q3", correct=False, difficulty=0.5, discrimination=1.0),
            Response(question_id="q4", correct=False, difficulty=1.0, discrimination=1.0),
        ]
        est = engine.estimate_ability(responses)
        assert -2.0 < est.theta < 2.0
        assert est.standard_error > 0

    def test_more_responses_lower_se(self, engine: IRTEngine):
        """More responses should generally reduce standard error."""
        responses_few = [
            Response(question_id="q1", correct=True, difficulty=-0.5, discrimination=1.0),
            Response(question_id="q2", correct=False, difficulty=0.5, discrimination=1.0),
        ]
        responses_many = responses_few + [
            Response(question_id="q3", correct=True, difficulty=0.0, discrimination=1.0),
            Response(question_id="q4", correct=False, difficulty=1.0, discrimination=1.0),
            Response(question_id="q5", correct=True, difficulty=-1.0, discrimination=1.0),
            Response(question_id="q6", correct=False, difficulty=0.5, discrimination=1.2),
        ]
        est_few = engine.estimate_ability(responses_few)
        est_many = engine.estimate_ability(responses_many)
        assert est_many.standard_error < est_few.standard_error

    def test_all_correct_theta_bounded(self, engine: IRTEngine):
        """All correct edge case should not exceed theta=3."""
        responses = [
            Response(question_id=f"q{i}", correct=True, difficulty=2.5, discrimination=1.0)
            for i in range(10)
        ]
        est = engine.estimate_ability(responses)
        assert est.theta <= 3.0

    def test_all_incorrect_theta_bounded(self, engine: IRTEngine):
        """All incorrect edge case should not go below theta=-3."""
        responses = [
            Response(question_id=f"q{i}", correct=False, difficulty=-2.5, discrimination=1.0)
            for i in range(10)
        ]
        est = engine.estimate_ability(responses)
        assert est.theta >= -3.0

    def test_confidence_interval(self, engine: IRTEngine):
        """Confidence interval should bracket the theta estimate."""
        responses = [
            Response(question_id="q1", correct=True, difficulty=0.0, discrimination=1.0),
            Response(question_id="q2", correct=False, difficulty=1.0, discrimination=1.0),
        ]
        est = engine.estimate_ability(responses)
        lo, hi = est.confidence_interval
        assert lo < est.theta < hi
        assert abs((hi - lo) / 2 - 1.96 * est.standard_error) < 0.001


# ---------------------------------------------------------------
# Question selection
# ---------------------------------------------------------------


class TestQuestionSelection:
    def test_max_info_selects_matching_difficulty(
        self, engine: IRTEngine, sample_questions: list[Question]
    ):
        """max_info should prefer questions near current theta."""
        ability = AbilityEstimate(theta=0.0)
        selected = engine.select_next_question(ability, sample_questions, "max_info")
        assert selected is not None
        assert selected.id == "medium"

    def test_target_50_selects_matching_difficulty(
        self, engine: IRTEngine, sample_questions: list[Question]
    ):
        """target_50 should select question closest to 50% probability."""
        ability = AbilityEstimate(theta=0.0)
        selected = engine.select_next_question(ability, sample_questions, "target_50")
        assert selected is not None
        p = engine.probability_correct(0.0, selected)
        assert abs(p - 0.5) < 0.2

    def test_returns_none_when_empty(self, engine: IRTEngine):
        """Should return None when no questions available."""
        ability = AbilityEstimate(theta=0.0)
        assert engine.select_next_question(ability, [], "max_info") is None

    def test_invalid_method_raises(self, engine: IRTEngine, sample_questions: list[Question]):
        """Unknown selection method should raise ValueError."""
        ability = AbilityEstimate(theta=0.0)
        with pytest.raises(ValueError, match="Unknown selection method"):
            engine.select_next_question(ability, sample_questions, "invalid")

    def test_selection_adapts_to_ability(self, engine: IRTEngine, sample_questions: list[Question]):
        """max_info should select different questions for different abilities."""
        low_ability = AbilityEstimate(theta=-2.0)
        high_ability = AbilityEstimate(theta=2.0)

        low_pick = engine.select_next_question(low_ability, sample_questions, "max_info")
        high_pick = engine.select_next_question(high_ability, sample_questions, "max_info")

        assert low_pick is not None
        assert high_pick is not None
        assert low_pick.id != high_pick.id


# ---------------------------------------------------------------
# Update ability
# ---------------------------------------------------------------


class TestUpdateAbility:
    def test_correct_answer_increases_theta(self, engine: IRTEngine):
        """Correct answer on a hard question should increase theta."""
        ability = AbilityEstimate(theta=0.0)
        q = Question(id="q1", difficulty=1.0, discrimination=1.0)
        updated = engine.update_ability(ability, q, correct=True)
        assert updated.theta > 0.0

    def test_incorrect_answer_decreases_theta(self, engine: IRTEngine):
        """Incorrect answer on an easy question should decrease theta."""
        ability = AbilityEstimate(theta=0.0)
        q = Question(id="q1", difficulty=-1.0, discrimination=1.0)
        updated = engine.update_ability(ability, q, correct=False)
        assert updated.theta < 0.0

    def test_response_history_grows(self, engine: IRTEngine):
        """Each update should add to the response history."""
        ability = AbilityEstimate(theta=0.0)
        q1 = Question(id="q1", difficulty=0.0, discrimination=1.0)
        q2 = Question(id="q2", difficulty=0.5, discrimination=1.0)

        ability = engine.update_ability(ability, q1, correct=True)
        assert len(ability.responses) == 1

        ability = engine.update_ability(ability, q2, correct=False)
        assert len(ability.responses) == 2


# ---------------------------------------------------------------
# Test summary
# ---------------------------------------------------------------


class TestGetTestSummary:
    def test_summary_fields(self, engine: IRTEngine):
        """Summary should contain all expected fields."""
        responses = [
            Response(question_id="q1", correct=True, difficulty=0.0, discrimination=1.0),
            Response(question_id="q2", correct=False, difficulty=0.5, discrimination=1.0),
        ]
        ability = engine.estimate_ability(responses)
        summary = engine.get_test_summary(ability)

        assert "theta" in summary
        assert "standard_error" in summary
        assert "confidence_interval" in summary
        assert "percentile" in summary
        assert "performance_level" in summary
        assert "questions_answered" in summary
        assert "correct" in summary
        assert "accuracy" in summary

    def test_summary_accuracy(self, engine: IRTEngine):
        """Accuracy should be correct/total * 100."""
        responses = [
            Response(question_id="q1", correct=True, difficulty=0.0, discrimination=1.0),
            Response(question_id="q2", correct=True, difficulty=0.5, discrimination=1.0),
            Response(question_id="q3", correct=False, difficulty=1.0, discrimination=1.0),
        ]
        ability = engine.estimate_ability(responses)
        summary = engine.get_test_summary(ability)
        assert summary["correct"] == 2
        assert summary["questions_answered"] == 3
        assert abs(summary["accuracy"] - 66.7) < 0.1

    def test_performance_levels(self, engine: IRTEngine):
        """Verify performance level thresholds."""
        levels = [
            (2.0, "Advanced"),
            (1.0, "Proficient"),
            (0.0, "Basic"),
            (-1.0, "Below Basic"),
            (-2.0, "Needs Support"),
        ]
        for theta, expected_level in levels:
            ability = AbilityEstimate(
                theta=theta,
                standard_error=0.3,
                responses=[Response("q1", True, 0.0, 1.0)],
            )
            summary = engine.get_test_summary(ability)
            assert summary["performance_level"] == expected_level, (
                f"theta={theta}: expected {expected_level}, got {summary['performance_level']}"
            )

    def test_no_responses_raises_value_error(self, engine: IRTEngine):
        """Empty response history should raise ValueError."""
        ability = AbilityEstimate()
        with pytest.raises(ValueError, match="No responses recorded"):
            engine.get_test_summary(ability)

    def test_percentile_reasonable(self, engine: IRTEngine):
        """Percentile should be 0-100 and reasonable for the theta."""
        responses = [
            Response(question_id="q1", correct=True, difficulty=0.0, discrimination=1.0),
        ]
        ability = AbilityEstimate(theta=0.0, standard_error=0.5, responses=responses)
        summary = engine.get_test_summary(ability)
        assert 45 < summary["percentile"] < 55


# ---------------------------------------------------------------
# Full adaptive session
# ---------------------------------------------------------------


class TestAdaptiveSession:
    def test_full_session_converges(self, engine: IRTEngine):
        """A full adaptive session should converge on the true ability."""
        import random

        random.seed(42)

        questions = [
            Question(
                id=f"q{i}",
                difficulty=random.uniform(-2.5, 2.5),
                discrimination=random.uniform(0.5, 2.0),
            )
            for i in range(50)
        ]

        true_theta = 1.0
        ability = AbilityEstimate()

        for _ in range(20):
            available = [q for q in questions if not q.used]
            if not available:
                break

            next_q = engine.select_next_question(ability, available, "max_info")
            if not next_q:
                break

            p = engine.probability_correct(true_theta, next_q)
            correct = random.random() < p
            next_q.used = True
            ability = engine.update_ability(ability, next_q, correct)

        assert abs(ability.theta - true_theta) < 1.5
        assert ability.standard_error < 1.0

    def test_add_and_retrieve_question(self, engine: IRTEngine):
        """Questions added to the engine should be retrievable."""
        q = Question(id="test_q", difficulty=0.5, discrimination=1.2)
        engine.add_question(q)
        assert "test_q" in engine.questions
        assert engine.questions["test_q"].difficulty == 0.5
