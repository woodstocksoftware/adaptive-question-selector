"""
Item Response Theory (IRT) Engine

Implements the 2-Parameter Logistic (2PL) model for adaptive testing.
Provides ability estimation via MLE and question selection via
Maximum Fisher Information criterion.
"""

import math
from dataclasses import dataclass, field

from scipy.optimize import minimize_scalar
from scipy.stats import norm


@dataclass
class Question:
    """A question with IRT parameters.

    Attributes:
        id: Unique question identifier.
        difficulty: Item difficulty parameter (b), range [-3, +3].
            Higher values = harder questions.
        discrimination: Item discrimination parameter (a), range [0.1, 3.0].
            Higher values = better at differentiating ability levels.
        content: Question text or display content.
        topic_id: Optional topic grouping for content balancing.
        used: Whether this question has been administered in the current session.
    """

    id: str
    difficulty: float
    discrimination: float = 1.0
    content: str = ""
    topic_id: str | None = None
    used: bool = False


@dataclass
class Response:
    """A student's response to a question.

    Attributes:
        question_id: ID of the answered question.
        correct: Whether the response was correct.
        difficulty: The question's difficulty parameter (b).
        discrimination: The question's discrimination parameter (a).
    """

    question_id: str
    correct: bool
    difficulty: float
    discrimination: float


@dataclass
class AbilityEstimate:
    """Current estimate of student ability.

    Attributes:
        theta: Estimated ability on the IRT scale, typically [-3, +3].
        standard_error: Uncertainty of the estimate. Lower = more precise.
        responses: History of all responses used in this estimate.
    """

    theta: float = 0.0
    standard_error: float = 1.0
    responses: list[Response] = field(default_factory=list)

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% confidence interval for the ability estimate."""
        margin = 1.96 * self.standard_error
        return (self.theta - margin, self.theta + margin)


class IRTEngine:
    """2-Parameter Logistic IRT Engine.

    Handles ability estimation and adaptive question selection using
    Maximum Likelihood Estimation and Fisher Information criteria.

    The 2PL model: P(X=1|theta) = 1 / (1 + e^(-a(theta - b)))

    Usage:
        engine = IRTEngine()
        engine.add_question(Question(id="q1", difficulty=0.5, discrimination=1.2))
        ability = AbilityEstimate()
        next_q = engine.select_next_question(ability, list(engine.questions.values()))
    """

    def __init__(self) -> None:
        self.questions: dict[str, Question] = {}

    def add_question(self, question: Question) -> None:
        """Add a question to the item pool.

        Args:
            question: Question with calibrated IRT parameters.
        """
        self.questions[question.id] = question

    def probability_correct(self, theta: float, question: Question) -> float:
        """Calculate probability of correct response using the 2PL model.

        P(X=1|theta) = 1 / (1 + e^(-a(theta - b)))

        Args:
            theta: Student ability estimate.
            question: Question with difficulty and discrimination parameters.

        Returns:
            Probability of a correct response, in [0, 1].
        """
        a = question.discrimination
        b = question.difficulty
        exponent = -a * (theta - b)

        # Prevent overflow
        if exponent > 700:
            return 0.0
        if exponent < -700:
            return 1.0

        return 1.0 / (1.0 + math.exp(exponent))

    def information(self, theta: float, question: Question) -> float:
        """Calculate Fisher information for a question at a given ability level.

        I(theta) = a^2 * P(theta) * Q(theta)

        Information is maximized when P(theta) = 0.5, i.e., when the question
        difficulty matches the student's ability.

        Args:
            theta: Student ability estimate.
            question: Question with IRT parameters.

        Returns:
            Fisher information value. Higher = more informative at this ability.
        """
        p = self.probability_correct(theta, question)
        q = 1 - p
        a = question.discrimination

        # Avoid numerical instability at extreme probabilities
        if p < 0.0001 or q < 0.0001:
            return 0.0

        return (a**2) * p * q

    def estimate_ability(self, responses: list[Response]) -> AbilityEstimate:
        """Estimate student ability using Maximum Likelihood Estimation (MLE).

        Uses scipy bounded optimization to find the theta that maximizes the
        log-likelihood of observed responses. A weak N(0, sigma^2) prior
        provides regularization.

        Edge cases:
            - No responses: returns theta=0, SE=1 (prior)
            - All correct: heuristic estimate above max difficulty
            - All incorrect: heuristic estimate below min difficulty

        Args:
            responses: List of student responses with IRT parameters.

        Returns:
            AbilityEstimate with theta, standard error, and response history.
        """
        if not responses:
            return AbilityEstimate(theta=0.0, standard_error=1.0, responses=[])

        # Check for all correct or all incorrect
        all_correct = all(r.correct for r in responses)
        all_incorrect = all(not r.correct for r in responses)

        if all_correct:
            max_diff = max(r.difficulty for r in responses)
            theta = min(max_diff + 1.0, 3.0)
            se = 1.0 / math.sqrt(len(responses))
            return AbilityEstimate(theta=theta, standard_error=se, responses=responses)

        if all_incorrect:
            min_diff = min(r.difficulty for r in responses)
            theta = max(min_diff - 1.0, -3.0)
            se = 1.0 / math.sqrt(len(responses))
            return AbilityEstimate(theta=theta, standard_error=se, responses=responses)

        # MLE estimation
        def neg_log_likelihood(theta: float) -> float:
            ll = 0.0
            for r in responses:
                q = Question(
                    id=r.question_id,
                    difficulty=r.difficulty,
                    discrimination=r.discrimination,
                )
                p = self.probability_correct(theta, q)

                # Epsilon clamp to prevent log(0)
                p = max(min(p, 0.9999), 0.0001)

                if r.correct:
                    ll += math.log(p)
                else:
                    ll += math.log(1 - p)

            # Weak prior centered at 0 for regularization
            ll -= 0.1 * (theta**2)

            return -ll

        # Find theta that maximizes likelihood
        result = minimize_scalar(neg_log_likelihood, bounds=(-4, 4), method="bounded")
        theta = float(result.x)

        # Guard against non-finite optimization results
        if not math.isfinite(theta):
            theta = 0.0

        # Standard error from inverse Fisher information
        total_info = sum(
            self.information(
                theta,
                Question(
                    id=r.question_id,
                    difficulty=r.difficulty,
                    discrimination=r.discrimination,
                ),
            )
            for r in responses
        )

        se = 1.0 / math.sqrt(total_info) if total_info > 0 else 1.0

        return AbilityEstimate(theta=theta, standard_error=se, responses=responses)

    def select_next_question(
        self,
        ability: AbilityEstimate,
        available_questions: list[Question],
        method: str = "max_info",
    ) -> Question | None:
        """Select the next optimal question for the student.

        Args:
            ability: Current ability estimate.
            available_questions: Pool of candidate questions (should be unused only).
            method: Selection strategy:
                - "max_info": Maximum Fisher Information (most precise).
                - "target_50": Target 50% correct probability (balanced challenge).

        Returns:
            The selected question, or None if no questions are available.

        Raises:
            ValueError: If method is not recognized.
        """
        if not available_questions:
            return None

        theta = ability.theta

        if method == "max_info":
            best_question: Question | None = None
            best_info = -1.0

            for q in available_questions:
                info = self.information(theta, q)
                if info > best_info:
                    best_info = info
                    best_question = q

            return best_question

        elif method == "target_50":
            best_question = None
            best_diff = float("inf")

            for q in available_questions:
                p = self.probability_correct(theta, q)
                diff = abs(p - 0.5)
                if diff < best_diff:
                    best_diff = diff
                    best_question = q

            return best_question

        else:
            raise ValueError(f"Unknown selection method: {method}")

    def update_ability(
        self,
        current: AbilityEstimate,
        question: Question,
        correct: bool,
    ) -> AbilityEstimate:
        """Update ability estimate after a new response.

        Appends the response to the history and re-estimates ability
        using the full response pattern.

        Args:
            current: Current ability estimate with response history.
            question: The question that was answered.
            correct: Whether the response was correct.

        Returns:
            Updated AbilityEstimate incorporating the new response.
        """
        new_response = Response(
            question_id=question.id,
            correct=correct,
            difficulty=question.difficulty,
            discrimination=question.discrimination,
        )

        all_responses = current.responses + [new_response]
        return self.estimate_ability(all_responses)

    def get_test_summary(self, ability: AbilityEstimate) -> dict:
        """Generate summary statistics for a completed or in-progress test.

        Args:
            ability: The current or final ability estimate.

        Returns:
            Dictionary with theta, standard_error, confidence_interval,
            percentile, performance_level, questions_answered, correct count,
            and accuracy percentage.

        Raises:
            ValueError: If no responses have been recorded.
        """
        if not ability.responses:
            raise ValueError("No responses recorded")

        correct = sum(1 for r in ability.responses if r.correct)
        total = len(ability.responses)

        # Convert theta to percentile (assuming normal distribution)
        percentile = float(norm.cdf(ability.theta) * 100)

        # Performance level classification
        if ability.theta >= 1.5:
            level = "Advanced"
        elif ability.theta >= 0.5:
            level = "Proficient"
        elif ability.theta >= -0.5:
            level = "Basic"
        elif ability.theta >= -1.5:
            level = "Below Basic"
        else:
            level = "Needs Support"

        return {
            "theta": round(ability.theta, 3),
            "standard_error": round(ability.standard_error, 3),
            "confidence_interval": [round(x, 3) for x in ability.confidence_interval],
            "percentile": round(percentile, 1),
            "performance_level": level,
            "questions_answered": total,
            "correct": correct,
            "accuracy": round(correct / total * 100, 1),
        }
