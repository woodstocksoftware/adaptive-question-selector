"""
Item Response Theory (IRT) Engine

Implements the 2-Parameter Logistic (2PL) model for adaptive testing.
"""

import math
import numpy as np
from scipy.optimize import minimize_scalar
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Question:
    """A question with IRT parameters."""
    id: str
    difficulty: float  # b parameter (-3 to +3)
    discrimination: float = 1.0  # a parameter (typically 0.5 to 2.5)
    content: str = ""
    topic_id: Optional[str] = None
    used: bool = False


@dataclass
class Response:
    """A student's response to a question."""
    question_id: str
    correct: bool
    difficulty: float
    discrimination: float


@dataclass
class AbilityEstimate:
    """Current estimate of student ability."""
    theta: float = 0.0  # ability estimate
    standard_error: float = 1.0  # uncertainty
    responses: List[Response] = field(default_factory=list)
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval."""
        margin = 1.96 * self.standard_error
        return (self.theta - margin, self.theta + margin)


class IRTEngine:
    """
    2-Parameter Logistic IRT Engine.
    
    Handles ability estimation and question selection using
    Maximum Fisher Information criterion.
    """
    
    def __init__(self):
        self.questions: dict[str, Question] = {}
    
    def add_question(self, question: Question):
        """Add a question to the pool."""
        self.questions[question.id] = question
    
    def probability_correct(self, theta: float, question: Question) -> float:
        """
        Calculate probability of correct response using 2PL model.
        
        P(X=1|θ) = 1 / (1 + e^(-a(θ - b)))
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
        """
        Calculate Fisher information for a question at given ability.
        
        I(θ) = a² * P(θ) * Q(θ)
        
        where Q(θ) = 1 - P(θ)
        """
        p = self.probability_correct(theta, question)
        q = 1 - p
        a = question.discrimination
        
        # Avoid division issues
        if p < 0.0001 or q < 0.0001:
            return 0.0
        
        return (a ** 2) * p * q
    
    def estimate_ability(self, responses: List[Response]) -> AbilityEstimate:
        """
        Estimate ability using Maximum Likelihood Estimation (MLE).
        
        For edge cases (all correct or all incorrect), uses Bayesian
        prior to regularize.
        """
        if not responses:
            return AbilityEstimate(theta=0.0, standard_error=1.0, responses=[])
        
        # Check for all correct or all incorrect
        all_correct = all(r.correct for r in responses)
        all_incorrect = all(not r.correct for r in responses)
        
        if all_correct:
            # Estimate high ability with uncertainty
            max_diff = max(r.difficulty for r in responses)
            theta = min(max_diff + 1.0, 3.0)
            se = 1.0 / math.sqrt(len(responses))
            return AbilityEstimate(theta=theta, standard_error=se, responses=responses)
        
        if all_incorrect:
            # Estimate low ability with uncertainty
            min_diff = min(r.difficulty for r in responses)
            theta = max(min_diff - 1.0, -3.0)
            se = 1.0 / math.sqrt(len(responses))
            return AbilityEstimate(theta=theta, standard_error=se, responses=responses)
        
        # MLE estimation
        def neg_log_likelihood(theta: float) -> float:
            ll = 0.0
            for r in responses:
                q = Question(id=r.question_id, difficulty=r.difficulty, 
                            discrimination=r.discrimination)
                p = self.probability_correct(theta, q)
                
                # Add small epsilon to prevent log(0)
                p = max(min(p, 0.9999), 0.0001)
                
                if r.correct:
                    ll += math.log(p)
                else:
                    ll += math.log(1 - p)
            
            # Add weak prior centered at 0
            ll -= 0.1 * (theta ** 2)
            
            return -ll
        
        # Find theta that maximizes likelihood
        result = minimize_scalar(neg_log_likelihood, bounds=(-4, 4), method='bounded')
        theta = result.x
        
        # Calculate standard error from Fisher information
        total_info = sum(
            self.information(theta, Question(id=r.question_id, difficulty=r.difficulty,
                                             discrimination=r.discrimination))
            for r in responses
        )
        
        se = 1.0 / math.sqrt(total_info) if total_info > 0 else 1.0
        
        return AbilityEstimate(theta=theta, standard_error=se, responses=responses)
    
    def select_next_question(
        self, 
        ability: AbilityEstimate,
        available_questions: List[Question],
        method: str = "max_info"
    ) -> Optional[Question]:
        """
        Select the next best question.
        
        Methods:
        - max_info: Maximum Fisher Information (most precise)
        - target_50: Target 50% probability (balanced)
        """
        if not available_questions:
            return None
        
        theta = ability.theta
        
        if method == "max_info":
            # Select question with maximum information at current theta
            best_question = None
            best_info = -1
            
            for q in available_questions:
                if q.used:
                    continue
                info = self.information(theta, q)
                if info > best_info:
                    best_info = info
                    best_question = q
            
            return best_question
        
        elif method == "target_50":
            # Select question closest to 50% probability
            best_question = None
            best_diff = float('inf')
            
            for q in available_questions:
                if q.used:
                    continue
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
        correct: bool
    ) -> AbilityEstimate:
        """Update ability estimate after a response."""
        new_response = Response(
            question_id=question.id,
            correct=correct,
            difficulty=question.difficulty,
            discrimination=question.discrimination
        )
        
        all_responses = current.responses + [new_response]
        return self.estimate_ability(all_responses)
    
    def get_test_summary(self, ability: AbilityEstimate) -> dict:
        """Generate summary statistics for completed test."""
        if not ability.responses:
            return {"error": "No responses recorded"}
        
        correct = sum(1 for r in ability.responses if r.correct)
        total = len(ability.responses)
        
        # Convert theta to percentile (assuming normal distribution)
        from scipy.stats import norm
        percentile = norm.cdf(ability.theta) * 100
        
        # Performance level
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
            "accuracy": round(correct / total * 100, 1)
        }
