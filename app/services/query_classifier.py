"""
Query intent classifier for routing queries to appropriate retrieval/generation strategies.
"""

from __future__ import annotations

import re

from enum import StrEnum
from typing import Any, ClassVar

from app.core.logging import get_logger

logger = get_logger(__name__)


class QueryIntent(StrEnum):
    """
    Query intent types for routing to specialized handlers.

    Intent routing enables:
    - Factual: Strict metadata filtering + citation requirements
    - Recommendation: Genre/era similarity + diversity ranking
    - Comparison: Dual-entity retrieval + contrastive synthesis
    - Creative: LLM creativity mode + relaxed retrieval
    - Meta: System capabilities + help responses
    """

    FACTUAL = "factual"
    RECOMMENDATION = "recommendation"
    COMPARISON = "comparison"
    CREATIVE = "creative"
    META = "meta"


class QueryClassifier:
    """
    Classifies user queries by intent using rule-based pattern matching.

    Design decisions (per spec):
    - Phase 2: Rule-based (fast, no training, interpretable)
    - Phase 5: Fine-tuned classifier (higher accuracy)
    - Returns confidence scores for thresholding
    - Extensible pattern registry for easy maintenance

    Usage:
        classifier = QueryClassifier()
        result = classifier.classify("Who directed Cowboy Bebop?")
        # {'intent': 'factual', 'confidence': 0.9, 'method': 'rule_based'}
    """

    # Intent detection patterns (ordered by priority)
    INTENT_PATTERNS: ClassVar[dict[QueryIntent, list[tuple[str, float]]]] = {
        QueryIntent.RECOMMENDATION: [
            (r"\b(recommend|suggest|suggestion)\b", 0.95),
            (r"\b(similar to|like\b.*\banime|watch next)\b", 0.9),
            (r"\b(best|top|must watch)\b.*\b(anime|series)\b", 0.85),
            (r"\banime\b.*\b(for|like|similar)\b", 0.85),
        ],
        QueryIntent.COMPARISON: [
            (r"\b(vs|versus|compare|comparison)\b", 0.95),
            (r"\b(difference between|how.*compare|which is better)\b", 0.9),
            (r"\b(and|vs)\b.*\b(anime|series)\b", 0.7),
        ],
        QueryIntent.CREATIVE: [
            (r"\b(write|create|compose|generate)\b", 0.9),
            (r"\b(story|poem|haiku|song)\b", 0.85),
            (r"\b(imagine|describe|what if)\b", 0.8),
            (r"\b(roleplay|rp|as if)\b", 0.9),
        ],
        QueryIntent.META: [
            (r"\b(what can you|how do you|what do you do)\b", 0.95),
            (r"\b(help|capabilities|features)\b", 0.9),
            (r"\b(about you|who are you)\b", 0.95),
        ],
    }

    # Factual query indicators (default fallback)
    FACTUAL_INDICATORS: ClassVar[list[str]] = [
        r"\b(who|what|when|where|why|how|which)\b",
        r"\b(directed by|created by|made by)\b",
        r"\b(plot|synopsis|story|about)\b",
        r"\b(character|genre|studio|year)\b",
        r"\b(episodes|seasons|runtime)\b",
    ]

    # Negative patterns that reduce confidence
    NEGATIVE_PATTERNS: ClassVar[list[tuple[str, float]]] = [
        (r"\b(not sure|maybe|perhaps|possibly)\b", -0.2),
        (r"\b(guess|think|believe)\b", -0.15),
        (r"\b(confused|don't know|unsure)\b", -0.25),
    ]

    def __init__(self):
        """Initialize query classifier with compiled regex patterns."""
        self._compiled_patterns: dict[QueryIntent, list[tuple[re.Pattern, float]]] = {}
        self._compiled_factual: list[re.Pattern] = []
        self._compiled_negative: list[tuple[re.Pattern, float]] = []
        self._compile_patterns()
        logger.info("QueryClassifier initialized with rule-based patterns")

    def _compile_patterns(self) -> None:
        """Compile regex patterns for performance."""
        for intent, patterns in self.INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                (re.compile(pattern, re.IGNORECASE), weight) for pattern, weight in patterns
            ]

        self._compiled_factual = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.FACTUAL_INDICATORS
        ]

        self._compiled_negative = [
            (re.compile(pattern, re.IGNORECASE), weight)
            for pattern, weight in self.NEGATIVE_PATTERNS
        ]

    def classify(self, query: str) -> dict[str, Any]:
        """
        Classify query intent with confidence scoring.

        Args:
            query: User query string

        Returns:
            Dictionary with intent, confidence, and method:
            {
                "intent": "factual",
                "confidence": 0.85,
                "method": "rule_based",
                "matched_pattern": "who directed"
            }
        """
        query_lower = query.lower().strip()

        # Try specific intent patterns first (highest priority)
        for intent in [
            QueryIntent.RECOMMENDATION,
            QueryIntent.COMPARISON,
            QueryIntent.CREATIVE,
            QueryIntent.META,
        ]:
            match = self._match_intent(intent, query_lower)
            if match:
                confidence, pattern = match
                confidence = self._apply_negative_adjustments(confidence, query_lower)
                return {
                    "intent": intent.value,
                    "confidence": round(confidence, 2),
                    "method": "rule_based",
                    "matched_pattern": pattern,
                }

        # Check for factual indicators
        factual_match = self._match_factual(query_lower)

        if factual_match:
            confidence, pattern = factual_match
            confidence = self._apply_negative_adjustments(confidence, query_lower)
            return {
                "intent": QueryIntent.FACTUAL.value,
                "confidence": round(confidence, 2),
                "method": "rule_based",
                "matched_pattern": pattern,
            }

        # Default to factual with lower confidence
        return {
            "intent": QueryIntent.FACTUAL.value,
            "confidence": 0.5,
            "method": "default",
            "matched_pattern": None,
        }

    def _match_intent(self, intent: QueryIntent, query: str) -> tuple[float, str] | None:
        """
        Match query against intent-specific patterns.

        Args:
            intent: Intent to match against
            query: Lowercase query string

        Returns:
            (confidence, matched_pattern) tuple or None
        """
        patterns = self._compiled_patterns.get(intent, [])

        for pattern, base_confidence in patterns:
            match_result = pattern.search(query)
            if match_result:
                matched_text = match_result.group(0)
                logger.debug(
                    f"Matched {intent.value} intent: '{matched_text}' "
                    f"(confidence: {base_confidence})"
                )
                return base_confidence, matched_text

        return None

    def _match_factual(self, query: str) -> tuple[float, str] | None:
        """
        Match query against factual indicators.

        Args:
            query: Lowercase query string

        Returns:
            (confidence, matched_pattern) tuple or None
        """
        for pattern in self._compiled_factual:
            match_result = pattern.search(query)
            if match_result:
                matched_text = match_result.group(0)
                base_confidence = (
                    0.8
                    if any(q in query for q in ["who", "what", "when", "where", "why", "how"])
                    else 0.65
                )
                logger.debug(
                    f"Matched factual intent: '{matched_text}' (confidence: {base_confidence})"
                )
                return base_confidence, matched_text

        return None

    def _apply_negative_adjustments(self, base_confidence: float, query: str) -> float:
        """
        Apply negative adjustments for uncertain language.

        Args:
            base_confidence: Base confidence score
            query: Lowercase query string

        Returns:
            Adjusted confidence score (clamped to 0.0-1.0)
        """
        adjusted = base_confidence

        for pattern, adjustment in self._compiled_negative:
            if pattern.search(query):
                adjusted += adjustment
                logger.debug(f"Applied negative adjustment {adjustment} for '{pattern.pattern}'")

        # Clamp to valid range
        return max(0.0, min(1.0, adjusted))

    def get_intent_stats(self) -> dict[str, Any]:
        """
        Get statistics about intent patterns.

        Returns:
            Dictionary with pattern counts per intent
        """
        stats = {
            "total_patterns": sum(len(patterns) for patterns in self.INTENT_PATTERNS.values()),
            "intents": {
                intent.value: len(patterns) for intent, patterns in self.INTENT_PATTERNS.items()
            },
            "factual_indicators": len(self.FACTUAL_INDICATORS),
            "negative_patterns": len(self.NEGATIVE_PATTERNS),
        }
        return stats


# Singleton instance for dependency injection
_classifier_instance: QueryClassifier | None = None


def get_query_classifier() -> QueryClassifier:
    """
    Get singleton QueryClassifier instance for dependency injection.

    Usage in FastAPI endpoints:
        @router.post("/query")
        async def query_endpoint(
            request: QueryRequest,
            classifier: QueryClassifier = Depends(get_query_classifier)
        ):
            classification = classifier.classify(request.query)
            ...
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryClassifier()
    return _classifier_instance
