"""
Unit tests for QueryClassifier.
"""

import pytest

from app.services.query_classifier import QueryClassifier, QueryIntent


class TestQueryClassifier:
    """Test QueryClassifier functionality."""

    @pytest.fixture
    def classifier(self) -> QueryClassifier:
        """Fixture for QueryClassifier instance."""
        return QueryClassifier()

    def test_classifier_initialization(self, classifier: QueryClassifier):
        """Test classifier initializes correctly."""
        assert classifier is not None
        assert hasattr(classifier, "_compiled_patterns")
        assert len(classifier._compiled_patterns) > 0

    def test_intent_enum_values(self):
        """Test QueryIntent enum has correct values."""
        assert QueryIntent.FACTUAL.value == "factual"
        assert QueryIntent.RECOMMENDATION.value == "recommendation"
        assert QueryIntent.COMPARISON.value == "comparison"
        assert QueryIntent.CREATIVE.value == "creative"
        assert QueryIntent.META.value == "meta"

    @pytest.mark.parametrize(
        "query,expected_intent,expected_confidence",
        [
            # Factual queries
            ("Who directed Cowboy Bebop?", "factual", 0.8),
            ("What is the plot of Attack on Titan?", "factual", 0.8),
            ("When was Naruto released?", "factual", 0.8),
            ("Where does One Piece take place?", "factual", 0.8),
            ("Why is Death Note so popular?", "factual", 0.8),
            ("Which studio made Fullmetal Alchemist?", "factual", 0.8),
            ("How many episodes does Bleach have?", "factual", 0.8),
            ("Who created Dragon Ball?", "factual", 0.8),
            ("What genre is My Hero Academia?", "factual", 0.65),
            ("Tell me about Studio Ghibli", "factual", 0.65),
            # Recommendation queries
            ("Recommend me a good anime", "recommendation", 0.95),
            ("What anime is similar to Attack on Titan?", "recommendation", 0.9),
            ("Suggest something like Cowboy Bebop", "recommendation", 0.9),
            ("What should I watch next?", "recommendation", 0.9),
            ("Best anime of 2023", "recommendation", 0.85),
            ("Top must-watch anime series", "recommendation", 0.85),
            ("Anime like Naruto for beginners", "recommendation", 0.85),
            # Comparison queries
            ("Naruto vs One Piece", "comparison", 0.95),
            ("Compare Attack on Titan and Demon Slayer", "comparison", 0.95),
            ("What's the difference between manga and anime?", "comparison", 0.9),
            ("Which is better: Death Note or Code Geass?", "comparison", 0.9),
            ("How does Frieren compare to Mushoku Tensei?", "comparison", 0.9),
            # Creative queries
            ("Write a haiku about Goku", "creative", 0.9),
            ("Create a story about a space bounty hunter", "creative", 0.9),
            ("Compose a poem about anime", "creative", 0.85),
            ("Imagine if Naruto met Luffy", "creative", 0.8),
            ("Roleplay as Spike Spiegel", "creative", 0.9),
            ("What if Goku was in One Piece?", "creative", 0.8),
            # Meta queries
            ("What can you do?", "meta", 0.95),
            ("How do you work?", "meta", 0.95),
            ("What are your capabilities?", "meta", 0.9),
            ("Help me find anime", "meta", 0.9),
            ("Tell me about yourself", "meta", 0.95),
        ],
    )
    def test_query_classification(
        self,
        classifier: QueryClassifier,
        query: str,
        expected_intent: str,
        expected_confidence: float,
    ):
        """Test query classification with various intents."""
        result = classifier.classify(query)

        assert result["intent"] == expected_intent
        assert result["confidence"] >= expected_confidence - 0.1  # Allow 0.1 tolerance
        assert result["method"] in ["rule_based", "default"]
        assert "matched_pattern" in result

    def test_factual_fallback(self, classifier: QueryClassifier):
        """Test that unknown queries default to factual."""
        query = "anime information please"
        result = classifier.classify(query)

        assert result["intent"] == "factual"
        assert result["confidence"] == 0.5
        assert result["method"] == "default"

    def test_confidence_clamping(self, classifier: QueryClassifier):
        """Test that confidence scores are clamped to 0.0-1.0."""
        # Query with multiple negative indicators
        query = "I'm not sure maybe you could guess what anime this is"
        result = classifier.classify(query)

        assert 0.0 <= result["confidence"] <= 1.0

    def test_case_insensitivity(self, classifier: QueryClassifier):
        """Test that classification is case-insensitive."""
        queries = [
            "WHO DIRECTED COWBOY BEBOP?",
            "Who Directed Cowboy Bebop?",
            "who directed cowboy bebop?",
        ]

        results = [classifier.classify(q) for q in queries]

        # All should have same intent
        intents = [r["intent"] for r in results]
        assert all(intent == "factual" for intent in intents)

    def test_get_intent_stats(self, classifier: QueryClassifier):
        """Test intent statistics retrieval."""
        stats = classifier.get_intent_stats()

        assert "total_patterns" in stats
        assert "intents" in stats
        assert "factual_indicators" in stats
        assert "negative_patterns" in stats

        # Verify counts are reasonable
        assert stats["total_patterns"] > 0
        assert stats["factual_indicators"] > 0
        assert stats["negative_patterns"] > 0

    def test_negative_adjustments(self, classifier: QueryClassifier):
        """Test that uncertain language reduces confidence."""
        # Query without uncertainty
        query1 = "Who directed Cowboy Bebop?"
        result1 = classifier.classify(query1)

        # Same query with uncertainty
        query2 = "I'm not sure but who directed Cowboy Bebop?"
        result2 = classifier.classify(query2)

        # Second query should have lower confidence
        assert result2["confidence"] < result1["confidence"]

    def test_singleton_pattern(self):
        """Test get_query_classifier singleton pattern."""
        from app.services.query_classifier import get_query_classifier

        classifier1 = get_query_classifier()
        classifier2 = get_query_classifier()

        # Should be same instance
        assert classifier1 is classifier2

    def test_empty_query(self, classifier: QueryClassifier):
        """Test handling of empty queries."""
        result = classifier.classify("")
        assert result["confidence"] == 0.0
        assert result["method"] == "invalid_input"

    def test_short_query(self, classifier: QueryClassifier):
        """Test handling of very short queries."""
        result = classifier.classify("hi")
        assert result["confidence"] == 0.3
        assert result["method"] == "short_query"

    def test_whitespace_query(self, classifier: QueryClassifier):
        """Test handling of whitespace-only queries."""
        result = classifier.classify("   ")
        assert result["confidence"] == 0.0
        assert result["method"] == "invalid_input"

    def test_uncertainty_phrases(self, classifier: QueryClassifier):
        """Test detection of uncertainty language."""
        queries = [
            "I think Cowboy Bebop was directed by...",
            "Maybe Attack on Titan is good?",
            "I'm not sure which anime to watch",
        ]
        for query in queries:
            result = classifier.classify(query)
            assert result["confidence"] < 0.7


class TestQueryIntent:
    """Test QueryIntent enum."""

    def test_all_intents_have_values(self):
        """Test all intents are defined."""
        assert QueryIntent.FACTUAL
        assert QueryIntent.RECOMMENDATION
        assert QueryIntent.COMPARISON
        assert QueryIntent.CREATIVE
        assert QueryIntent.META

    def test_intent_string_comparison(self):
        """Test intent can be compared as string."""
        assert QueryIntent.FACTUAL == "factual"
        assert QueryIntent.RECOMMENDATION == "recommendation"
