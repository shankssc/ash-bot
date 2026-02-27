"""Unit tests for CircuitBreaker pattern implementation."""

import time

import pytest

from app.retrieval.vector_store import CircuitBreaker, CircuitBreakerOpenError


class TestCircuitBreaker:
    """Test circuit breaker state transitions and behavior."""

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker()
        assert cb._state == "CLOSED"
        assert cb._failure_count == 0

    def test_opens_after_threshold_failures(self):
        """Circuit opens after reaching failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)

        for _ in range(3):
            cb._record_failure(Exception("test error"))

        assert cb._state == "OPEN"
        assert cb._failure_count == 3

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)

        # Open the circuit
        cb._record_failure(Exception("error"))
        cb._record_failure(Exception("error"))
        assert cb._state == "OPEN"

        # Wait for timeout
        time.sleep(0.15)

        # Should allow request and transition to HALF_OPEN
        assert cb._should_allow_request() is True
        assert cb._state == "HALF_OPEN"

    def test_closes_after_successful_half_open_attempts(self):
        """Circuit closes after successful attempts in HALF_OPEN."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1, half_open_attempts=2)

        # Open circuit
        cb._record_failure(Exception("error"))
        cb._record_failure(Exception("error"))

        # Transition to HALF_OPEN
        time.sleep(0.15)
        cb._should_allow_request()
        assert cb._state == "HALF_OPEN"

        # Record successful attempts
        cb._record_success()
        cb._record_success()

        assert cb._state == "CLOSED"
        assert cb._failure_count == 0

    def test_decorator_blocks_when_open(self):
        """Circuit breaker decorator blocks calls when open."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10.0)
        cb._record_failure(Exception("initial"))

        @cb
        async def protected_func():
            return "success"

        with pytest.raises(CircuitBreakerOpenError):
            import asyncio

            asyncio.run(protected_func())

    def test_decorator_records_success(self):
        """Circuit breaker decorator records successful calls."""
        cb = CircuitBreaker(failure_threshold=3)

        @cb
        async def protected_func():
            return "ok"

        import asyncio

        result = asyncio.run(protected_func())

        assert result == "ok"
        assert cb._state == "CLOSED"
