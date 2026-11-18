"""Resilience primitives for 10,000-concurrency traffic: rate limiting, backpressure, and circuit breaking."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import time
from typing import AsyncIterator


class TokenBucketRateLimiter:
    """High-throughput non-blocking token bucket rate limiter."""

    def __init__(self, rps: float = 15000.0, burst: float = 12000.0) -> None:
        self.rps = max(1.0, float(rps))
        self.burst = max(1.0, float(burst))
        self._tokens = self.burst
        self._last_refill = time.monotonic()
        self.allowed_count: int = 0
        self.rejected_count: int = 0

    def allow(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.burst, self._tokens + elapsed * self.rps)
            self._last_refill = now
        if self._tokens >= cost:
            self._tokens -= cost
            self.allowed_count += 1
            return True
        self.rejected_count += 1
        return False

    def reset_stats(self) -> None:
        self._tokens = self.burst
        self._last_refill = time.monotonic()
        self.allowed_count = 0
        self.rejected_count = 0


class ConcurrencyLimitExceeded(RuntimeError):
    """Raised when the async concurrency gate queue times out under overload."""


class AsyncConcurrencyGate:
    """Bounded async semaphore with queue timeout and peak concurrency tracking."""

    def __init__(self, max_concurrent: int = 256, queue_timeout_seconds: float = 5.0) -> None:
        self.max_concurrent = max_concurrent
        self.queue_timeout_seconds = queue_timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.active_count: int = 0
        self.peak_concurrency: int = 0
        self.total_acquired: int = 0
        self.total_timeouts: int = 0

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[float]:
        """Acquire a concurrency slot, yielding queue wait time in milliseconds."""
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.queue_timeout_seconds)
        except asyncio.TimeoutError as exc:
            self.total_timeouts += 1
            raise ConcurrencyLimitExceeded(
                f"Backend concurrency gate timed out after {self.queue_timeout_seconds}s"
            ) from exc

        wait_ms = (time.perf_counter() - t0) * 1000.0
        self.active_count += 1
        self.total_acquired += 1
        if self.active_count > self.peak_concurrency:
            self.peak_concurrency = self.active_count
        try:
            yield wait_ms
        finally:
            self.active_count -= 1
            self._semaphore.release()

    def reset_stats(self) -> None:
        self.peak_concurrency = 0
        self.total_acquired = 0
        self.total_timeouts = 0


class CircuitBreaker:
    """Tri-state circuit breaker protecting upstream synthesis from cascading failures."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: int = 10, reset_seconds: float = 15.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self.state: str = self.CLOSED
        self.consecutive_failures: int = 0
        self.opened_at: float = 0.0
        self.trip_count: int = 0

    def allow_upstream(self) -> bool:
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            if (time.monotonic() - self.opened_at) >= self.reset_seconds:
                self.state = self.HALF_OPEN
                return True
            return False
        # HALF_OPEN allows probe calls
        return True

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.state = self.CLOSED

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.failure_threshold:
            if self.state != self.OPEN:
                self.trip_count += 1
            self.state = self.OPEN
            self.opened_at = time.monotonic()

    def reset(self) -> None:
        self.state = self.CLOSED
        self.consecutive_failures = 0
        self.opened_at = 0.0
        self.trip_count = 0
