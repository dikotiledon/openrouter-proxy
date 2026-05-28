#!/usr/bin/env python3
"""
Provider-wide request pacing helpers.
"""

import asyncio
import time

from config import logger


class ProviderRequestPacer:
    """Enforce a minimum interval between upstream dispatches for one provider."""

    def __init__(self, delay_seconds: float, label: str = "provider"):
        try:
            parsed_delay = float(delay_seconds)
        except (TypeError, ValueError):
            parsed_delay = 0.0

        self.delay_seconds = max(0.0, parsed_delay)
        self.label = label
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    async def wait(self) -> None:
        if self.delay_seconds <= 0:
            return

        async with self._lock:
            now = time.monotonic()
            if now < self._next_allowed_at:
                wait_seconds = self._next_allowed_at - now
                logger.info(
                    "[%s] Waiting %.2f seconds before the next upstream request.",
                    self.label,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
                now = time.monotonic()

            self._next_allowed_at = now + self.delay_seconds
