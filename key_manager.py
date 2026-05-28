#!/usr/bin/env python3
"""
API Key management module for OpenRouter API Proxy.
Implements key rotation and rate limit handling.
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import HTTPException

from config import logger


def mask_key(key: str) -> str:
    """Mask an API key for logging purposes."""
    if not key:
        return key
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]


class KeyManager:
    """Manages provider API keys, including rotation and rate limit handling."""
    def __init__(
        self,
        keys: List[str],
        cooldown_seconds: int,
        strategy: str,
        opts: list[str],
        label: str = "provider",
    ):
        self.keys = keys
        self.cooldown_seconds = cooldown_seconds
        self.current_index = 0
        self.disabled_until: Dict[str, datetime] = {}
        self.strategy = strategy
        self.use_last_key = "same" in opts
        self.last_key = None
        self.lock = asyncio.Lock()
        self.label = label

        if not keys:
            logger.info("No API keys provided in configuration for %s; requests will be sent without Authorization.", self.label)

    async def get_next_key(self) -> str:
        """Get the next available API key using round-robin selection."""
        if not self.keys:
            return ""

        available_keys = []
        async with self.lock:
            now_ = datetime.now()
            for key in self.keys:
                # Check if the key is disabled
                if key in self.disabled_until:
                    if now_ >= self.disabled_until[key]:
                        # Key cooldown period has expired
                        del self.disabled_until[key]
                        logger.info("[%s] API key %s is now enabled again.", self.label, mask_key(key))
                        available_keys.append(key)
                else:
                    # Key is not disabled
                    available_keys.append(key)

            # All keys are disabled
            if not available_keys:
                soonest_available = min(self.disabled_until.values())
                wait_seconds = (soonest_available - now_).total_seconds()
                logger.error(
                    "[%s] All API keys are currently disabled. The next key will be available in %.2f seconds.",
                    self.label,
                    wait_seconds,
                )
                raise HTTPException(
                    status_code=503,
                    detail="All API keys are currently disabled due to rate limits. Please try again later."
                )

            available_keys_set = set(available_keys)
            if self.use_last_key and self.last_key in available_keys_set:
                selected_key = self.last_key
            elif self.strategy == "round-robin":
                for _ in self.keys:
                    key = self.keys[self.current_index]
                    self.current_index = (self.current_index + 1) % len(self.keys)
                    if key in available_keys_set:
                        selected_key = key
                        break
            elif self.strategy == "first":
                selected_key = available_keys[0]
            elif self.strategy == "random":
                selected_key = random.choice(available_keys)
            else:
                raise RuntimeError(f"Unknown key selection strategy: {self.strategy}")
            self.last_key = selected_key
            return selected_key

    async def disable_key(self, key: str, reset_time_ms: Optional[int] = None):
        """
        Disable a key until reset time or for the configured cooldown period.

        Args:
            key: The API key to disable
            reset_time_ms: Optional reset time in milliseconds since epoch. If provided,
                          the key will be disabled until this time. Otherwise, the default
                          cooldown period will be used.
        """
        async with self.lock:
            now_ = datetime.now()
            if reset_time_ms:
                try:
                    # Convert milliseconds to seconds and create datetime
                    reset_datetime = datetime.fromtimestamp(reset_time_ms / 1000)

                    # Ensure reset time is in the future
                    if reset_datetime > now_:
                        disabled_until = reset_datetime
                        logger.info("[%s] Using server-provided reset time: %s", self.label, str(disabled_until))
                    else:
                        # Fallback to default cooldown if reset time is in the past
                        disabled_until = now_ + timedelta(seconds=self.cooldown_seconds)
                        logger.warning(
                            "[%s] Server-provided reset time is in the past, using default cooldown of %s seconds",
                            self.label,
                            self.cooldown_seconds,
                        )
                except Exception as e:
                    # Fallback to default cooldown on error
                    disabled_until = now_ + timedelta(seconds=self.cooldown_seconds)
                    logger.error(
                        "[%s] Error processing reset time %s, using default cooldown: %s",
                        self.label,
                        reset_time_ms,
                        e,
                    )
            else:
                # Use default cooldown period
                disabled_until = now_ + timedelta(seconds=self.cooldown_seconds)
                logger.info(
                    "[%s] No reset time provided, using default cooldown of %s seconds",
                    self.label,
                    self.cooldown_seconds,
                )

            self.disabled_until[key] = disabled_until
            logger.warning(
                "[%s] API key %s has been disabled until %s.",
                self.label,
                mask_key(key),
                disabled_until,
            )
