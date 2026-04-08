#!/usr/bin/env python3
"""
Constants used in OpenRouter API Proxy.
"""

# Config
CONFIG_FILE = "config.yml"

# Rate limit error code
RATE_LIMIT_ERROR_CODE = 429

# Canonical OpenAI-style model listing paths.
MODELS_ENDPOINTS = ["/models"]

GLOBAL_LIMIT_PATTERN = "is temporarily rate-limited upstream"

GOOGLE_LIMIT_ERROR = "Google returned RESOURCE_EXHAUSTED code"
GLOBAL_LIMIT_ERROR = "Model is temporarily rate-limited upstream"
