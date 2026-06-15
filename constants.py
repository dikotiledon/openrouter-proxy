#!/usr/bin/env python3
"""
Constants used in OpenRouter API Proxy.
"""

# Config
CONFIG_FILE = "config.yml"

# Rate limit error code
RATE_LIMIT_ERROR_CODE = 429

# Client-facing API path prefixes
CLIENT_API_PREFIXES = ("/api/v1", "/v1")

# Reasoning effort → thinking budget_tokens mapping
REASONING_EFFORT_BUDGET_MAP: dict[str, int] = {
    "minimal": 512,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "xhigh": 32768,
}
VALID_REASONING_EFFORTS = set(REASONING_EFFORT_BUDGET_MAP.keys())
DEFAULT_REASONING_BUDGET = 4096
REASONING_EFFORT_TO_BUDGET = REASONING_EFFORT_BUDGET_MAP

# Canonical OpenAI-style model listing paths.
MODELS_ENDPOINTS = ["/models"]

GLOBAL_LIMIT_PATTERN = "is temporarily rate-limited upstream"

GOOGLE_LIMIT_ERROR = "Google returned RESOURCE_EXHAUSTED code"
GLOBAL_LIMIT_ERROR = "Model is temporarily rate-limited upstream"

# Gemini API
GEMINI_FINISH_MAP: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
    "FINISH_REASON_STOP": "stop",
    "FINISH_REASON_UNSPECIFIED": "stop",
}
OPENAI_TO_GEMINI_FINISH: dict[str, str] = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "tool_calls": "STOP",
    "content_filter": "SAFETY",
}
ANTHROPIC_TO_GEMINI_FINISH: dict[str, str] = {
    "end_turn": "STOP",
    "stop_sequence": "STOP",
    "max_tokens": "MAX_TOKENS",
    "tool_use": "STOP",
}
DEFAULT_GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
