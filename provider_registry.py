#!/usr/bin/env python3
"""
Provider registry and model normalization helpers.
"""

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import HTTPException

from config import logger
from constants import CLIENT_API_PREFIXES, REASONING_EFFORT_BUDGET_MAP, VALID_REASONING_EFFORTS
from key_manager import KeyManager
from request_pacer import ProviderRequestPacer


OPENROUTER_ONLY_REQUEST_FIELDS = {"models", "provider", "plugins"}

# How the proxy injects reasoning/thinking parameters for the upstream provider
REASONING_MODE_OPENAI = "openai_reasoning"          # reasoning: {effort: "high"}
REASONING_MODE_ANTHROPIC = "anthropic_thinking"      # thinking: {type: "enabled", budget_tokens: N}
REASONING_MODE_ENABLE_FLAG = "enable_thinking_flag"  # enable_thinking: true
REASONING_MODE_CHAT_TEMPLATE = "chat_template"       # injected into chat_template_kwargs
REASONING_MODES = {
    REASONING_MODE_OPENAI,
    REASONING_MODE_ANTHROPIC,
    REASONING_MODE_ENABLE_FLAG,
    REASONING_MODE_CHAT_TEMPLATE,
}

DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": "https://ollama.com/v1",
}

DEFAULT_PUBLIC_ENDPOINTS = ["/models"]


def normalize_public_endpoint(path: str) -> str:
    """Normalize client-facing public endpoint definitions to a canonical path."""
    normalized = path.split("?", 1)[0].strip()
    if not normalized:
        return "/"
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    for prefix in CLIENT_API_PREFIXES:
        if normalized == prefix:
            return "/"
        if normalized.startswith(prefix + "/"):
            normalized = normalized[len(prefix):]
            break
    if normalized != "/":
        normalized = normalized.rstrip("/") or "/"
    return normalized


def path_matches_prefix(path: str, prefix: str) -> bool:
    """Match an endpoint prefix without accidentally matching sibling paths."""
    if prefix == "/":
        return True
    return path == prefix or path.startswith(prefix + "/")


def split_provider_model_id(model_id: str) -> tuple[str, str]:
    """Split a provider-qualified model id into provider and upstream model."""
    if not isinstance(model_id, str):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Model must be a string",
                "code": "invalid_request_error",
                "param": "model",
            },
        )

    provider_id, separator, upstream_model = model_id.partition("/")
    if not separator or not provider_id.strip() or not upstream_model.strip():
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Model must be provider-qualified as <provider>/<model>",
                "code": "invalid_request_error",
                "param": "model",
            },
        )

    return provider_id.strip(), upstream_model.strip()


def provider_model_id(provider_id: str, upstream_model: str) -> str:
    """Build a provider-qualified model identifier."""
    if upstream_model.startswith(provider_id + "/"):
        return upstream_model
    return f"{provider_id}/{upstream_model}"


def model_is_free(model_payload: dict[str, Any]) -> bool:
    """Return True when model pricing indicates a free model."""
    pricing = model_payload.get("pricing", {})
    if not isinstance(pricing, dict):
        return False

    for key in ("prompt", "completion", "request", "image", "web_search", "internal_reasoning"):
        value = pricing.get(key)
        if value is None:
            return False
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return False
        try:
            if float(value) != 0.0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def prefix_model_payload(provider_id: str, model_payload: dict[str, Any]) -> dict[str, Any]:
    """Prefix a model payload with its provider id."""
    prefixed_payload = copy.deepcopy(model_payload)
    model_name = prefixed_payload.get("id") or prefixed_payload.get("name")
    if isinstance(model_name, str) and model_name:
        prefixed_name = provider_model_id(provider_id, model_name)
        prefixed_payload["id"] = prefixed_name
        if prefixed_payload.get("name") is not None:
            prefixed_payload["name"] = prefixed_name
    return prefixed_payload


@dataclass(slots=True)
class ProviderRuntime:
    """Normalized provider configuration and request handling runtime."""

    provider_id: str
    base_url: str
    keys: list[str]
    key_selection_strategy: str
    key_selection_opts: list[str]
    allowed_models: tuple[str, ...]
    public_endpoints: tuple[str, ...]
    rate_limit_cooldown: int
    global_rate_delay: float
    free_only: bool
    supports_model_fallbacks: bool = False
    supports_stateful_responses: bool = True
    # Upstream protocol format: "openai" (default) or "anthropic"
    upstream_format: str = "openai"
    # Upstream path for chat completions (overrides provider default)
    upstream_chat_path: str = ""
    # Upstream path for messages (anthropic mode)
    upstream_messages_path: str = ""
    # Upstream path for generate content (gemini mode)
    upstream_generate_path: str = ""
    # Gemini API version (v1 or v1beta)
    gemini_api_version: str = "v1beta"
    # Model prefix to strip before sending upstream (e.g. "agentrouter/")
    model_prefix: str = ""
    # Default max_tokens when client doesn't specify
    default_max_tokens: int = 4096
    # Inject billing header for anthropic mode
    inject_billing: bool = False
    # Fingerprint overrides for anthropic headers (anthropic_version, user_agent, etc.)
    fingerprint: dict[str, Any] = field(default_factory=dict)
# Extra headers merged on top of the fingerprint (take precedence)
    custom_headers: dict[str, str] = field(default_factory=dict)
    # Reasoning/thinking configuration
    reasoning_mode: str = ""              # see REASONING_MODES
    reasoning_effort: str = ""            # low/medium/high/xhigh/minimal/none
    key_manager: KeyManager = field(init=False)
    request_pacer: ProviderRequestPacer = field(init=False)

    def __post_init__(self) -> None:
        self.key_manager = KeyManager(
            keys=self.keys,
            cooldown_seconds=self.rate_limit_cooldown,
            strategy=self.key_selection_strategy,
            opts=self.key_selection_opts,
            label=self.provider_id,
        )
        self.request_pacer = ProviderRequestPacer(self.global_rate_delay, label=self.provider_id)

    def path_is_public(self, normalized_path: str) -> bool:
        return any(path_matches_prefix(normalized_path, endpoint) for endpoint in self.public_endpoints)

    def allows_model(self, model_id: str) -> bool:
        return not self.allowed_models or model_id in self.allowed_models


def _normalize_string_list(values: Any, *, field_name: str, provider_id: str) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        logger.warning("'providers.%s.%s' is invalid. Using empty list.", provider_id, field_name)
        return []

    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            logger.warning(
                "Item %d in 'providers.%s.%s' is not a string. Skipping.",
                index,
                provider_id,
                field_name,
            )
            continue
        stripped = value.strip()
        if not stripped:
            logger.warning(
                "Item %d in 'providers.%s.%s' is empty. Skipping.",
                index,
                provider_id,
                field_name,
            )
            continue
        if stripped not in normalized:
            normalized.append(stripped)
    return normalized


def _normalize_provider_keys(provider_id: str, provider_config: dict[str, Any]) -> list[str]:
    raw_keys = provider_config.get("keys")
    if raw_keys is None:
        raw_keys = provider_config.get("api_keys")
    if raw_keys is None and isinstance(provider_config.get("key"), str):
        raw_keys = [provider_config["key"]]

    keys = _normalize_string_list(raw_keys, field_name="keys", provider_id=provider_id)
    if not keys:
        if provider_id == "ollama":
            logger.info("'providers.%s.keys' is empty. Requests will be sent without Authorization.", provider_id)
        else:
            logger.warning("'providers.%s.keys' is empty. Proxy will not work for authenticated endpoints.", provider_id)
    return keys


def _normalize_provider_models(provider_id: str, provider_config: dict[str, Any]) -> tuple[str, ...]:
    raw_models = provider_config.get("allowed_models")
    if raw_models is None:
        return ()

    if isinstance(raw_models, str):
        raw_models = [raw_models]

    if not isinstance(raw_models, list):
        logger.warning(
            "'providers.%s.allowed_models' is invalid. Using empty list.",
            provider_id,
        )
        return ()

    normalized_models: list[str] = []
    prefix = provider_id + "/"
    for index, model_name in enumerate(raw_models):
        if not isinstance(model_name, str):
            logger.warning(
                "Item %d in 'providers.%s.allowed_models' is not a string. Skipping.",
                index,
                provider_id,
            )
            continue
        normalized = model_name.strip()
        if not normalized:
            logger.warning(
                "Item %d in 'providers.%s.allowed_models' is empty. Skipping.",
                index,
                provider_id,
            )
            continue
        if not normalized.startswith(prefix):
            logger.warning(
                "Item %d in 'providers.%s.allowed_models' must start with '%s'. Skipping.",
                index,
                provider_id,
                prefix,
            )
            continue
        if normalized not in normalized_models:
            normalized_models.append(normalized)

    return tuple(normalized_models)


def _normalize_public_endpoints(provider_id: str, provider_config: dict[str, Any]) -> tuple[str, ...]:
    raw_endpoints = provider_config.get("public_endpoints")
    if raw_endpoints is None:
        raw_endpoints = DEFAULT_PUBLIC_ENDPOINTS
    normalized_endpoints = [normalize_public_endpoint(endpoint) for endpoint in _normalize_string_list(
        raw_endpoints,
        field_name="public_endpoints",
        provider_id=provider_id,
    )]
    return tuple(normalized_endpoints)


def _normalize_provider_block(provider_id: str, provider_config: Any) -> dict[str, Any]:
    if not isinstance(provider_config, dict):
        logger.warning("'providers.%s' is missing or invalid. Using defaults.", provider_id)
        provider_config = {}

    normalized = copy.deepcopy(provider_config)

    base_url = normalized.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        base_url = DEFAULT_BASE_URLS.get(provider_id, "")
        if base_url:
            logger.warning(
                "'providers.%s.base_url' missing or invalid. Using default: %s",
                provider_id,
                base_url,
            )
        else:
            logger.warning(
                "'providers.%s.base_url' missing or invalid and no default is known. Using empty string.",
                provider_id,
            )
    normalized["base_url"] = base_url.rstrip("/")

    normalized["keys"] = _normalize_provider_keys(provider_id, normalized)
    normalized["allowed_models"] = list(_normalize_provider_models(provider_id, normalized))
    normalized["public_endpoints"] = list(_normalize_public_endpoints(provider_id, normalized))

    key_selection_strategy = normalized.get("key_selection_strategy")
    if not isinstance(key_selection_strategy, str) or key_selection_strategy not in {"round-robin", "first", "random"}:
        key_selection_strategy = "round-robin"
        logger.warning(
            "'providers.%s.key_selection_strategy' is invalid. Using default: %s",
            provider_id,
            key_selection_strategy,
        )
    normalized["key_selection_strategy"] = key_selection_strategy

    key_selection_opts = normalized.get("key_selection_opts")
    if not isinstance(key_selection_opts, list):
        key_selection_opts = []
        logger.warning(
            "'providers.%s.key_selection_opts' is invalid. Using empty list.",
            provider_id,
        )
    else:
        key_selection_opts = [option for option in key_selection_opts if isinstance(option, str) and option.strip()]
    normalized["key_selection_opts"] = key_selection_opts

    rate_limit_cooldown = normalized.get("rate_limit_cooldown")
    if not isinstance(rate_limit_cooldown, (int, float)):
        rate_limit_cooldown = 14400
        logger.warning(
            "'providers.%s.rate_limit_cooldown' is invalid. Using default: %s",
            provider_id,
            rate_limit_cooldown,
        )
    normalized["rate_limit_cooldown"] = int(rate_limit_cooldown)

    global_rate_delay = normalized.get("global_rate_delay")
    if not isinstance(global_rate_delay, (int, float)):
        logger.warning(
            "'providers.%s.global_rate_delay' is invalid. Using default: %s",
            provider_id,
            0,
        )
        global_rate_delay = 0
    elif global_rate_delay < 0:
        logger.warning(
            "'providers.%s.global_rate_delay' cannot be negative. Using default: %s",
            provider_id,
            0,
        )
        global_rate_delay = 0
    normalized["global_rate_delay"] = float(global_rate_delay)

    free_only = normalized.get("free_only")
    if not isinstance(free_only, bool):
        free_only = False
    normalized["free_only"] = free_only

    supports_model_fallbacks = normalized.get("supports_model_fallbacks")
    if not isinstance(supports_model_fallbacks, bool):
        supports_model_fallbacks = provider_id == "openrouter"
    normalized["supports_model_fallbacks"] = supports_model_fallbacks

    supports_stateful_responses = normalized.get("supports_stateful_responses")
    if not isinstance(supports_stateful_responses, bool):
        supports_stateful_responses = provider_id != "ollama"
    normalized["supports_stateful_responses"] = supports_stateful_responses

    # Upstream format: "openai" (default), "anthropic", or "gemini"
    # Accept both "upstream_format" and legacy "upstream_mode"
    upstream_format = normalized.get("upstream_format") or normalized.get("upstream_mode", "openai")
    if not isinstance(upstream_format, str) or upstream_format not in ("openai", "anthropic", "gemini"):
        upstream_format = "openai"
    normalized["upstream_format"] = upstream_format

    # Upstream chat completions path
    upstream_chat_path = normalized.get("upstream_chat_path", "")
    if not isinstance(upstream_chat_path, str):
        upstream_chat_path = ""
    normalized["upstream_chat_path"] = upstream_chat_path.strip()

    # Upstream messages path (anthropic mode)
    upstream_messages_path = normalized.get("upstream_messages_path", "")
    if not isinstance(upstream_messages_path, str):
        upstream_messages_path = ""
    normalized["upstream_messages_path"] = upstream_messages_path.strip()

    # Upstream generate content path (gemini mode)
    upstream_generate_path = normalized.get("upstream_generate_path", "")
    if not isinstance(upstream_generate_path, str):
        upstream_generate_path = ""
    normalized["upstream_generate_path"] = upstream_generate_path.strip()

    # Gemini API version (v1 or v1beta)
    gemini_api_version = normalized.get("gemini_api_version", "v1beta")
    if not isinstance(gemini_api_version, str) or gemini_api_version not in ("v1", "v1beta"):
        gemini_api_version = "v1beta"
    normalized["gemini_api_version"] = gemini_api_version

    # Model prefix to strip before sending upstream
    model_prefix = normalized.get("model_prefix", "")
    if not isinstance(model_prefix, str):
        model_prefix = ""
    normalized["model_prefix"] = model_prefix

    # Default max tokens
    default_max_tokens = normalized.get("default_max_tokens", 4096)
    if not isinstance(default_max_tokens, (int, float)) or default_max_tokens <= 0:
        default_max_tokens = 4096
    normalized["default_max_tokens"] = int(default_max_tokens)

    # Inject billing header (anthropic mode)
    inject_billing = normalized.get("inject_billing", False)
    if not isinstance(inject_billing, bool):
        inject_billing = False
    normalized["inject_billing"] = inject_billing

    # Fingerprint overrides (anthropic_version, user_agent, stainless_*, etc.)
    fingerprint = normalized.get("fingerprint", {})
    if not isinstance(fingerprint, dict):
        logger.warning("'providers.%s.fingerprint' is invalid. Using empty dict.", provider_id)
        fingerprint = {}
    normalized["fingerprint"] = {str(k): v for k, v in fingerprint.items() if isinstance(k, str) and v is not None}

# Custom headers (merged on top of fingerprint headers)
    custom_headers = normalized.get("custom_headers", {})
    if not isinstance(custom_headers, dict):
        logger.warning("'providers.%s.custom_headers' is invalid. Using empty dict.", provider_id)
        custom_headers = {}
    normalized["custom_headers"] = {str(k): str(v) for k, v in custom_headers.items() if isinstance(k, str) and v is not None}

    # Reasoning mode
    reasoning_mode = normalized.get("reasoning_mode", "")
    if not isinstance(reasoning_mode, str) or reasoning_mode not in REASONING_MODES:
        reasoning_mode = ""
    normalized["reasoning_mode"] = reasoning_mode

    # Reasoning effort (acts as default when client doesn't specify)
    reasoning_effort = normalized.get("reasoning_effort", "")
    if not isinstance(reasoning_effort, str):
        reasoning_effort = ""
    if reasoning_effort and reasoning_effort not in VALID_REASONING_EFFORTS:
        logger.warning(
            "'providers.%s.reasoning_effort' unrecognized: '%s'. Must be one of %s.",
            provider_id, reasoning_effort, sorted(VALID_REASONING_EFFORTS),
        )
        reasoning_effort = ""
    normalized["reasoning_effort"] = reasoning_effort

    return normalized


def build_provider_registry(config_data: dict[str, Any]) -> dict[str, ProviderRuntime]:
    """Create ProviderRuntime instances from normalized configuration."""
    raw_providers = config_data.get("providers")
    if not isinstance(raw_providers, dict) or not raw_providers:
        legacy_openrouter = config_data.get("openrouter")
        if isinstance(legacy_openrouter, dict):
            raw_providers = {"openrouter": legacy_openrouter}
        else:
            raise RuntimeError("No providers are configured.")

    registry: dict[str, ProviderRuntime] = {}
    for provider_id, provider_config in raw_providers.items():
        normalized = _normalize_provider_block(provider_id, provider_config)
        registry[provider_id] = ProviderRuntime(
            provider_id=provider_id,
            base_url=normalized["base_url"],
            keys=normalized["keys"],
            key_selection_strategy=normalized["key_selection_strategy"],
            key_selection_opts=normalized["key_selection_opts"],
            allowed_models=tuple(normalized["allowed_models"]),
            public_endpoints=tuple(normalized["public_endpoints"]),
            rate_limit_cooldown=int(normalized["rate_limit_cooldown"]),
            global_rate_delay=float(normalized["global_rate_delay"]),
            free_only=bool(normalized["free_only"]),
            supports_model_fallbacks=bool(normalized["supports_model_fallbacks"]),
            supports_stateful_responses=bool(normalized["supports_stateful_responses"]),
            upstream_format=str(normalized["upstream_format"]),
            upstream_chat_path=str(normalized["upstream_chat_path"]),
            upstream_messages_path=str(normalized["upstream_messages_path"]),
            upstream_generate_path=str(normalized.get("upstream_generate_path", "")),
            gemini_api_version=str(normalized.get("gemini_api_version", "v1beta")),
            model_prefix=str(normalized["model_prefix"]),
            default_max_tokens=int(normalized["default_max_tokens"]),
            inject_billing=bool(normalized["inject_billing"]),
            fingerprint=dict(normalized.get("fingerprint", {})),
            custom_headers=dict(normalized.get("custom_headers", {})),
            reasoning_mode=str(normalized["reasoning_mode"]),
            reasoning_effort=str(normalized["reasoning_effort"]),
        )

    logger.info("Loaded providers: %s", ", ".join(sorted(registry.keys())))
    return registry


def resolve_provider_hint(request_provider: Optional[str], provider_registry: dict[str, ProviderRuntime]) -> Optional[ProviderRuntime]:
    """Resolve an explicit provider hint into a runtime instance."""
    if not request_provider:
        return None
    provider_name = request_provider.strip()
    if not provider_name:
        return None
    provider = provider_registry.get(provider_name)
    if provider is None:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unknown provider '{provider_name}'",
                "code": "invalid_request_error",
                "param": "provider",
            },
        )
    return provider


def normalize_request_body_for_provider(
    provider: ProviderRuntime,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Validate and strip provider prefixes from request model fields."""
    body = copy.deepcopy(request_body)

    requested_model = body.get("model")
    requested_models = body.get("models")

    if requested_model is None and requested_models is None:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "A provider-qualified model is required",
                "code": "invalid_request_error",
                "param": "model",
            },
        )

    if requested_models is not None:
        if not provider.supports_model_fallbacks:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Provider '{provider.provider_id}' does not support the 'models' fallback array",
                    "code": "invalid_request_error",
                    "param": "models",
                },
            )
        if not isinstance(requested_models, list) or not requested_models:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid models value",
                    "code": "invalid_request_error",
                    "param": "models",
                },
            )

        normalized_models: list[str] = []
        for index, model_id in enumerate(requested_models):
            model_provider, upstream_model = split_provider_model_id(model_id)
            if model_provider != provider.provider_id:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": (
                            f"All fallback models must use provider '{provider.provider_id}'. "
                            f"Item {index} used '{model_provider}'."
                        ),
                        "code": "invalid_request_error",
                        "param": "models",
                    },
                )
            qualified_model = provider_model_id(provider.provider_id, upstream_model)
            if not provider.allows_model(qualified_model):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": f"Model '{qualified_model}' is not allowed by this proxy",
                        "code": "invalid_request_error",
                        "param": "models",
                    },
                )
            normalized_models.append(upstream_model)

        if requested_model is None:
            body["model"] = normalized_models[0]

        body["models"] = normalized_models

    if requested_model is not None:
        model_provider, upstream_model = split_provider_model_id(requested_model)
        if model_provider != provider.provider_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Model must use provider '{provider.provider_id}'",
                    "code": "invalid_request_error",
                    "param": "model",
                },
            )
        qualified_model = provider_model_id(provider.provider_id, upstream_model)
        if not provider.allows_model(qualified_model):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Model '{qualified_model}' is not allowed by this proxy",
                    "code": "invalid_request_error",
                    "param": "model",
                },
            )
        body["model"] = upstream_model

    if provider.provider_id != "openrouter":
        for field_name in OPENROUTER_ONLY_REQUEST_FIELDS:
            body.pop(field_name, None)

    # Inject reasoning/thinking parameter if provider has reasoning configured
    # and client didn't specify any reasoning parameter
    if provider.reasoning_effort and not body.get("reasoning") and not body.get("reasoning_effort"):
        mode = provider.reasoning_mode
        effort = provider.reasoning_effort
        body = _inject_reasoning(body, mode, effort)

    return body


def _inject_reasoning(
    body: dict[str, Any],
    mode: str,
    effort: str,
) -> dict[str, Any]:
    """Inject the reasoning/thinking parameter in the provider-specific format."""
    budget = REASONING_EFFORT_BUDGET_MAP.get(effort, 4096)

    if mode == REASONING_MODE_ENABLE_FLAG:
        body["enable_thinking"] = True
    elif mode == REASONING_MODE_CHAT_TEMPLATE:
        if "chat_template_kwargs" not in body:
            body["chat_template_kwargs"] = {}
        body["chat_template_kwargs"]["enable_thinking"] = True
        body["chat_template_kwargs"]["thinking_budget"] = budget
    else:
        # openai_reasoning (default), anthropic_thinking — both use reasoning object
        # for anthropic, the protocol translator converts reasoning → thinking later
        body["reasoning"] = {"effort": effort}

    return body


def normalize_models_payload_for_client(
    provider: ProviderRuntime,
    body: bytes,
) -> bytes:
    """Filter and prefix a provider model listing payload."""
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return body

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return body

    filtered_models: list[dict[str, Any]] = []
    allowed_model_set = set(provider.allowed_models)
    for model_payload in payload["data"]:
        if not isinstance(model_payload, dict):
            continue
        model_id = model_payload.get("id") or model_payload.get("name")
        if not isinstance(model_id, str) or not model_id.strip():
            continue

        qualified_model_id = provider_model_id(provider.provider_id, model_id)
        if allowed_model_set and qualified_model_id not in allowed_model_set:
            continue
        if provider.free_only and not model_is_free(model_payload):
            continue

        filtered_models.append(prefix_model_payload(provider.provider_id, model_payload))

    payload["data"] = filtered_models
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def aggregate_model_payloads(payloads: list[bytes]) -> bytes:
    """Merge multiple model list payloads into one response."""
    aggregate: list[dict[str, Any]] = []
    response_template: dict[str, Any] | None = None

    for body in payloads:
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            continue

        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            continue

        if response_template is None:
            response_template = copy.deepcopy(payload)

        for item in payload["data"]:
            if isinstance(item, dict):
                aggregate.append(item)

    if response_template is None:
        response_template = {"object": "list"}

    response_template["data"] = aggregate
    return json.dumps(response_template, ensure_ascii=False).encode("utf-8")