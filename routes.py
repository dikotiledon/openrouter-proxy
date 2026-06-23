#!/usr/bin/env python3
"""
API routes for the upstream provider proxy.
"""

import asyncio
import json
import time
import hashlib
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Request, Header, HTTPException, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, Response

from config import config, logger
from constants import MODELS_ENDPOINTS, CLIENT_API_PREFIXES
from key_manager import mask_key
from protocol_adapter import (
    AnthropicStreamDecoder,
    AnthropicMessageAggregator,
    OpenAIToAnthropicSSETranslator,
    build_anthropic_headers,
    deduplicate_anthropic_tool_use_ids,
    deduplicate_kiro_conversation_state,
    deduplicate_openai_tool_call_ids,
    translate_openai_to_anthropic,
    translate_anthropic_to_openai,
    translate_request,
    translate_openai_chat_to_responses,
    translate_openai_responses_to_chat,
    translate_anthropic_to_responses,
    translate_responses_to_anthropic,
    _random_uuid_like,
    _random_hex,
)
from gemini_adapter import (
    deduplicate_gemini_function_call_ids,
    translate_openai_to_gemini,
    translate_gemini_to_openai,
    translate_anthropic_to_gemini,
    translate_gemini_to_anthropic,
    translate_gemini_to_responses,
    translate_responses_to_gemini,
    translate_gemini_response_to_openai,
    translate_gemini_response_to_anthropic,
    GeminiStreamDecoder,
    GeminiAnthropicStreamDecoder,
    OpenAIToGeminiSSETranslator,
    AnthropicToGeminiSSETranslator,
    build_gemini_tool_config,
)
from provider_registry import (
    ProviderRuntime,
    aggregate_model_payloads,
    build_provider_registry,
    normalize_models_payload_for_client,
    normalize_request_body_for_provider,
    path_matches_prefix,
    resolve_provider_hint,
)
from utils import verify_access_key, check_rate_limit

# Create router
router = APIRouter()

SUPPORTED_PROXY_METHODS = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB request body limit
HOP_BY_HOP_REQUEST_HEADERS = {
    "authorization",
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
HOP_BY_HOP_RESPONSE_HEADERS = {
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

# Session ID cache for Claude Code conversation continuity.
# Keyed by (provider_id, conv_key) → (session_uuid, expiry_timestamp).
_SESSION_CACHE: dict[tuple[str, str], tuple[str, float]] = {}
SESSION_TTL: float = 1800.0  # 30 minutes of inactivity
SESSION_CACHE_MAX_SIZE: int = 10000


def _make_conv_key(body: dict[str, Any]) -> str:
    """Derive a stable conversation key from the request body.

    Uses the model name + first user message text. Falls back to all-messages
    hash if no user message is found. Stable across same-conversation requests.
    """
    model = body.get("model", "")
    messages = body.get("messages", [])
    user_text = ""
    for msg in messages:
        role = msg.get("role", "").lower().strip()
        if role != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            user_text = content
            break
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    user_text = part.get("text", "")
                    break
            if user_text:
                break

    # Hash the first ~200 chars of user text + model for a compact key
    snippet = user_text[:200] if user_text else json.dumps(messages, ensure_ascii=False, sort_keys=True)[:200]
    raw = f"{model}:{snippet}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_or_create_session_id(
    provider_id: str,
    body: dict[str, Any],
    *,
    client_hint: Optional[str] = None,
) -> str:
    """Return a session ID, preferring client-provided hint otherwise cached.

    Hybrid mode (Option C):
    - Client passes X-Claude-Code-Session-Id → use it verbatim (no caching).
    - No client hint → derive conv_key from body, use cached UUID if present,
      otherwise generate new UUID and cache it with TTL.
    """
    if client_hint:
        return client_hint

    conv_key = _make_conv_key(body)
    cache_key = (provider_id, conv_key)
    now = time.time()

    if cache_key in _SESSION_CACHE:
        sid, expiry = _SESSION_CACHE[cache_key]
        if now < expiry:
            _SESSION_CACHE[cache_key] = (sid, now + SESSION_TTL)
            return sid

    sid = _random_uuid_like()
    _SESSION_CACHE[cache_key] = (sid, now + SESSION_TTL)

    # Evict expired entries if cache is too large
    if len(_SESSION_CACHE) > SESSION_CACHE_MAX_SIZE:
        expired = [k for k, (_, exp) in _SESSION_CACHE.items() if now >= exp]
        for k in expired:
            del _SESSION_CACHE[k]
        # If still too large after eviction, drop oldest entries
        if len(_SESSION_CACHE) > SESSION_CACHE_MAX_SIZE:
            sorted_keys = sorted(_SESSION_CACHE, key=lambda k: _SESSION_CACHE[k][1])
            for k in sorted_keys[: len(_SESSION_CACHE) - SESSION_CACHE_MAX_SIZE]:
                del _SESSION_CACHE[k]

    return sid


def normalize_proxy_path(path: str) -> str:
    """Normalize a request or configured endpoint to a canonical API path."""
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


def get_client_prefix(path: str) -> Optional[str]:
    """Return which client-facing prefix matched the incoming path."""
    for prefix in CLIENT_API_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return prefix
    return None


def translate_request_path(path: str, client_prefix: str) -> str:
    """Translate a client path to the upstream OpenRouter path suffix."""
    if path == client_prefix or path == f"{client_prefix}/":
        return ""
    if path.startswith(client_prefix + "/"):
        return path[len(client_prefix):]
    raise HTTPException(status_code=404, detail="Unsupported API prefix")


def split_response_headers(headers: httpx.Headers | dict, default_content_type: str) -> tuple[dict, str]:
    """Remove hop-by-hop headers and extract the upstream media type."""
    filtered_headers: dict[str, str] = {}
    content_type = default_content_type
    for header_name, header_value in dict(headers).items():
        lowered = header_name.lower()
        if lowered in HOP_BY_HOP_RESPONSE_HEADERS:
            continue
        if lowered == "content-type":
            content_type = header_value
            continue
        filtered_headers[header_name] = header_value
    return filtered_headers, content_type


def openai_error_type(status_code: int) -> str:
    """Map HTTP status codes to the OpenAI error taxonomy."""
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code in (400, 404, 405, 409, 422):
        return "invalid_request_error"
    if status_code >= 500:
        return "server_error"
    return "invalid_request_error"


def extract_error_details(body: bytes | str, fallback_status: int) -> tuple[str, Any, Optional[str]]:
    """Extract a human-readable message from an upstream error payload."""
    if isinstance(body, bytes):
        text = body.decode("utf-8", errors="replace")
    else:
        text = body
    message = text.strip() or "Upstream API request failed"
    code: Any = fallback_status
    param: Optional[str] = None

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return message, code, param

    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("detail") or payload.get("message") or message)
            code = error.get("code", code)
            param = error.get("param")
            return message, code, param
        message = str(payload.get("message") or payload.get("detail") or message)

    return message, code, param


def build_openai_error_response(
    status_code: int,
    message: str,
    *,
    code: Any = None,
    param: Optional[str] = None,
    headers: Optional[dict] = None,
) -> JSONResponse:
    """Return an OpenAI-shaped error response for client-facing compatibility."""
    payload = {
        "error": {
            "message": message,
            "type": openai_error_type(status_code),
            "param": param,
            "code": code if code is not None else status_code,
        }
    }
    return JSONResponse(status_code=status_code, content=payload, headers=headers)


def build_anthropic_error_response(
    status_code: int,
    message: str,
    error_type: str = "api_error",
) -> JSONResponse:
    """Return an Anthropic Messages API-shaped error response."""
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {"type": error_type, "message": message},
        },
    )


async def _handle_upstream_error(
    upstream_resp: httpx.Response,
    provider: ProviderRuntime,
    api_key: str,
) -> HTTPException:
    """Read upstream error body, check rate limits, and return an HTTPException."""
    try:
        error_body = await upstream_resp.aread()
    except Exception:
        error_body = b""
    await upstream_resp.aclose()
    await check_httpx_err(error_body, provider, api_key)
    message, code, param = extract_error_details(error_body, upstream_resp.status_code)
    return HTTPException(
        status_code=upstream_resp.status_code,
        detail={"message": message, "code": code, "param": param},
    )


PROVIDER_REGISTRY = build_provider_registry(config)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    client_kwargs = {
        "timeout": httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
        "limits": httpx.Limits(max_connections=200, max_keepalive_connections=50),
    }
    if config["requestProxy"]["enabled"]:
        proxy_url = config["requestProxy"]["url"]
        masked = proxy_url.split("@")[-1] if "@" in proxy_url else proxy_url
        logger.info("Using proxy for httpx client: %s", masked)
        client_kwargs["proxy"] = proxy_url
    app_.state.http_client = httpx.AsyncClient(**client_kwargs)
    yield
    await app_.state.http_client.aclose()


async def get_async_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


async def check_httpx_err(body: str | bytes, provider: ProviderRuntime, api_key: Optional[str]):
    # too big or small for error
    if len(body) < 10 or len(body) > 4000 or not api_key:
        return
    has_rate_limit_error, reset_time_ms = await check_rate_limit(body, provider.global_rate_delay)
    if has_rate_limit_error:
        await provider.key_manager.disable_key(api_key, reset_time_ms)

def prepare_forward_headers(request: Request) -> dict:
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP_REQUEST_HEADERS
    }


def prepare_forward_params(request: Request) -> dict:
    return {
        key: value
        for key, value in request.query_params.items()
        if key != "provider"
    }


def get_provider_by_hint(provider_hint: Optional[str]) -> Optional[ProviderRuntime]:
    if not provider_hint:
        return None
    return resolve_provider_hint(provider_hint, PROVIDER_REGISTRY)


def resolve_request_provider(
    request: Request,
    request_body: Optional[dict[str, Any]],
    normalized_path: str,
) -> Optional[ProviderRuntime]:
    provider = get_provider_by_hint(request.query_params.get("provider"))

    if request_body:
        requested_model = request_body.get("model")
        requested_models = request_body.get("models")

        if requested_model is not None:
            provider_name, _ = requested_model.split("/", 1) if isinstance(requested_model, str) and "/" in requested_model else (None, None)
            if provider_name is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Model must be provider-qualified as <provider>/<model>",
                        "code": "invalid_request_error",
                        "param": "model",
                    },
                )
            body_provider = get_provider_by_hint(provider_name)
            if provider and provider.provider_id != body_provider.provider_id:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": f"Provider hint '{provider.provider_id}' does not match request model provider '{body_provider.provider_id}'",
                        "code": "invalid_request_error",
                        "param": "provider",
                    },
                )
            provider = provider or body_provider

        elif isinstance(requested_models, list) and requested_models:
            first_model = requested_models[0]
            if not isinstance(first_model, str) or "/" not in first_model:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Fallback models must be provider-qualified as <provider>/<model>",
                        "code": "invalid_request_error",
                        "param": "models",
                    },
                )
            provider_name, _ = first_model.split("/", 1)
            body_provider = get_provider_by_hint(provider_name)
            if provider and provider.provider_id != body_provider.provider_id:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": f"Provider hint '{provider.provider_id}' does not match request models provider '{body_provider.provider_id}'",
                        "code": "invalid_request_error",
                        "param": "provider",
                    },
                )
            provider = provider or body_provider

    if provider is None and normalized_path in MODELS_ENDPOINTS:
        return None

    if provider is None:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "A provider hint or provider-qualified model is required",
                "code": "invalid_request_error",
                "param": "provider",
            },
        )

    return provider


async def verify_client_access(provider: ProviderRuntime, normalized_path: str, authorization: Optional[str]) -> None:
    if provider.path_is_public(normalized_path):
        return
    await verify_access_key(authorization=authorization)


async def verify_optional_client_access(authorization: Optional[str]) -> bool:
    if not authorization:
        return False
    await verify_access_key(authorization=authorization)
    return True


def is_responses_path(normalized_path: str) -> bool:
    return normalized_path == "/responses" or normalized_path.startswith("/responses/")


async def send_provider_request(
    request: Request,
    provider: ProviderRuntime,
    upstream_path: str,
    api_key: str,
    *,
    is_stream: bool,
    content: Optional[bytes] = None,
    headers_override: Optional[dict[str, str]] = None,
    params: Optional[dict[str, Any]] = None,
) -> httpx.Response:
    await provider.request_pacer.wait()

    req_kwargs = {
        "method": request.method,
        "url": f"{provider.base_url}{upstream_path}",
        "headers": prepare_forward_headers(request),
        "content": await request.body() if content is None else content,
        "params": prepare_forward_params(request) if params is None else params,
    }
    if headers_override:
        req_kwargs["headers"].update(headers_override)
    if api_key:
        req_kwargs["headers"]["Authorization"] = f"Bearer {api_key}"

    client = await get_async_client(request)
    openrouter_req = client.build_request(**req_kwargs)
    return await client.send(openrouter_req, stream=is_stream)


def _build_local_models_payload(provider: ProviderRuntime) -> bytes:
    """Build a synthetic OpenAI-format /models payload from allowed_models."""
    now = int(time.time())
    models = []
    for model_id in provider.allowed_models:
        models.append({
            "id": model_id,
            "object": "model",
            "created": now,
            "owned_by": provider.provider_id,
        })
    if not models:
        models.append({
            "id": f"{provider.provider_id}/default",
            "object": "model",
            "created": now,
            "owned_by": provider.provider_id,
        })
    return json.dumps({"object": "list", "data": models}, ensure_ascii=False).encode("utf-8")


async def handle_models_endpoint(
    request: Request,
    request_path: str,
    client_prefix: str,
    authorization: Optional[str],
) -> Response:
    """Aggregate provider model lists into a single provider-qualified response."""
    normalized_path = normalize_proxy_path(request_path)
    provider_hint = request.query_params.get("provider")
    params = prepare_forward_params(request)
    upstream_path = translate_request_path(request_path, client_prefix)

    selected_provider = get_provider_by_hint(provider_hint) if provider_hint else None
    if selected_provider is not None:
        await verify_client_access(selected_provider, normalized_path, authorization)
        providers = [selected_provider]
    else:
        if await verify_optional_client_access(authorization):
            providers = list(PROVIDER_REGISTRY.values())
        else:
            providers = [provider for provider in PROVIDER_REGISTRY.values() if provider.path_is_public(normalized_path)]

    if not providers:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing",
        )

    async def fetch_provider_models(provider: ProviderRuntime) -> Optional[bytes]:
        # Anthropic-mode providers don't have a real /models endpoint;
        # return a synthetic list from configured allowed_models.
        if provider.upstream_format == "anthropic":
            return _build_local_models_payload(provider)

        # Gemini-mode providers don't have a real /models endpoint either.
        if provider.upstream_format == "gemini":
            return _build_local_models_payload(provider)

        api_key = await provider.key_manager.get_next_key()
        response = await send_provider_request(
            request,
            provider,
            upstream_path,
            api_key,
            is_stream=False,
            params=params,
        )
        try:
            if response.status_code >= 400:
                await check_httpx_err(response.content, provider, api_key)
                if selected_provider is not None or len(providers) == 1:
                    message, code, param = extract_error_details(response.content, response.status_code)
                    headers, _ = split_response_headers(response.headers, "application/json")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail={"message": message, "code": code, "param": param},
                        headers=headers,
                    )
                logger.warning(
                    "Skipping provider %s models due to upstream status %s",
                    provider.provider_id,
                    response.status_code,
                )
                return None

            body = response.content
            await check_httpx_err(body, provider, api_key)
            return normalize_models_payload_for_client(provider, body)
        finally:
            await response.aclose()

    results = await asyncio.gather(*(fetch_provider_models(provider) for provider in providers), return_exceptions=True)

    successful_payloads: list[bytes] = []
    for result in results:
        if isinstance(result, Exception):
            if selected_provider is not None or len(providers) == 1:
                raise result
            logger.exception("Skipping provider models aggregation error")
            continue
        if result is not None:
            successful_payloads.append(result)

    if not successful_payloads:
        raise HTTPException(
            status_code=502,
            detail="Unable to retrieve models from any configured provider",
        )

    return Response(
        content=aggregate_model_payloads(successful_payloads),
        status_code=200,
        media_type="application/json",
    )


@router.api_route("/api/v1", methods=SUPPORTED_PROXY_METHODS, include_in_schema=False)
@router.api_route("/api/v1{path:path}", methods=SUPPORTED_PROXY_METHODS, include_in_schema=False)
@router.api_route("/v1", methods=SUPPORTED_PROXY_METHODS, include_in_schema=False)
@router.api_route("/v1{path:path}", methods=SUPPORTED_PROXY_METHODS, include_in_schema=False)
async def proxy_endpoint(
    request: Request, path: str, authorization: Optional[str] = Header(None)
):
    """Main proxy endpoint for handling provider-qualified client paths."""
    try:
        request_path = request.url.path
        client_prefix = get_client_prefix(request_path)
        if client_prefix is None:
            raise HTTPException(status_code=404, detail="Unsupported API prefix")

        normalized_path = normalize_proxy_path(request_path)
        is_stream = False
        body_bytes: Optional[bytes] = None
        request_body: Optional[dict[str, Any]] = None
        if request.method in {"POST", "PUT", "PATCH"}:
            try:
                body_bytes = await request.body()
                if body_bytes:
                    if len(body_bytes) > MAX_BODY_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail={"message": f"Request body too large (max {MAX_BODY_SIZE // (1024*1024)}MB)", "code": "invalid_request_error"},
                        )
                    parsed_body = json.loads(body_bytes)
                    if isinstance(parsed_body, dict):
                        request_body = parsed_body
            except HTTPException:
                raise
            except Exception as e:
                logger.debug("Could not parse request body: %s", str(e))

        if normalized_path in MODELS_ENDPOINTS and request.method == "GET":
            return await handle_models_endpoint(request, request_path, client_prefix, authorization)

        # Anthropic Messages API — delegate to dedicated handler
        if normalized_path == "/messages" and request.method == "POST":
            return await handle_anthropic_messages(request, request_body, body_bytes, authorization)

        provider = resolve_request_provider(request, request_body, normalized_path)
        if provider is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "A provider hint or provider-qualified model is required",
                    "code": "invalid_request_error",
                    "param": "provider",
                },
            )

        await verify_client_access(provider, normalized_path, authorization)

        # Log the full request URL including query parameters
        full_url = str(request.url).replace(str(request.base_url), "/")

        # Get API key to use
        api_key = await provider.key_manager.get_next_key()

        logger.info(
            "Proxying request to %s (provider: %s, key: %s)",
            full_url,
            provider.provider_id,
            mask_key(api_key),
        )

        if request_body is not None:
            request_body = normalize_request_body_for_provider(provider, request_body)
            # Deduplicate tool_call IDs in conversation history (Kiro/Bedrock phantom block workaround)
            request_body = deduplicate_openai_tool_call_ids(request_body)
            # Deduplicate Kiro conversationState toolUseId values (Bedrock TOOL_DUPLICATE workaround)
            request_body = deduplicate_kiro_conversation_state(request_body)
            body_bytes = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
            is_stream = bool(request_body.get("stream", False))
            if is_stream:
                logger.info("[%s] Detected streaming request", provider.provider_id)
            if model := request_body.get("model"):
                logger.info("[%s] Using model: %s", provider.provider_id, model)

        if is_responses_path(normalized_path) and normalized_path != "/responses" and not provider.supports_stateful_responses:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Provider '{provider.provider_id}' does not support stateful Responses endpoints",
                    "code": "invalid_request_error",
                    "param": "provider",
                },
            )

        if is_responses_path(normalized_path):
            return await handle_responses_endpoint(
                request,
                provider,
                normalized_path,
                api_key,
                body_bytes,
                is_stream,
            )

        upstream_path = translate_request_path(request_path, client_prefix)

        model_name = request_body.get("model", "") if request_body else ""
        upstream_model = model_name.split("/")[-1].lower() if "/" in model_name else model_name.lower()
        effective_upstream_format = provider.upstream_format
        if effective_upstream_format in ("gemini", "auto"):
            if "claude" in upstream_model or "sonnet" in upstream_model or "opus" in upstream_model:
                effective_upstream_format = "anthropic"
            elif "gemini" in upstream_model:
                effective_upstream_format = "gemini"
            elif effective_upstream_format == "auto":
                effective_upstream_format = "openai"

        # Anthropic-mode providers (AgentRouter) need full request translation
        if effective_upstream_format == "anthropic" and normalized_path == "/chat/completions":
            return await proxy_anthropic_with_httpx(
                request,
                provider,
                normalized_path,
                api_key,
                is_stream,
                body_bytes=body_bytes,
                request_body=request_body,
            )

        # Gemini-mode providers need request translation
        if effective_upstream_format == "gemini" and normalized_path == "/chat/completions":
            return await proxy_openai_to_gemini(
                request,
                provider,
                normalized_path,
                api_key,
                is_stream,
                body_bytes=body_bytes,
                request_body=request_body,
            )

        return await proxy_with_httpx(
            request,
            provider,
            upstream_path,
            normalized_path,
            api_key,
            is_stream,
            content_bytes=body_bytes,
        )
    except HTTPException as exc:
        detail = exc.detail
        if isinstance(detail, dict):
            message = str(detail.get("message") or detail.get("detail") or detail)
            code = detail.get("code")
            param = detail.get("param")
        else:
            message = str(detail)
            code = None
            param = None
        return build_openai_error_response(
            exc.status_code,
            message,
            code=code,
            param=param,
            headers=exc.headers,
        )
    except Exception:
        logger.exception("Unexpected proxy error")
        return build_openai_error_response(500, "Internal Proxy Error")


async def handle_responses_endpoint(
    request: Request,
    provider: ProviderRuntime,
    normalized_path: str,
    api_key: str,
    content_bytes: Optional[bytes],
    is_stream: bool,
) -> Response:
    """Proxy OpenAI Responses API requests to the selected provider.

    Handles translation based on provider.upstream_format:
      openai    → passthrough (Responses API)
      anthropic → translate Responses→Anthropic
      gemini    → translate Responses→Gemini
    """
    if content_bytes is None:
        content_bytes = await request.body()
    try:
        request_body = json.loads(content_bytes)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model_name = request_body.get("model", "")
    upstream_model = model_name.split("/")[-1].lower() if "/" in model_name else model_name.lower()
    
    effective_upstream_format = provider.upstream_format
    if effective_upstream_format in ("gemini", "auto"):
        if "claude" in upstream_model or "sonnet" in upstream_model or "opus" in upstream_model:
            effective_upstream_format = "anthropic"
        elif "gemini" in upstream_model:
            effective_upstream_format = "gemini"
        elif effective_upstream_format == "auto":
            effective_upstream_format = "openai"

    if effective_upstream_format == "anthropic":
        try:
            anthropic_body = translate_responses_to_anthropic(
                request_body,
                default_max_tokens=provider.default_max_tokens,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return await proxy_anthropic_with_httpx(
            request, provider, normalized_path, api_key, is_stream,
            body_bytes=json.dumps(anthropic_body, ensure_ascii=False).encode("utf-8"),
            request_body=anthropic_body,
        )

    if effective_upstream_format == "gemini":
        return await proxy_responses_to_gemini(request, provider, request_body, api_key, is_stream)

    return await proxy_with_httpx(
        request,
        provider,
        normalized_path,
        normalized_path,
        api_key,
        is_stream,
        content_bytes=content_bytes,
    )


async def proxy_with_httpx(
    request: Request,
    provider: ProviderRuntime,
    upstream_path: str,
    normalized_path: str,
    api_key: str,
    is_stream: bool,
    content_bytes: Optional[bytes] = None,
) -> Response:
    """Core logic to proxy requests."""
    try:
        openrouter_resp = await send_provider_request(
            request,
            provider,
            upstream_path,
            api_key,
            is_stream=is_stream,
            content=content_bytes,
        )
    except httpx.ConnectError as e:
        logger.error("Connection error to %s: %s", provider.provider_id, str(e))
        raise HTTPException(503, "Unable to connect to the upstream provider") from e
    except httpx.TimeoutException as e:
        logger.error("Timeout connecting to %s: %s", provider.provider_id, str(e))
        raise HTTPException(504, "Upstream provider request timed out") from e

    try:
        if openrouter_resp.status_code >= 400:
            if is_stream:
                try:
                    await openrouter_resp.aread()
                except Exception:
                    pass
            await check_httpx_err(openrouter_resp.content, provider, api_key)
            message, code, param = extract_error_details(openrouter_resp.content, openrouter_resp.status_code)
            headers, _ = split_response_headers(openrouter_resp.headers, "application/json")
            raise HTTPException(
                status_code=openrouter_resp.status_code,
                detail={"message": message, "code": code, "param": param},
                headers=headers,
            )

        default_content_type = "text/event-stream" if is_stream else "application/json"
        headers, content_type = split_response_headers(openrouter_resp.headers, default_content_type)

        if not is_stream:
            body = openrouter_resp.content
            await check_httpx_err(body, provider, api_key)
            if normalized_path in MODELS_ENDPOINTS:
                body = normalize_models_payload_for_client(provider, body)
            return Response(
                content=body,
                status_code=openrouter_resp.status_code,
                media_type=content_type,
                headers=headers,
            )

        async def sse_stream():
            last_json = ""
            try:
                async for line in openrouter_resp.aiter_lines():
                    if line:
                        if line.startswith("data: {"):
                            last_json = line[6:]
                        yield f"{line}\n\n".encode("utf-8")
            except Exception:
                logger.exception("sse_stream error")
            finally:
                try:
                    await check_httpx_err(last_json, provider, api_key)
                finally:
                    await openrouter_resp.aclose()

        return StreamingResponse(
            sse_stream(),
            status_code=openrouter_resp.status_code,
            media_type=content_type,
            headers=headers,
        )
    except HTTPException:
        await openrouter_resp.aclose()
        raise
    except Exception:
        await openrouter_resp.aclose()
        raise


async def proxy_anthropic_with_httpx(
    request: Request,
    provider: ProviderRuntime,
    normalized_path: str,
    api_key: str,
    is_stream: bool,
    *,
    body_bytes: Optional[bytes] = None,
    request_body: Optional[dict[str, Any]] = None,
) -> Response:
    """Proxy to an Anthropic Messages API upstream (AgentRouter).

    Translates OpenAI chat completion request → Anthropic Messages format,
    spoofs Claude Code headers, and translates the response back.
    """
    if request_body is None:
        if body_bytes is None:
            body_bytes = await request.body()
        try:
            request_body = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Deduplicate tool_call IDs in OpenAI request history
    request_body = deduplicate_openai_tool_call_ids(request_body)

    # DEBUG: verify dedup ran
    _debug_ids = []
    for _msg in request_body.get("messages", []):
        if _msg.get("role") == "assistant" and _msg.get("tool_calls"):
            for _tc in _msg["tool_calls"]:
                _debug_ids.append(_tc.get("id", ""))
    logger.info("[DEBUG] After dedup, tool_call IDs: %s", _debug_ids)

    # Translate request
    try:
        billing_hdr = str(provider.fingerprint.get("billing_header", "")) if provider.fingerprint else ""
        anthropic_body, model_name = translate_request(
            request_body,
            model_prefix=provider.model_prefix or provider.provider_id + "/",
            default_max_tokens=provider.default_max_tokens,
            inject_billing=provider.inject_billing,
            billing_header=billing_hdr,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    anthropic_payload = json.dumps(anthropic_body, ensure_ascii=False).encode("utf-8")

    # DEBUG: verify translated anthropic tool_use IDs
    _debug_aids = []
    for _msg in anthropic_body.get("messages", []):
        for _block in _msg.get("content", []):
            if isinstance(_block, dict) and _block.get("type") == "tool_use":
                _debug_aids.append(_block.get("id", ""))
    logger.info("[DEBUG] Anthropic tool_use IDs: %s", _debug_aids)
    if len(_debug_aids) != len(set(_debug_aids)):
        logger.error("[DEBUG] DUPLICATE DETECTED in Anthropic output!")

    # Determine upstream URL
    messages_path = provider.upstream_messages_path or "/v1/messages?beta=true"
    upstream_url = f"{provider.base_url}{messages_path}"

    # Build Claude Code headers
    client_session_id = request.headers.get("X-Claude-Code-Session-Id")
    session_id = _get_or_create_session_id(
        provider.provider_id,
        request_body,
        client_hint=client_session_id,
    )
    headers = build_anthropic_headers(
        api_key,
        stream=True,
        fingerprint=provider.fingerprint or None,
        custom_headers=provider.custom_headers or None,
        session_id=session_id,
    )

    await provider.request_pacer.wait()

    client = await get_async_client(request)
    upstream_req = client.build_request(
        method="POST",
        url=upstream_url,
        headers=headers,
        content=anthropic_payload,
    )

    logger.info(
        "[%s] Anthropic proxy: %s (model: %s, key: %s)",
        provider.provider_id,
        upstream_url,
        model_name,
        mask_key(api_key),
    )

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectError as e:
        logger.error("Connection error to %s: %s", provider.provider_id, str(e))
        raise HTTPException(503, "Unable to connect to the upstream provider") from e
    except httpx.TimeoutException as e:
        logger.error("Timeout connecting to %s: %s", provider.provider_id, str(e))
        raise HTTPException(504, "Upstream provider request timed out") from e

    if upstream_resp.status_code >= 400:
        try:
            error_body = await upstream_resp.aread()
        except Exception:
            error_body = b""
        await upstream_resp.aclose()
        message = error_body.decode("utf-8", errors="replace").strip() or f"Upstream error ({upstream_resp.status_code})"
        logger.error("[%s] Upstream error %s: %s", provider.provider_id, upstream_resp.status_code, message[:500])
        await check_httpx_err(message, provider, api_key)
        raise HTTPException(
            status_code=upstream_resp.status_code,
            detail={"message": message, "code": upstream_resp.status_code, "param": None},
        )

    # Non-streaming: aggregate all SSE events into a single response
    if not is_stream:
        decoder = AnthropicStreamDecoder(model_name)
        event_name = ""
        data_lines: list[str] = []
        try:
            async for line in upstream_resp.aiter_lines():
                line = line.rstrip("\r")
                if not line:
                    decoder.process_sse_lines(data_lines)
                    event_name = ""
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        except Exception as e:
            logger.error("[%s] Error reading anthropic stream: %s", provider.provider_id, e)
        finally:
            await upstream_resp.aclose()

        result = decoder.build_final_response()
        if is_responses_path(normalized_path):
            result = _openai_chat_response_to_responses(result)
        return JSONResponse(content=result, status_code=200)

    # Streaming: translate Anthropic SSE → OpenAI SSE chunks
    async def translated_sse():
        decoder = AnthropicStreamDecoder(model_name)
        is_responses = is_responses_path(normalized_path)
        event_name = ""
        data_lines: list[str] = []

        def _emit(chunk: dict[str, Any]) -> list[bytes]:
            parts: list[bytes] = []
            if is_responses:
                for resp_chunk in _openai_chat_chunk_to_responses(chunk):
                    parts.append(f"data: {json.dumps(resp_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            else:
                parts.append(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            return parts

        def _emit_with_pending(chunk: Optional[dict[str, Any]]) -> list[bytes]:
            parts: list[bytes] = []
            if chunk is not None:
                parts.extend(_emit(chunk))
            for pending in decoder.drain_pending():
                parts.extend(_emit(pending))
            return parts

        try:
            async for line in upstream_resp.aiter_lines():
                line = line.rstrip("\r")
                if not line:
                    chunk = decoder.process_sse_lines(data_lines)
                    for b in _emit_with_pending(chunk):
                        yield b
                    event_name = ""
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())

            if data_lines:
                chunk = decoder.process_sse_lines(data_lines)
                for b in _emit_with_pending(chunk):
                    yield b
            yield b"data: [DONE]\n\n"
        except Exception as e:
            logger.exception("[%s] SSE translation error", provider.provider_id)
        finally:
            try:
                await upstream_resp.aclose()
            except Exception:
                pass

    return StreamingResponse(
        translated_sse(),
        status_code=200,
        media_type="text/event-stream",
    )


async def handle_anthropic_messages(
    request: Request,
    request_body: Optional[dict[str, Any]],
    body_bytes: Optional[bytes],
    authorization: Optional[str],
) -> Response:
    """Handle Anthropic Messages API requests (client → proxy).

    Routes based on the model prefix to the correct provider, then translates
    based on the provider's upstream_format.
    """
    if request_body is None:
        return build_anthropic_error_response(400, "Invalid JSON body")

    model = request_body.get("model", "")
    if not model:
        return build_anthropic_error_response(400, "model is required")

    # Resolve provider from model prefix
    provider_name = model.split("/")[0] if "/" in model else None
    if not provider_name:
        return build_anthropic_error_response(
            400, "Model must be provider-qualified as <provider>/<model>",
        )

    provider = get_provider_by_hint(provider_name)
    if provider is None:
        return build_anthropic_error_response(400, "Provider not found for the given model")

    await verify_client_access(provider, "/messages", authorization)
    api_key = await provider.key_manager.get_next_key()

    is_stream = request_body.get("stream", False)

    # ── Upstream is OpenAI format → translate Anthropic→OpenAI → send → translate back
    if provider.upstream_format == "openai":
        try:
            openai_body, model_name = translate_anthropic_to_openai(
                request_body,
                model_prefix=provider.model_prefix,
            )
        except ValueError as e:
            return build_anthropic_error_response(400, str(e))

        openai_payload = json.dumps(openai_body, ensure_ascii=False).encode("utf-8")
        chat_path = provider.upstream_chat_path or "/v1/chat/completions"
        upstream_url = f"{provider.base_url}{chat_path}"

        await provider.request_pacer.wait()
        client = await get_async_client(request)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP_REQUEST_HEADERS}
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"

        upstream_req = client.build_request(
            method="POST", url=upstream_url, headers=headers, content=openai_payload,
        )
        try:
            upstream_resp = await client.send(upstream_req, stream=True)
        except httpx.ConnectError as e:
            raise HTTPException(503, "Unable to connect to upstream") from e
        except httpx.TimeoutException as e:
            raise HTTPException(504, "Upstream timeout") from e

        if upstream_resp.status_code >= 400:
            error_body = await upstream_resp.aread()
            await upstream_resp.aclose()
            msg = error_body.decode("utf-8", errors="replace").strip() or f"Upstream error ({upstream_resp.status_code})"
            return build_anthropic_error_response(upstream_resp.status_code, msg)

        if is_stream:
            translator = OpenAIToAnthropicSSETranslator(model_name)

            async def anthropic_sse_from_openai():
                data_lines: list[str] = []
                try:
                    async for line in upstream_resp.aiter_lines():
                        line = line.rstrip("\r")
                        if not line:
                            if data_lines:
                                data_str = "\n".join(data_lines)
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    events = translator.translate_chunk(chunk)
                                    if events:
                                        yield events.encode("utf-8")
                                except (json.JSONDecodeError, TypeError):
                                    pass
                                data_lines = []
                            continue
                        if line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                    if not translator._finished:
                        finish_events = translator.finish_message()
                        if finish_events:
                            yield finish_events.encode("utf-8")
                except Exception:
                    logger.exception("[%s] OpenAI→Anthropic SSE error", provider.provider_id)
                finally:
                    await upstream_resp.aclose()

            return StreamingResponse(anthropic_sse_from_openai(), status_code=200, media_type="text/event-stream")

        # Non-streaming
        data_lines = []
        openai_resp_data = None
        async for line in upstream_resp.aiter_lines():
            line = line.rstrip("\r")
            if not line:
                if data_lines:
                    try:
                        openai_resp_data = json.loads("\n".join(data_lines))
                    except (json.JSONDecodeError, TypeError):
                        pass
                    data_lines = []
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        await upstream_resp.aclose()

        if openai_resp_data:
            translator = OpenAIToAnthropicSSETranslator(model_name)
            return JSONResponse(content=translator.translate_complete_response(openai_resp_data))

        return build_anthropic_error_response(502, "No response from upstream")

    # ── Upstream is Gemini format → translate Anthropic→Gemini
    if provider.upstream_format == "gemini":
        is_stream = request_body.get("stream", False)
        return await proxy_anthropic_to_gemini(request, provider, request_body, api_key, is_stream)

    # ── Upstream is Anthropic format → passthrough with correct headers
    else:
        if provider.model_prefix and request_body.get("model", "").startswith(provider.model_prefix):
            request_body["model"] = request_body["model"][len(provider.model_prefix):]

        # Deduplicate tool_use IDs in conversation history (Kiro/Bedrock phantom block workaround)
        request_body = deduplicate_anthropic_tool_use_ids(request_body)

        # Always force stream: true upstream (AgentRouter/Anthropic requires it)
        request_body["stream"] = True
        body_bytes = json.dumps(request_body, ensure_ascii=False).encode("utf-8")

        messages_path = provider.upstream_messages_path or "/v1/messages"
        upstream_url = f"{provider.base_url}{messages_path}"
        if "?" not in messages_path:
            upstream_url += "?beta=true"

        headers = build_anthropic_headers(
            api_key, stream=True,
            fingerprint=provider.fingerprint or None,
            custom_headers=provider.custom_headers or None,
            session_id=_get_or_create_session_id(
                provider.provider_id,
                request_body,
                client_hint=request.headers.get("X-Claude-Code-Session-Id"),
            ),
        )

        await provider.request_pacer.wait()
        client = await get_async_client(request)
        upstream_req = client.build_request(
            method="POST", url=upstream_url, headers=headers, content=body_bytes,
        )

        logger.info("[%s] Anthropic passthrough: %s (key: %s)", provider.provider_id, upstream_url, mask_key(api_key))

        try:
            upstream_resp = await client.send(upstream_req, stream=True)
        except httpx.ConnectError as e:
            raise HTTPException(503, "Unable to connect to upstream") from e
        except httpx.TimeoutException as e:
            raise HTTPException(504, "Upstream timeout") from e

        if upstream_resp.status_code >= 400:
            error_body = await upstream_resp.aread()
            await upstream_resp.aclose()
            msg = error_body.decode("utf-8", errors="replace").strip() or f"Upstream error ({upstream_resp.status_code})"
            logger.error("[%s] Upstream error %s: %s", provider.provider_id, upstream_resp.status_code, msg[:300])
            return build_anthropic_error_response(upstream_resp.status_code, msg)

        # Client wants streaming → pipe through directly
        if is_stream:
            async def passthrough_sse():
                try:
                    async for line in upstream_resp.aiter_lines():
                        yield f"{line}\n".encode("utf-8")
                finally:
                    await upstream_resp.aclose()
            return StreamingResponse(passthrough_sse(), status_code=200, media_type="text/event-stream")

        # Client wants non-streaming → aggregate SSE into single Anthropic response
        aggregator = AnthropicMessageAggregator()
        data_lines: list[str] = []
        try:
            async for line in upstream_resp.aiter_lines():
                line = line.rstrip("\r")
                if not line:
                    aggregator.process_sse_lines(data_lines)
                    data_lines = []
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        except Exception:
            logger.error("[%s] Error reading anthropic passthrough stream", provider.provider_id)
        finally:
            await upstream_resp.aclose()

        return JSONResponse(content=aggregator.build_response())


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


# ── Anthropic client-facing endpoint ─────────────────────────────────────────


@router.post("/v1/messages")
@router.post("/api/v1/messages")
async def anthropic_messages_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Client-facing Anthropic Messages API endpoint.

    Accepts Anthropic Messages format from clients, routes to the correct
    provider based on the model field, and translates based on upstream_format.

    Translation matrix:
      Client Anthropic → Upstream OpenAI  : translate_anthropic_to_openai()
      Client Anthropic → Upstream Anthropic: passthrough (just fix headers)
    """
    try:
        body_bytes = await request.body()
        try:
            request_body = json.loads(body_bytes) if body_bytes else {}
        except (json.JSONDecodeError, ValueError):
            return build_anthropic_error_response(400, "Invalid JSON body")

        model = request_body.get("model", "")
        if not model:
            return build_anthropic_error_response(400, "model is required")

        # Resolve provider from model prefix
        provider_name = model.split("/")[0] if "/" in model else None
        if not provider_name:
            return build_anthropic_error_response(
                400, "Model must be provider-qualified as <provider>/<model>",
            )

        provider = get_provider_by_hint(provider_name)
        if provider is None:
            return build_anthropic_error_response(400, "Provider not found for the given model")

        await verify_client_access(provider, "/messages", authorization)
        api_key = await provider.key_manager.get_next_key()

        # ── Upstream is OpenAI format ───────────────────────────────────
        if provider.upstream_format == "openai":
            # Deduplicate tool_use IDs before translation
            request_body = deduplicate_anthropic_tool_use_ids(request_body)
            # Translate Anthropic → OpenAI
            try:
                openai_body, model_name = translate_anthropic_to_openai(
                    request_body,
                    model_prefix=provider.model_prefix,
                )
            except ValueError as e:
                return build_anthropic_error_response(400, str(e))

            openai_payload = json.dumps(openai_body, ensure_ascii=False).encode("utf-8")
            is_stream = request_body.get("stream", False)

            # Use the chat completions upstream path
            chat_path = provider.upstream_chat_path or "/v1/chat/completions"
            upstream_url = f"{provider.base_url}{chat_path}"

            await provider.request_pacer.wait()
            client = await get_async_client(request)
            headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP_REQUEST_HEADERS}
            headers["Authorization"] = f"Bearer {api_key}"
            headers["Content-Type"] = "application/json"

            upstream_req = client.build_request(
                method="POST", url=upstream_url, headers=headers, content=openai_payload,
            )

            try:
                upstream_resp = await client.send(upstream_req, stream=True)
            except httpx.ConnectError as e:
                raise HTTPException(503, "Unable to connect to upstream") from e
            except httpx.TimeoutException as e:
                raise HTTPException(504, "Upstream timeout") from e

            if upstream_resp.status_code >= 400:
                error_body = await upstream_resp.aread()
                await upstream_resp.aclose()
                msg = error_body.decode("utf-8", errors="replace").strip() or f"Upstream error ({upstream_resp.status_code})"
                return build_anthropic_error_response(upstream_resp.status_code, msg)

            # Streaming: translate OpenAI SSE → Anthropic SSE
            if is_stream:
                translator = OpenAIToAnthropicSSETranslator(model_name)

                async def anthropic_sse_from_openai():
                    data_lines: list[str] = []
                    try:
                        async for line in upstream_resp.aiter_lines():
                            line = line.rstrip("\r")
                            if not line:
                                if data_lines:
                                    data_str = "\n".join(data_lines)
                                    if data_str.strip() == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(data_str)
                                        events = translator.translate_chunk(chunk)
                                        if events:
                                            yield events.encode("utf-8")
                                    except (json.JSONDecodeError, TypeError):
                                        pass
                                    data_lines = []
                                continue
                            if line.startswith("data:"):
                                data_lines.append(line[5:].lstrip())
                        # Ensure message is properly finished
                        if not translator._finished:
                            finish_events = translator.finish_message()
                            if finish_events:
                                yield finish_events.encode("utf-8")
                    except Exception:
                        logger.exception("[%s] OpenAI→Anthropic SSE error", provider.provider_id)
                    finally:
                        await upstream_resp.aclose()

                return StreamingResponse(
                    anthropic_sse_from_openai(),
                    status_code=200,
                    media_type="text/event-stream",
                )

            # Non-streaming: aggregate OpenAI response → Anthropic response
            data_lines = []
            openai_resp_data = None
            async for line in upstream_resp.aiter_lines():
                line = line.rstrip("\r")
                if not line:
                    if data_lines:
                        data_str = "\n".join(data_lines)
                        try:
                            openai_resp_data = json.loads(data_str)
                        except (json.JSONDecodeError, TypeError):
                            pass
                        data_lines = []
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            await upstream_resp.aclose()

            if openai_resp_data:
                translator = OpenAIToAnthropicSSETranslator(model_name)
                anthropic_resp = translator.translate_complete_response(openai_resp_data)
                return JSONResponse(content=anthropic_resp)

            return build_anthropic_error_response(502, "No response from upstream")

        # ── Upstream is Gemini format → translate Anthropic→Gemini ─────
        if provider.upstream_format == "gemini":
            is_stream = request_body.get("stream", False)
            return await proxy_anthropic_to_gemini(request, provider, request_body, api_key, is_stream)

        # ── Upstream is Anthropic format ────────────────────────────────
        else:
            # Passthrough — client format matches upstream format
            # Strip model prefix before sending upstream
            if provider.model_prefix and request_body.get("model", "").startswith(provider.model_prefix):
                request_body["model"] = request_body["model"][len(provider.model_prefix):]

            # Deduplicate tool_use IDs in conversation history (Kiro/Bedrock phantom block workaround)
            request_body = deduplicate_anthropic_tool_use_ids(request_body)
            body_bytes = json.dumps(request_body, ensure_ascii=False).encode("utf-8")

            messages_path = provider.upstream_messages_path or "/v1/messages"
            if "?" in messages_path:
                upstream_url = f"{provider.base_url}{messages_path}"
            else:
                upstream_url = f"{provider.base_url}{messages_path}?beta=true"

            is_stream = request_body.get("stream", False)

            headers = build_anthropic_headers(
                api_key,
                stream=is_stream,
                fingerprint=provider.fingerprint or None,
                custom_headers=provider.custom_headers or None,
                session_id=_get_or_create_session_id(
                    provider.provider_id,
                    request_body,
                    client_hint=request.headers.get("X-Claude-Code-Session-Id"),
                ),
            )

            await provider.request_pacer.wait()
            client = await get_async_client(request)
            upstream_req = client.build_request(
                method="POST", url=upstream_url, headers=headers, content=body_bytes,
            )

            try:
                upstream_resp = await client.send(upstream_req, stream=True)
            except httpx.ConnectError as e:
                raise HTTPException(503, "Unable to connect to upstream") from e
            except httpx.TimeoutException as e:
                raise HTTPException(504, "Upstream timeout") from e

            if upstream_resp.status_code >= 400:
                error_body = await upstream_resp.aread()
                await upstream_resp.aclose()
                msg = error_body.decode("utf-8", errors="replace").strip() or f"Upstream error ({upstream_resp.status_code})"
                return build_anthropic_error_response(upstream_resp.status_code, msg)

            if is_stream:
                async def passthrough_sse():
                    try:
                        async for line in upstream_resp.aiter_lines():
                            yield f"{line}\n".encode("utf-8")
                    except Exception:
                        logger.exception("[%s] Anthropic passthrough SSE error", provider.provider_id)
                    finally:
                        await upstream_resp.aclose()

                return StreamingResponse(
                    passthrough_sse(),
                    status_code=200,
                    media_type="text/event-stream",
                )

            # Non-streaming passthrough
            resp_body = await upstream_resp.aread()
            await upstream_resp.aclose()
            return Response(
                content=resp_body,
                status_code=upstream_resp.status_code,
                media_type="application/json",
            )

    except HTTPException as exc:
        detail = exc.detail
        if isinstance(detail, dict):
            message = str(detail.get("message") or detail.get("detail") or detail)
        else:
            message = str(detail)
        return build_anthropic_error_response(exc.status_code, message)
    except Exception:
        logger.exception("Anthropic endpoint error")
        return build_anthropic_error_response(500, "Internal Proxy Error")


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI Chat → Gemini upstream proxy
# ══════════════════════════════════════════════════════════════════════════════


async def proxy_openai_to_gemini(
    request: Request,
    provider: ProviderRuntime,
    normalized_path: str,
    api_key: str,
    is_stream: bool,
    *,
    body_bytes: Optional[bytes] = None,
    request_body: Optional[dict[str, Any]] = None,
) -> Response:
    """Proxy OpenAI Chat Completions request to a Gemini upstream."""
    if request_body is None:
        if body_bytes is None:
            body_bytes = await request.body()
        try:
            request_body = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Deduplicate tool_call IDs before translation
    request_body = deduplicate_openai_tool_call_ids(request_body)

    try:
        gemini_body, model_name = translate_openai_to_gemini(
            request_body,
            model_prefix=provider.model_prefix or provider.provider_id + "/",
            default_max_tokens=provider.default_max_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    api_version = provider.gemini_api_version or "v1beta"
    endpoint = ":streamGenerateContent?alt=sse" if is_stream else ":generateContent"
    generate_path = provider.upstream_generate_path or f"/{api_version}/models/{model_name}{endpoint}"
    upstream_url = f"{provider.base_url}{generate_path}"

    gemini_payload = json.dumps(gemini_body, ensure_ascii=False).encode("utf-8")

    await provider.request_pacer.wait()
    client = await get_async_client(request)
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    upstream_req = client.build_request(method="POST", url=upstream_url, headers=headers, content=gemini_payload)
    logger.info("[%s] OpenAI→Gemini: %s (model: %s, key: %s)", provider.provider_id, upstream_url, model_name, mask_key(api_key))

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectError as e:
        raise HTTPException(503, "Unable to connect to upstream") from e
    except httpx.TimeoutException as e:
        raise HTTPException(504, "Upstream timeout") from e

    if upstream_resp.status_code >= 400:
        error_body = await upstream_resp.aread()
        await upstream_resp.aclose()
        msg = error_body.decode("utf-8", errors="replace").strip() or f"Upstream error ({upstream_resp.status_code})"
        raise HTTPException(status_code=upstream_resp.status_code, detail={"message": msg, "code": upstream_resp.status_code})

    if not is_stream:
        resp_data = await _read_gemini_response(upstream_resp)
        if resp_data:
            openai_resp = translate_gemini_response_to_openai(resp_data, model_name)
            return JSONResponse(content=openai_resp)
        raise HTTPException(502, "No response from upstream")

    decoder = GeminiStreamDecoder(model_name)

    async def translated_sse():
        try:
            async for line in upstream_resp.aiter_lines():
                line = line.rstrip("\r")
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        chunks = decoder.process_sse_data(data)
                        for chunk in chunks:
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                    except (json.JSONDecodeError, TypeError):
                        pass
            yield b"data: [DONE]\n\n"
        except Exception:
            logger.exception("[%s] Gemini→OpenAI SSE error", provider.provider_id)
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(translated_sse(), status_code=200, media_type="text/event-stream")


async def _read_gemini_response(upstream_resp: httpx.Response) -> Optional[dict[str, Any]]:
    """Read a Gemini response (plain JSON) and return the parsed data."""
    resp_data: Optional[dict[str, Any]] = None
    try:
        content = await upstream_resp.aread()
        if content:
            try:
                resp_data = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                logger.debug("Failed to decode JSON from upstream")
    except Exception:
        logger.debug("Error reading Gemini response body")
    finally:
        await upstream_resp.aclose()
    return resp_data


# ══════════════════════════════════════════════════════════════════════════════
# Anthropic client → Gemini upstream proxy
# ══════════════════════════════════════════════════════════════════════════════


async def proxy_anthropic_to_gemini(
    request: Request,
    provider: ProviderRuntime,
    request_body: dict[str, Any],
    api_key: str,
    is_stream: bool,
) -> Response:
    """Proxy Anthropic Messages request to a Gemini upstream."""
    # Deduplicate tool_use IDs before translation
    request_body = deduplicate_anthropic_tool_use_ids(request_body)

    try:
        gemini_body, model_name = translate_anthropic_to_gemini(
            request_body,
            model_prefix=provider.model_prefix or provider.provider_id + "/",
            default_max_tokens=provider.default_max_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    api_version = provider.gemini_api_version or "v1beta"
    endpoint = ":streamGenerateContent?alt=sse" if is_stream else ":generateContent"
    generate_path = provider.upstream_generate_path or f"/{api_version}/models/{model_name}{endpoint}"
    upstream_url = f"{provider.base_url}{generate_path}"

    gemini_payload = json.dumps(gemini_body, ensure_ascii=False).encode("utf-8")

    await provider.request_pacer.wait()
    client = await get_async_client(request)
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    upstream_req = client.build_request(method="POST", url=upstream_url, headers=headers, content=gemini_payload)
    logger.info("[%s] Anthropic→Gemini: %s (model: %s, key: %s)", provider.provider_id, upstream_url, model_name, mask_key(api_key))

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectError as e:
        raise HTTPException(503, "Unable to connect to upstream") from e
    except httpx.TimeoutException as e:
        raise HTTPException(504, "Upstream timeout") from e

    if upstream_resp.status_code >= 400:
        error_body = await upstream_resp.aread()
        await upstream_resp.aclose()
        msg = error_body.decode("utf-8", errors="replace").strip() or f"Upstream error ({upstream_resp.status_code})"
        raise HTTPException(status_code=upstream_resp.status_code, detail={"message": msg, "code": upstream_resp.status_code})

    if not is_stream:
        resp_data = await _read_gemini_response(upstream_resp)
        if resp_data:
            anthropic_resp = translate_gemini_response_to_anthropic(resp_data, model_name)
            return JSONResponse(content=anthropic_resp)
        raise HTTPException(502, "No response from upstream")

    decoder = GeminiAnthropicStreamDecoder(model_name)

    async def translated_sse():
        try:
            async for line in upstream_resp.aiter_lines():
                line = line.rstrip("\r")
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        events = decoder.process_sse_data(data)
                        if events:
                            yield events.encode("utf-8")
                    except (json.JSONDecodeError, TypeError):
                        pass
            finish_events = decoder.finish_message()
            if finish_events:
                yield finish_events.encode("utf-8")
        except Exception:
            logger.exception("[%s] Gemini→Anthropic SSE error", provider.provider_id)
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(translated_sse(), status_code=200, media_type="text/event-stream")


# ══════════════════════════════════════════════════════════════════════════════
# Responses client → Gemini upstream proxy
# ══════════════════════════════════════════════════════════════════════════════


async def proxy_responses_to_gemini(
    request: Request,
    provider: ProviderRuntime,
    request_body: dict[str, Any],
    api_key: str,
    is_stream: bool,
) -> Response:
    """Proxy OpenAI Responses API request to a Gemini upstream."""
    try:
        gemini_body, model_name = translate_responses_to_gemini(
            request_body,
            model_prefix=provider.model_prefix or provider.provider_id + "/",
            default_max_tokens=provider.default_max_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    api_version = provider.gemini_api_version or "v1beta"
    endpoint = ":streamGenerateContent?alt=sse" if is_stream else ":generateContent"
    generate_path = provider.upstream_generate_path or f"/{api_version}/models/{model_name}{endpoint}"
    upstream_url = f"{provider.base_url}{generate_path}"

    gemini_payload = json.dumps(gemini_body, ensure_ascii=False).encode("utf-8")

    await provider.request_pacer.wait()
    client = await get_async_client(request)
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    upstream_req = client.build_request(method="POST", url=upstream_url, headers=headers, content=gemini_payload)
    logger.info("[%s] Responses→Gemini: %s (model: %s, key: %s)", provider.provider_id, upstream_url, model_name, mask_key(api_key))

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectError as e:
        raise HTTPException(503, "Unable to connect to upstream") from e
    except httpx.TimeoutException as e:
        raise HTTPException(504, "Upstream timeout") from e

    if upstream_resp.status_code >= 400:
        error_body = await upstream_resp.aread()
        await upstream_resp.aclose()
        msg = error_body.decode("utf-8", errors="replace").strip() or f"Upstream error ({upstream_resp.status_code})"
        raise HTTPException(status_code=upstream_resp.status_code, detail={"message": msg, "code": upstream_resp.status_code})

    if not is_stream:
        resp_data = await _read_gemini_response(upstream_resp)
        if resp_data:
            # Convert Gemini response to OpenAI Responses format via Chat
            openai_chat_resp = translate_gemini_response_to_openai(resp_data, model_name)
            # For Responses API, wrap in responses-style envelope
            responses_resp = _openai_chat_response_to_responses(openai_chat_resp)
            return JSONResponse(content=responses_resp)
        raise HTTPException(502, "No response from upstream")

    # Streaming: translate Gemini SSE → OpenAI Responses SSE
    decoder = GeminiStreamDecoder(model_name)

    async def translated_sse():
        try:
            async for line in upstream_resp.aiter_lines():
                line = line.rstrip("\r")
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        chunks = decoder.process_sse_data(data)
                        for chunk in chunks:
                            resp_events = _openai_chat_chunk_to_responses(chunk)
                            for resp_chunk in resp_events:
                                yield f"data: {json.dumps(resp_chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                    except (json.JSONDecodeError, TypeError):
                        pass
            yield b"data: [DONE]\n\n"
        except Exception:
            logger.exception("[%s] Gemini→Responses SSE error", provider.provider_id)
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(translated_sse(), status_code=200, media_type="text/event-stream")


def _openai_chat_response_to_responses(chat_resp: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Chat Completion response to Responses API envelope."""
    choices = chat_resp.get("choices", [])
    usage = chat_resp.get("usage", {})

    output_items: list[dict[str, Any]] = []
    if choices:
        msg = choices[0].get("message", {})
        content_parts: list[dict[str, Any]] = []
        text = msg.get("content")
        if text:
            content_parts.append({"type": "output_text", "text": text})
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                content_parts.append({
                    "type": "tool_call",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                })
        reasoning_details = msg.get("reasoning_details")
        if reasoning_details:
            for rd in reasoning_details:
                if isinstance(rd, dict) and rd.get("type") == "reasoning.text":
                    content_parts.append({"type": "reasoning_text", "text": rd.get("text", "")})

        output_items.append({
            "type": "message",
            "role": "assistant",
            "content": content_parts,
        })

    return {
        "id": f"resp_{_random_hex(12)}",
        "object": "response",
        "created_at": chat_resp.get("created", 0),
        "model": chat_resp.get("model", ""),
        "output": output_items,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "status": "completed",
    }


def _openai_chat_chunk_to_responses(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an OpenAI Chat Completion chunk to Responses SSE events."""
    choices = chunk.get("choices", [])
    if not choices:
        return []

    choice = choices[0]
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    events: list[dict[str, Any]] = []

    content = delta.get("content")
    if content:
        events.append({
            "type": "response.output_text.delta",
            "delta": content,
        })

    reasoning_details = delta.get("reasoning_details")
    if reasoning_details:
        for rd in reasoning_details:
            if isinstance(rd, dict) and rd.get("type") == "reasoning.text":
                events.append({
                    "type": "response.reasoning_text.delta",
                    "delta": rd.get("text", ""),
                })

    if finish_reason:
        # Build a pseudo-complete response from accumulated state
        # The caller should track accumulated content for a proper response
        events.append({
            "type": "response.completed",
            "response": {
                "id": f"resp_{_random_hex(12)}",
                "object": "response",
                "status": "completed",
            },
        })

    return events


# ══════════════════════════════════════════════════════════════════════════════
# Gemini client-facing endpoint
# ══════════════════════════════════════════════════════════════════════════════


@router.post("/v1/models/{model_path:path}:generateContent")
@router.post("/api/v1/models/{model_path:path}:generateContent")
@router.post("/v1/models/{model_path:path}:streamGenerateContent")
@router.post("/api/v1/models/{model_path:path}:streamGenerateContent")
@router.post("/v1beta/models/{model_path:path}:generateContent")
@router.post("/v1beta/models/{model_path:path}:streamGenerateContent")
async def gemini_generate_endpoint(
    request: Request,
    model_path: str,
    authorization: Optional[str] = Header(None),
):
    """Client-facing Gemini GenerateContent API endpoint.

    Accepts Gemini format from clients, routes to the correct provider,
    and translates based on upstream_format.

    Translation matrix (Gemini client):
      Gemini → Upstream OpenAI    : translate_gemini_to_openai() → send → translate back
      Gemini → Upstream Anthropic : translate_gemini_to_anthropic() → send → translate back
      Gemini → Upstream Gemini    : passthrough
    """
    try:
        body_bytes = await request.body()
        try:
            request_body = json.loads(body_bytes) if body_bytes else {}
        except (json.JSONDecodeError, ValueError):
            return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON body", "code": 400}})

        is_stream = ":streamGenerateContent" in request.url.path

        model_name = model_path
        if request_body.get("model"):
            model_name = request_body["model"]
        else:
            request_body["model"] = model_name

        provider_name = model_name.split("/")[0] if "/" in model_name else None
        if not provider_name:
            return JSONResponse(status_code=400, content={
                "error": {"message": "Model must be provider-qualified as <provider>/<model>", "code": 400},
            })

        provider = get_provider_by_hint(provider_name)
        if provider is None:
            return JSONResponse(status_code=400, content={
                "error": {"message": "Provider not found for the given model", "code": 400},
            })

        await verify_client_access(provider, "/generateContent", authorization)
        api_key = await provider.key_manager.get_next_key()

        return await _handle_gemini_upstream(request, provider, request_body, api_key, is_stream)

    except HTTPException as exc:
        detail = exc.detail
        msg = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
        return JSONResponse(status_code=exc.status_code, content={"error": {"message": msg, "code": exc.status_code}})
    except Exception:
        logger.exception("Gemini endpoint error")
        return JSONResponse(status_code=500, content={"error": {"message": "Internal Proxy Error", "code": 500}})


async def _handle_gemini_upstream(
    request: Request,
    provider: ProviderRuntime,
    request_body: dict[str, Any],
    api_key: str,
    is_stream: bool,
) -> Response:
    """Route a Gemini client request to the appropriate upstream format."""

    if provider.upstream_format == "gemini":
        return await _proxy_gemini_passthrough(request, provider, request_body, api_key, is_stream)

    if provider.upstream_format == "openai":
        return await _proxy_gemini_to_openai_upstream(request, provider, request_body, api_key, is_stream)

    if provider.upstream_format == "anthropic":
        return await _proxy_gemini_to_anthropic_upstream(request, provider, request_body, api_key, is_stream)

    return JSONResponse(status_code=500, content={
        "error": {"message": f"Unknown upstream format: {provider.upstream_format}", "code": 500},
    })


async def _proxy_gemini_passthrough(
    request: Request,
    provider: ProviderRuntime,
    request_body: dict[str, Any],
    api_key: str,
    is_stream: bool,
) -> Response:
    """Passthrough a Gemini request to a Gemini upstream."""
    model = request_body.get("model", "")
    if provider.model_prefix and model.startswith(provider.model_prefix):
        request_body["model"] = model[len(provider.model_prefix):]

    # Inject toolConfig and tool_config when tools are present
    if request_body.get("tools"):
        # Strip built-in tools like googleSearch because the 8045 proxy (Vertex AI SDK)
        # drops includeServerSideToolInvocations, causing Vertex AI to throw an error
        # when custom functions are mixed with built-in tools.
        cleaned_tools = []
        for tool in request_body["tools"]:
            if isinstance(tool, dict):
                # Keep only functionDeclarations
                if "functionDeclarations" in tool:
                    cleaned_tools.append({"functionDeclarations": tool["functionDeclarations"]})
        request_body["tools"] = cleaned_tools

        mode = "AUTO"
        allowed_names = None

        tc = request_body.get("toolConfig")
        if isinstance(tc, dict):
            fcc = tc.get("functionCallingConfig")
            if isinstance(fcc, dict):
                mode = fcc.get("mode") or "AUTO"
                allowed_names = fcc.get("allowedFunctionNames")

        tc_s = request_body.get("tool_config")
        if isinstance(tc_s, dict):
            fcc_s = tc_s.get("function_calling_config")
            if isinstance(fcc_s, dict):
                mode = fcc_s.get("mode") or mode or "AUTO"
                allowed_names = fcc_s.get("allowed_function_names") or allowed_names

        tc_camel_clean = build_gemini_tool_config(mode, allowed_names)
        request_body["toolConfig"] = tc_camel_clean
        if "tool_config" in request_body:
            del request_body["tool_config"]

    api_version = provider.gemini_api_version or "v1beta"
    endpoint = ":streamGenerateContent?alt=sse" if is_stream else ":generateContent"
    generate_path = provider.upstream_generate_path or f"/{api_version}/models/{request_body.get('model', 'unknown')}{endpoint}"
    upstream_url = f"{provider.base_url}{generate_path}"

    body_bytes = json.dumps(request_body, ensure_ascii=False).encode("utf-8")

    await provider.request_pacer.wait()
    client = await get_async_client(request)
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    upstream_req = client.build_request(method="POST", url=upstream_url, headers=headers, content=body_bytes)
    logger.info("[%s] Gemini passthrough: %s (key: %s)", provider.provider_id, upstream_url, mask_key(api_key))

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectError as e:
        raise HTTPException(503, "Unable to connect to upstream") from e
    except httpx.TimeoutException as e:
        raise HTTPException(504, "Upstream timeout") from e

    if upstream_resp.status_code >= 400:
        error_body = await upstream_resp.aread()
        await upstream_resp.aclose()
        msg = error_body.decode("utf-8", errors="replace").strip() or f"Upstream error ({upstream_resp.status_code})"
        return JSONResponse(status_code=upstream_resp.status_code, content={"error": {"message": msg, "code": upstream_resp.status_code}})

    if is_stream:
        async def passthrough_sse():
            try:
                async for line in upstream_resp.aiter_lines():
                    yield f"{line}\n".encode("utf-8")
            finally:
                await upstream_resp.aclose()
        return StreamingResponse(passthrough_sse(), status_code=200, media_type="text/event-stream")

    resp_body = await upstream_resp.aread()
    await upstream_resp.aclose()
    return Response(content=resp_body, status_code=upstream_resp.status_code, media_type="application/json")


async def _proxy_gemini_to_openai_upstream(
    request: Request,
    provider: ProviderRuntime,
    request_body: dict[str, Any],
    api_key: str,
    is_stream: bool,
) -> Response:
    """Proxy a Gemini client request to an OpenAI upstream."""
    try:
        openai_body, model_name = translate_gemini_to_openai(
            request_body,
            model_prefix=provider.model_prefix,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    openai_payload = json.dumps(openai_body, ensure_ascii=False).encode("utf-8")
    chat_path = provider.upstream_chat_path or "/v1/chat/completions"
    upstream_url = f"{provider.base_url}{chat_path}"

    await provider.request_pacer.wait()
    client = await get_async_client(request)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP_REQUEST_HEADERS}
    headers["Authorization"] = f"Bearer {api_key}"
    headers["Content-Type"] = "application/json"

    upstream_req = client.build_request(method="POST", url=upstream_url, headers=headers, content=openai_payload)

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectError as e:
        raise HTTPException(503, "Unable to connect to upstream") from e
    except httpx.TimeoutException as e:
        raise HTTPException(504, "Upstream timeout") from e

    if upstream_resp.status_code >= 400:
        error_body = await upstream_resp.aread()
        await upstream_resp.aclose()
        msg = error_body.decode("utf-8", errors="replace").strip() or f"Upstream error ({upstream_resp.status_code})"
        return JSONResponse(status_code=upstream_resp.status_code, content={"error": {"message": msg, "code": upstream_resp.status_code}})

    if is_stream:
        translator = OpenAIToGeminiSSETranslator(model_name)

        async def gemini_sse_from_openai():
            data_lines: list[str] = []
            try:
                async for line in upstream_resp.aiter_lines():
                    line = line.rstrip("\r")
                    if not line:
                        if data_lines:
                            data_str = "\n".join(data_lines)
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                gemini_obj = translator.translate_chunk(chunk)
                                if gemini_obj:
                                    yield f"data: {json.dumps(gemini_obj, ensure_ascii=False)}\n\n".encode("utf-8")
                            except (json.JSONDecodeError, TypeError):
                                pass
                            data_lines = []
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
            except Exception:
                logger.exception("[%s] Gemini→OpenAI upstream SSE error", provider.provider_id)
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(gemini_sse_from_openai(), status_code=200, media_type="text/event-stream")

    # Non-streaming: read OpenAI response, convert to Gemini
    data_lines = []
    openai_resp_data = None
    try:
        async for line in upstream_resp.aiter_lines():
            line = line.rstrip("\r")
            if not line:
                if data_lines:
                    try:
                        openai_resp_data = json.loads("\n".join(data_lines))
                    except (json.JSONDecodeError, TypeError):
                        pass
                    data_lines = []
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
    except Exception:
        logger.exception("[%s] Error reading OpenAI response for Gemini translation", provider.provider_id)
    finally:
        await upstream_resp.aclose()

    if openai_resp_data:
        translator = OpenAIToGeminiSSETranslator(model_name)
        translator.translate_chunk(openai_resp_data)
        gemini_resp = translator.build_final_response()
        return JSONResponse(content=gemini_resp)

    return JSONResponse(status_code=502, content={"error": {"message": "No response from upstream", "code": 502}})


# ══════════════════════════════════════════════════════════════════════════════
# Anthropic client → Gemini upstream (added to handle_anthropic_messages)
# ══════════════════════════════════════════════════════════════════════════════


# The Gemini upstream branch is added inside handle_anthropic_messages() above.
# When provider.upstream_format == "gemini", it calls proxy_anthropic_to_gemini().


async def _proxy_gemini_to_anthropic_upstream(
    request: Request,
    provider: ProviderRuntime,
    request_body: dict[str, Any],
    api_key: str,
    is_stream: bool,
) -> Response:
    """Proxy a Gemini client request to an Anthropic upstream."""
    try:
        anthropic_body, model_name = translate_gemini_to_anthropic(
            request_body,
            model_prefix=provider.model_prefix or provider.provider_id + "/",
            default_max_tokens=provider.default_max_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    anthropic_payload = json.dumps(anthropic_body, ensure_ascii=False).encode("utf-8")
    messages_path = provider.upstream_messages_path or "/v1/messages"
    upstream_url = f"{provider.base_url}{messages_path}"
    if "?" not in messages_path:
        upstream_url += "?beta=true"

    headers = build_anthropic_headers(
        api_key, stream=True,
        fingerprint=provider.fingerprint or None,
        custom_headers=provider.custom_headers or None,
        session_id=_get_or_create_session_id(
            provider.provider_id, request_body,
            client_hint=request.headers.get("X-Claude-Code-Session-Id"),
        ),
    )

    await provider.request_pacer.wait()
    client = await get_async_client(request)
    upstream_req = client.build_request(method="POST", url=upstream_url, headers=headers, content=anthropic_payload)

    logger.info("[%s] Gemini→Anthropic: %s (model: %s, key: %s)", provider.provider_id, upstream_url, model_name, mask_key(api_key))

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.ConnectError as e:
        raise HTTPException(503, "Unable to connect to upstream") from e
    except httpx.TimeoutException as e:
        raise HTTPException(504, "Upstream timeout") from e

    if upstream_resp.status_code >= 400:
        error_body = await upstream_resp.aread()
        await upstream_resp.aclose()
        msg = error_body.decode("utf-8", errors="replace").strip() or f"Upstream error ({upstream_resp.status_code})"
        return JSONResponse(status_code=upstream_resp.status_code, content={"error": {"message": msg, "code": upstream_resp.status_code}})

    if not is_stream:
        # Aggregate Anthropic SSE → single Gemini response
        from protocol_adapter import AnthropicMessageAggregator
        aggregator = AnthropicMessageAggregator()
        data_lines: list[str] = []
        try:
            async for line in upstream_resp.aiter_lines():
                line = line.rstrip("\r")
                if not line:
                    aggregator.process_sse_lines(data_lines)
                    data_lines = []
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        except Exception:
            logger.error("[%s] Error reading anthropic stream for Gemini translation", provider.provider_id)
        finally:
            await upstream_resp.aclose()

        anthropic_resp = aggregator.build_response()
        gemini_resp = _anthropic_response_to_gemini(anthropic_resp, model_name)
        return JSONResponse(content=gemini_resp)

    # Streaming: translate Anthropic SSE → Gemini SSE
    translator = AnthropicToGeminiSSETranslator(model_name)

    async def translated_sse():
        event_name = ""
        data_lines: list[str] = []
        try:
            async for line in upstream_resp.aiter_lines():
                line = line.rstrip("\r")
                if not line:
                    if data_lines:
                        data_str = "\n".join(data_lines)
                        try:
                            data = json.loads(data_str)
                        except (json.JSONDecodeError, TypeError):
                            data = {}
                        event_name_for_translate = event_name
                        gemini_obj = translator.translate_event(event_name_for_translate, data)
                        if gemini_obj:
                            yield f"data: {json.dumps(gemini_obj, ensure_ascii=False)}\n\n".encode("utf-8")
                        data_lines = []
                    event_name = ""
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        except Exception:
            logger.exception("[%s] Gemini→Anthropic upstream SSE error", provider.provider_id)
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(translated_sse(), status_code=200, media_type="text/event-stream")


def _anthropic_response_to_gemini(anthropic_resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert an Anthropic Messages response to a Gemini response object."""
    parts: list[dict[str, Any]] = []
    for block in anthropic_resp.get("content", []):
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            parts.append({"text": block.get("text", "")})
        elif btype == "thinking":
            parts.append({"thought": True, "text": block.get("thinking", "")})
        elif btype == "tool_use":
            parts.append({"functionCall": {"name": block.get("name", ""), "args": block.get("input", {}), "id": block.get("id")}})

    stop_reason = anthropic_resp.get("stop_reason", "end_turn")
    stop_map = {"end_turn": "STOP", "max_tokens": "MAX_TOKENS", "tool_use": "STOP"}
    usage = anthropic_resp.get("usage", {})

    return {
        "candidates": [{
            "content": {"role": "model", "parts": parts or [{"text": ""}]},
            "finishReason": stop_map.get(stop_reason, "STOP"),
            "index": 0,
        }],
        "usageMetadata": {
            "promptTokenCount": usage.get("input_tokens", 0),
            "candidatesTokenCount": usage.get("output_tokens", 0),
            "totalTokenCount": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
        "modelVersion": model,
    }

