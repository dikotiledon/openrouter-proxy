#!/usr/bin/env python3
"""
API routes for the upstream provider proxy.
"""

import asyncio
import json
import copy
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Request, Header, HTTPException, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, Response

from config import config, logger
from constants import MODELS_ENDPOINTS
from key_manager import mask_key
from protocol_adapter import (
    AnthropicStreamDecoder,
    AnthropicMessageAggregator,
    OpenAIToAnthropicSSETranslator,
    build_anthropic_headers,
    translate_openai_to_anthropic,
    translate_anthropic_to_openai,
    translate_request,
    sse_lines_to_openai_chunk,
)
from provider_registry import (
    ProviderRuntime,
    aggregate_model_payloads,
    build_provider_registry,
    normalize_models_payload_for_client,
    normalize_request_body_for_provider,
    resolve_provider_hint,
)
from utils import verify_access_key, check_rate_limit

# Create router
router = APIRouter()

CLIENT_API_PREFIXES = ("/api/v1", "/v1")
SUPPORTED_PROXY_METHODS = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
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


def path_matches_prefix(path: str, prefix: str) -> bool:
    """Match an endpoint prefix without accidentally matching sibling paths."""
    if prefix == "/":
        return True
    return path == prefix or path.startswith(prefix + "/")


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


PROVIDER_REGISTRY = build_provider_registry(config)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    client_kwargs = {"timeout": 600.0}  # Increase default timeout
    # Add proxy configuration if enabled
    if config["requestProxy"]["enabled"]:
        proxy_url = config["requestProxy"]["url"]
        client_kwargs["proxy"] = proxy_url
        logger.info("Using proxy for httpx client: %s", proxy_url)
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
                    parsed_body = json.loads(body_bytes)
                    if isinstance(parsed_body, dict):
                        request_body = parsed_body
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

        # Anthropic-mode providers (AgentRouter) need full request translation
        if provider.upstream_format == "anthropic" and normalized_path == "/chat/completions":
            return await proxy_anthropic_with_httpx(
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
    """Proxy OpenAI Responses API requests directly to the selected provider."""
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

        if openrouter_resp.status_code >= 400:
            if is_stream:
                try:
                    await openrouter_resp.aread()
                except Exception as e:
                    await openrouter_resp.aclose()
                    raise e
            await check_httpx_err(openrouter_resp.content, provider, api_key)
            message, code, param = extract_error_details(openrouter_resp.content, openrouter_resp.status_code)
            headers, _ = split_response_headers(openrouter_resp.headers, "application/json")
            await openrouter_resp.aclose()
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
            await openrouter_resp.aclose()
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
                    if line.startswith("data: {"):  # get json only
                        last_json = line[6:]
                    yield f"{line}\n\n".encode("utf-8")
            except Exception as err:
                logger.exception("sse_stream error")
                raise
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
    except httpx.ConnectError as e:
        logger.error("Connection error to %s: %s", provider.provider_id, str(e))
        raise HTTPException(503, "Unable to connect to the upstream provider") from e
    except httpx.TimeoutException as e:
        logger.error("Timeout connecting to %s: %s", provider.provider_id, str(e))
        raise HTTPException(504, "Upstream provider request timed out") from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Internal error: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Proxy Error") from e


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

    # Determine upstream URL
    messages_path = provider.upstream_messages_path or "/v1/messages?beta=true"
    upstream_url = f"{provider.base_url}{messages_path}"

    # Build Claude Code headers
    headers = build_anthropic_headers(
        api_key,
        stream=True,
        fingerprint=provider.fingerprint or None,
        custom_headers=provider.custom_headers or None,
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
        return JSONResponse(content=result, status_code=200)

    # Streaming: translate Anthropic SSE → OpenAI SSE chunks
    async def translated_sse():
        decoder = AnthropicStreamDecoder(model_name)
        event_name = ""
        data_lines: list[str] = []
        try:
            async for line in upstream_resp.aiter_lines():
                line = line.rstrip("\r")
                if not line:
                    # Process accumulated event
                    chunk = decoder.process_sse_lines(data_lines)
                    if chunk is not None:
                        output = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        yield output.encode("utf-8")
                    event_name = ""
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())

            # Process any remaining data
            if data_lines:
                chunk = decoder.process_sse_lines(data_lines)
                if chunk is not None:
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
        except Exception as e:
            logger.exception("[%s] SSE translation error", provider.provider_id)
        finally:
            yield b"data: [DONE]\n\n"
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
        return build_openai_error_response(400, "Invalid JSON body")

    model = request_body.get("model", "")
    if not model:
        return build_openai_error_response(400, "model is required", param="model")

    # Resolve provider from model prefix
    provider_name = model.split("/")[0] if "/" in model else None
    if not provider_name:
        return build_openai_error_response(
            400, "Model must be provider-qualified as <provider>/<model>", param="model",
        )

    provider = get_provider_by_hint(provider_name)
    if provider is None:
        return build_openai_error_response(400, f"Unknown provider '{provider_name}'", param="model")

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
            return build_openai_error_response(400, str(e))

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
            return build_openai_error_response(upstream_resp.status_code, msg)

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
                except Exception:
                    logger.exception("[%s] OpenAI→Anthropic SSE error", provider.provider_id)
                finally:
                    if not translator._finished:
                        finish_events = translator.finish_message()
                        if finish_events:
                            yield finish_events.encode("utf-8")
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

        return build_openai_error_response(502, "No response from upstream")

    # ── Upstream is Anthropic format → passthrough with correct headers
    else:
        if provider.model_prefix and request_body.get("model", "").startswith(provider.model_prefix):
            request_body["model"] = request_body["model"][len(provider.model_prefix):]

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
            return build_openai_error_response(upstream_resp.status_code, msg)

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
            return build_openai_error_response(400, "Invalid JSON body")

        model = request_body.get("model", "")
        if not model:
            return build_openai_error_response(400, "model is required", param="model")

        # Resolve provider from model prefix
        provider_name = model.split("/")[0] if "/" in model else None
        if not provider_name:
            return build_openai_error_response(
                400, "Model must be provider-qualified as <provider>/<model>", param="model",
            )

        provider = get_provider_by_hint(provider_name)
        if provider is None:
            return build_openai_error_response(400, f"Unknown provider '{provider_name}'", param="model")

        await verify_client_access(provider, "/messages", authorization)
        api_key = await provider.key_manager.get_next_key()

        # ── Upstream is OpenAI format ───────────────────────────────────
        if provider.upstream_format == "openai":
            # Translate Anthropic → OpenAI
            try:
                openai_body, model_name = translate_anthropic_to_openai(
                    request_body,
                    model_prefix=provider.model_prefix,
                )
            except ValueError as e:
                return build_openai_error_response(400, str(e))

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
                return build_openai_error_response(upstream_resp.status_code, msg)

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
                    except Exception:
                        logger.exception("[%s] OpenAI→Anthropic SSE error", provider.provider_id)
                    finally:
                        # Ensure message is properly finished
                        if not translator._finished:
                            finish_events = translator.finish_message()
                            if finish_events:
                                yield finish_events.encode("utf-8")
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

            return build_openai_error_response(502, "No response from upstream")

        # ── Upstream is Anthropic format ────────────────────────────────
        else:
            # Passthrough — client format matches upstream format
            # Strip model prefix before sending upstream
            if provider.model_prefix and request_body.get("model", "").startswith(provider.model_prefix):
                request_body["model"] = request_body["model"][len(provider.model_prefix):]
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
                return build_openai_error_response(upstream_resp.status_code, msg)

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
        return build_openai_error_response(exc.status_code, message)
    except Exception:
        logger.exception("Anthropic endpoint error")
        return build_openai_error_response(500, "Internal Proxy Error")
