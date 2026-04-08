#!/usr/bin/env python3
"""
API routes for the OpenRouter upstream proxy.
"""

import json
import copy
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Request, Header, HTTPException, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, Response

from config import config, logger
from constants import MODELS_ENDPOINTS
from key_manager import KeyManager, mask_key
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


PUBLIC_ENDPOINTS = tuple(normalize_proxy_path(endpoint) for endpoint in config["openrouter"]["public_endpoints"])
ALLOWED_MODELS = tuple(config["openrouter"].get("allowed_models", []))

# Initialize key manager
key_manager = KeyManager(
    keys=config["openrouter"]["keys"],
    cooldown_seconds=config["openrouter"]["rate_limit_cooldown"],
    strategy=config["openrouter"]["key_selection_strategy"],
    opts=config["openrouter"]["key_selection_opts"],
)


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


async def check_httpx_err(body: str | bytes, api_key: Optional[str]):
    # too big or small for error
    if len(body) < 10 or len(body) > 4000 or not api_key:
        return
    has_rate_limit_error, reset_time_ms = await check_rate_limit(body)
    if has_rate_limit_error:
        await key_manager.disable_key(api_key, reset_time_ms)

def prepare_forward_headers(request: Request) -> dict:
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP_REQUEST_HEADERS
    }


def _invalid_model_response(model_name: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "message": f"Model '{model_name}' is not allowed by this proxy",
            "code": "invalid_request_error",
            "param": "model",
        },
    )


def enforce_allowed_models(request_body: dict[str, Any]) -> dict[str, Any]:
    """Restrict client-selected models to the configured allowlist."""
    if not ALLOWED_MODELS:
        return request_body

    body = copy.deepcopy(request_body)
    requested_model = body.get("model")
    if requested_model is not None:
        if not isinstance(requested_model, str) or not requested_model.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid model value",
                    "code": "invalid_request_error",
                    "param": "model",
                },
            )
        if requested_model not in ALLOWED_MODELS:
            raise _invalid_model_response(requested_model)
        return body

    requested_models = body.get("models")
    if isinstance(requested_models, list) and requested_models:
        invalid_models = [model_name for model_name in requested_models if not isinstance(model_name, str) or model_name not in ALLOWED_MODELS]
        if invalid_models:
            raise _invalid_model_response(str(invalid_models[0]))
        return body

    body["model"] = ALLOWED_MODELS[0]
    return body


def filter_models_response(body: bytes) -> bytes:
    """Filter /models responses by free-only and/or model allowlist configuration."""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Error models deserialize: %s", str(e))
        return body

    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        return body

    filtered_data = []
    allowed_model_set = set(ALLOWED_MODELS)
    prices = ["prompt", "completion", "request", "image", "web_search", "internal_reasoning"]

    for model in data["data"]:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id") or model.get("name")
        if allowed_model_set and model_id not in allowed_model_set:
            continue
        if config["openrouter"]["free_only"] and not all(model.get("pricing", {}).get(k, "1") == "0" for k in prices):
            continue
        filtered_data.append(model)

    data["data"] = filtered_data
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def is_responses_path(normalized_path: str) -> bool:
    return normalized_path == "/responses" or normalized_path.startswith("/responses/")


async def send_openrouter_request(
    request: Request,
    upstream_path: str,
    api_key: str,
    *,
    is_stream: bool,
    content: Optional[bytes] = None,
    headers_override: Optional[dict[str, str]] = None,
    params: Optional[dict[str, Any]] = None,
) -> httpx.Response:
    req_kwargs = {
        "method": request.method,
        "url": f"{config['openrouter']['base_url']}{upstream_path}",
        "headers": prepare_forward_headers(request),
        "content": await request.body() if content is None else content,
        "params": request.query_params if params is None else params,
    }
    if headers_override:
        req_kwargs["headers"].update(headers_override)
    if api_key:
        req_kwargs["headers"]["Authorization"] = f"Bearer {api_key}"

    client = await get_async_client(request)
    openrouter_req = client.build_request(**req_kwargs)
    return await client.send(openrouter_req, stream=is_stream)


@router.api_route("/api/v1", methods=SUPPORTED_PROXY_METHODS, include_in_schema=False)
@router.api_route("/api/v1{path:path}", methods=SUPPORTED_PROXY_METHODS, include_in_schema=False)
@router.api_route("/v1", methods=SUPPORTED_PROXY_METHODS, include_in_schema=False)
@router.api_route("/v1{path:path}", methods=SUPPORTED_PROXY_METHODS, include_in_schema=False)
async def proxy_endpoint(
    request: Request, path: str, authorization: Optional[str] = Header(None)
):
    """Main proxy endpoint for handling both OpenAI and OpenRouter client paths."""
    try:
        request_path = request.url.path
        client_prefix = get_client_prefix(request_path)
        if client_prefix is None:
            raise HTTPException(status_code=404, detail="Unsupported API prefix")

        normalized_path = normalize_proxy_path(request_path)
        is_public = any(path_matches_prefix(normalized_path, endpoint) for endpoint in PUBLIC_ENDPOINTS)

        # Verify authorization for non-public endpoints
        if not is_public:
            await verify_access_key(authorization=authorization)

        # Log the full request URL including query parameters
        full_url = str(request.url).replace(str(request.base_url), "/")

        # Get API key to use
        api_key = "" if is_public else await key_manager.get_next_key()

        logger.info("Proxying request to %s (Public: %s, key: %s)", full_url, is_public, mask_key(api_key))

        is_stream = False
        body_bytes: Optional[bytes] = None
        if request.method in {"POST", "PUT", "PATCH"}:
            try:
                if body_bytes := await request.body():
                    request_body = json.loads(body_bytes)
                    if isinstance(request_body, dict):
                        request_body = enforce_allowed_models(request_body)
                        body_bytes = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
                        if is_stream := bool(request_body.get("stream", False)):
                            logger.info("Detected streaming request")
                        if model := request_body.get("model"):
                            logger.info("Using model: %s", model)
            except Exception as e:
                logger.debug("Could not parse request body: %s", str(e))

        if is_responses_path(normalized_path):
            return await handle_responses_endpoint(request, normalized_path, api_key, body_bytes, is_stream)

        upstream_path = translate_request_path(request_path, client_prefix)
        return await proxy_with_httpx(request, upstream_path, normalized_path, api_key, is_stream, content_bytes=body_bytes)
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
    normalized_path: str,
    api_key: str,
    content_bytes: Optional[bytes],
    is_stream: bool,
) -> Response:
    """Proxy OpenAI Responses API requests directly to OpenRouter Responses endpoints."""
    return await proxy_with_httpx(
        request,
        normalized_path,
        normalized_path,
        api_key,
        is_stream,
        content_bytes=content_bytes,
    )


async def proxy_with_httpx(
    request: Request,
    upstream_path: str,
    normalized_path: str,
    api_key: str,
    is_stream: bool,
    content_bytes: Optional[bytes] = None,
) -> Response:
    """Core logic to proxy requests."""
    try:
        openrouter_resp = await send_openrouter_request(
            request,
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
            await check_httpx_err(openrouter_resp.content, api_key)
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
            await check_httpx_err(body, api_key)
            if normalized_path in MODELS_ENDPOINTS:
                filtered_body = filter_models_response(body)
                if filtered_body != body:
                    body = filtered_body
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
                logger.error("sse_stream error: %s", err)
            finally:
                await openrouter_resp.aclose()
            await check_httpx_err(last_json, api_key)


        return StreamingResponse(
            sse_stream(),
            status_code=openrouter_resp.status_code,
            media_type=content_type,
            headers=headers,
        )
    except httpx.ConnectError as e:
        logger.error("Connection error to OpenRouter: %s", str(e))
        raise HTTPException(503, "Unable to connect to OpenRouter API") from e
    except httpx.TimeoutException as e:
        logger.error("Timeout connecting to OpenRouter: %s", str(e))
        raise HTTPException(504, "OpenRouter API request timed out") from e
    except Exception as e:
        logger.error("Internal error: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal Proxy Error") from e


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
