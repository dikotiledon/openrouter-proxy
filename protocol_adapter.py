#!/usr/bin/env python3
"""
Bidirectional OpenAI ↔ Anthropic protocol adapter.

Supports the full translation matrix:
  - OpenAI → Anthropic: translate_openai_to_anthropic() + AnthropicStreamDecoder
  - Anthropic → OpenAI: translate_anthropic_to_openai() + OpenAIToAnthropicSSETranslator
  - Both directions support streaming and non-streaming.
  - Header spoofing for Claude Code client fingerprinting (configurable via YAML).

Reference: https://github.com/lutfi238/proxy-agentrouter
"""

import hashlib
import json
import time
import uuid
import random
from collections import deque

from typing import Any, Optional

from constants import REASONING_EFFORT_TO_BUDGET, DEFAULT_REASONING_BUDGET


# ── Claude Code client fingerprint defaults ───────────────────────────────────

CC_VERSION = "2.1.145"
BILLING_SALT = "59cf53e54c78"

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_BETA = (
    "claude-code-20250219,interleaved-thinking-2025-05-14,"
    "context-management-2025-06-27,prompt-caching-scope-2026-01-05,"
    "advisor-tool-2026-03-01,effort-2025-11-24"
)
DEFAULT_USER_AGENT = "claude-cli/2.1.145 (external, sdk-cli)"
DEFAULT_BILLING_HEADER = (
    "x-anthropic-billing-header: cc_version=2.1.145.560; "
    "cc_entrypoint=sdk-cli; cch=00000;"
)
DEFAULT_MAX_TOKENS = 4096


def _sha256(s: str) -> str:
    """Return the full SHA-256 hex digest of a string."""
    return hashlib.sha256(s.encode()).hexdigest()


def compute_billing_header(
    message_text: str,
    *,
    cc_version: str = CC_VERSION,
    billing_salt: str = BILLING_SALT,
    entrypoint: str = "sdk-cli",
) -> str:
    """Compute a per-request billing header based on the first user message.

    Algorithm (from Claude Code source):
    1. cch = SHA-256(message_text)[:5]
    2. sampled = message[4] + message[7] + message[20], pad with "0" if short
    3. version_hash = SHA-256(salt + sampled + version)[:3]
    4. Format: "x-anthropic-billing-header: cc_version=V.hash; cc_entrypoint=E; cch=H;"
    """
    # Step 1: content hash
    cch = _sha256(message_text)[:5]

    # Step 2: sample characters
    sampled = "".join(
        message_text[i] if i < len(message_text) else "0"
        for i in (4, 7, 20)
    )

    # Step 3: version integrity hash
    version_hash = _sha256(f"{billing_salt}{sampled}{cc_version}")[:3]

    # Step 4: format
    return (
        f"x-anthropic-billing-header: "
        f"cc_version={cc_version}.{version_hash}; "
        f"cc_entrypoint={entrypoint}; "
        f"cch={cch};"
    )

# All overridable fingerprint fields with their defaults
FINGERPRINT_DEFAULTS: dict[str, Any] = {
    "anthropic_version": DEFAULT_ANTHROPIC_VERSION,
    "anthropic_beta": DEFAULT_ANTHROPIC_BETA,
    "user_agent": DEFAULT_USER_AGENT,
    "anthropic_browser_access": "true",
    "x_app": "cli",
    "stainless_arch": "x64",
    "stainless_lang": "js",
    "stainless_os": "Windows",
    "stainless_package_version": "0.93.0",
    "stainless_runtime": "node",
    "stainless_runtime_version": "v24.3.0",
    "stainless_timeout": "300",
    "billing_header": DEFAULT_BILLING_HEADER,
}


def _random_hex(n: int = 16) -> str:
    """Return a random hex string of *n* random bytes."""
    return uuid.uuid4().hex[: n * 2]


def _random_uuid_like() -> str:
    h = _random_hex(16)
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ── Header helpers ────────────────────────────────────────────────────────────


def build_anthropic_headers(
    api_key: str,
    *,
    stream: bool = False,
    fingerprint: Optional[dict[str, Any]] = None,
    custom_headers: Optional[dict[str, str]] = None,
    session_id: Optional[str] = None,
) -> dict[str, str]:
    """Return the full set of Claude-Code-shaped upstream headers.

    Args:
        api_key: The API key for x-api-key header.
        stream: Whether to set Accept to text/event-stream.
        fingerprint: Override any fingerprint field (see FINGERPRINT_DEFAULTS).
        custom_headers: Extra headers merged on top (take precedence).
        session_id: X-Claude-Code-Session-Id value. If None, random UUID generated.
    """
    fp = {**FINGERPRINT_DEFAULTS, **(fingerprint or {})}

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": api_key,
        "anthropic-version": str(fp["anthropic_version"]),
        "anthropic-beta": str(fp["anthropic_beta"]),
        "anthropic-dangerous-direct-browser-access": str(fp["anthropic_browser_access"]),
        "x-app": str(fp["x_app"]),
        "User-Agent": str(fp["user_agent"]),
        "X-Claude-Code-Session-Id": session_id or _random_uuid_like(),
        "X-Stainless-Arch": str(fp["stainless_arch"]),
        "X-Stainless-Lang": str(fp["stainless_lang"]),
        "X-Stainless-OS": str(fp["stainless_os"]),
        "X-Stainless-Package-Version": str(fp["stainless_package_version"]),
        "X-Stainless-Retry-Count": "0",
        "X-Stainless-Runtime": str(fp["stainless_runtime"]),
        "X-Stainless-Runtime-Version": str(fp["stainless_runtime_version"]),
        "X-Stainless-Timeout": str(fp["stainless_timeout"]),
    }
    if stream:
        headers["Accept"] = "text/event-stream"
    # Custom headers take precedence over everything
    if custom_headers:
        _BLOCKED_HEADERS = {"host", "content-length", "authorization", "connection", "proxy-authorization"}
        for k, v in custom_headers.items():
            if k.lower() not in _BLOCKED_HEADERS:
                headers[k] = v
    return headers


# ── OpenAI → Anthropic request translation ───────────────────────────────────


def _content_to_anthropic_blocks(content: Any) -> list[dict[str, Any]]:
    """Convert an OpenAI message content field to Anthropic content blocks."""
    if content is None:
        return [{"type": "text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                blocks.append({"type": "text", "text": str(part)})
                continue
            ptype = part.get("type", "")
            if ptype in ("text", "input_text", "output_text", "summary_text"):
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                url = part.get("image_url", {})
                if isinstance(url, dict):
                    url = url.get("url", "")
                blocks.append({
                    "type": "image",
                    "source": _parse_image_url(url),
                })
            elif ptype == "input_image":
                url = part.get("image_url", "")
                blocks.append({
                    "type": "image",
                    "source": _parse_image_url(url),
                })
            else:
                blocks.append({"type": "text", "text": json.dumps(part)})
        return blocks if blocks else [{"type": "text", "text": ""}]
    return [{"type": "text", "text": str(content)}]


def _parse_image_url(url: str) -> dict[str, Any]:
    """Parse a data URL or wrap a remote URL for Anthropic."""
    if url.startswith("data:") and ";base64," in url:
        meta, data = url.split(";base64,", 1)
        media_type = meta.split(":", 1)[1] if ":" in meta else "application/octet-stream"
        return {"type": "base64", "media_type": media_type, "data": data}
    return {"type": "url", "url": url}


def _tool_calls_to_anthropic_blocks(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool_calls to Anthropic tool_use blocks."""
    blocks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for tc in tool_calls or []:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            input_data = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            input_data = {}
        call_id = tc.get("id", f"toolu_{_random_hex(12)}")
        if call_id in seen_ids:
            call_id = f"toolu_{_random_hex(12)}"
        seen_ids.add(call_id)
        blocks.append({
            "type": "tool_use",
            "id": call_id,
            "name": name,
            "input": input_data if isinstance(input_data, dict) else {},
        })
    return blocks


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool definitions to Anthropic tool definitions."""
    result: list[dict[str, Any]] = []
    for tool in tools or []:
        if tool.get("type", "function") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        if not name:
            continue
        params = func.get("parameters")
        if params is None:
            params = {"type": "object", "properties": {}}
        result.append({
            "name": name,
            "description": func.get("description", ""),
            "input_schema": params,
        })
    return result


def _tool_choice_to_anthropic(choice: Any) -> Optional[dict[str, Any]]:
    """Convert OpenAI tool_choice to Anthropic format."""
    if choice is None:
        return None
    if isinstance(choice, str):
        if choice == "none":
            return {"type": "none"}
        if choice == "required":
            return {"type": "any"}
        if choice in ("auto", "any"):
            return {"type": choice}
        return None
    if isinstance(choice, dict):
        func = choice.get("function", {})
        if isinstance(func, dict) and func.get("name"):
            return {"type": "tool", "name": func["name"]}
        if choice.get("name"):
            return {"type": "tool", "name": choice["name"]}
    return None


def _normalize_stop(stop: Any) -> list[str]:
    """Normalize OpenAI stop parameter to a list of strings."""
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop] if stop else []
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str) and s]
    return []


def _has_billing_header(blocks: list[dict[str, Any]]) -> bool:
    """Check if billing header is already injected into system blocks."""
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            if "x-anthropic-billing-header:" in block.get("text", "").lower():
                return True
    return False


def deduplicate_anthropic_tool_use_ids(body: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate tool_use IDs in an Anthropic Messages request body.

    Some upstreams (Kiro/Bedrock) emit phantom tool_use blocks with duplicate IDs.
    When clients store these in conversation history and send them back, Bedrock
    rejects the request with TOOL_DUPLICATE. This function re-assigns unique IDs
    to duplicate tool_use blocks and patches matching tool_result references.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body

    seen_ids: dict[str, int] = {}  # id → occurrence count
    id_map: dict[str, str] = {}    # old_id → new_id (for duplicates)

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tid = block.get("id", "")
                if not tid:
                    continue
                if tid in seen_ids:
                    seen_ids[tid] += 1
                    new_id = f"toolu_{_random_hex(12)}"
                    id_map[f"{tid}:{seen_ids[tid]}"] = new_id
                    block["id"] = new_id
                else:
                    seen_ids[tid] = 1

    if not id_map:
        return body

    # Patch tool_result blocks that reference remapped tool_use IDs
    # We need to track which occurrence of each original ID each tool_result matches
    occurrence_counters: dict[str, int] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                tid = block.get("tool_use_id", "")
                if not tid:
                    continue
                if tid in seen_ids and seen_ids[tid] > 1:
                    occurrence_counters[tid] = occurrence_counters.get(tid, 0) + 1
                    map_key = f"{tid}:{occurrence_counters[tid]}"
                    if map_key in id_map:
                        block["tool_use_id"] = id_map[map_key]

    return body


def deduplicate_kiro_conversation_state(body: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate toolUseId values in a Kiro conversationState payload.

    Kiro IDE sends requests in its own ``conversationState`` format (not OpenAI
    ``messages``).  When the upstream Bedrock-based model emits duplicate tool_use
    IDs, Kiro stores them verbatim.  On the next turn the proxy must deduplicate
    these IDs before forwarding, otherwise Bedrock rejects with ``TOOL_DUPLICATE``.

    Handles two paths:
      - ``conversationState.history[].assistantResponseMessage.toolUses[].toolUseId``
      - ``conversationState.currentMessage.userInputMessage.toolResults[].toolUseId``
    """
    cs = body.get("conversationState")
    if not isinstance(cs, dict):
        return body

    seen_ids: dict[str, int] = {}
    id_map: dict[str, str] = {}

    # ── Scan toolUses in history ──────────────────────────────────────────
    for entry in cs.get("history", []):
        if not isinstance(entry, dict):
            continue
        arm = entry.get("assistantResponseMessage")
        if not isinstance(arm, dict):
            continue
        tool_uses = arm.get("toolUses")
        if not isinstance(tool_uses, list):
            continue
        for tu in tool_uses:
            if not isinstance(tu, dict):
                continue
            tid = tu.get("toolUseId", "")
            if not tid:
                continue
            if tid in seen_ids:
                seen_ids[tid] += 1
                new_id = f"tooluse_{_random_hex(12)}"
                id_map[f"{tid}:{seen_ids[tid]}"] = new_id
                tu["toolUseId"] = new_id
            else:
                seen_ids[tid] = 1

    if not id_map:
        return body

    # ── Patch toolResults in currentMessage ───────────────────────────────
    # Kiro nests toolResults inside userInputMessageContext:
    #   currentMessage.userInputMessage.userInputMessageContext.toolResults
    cm = cs.get("currentMessage")
    if isinstance(cm, dict):
        uim = cm.get("userInputMessage")
        if isinstance(uim, dict):
            # Try both direct toolResults and nested under userInputMessageContext
            tool_results = uim.get("toolResults")
            if not isinstance(tool_results, list):
                ctx = uim.get("userInputMessageContext")
                if isinstance(ctx, dict):
                    tool_results = ctx.get("toolResults")
            if isinstance(tool_results, list):
                occurrence_counters: dict[str, int] = {}
                for tr in tool_results:
                    if not isinstance(tr, dict):
                        continue
                    tid = tr.get("toolUseId", "")
                    if not tid or tid not in seen_ids or seen_ids[tid] <= 1:
                        continue
                    occurrence_counters[tid] = occurrence_counters.get(tid, 0) + 1
                    map_key = f"{tid}:{occurrence_counters[tid]}"
                    if map_key in id_map:
                        tr["toolUseId"] = id_map[map_key]

    return body


def deduplicate_openai_tool_call_ids(body: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate tool_call IDs in an OpenAI Chat Completions request body.

    When upstreams emit duplicate tool_call IDs (e.g. Kiro/Bedrock phantom blocks),
    clients store them in history. This function re-assigns unique IDs to duplicate
    tool_calls and patches matching tool result references.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body

    seen_ids: dict[str, int] = {}
    id_map: dict[str, str] = {}

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tid = tc.get("id", "")
            if not tid:
                continue
            if tid in seen_ids:
                seen_ids[tid] += 1
                new_id = f"call_{_random_hex(12)}"
                id_map[f"{tid}:{seen_ids[tid]}"] = new_id
                tc["id"] = new_id
            else:
                seen_ids[tid] = 1

    if not id_map:
        return body

    occurrence_counters: dict[str, int] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        tid = msg.get("tool_call_id", "")
        if not tid or tid not in seen_ids or seen_ids[tid] <= 1:
            continue
        occurrence_counters[tid] = occurrence_counters.get(tid, 0) + 1
        map_key = f"{tid}:{occurrence_counters[tid]}"
        if map_key in id_map:
            msg["tool_call_id"] = id_map[map_key]

    return body


def translate_openai_to_anthropic(
    openai_body: dict[str, Any],
    *,
    model_prefix: str = "",
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
    inject_billing: bool = True,
    billing_header: str = "",
) -> tuple[dict[str, Any], str]:
    """Translate an OpenAI chat completion request to Anthropic Messages format.

    Returns (anthropic_body, upstream_model_name).
    """
    model = openai_body.get("model", "")
    # Strip provider prefix (e.g. "agentrouter/glm-5.1" → "glm-5.1")
    if model_prefix and model.startswith(model_prefix):
        model = model[len(model_prefix):]

    messages = openai_body.get("messages", [])
    if not messages:
        raise ValueError("messages array is required and must not be empty")

    system_blocks: list[dict[str, Any]] = []
    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "").lower().strip()
        content_blocks = _content_to_anthropic_blocks(msg.get("content"))
        tool_calls = msg.get("tool_calls")

        if role in ("system", "developer"):
            system_blocks.extend(content_blocks)
        elif role == "assistant":
            if tool_calls:
                content_blocks.extend(_tool_calls_to_anthropic_blocks(tool_calls))
            anthropic_messages = _append_or_merge(anthropic_messages, "assistant", content_blocks)
        elif role in ("tool", "function"):
            tool_call_id = msg.get("tool_call_id", "")
            text = _blocks_to_text(content_blocks)
            if tool_call_id:
                anthropic_messages = _append_or_merge(
                    anthropic_messages, "user",
                    [{"type": "tool_result", "tool_use_id": tool_call_id, "content": text}],
                )
            else:
                label = msg.get("name", role)
                anthropic_messages = _append_or_merge(
                    anthropic_messages, "user",
                    [{"type": "text", "text": f"{label} result:\n{text}"}],
                )
        else:
            # user and anything else
            anthropic_messages = _append_or_merge(anthropic_messages, "user", content_blocks)

    if not anthropic_messages:
        raise ValueError("At least one non-system message is required")

    # Inject billing header into system blocks if configured
    if inject_billing and not _has_billing_header(system_blocks):
        # Extract first user message text for the content hash
        first_user_text = ""
        for msg in anthropic_messages:
            if msg.get("role") == "user":
                blocks = msg.get("content", [])
                if isinstance(blocks, list):
                    for b in blocks:
                        if isinstance(b, dict) and b.get("type") == "text":
                            first_user_text = b.get("text", "")
                            break
                break

        if billing_header:
            # Use static billing header if explicitly provided
            system_blocks.insert(0, {"type": "text", "text": billing_header})
        else:
            # Compute dynamic billing header from user message
            header = compute_billing_header(first_user_text)
            system_blocks.insert(0, {"type": "text", "text": header})

    # Translate reasoning → Anthropic thinking
    reasoning = openai_body.get("reasoning")
    reasoning_effort = openai_body.get("reasoning_effort")
    enable_thinking = openai_body.get("enable_thinking")
    thinking: Optional[dict[str, Any]] = None
    if isinstance(reasoning, dict):
        exclude = reasoning.get("exclude")
        if reasoning.get("enabled") is False or exclude is True:
            thinking = {"type": "disabled"}
        else:
            effort = reasoning.get("effort")
            budget = reasoning.get("max_tokens")
            if budget is not None and isinstance(budget, (int, float)) and budget > 0:
                thinking = {"type": "enabled", "budget_tokens": int(budget)}
            elif isinstance(effort, str) and effort in REASONING_EFFORT_TO_BUDGET:
                thinking = {"type": "enabled", "budget_tokens": REASONING_EFFORT_TO_BUDGET[effort]}
            else:
                thinking = {"type": "enabled", "budget_tokens": DEFAULT_REASONING_BUDGET}
    elif isinstance(reasoning_effort, str) and reasoning_effort in REASONING_EFFORT_TO_BUDGET:
        thinking = {"type": "enabled", "budget_tokens": REASONING_EFFORT_TO_BUDGET[reasoning_effort]}
    elif enable_thinking is True:
        thinking = {"type": "enabled", "budget_tokens": DEFAULT_REASONING_BUDGET}

    # Max tokens (required by Anthropic)
    max_tokens = default_max_tokens
    if openai_body.get("max_tokens") and openai_body["max_tokens"] > 0:
        max_tokens = openai_body["max_tokens"]
    if openai_body.get("max_completion_tokens") and openai_body["max_completion_tokens"] > 0:
        max_tokens = openai_body["max_completion_tokens"]

    body: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
        "stream": True,  # Always stream from Anthropic side
    }

    if thinking is not None:
        body["thinking"] = thinking

    if system_blocks:
        body["system"] = system_blocks

    if openai_body.get("temperature") is not None:
        body["temperature"] = openai_body["temperature"]
    if openai_body.get("top_p") is not None:
        body["top_p"] = openai_body["top_p"]

    stop = _normalize_stop(openai_body.get("stop"))
    if stop:
        body["stop_sequences"] = stop

    tools = _openai_tools_to_anthropic(openai_body.get("tools"))
    if tools:
        body["tools"] = tools
        tc = _tool_choice_to_anthropic(openai_body.get("tool_choice"))
        if tc is not None:
            body["tool_choice"] = tc

    return body, model


# ── Anthropic SSE → OpenAI response translation ──────────────────────────────


def _blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    """Extract text from Anthropic content blocks."""
    parts: list[str] = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def _append_or_merge(
    messages: list[dict[str, Any]],
    role: str,
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append a message or merge with the last one if same role."""
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] = list(messages[-1]["content"]) + blocks
        return messages
    return messages + [{"role": role, "content": blocks}]


def _map_stop_reason(reason: str) -> str:
    """Map Anthropic stop_reason to OpenAI finish_reason."""
    mapping = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "stop": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
    return mapping.get(reason, reason if reason else "")


def _safe_int(v: Any) -> int:
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return 0
    return 0


def _build_openai_chunk(
    chunk_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: Optional[str] = None,
    usage: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a single OpenAI streaming chunk."""
    chunk: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _build_openai_response(
    response_id: str,
    created: int,
    model: str,
    content_text: str,
    tool_calls: Optional[list[dict[str, Any]]],
    finish_reason: str,
    usage: Optional[dict[str, Any]] = None,
    reasoning_details: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Build a complete OpenAI non-streaming response."""
    message: dict[str, Any] = {"role": "assistant"}
    if content_text:
        message["content"] = content_text
    else:
        message["content"] = None
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_details:
        message["reasoning_details"] = reasoning_details
    resp: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
    }
    if usage is not None:
        resp["usage"] = usage
    return resp


class AnthropicStreamDecoder:
    """Parses Anthropic SSE events and yields OpenAI-format chunks.

    Two modes:
      - streaming: call process_line() for each SSE line, yields chunks
      - non-streaming: call process_all() on complete body, returns full response
    """

    def __init__(self, model: str, *, request_id: Optional[str] = None):
        self.model = model
        self.chunk_id = request_id or f"chatcmpl-{_random_hex(12)}"
        self.created = int(time.time())

        # Streaming state
        self._content_parts: list[str] = []
        self._tool_calls_by_index: dict[int, dict[str, Any]] = {}
        self._tool_name_by_index: dict[int, str] = {}
        self._tool_args_by_index: dict[int, str] = {}
        self._reasoning_parts: list[str] = []
        self._reasoning_details: list[dict[str, Any]] = []
        self._block_type_by_index: dict[int, str] = {}
        self._stopped_blocks: set[int] = set()
        self._seen_tool_ids: set[str] = set()
        self._pending_tool_starts: dict[int, dict[str, Any]] = {}
        self._pending_chunks: deque[dict[str, Any]] = deque()
        self._finish_reason: str = "stop"
        self._usage: Optional[dict[str, Any]] = None

    def _reset_block(self, index: int):
        self._tool_name_by_index.pop(index, None)
        self._tool_args_by_index.pop(index, None)

    def drain_pending(self) -> list[dict[str, Any]]:
        """Return all buffered chunks from deferred tool_use starts."""
        chunks = list(self._pending_chunks)
        self._pending_chunks.clear()
        return chunks

    def _extract_usage(self, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Extract usage from an Anthropic message_delta/message_start event."""
        usage_raw = data.get("usage")
        if not isinstance(usage_raw, dict):
            return None
        prompt = _safe_int(usage_raw.get("input_tokens"))
        completion = _safe_int(usage_raw.get("output_tokens"))
        if prompt == 0 and completion == 0:
            return None
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    # ── Streaming: process one SSE line ───────────────────────────────────

    def process_sse_lines(self, lines: list[str]) -> Optional[dict[str, Any]]:
        """Process a complete SSE event (multiple data: lines). Returns an OpenAI chunk or None."""
        if not lines:
            return None
        data_str = "\n".join(lines)
        if data_str.strip() == "[DONE]":
            return None

        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, TypeError):
            return None

        event_type = data.get("type", "")

        if event_type == "message_start":
            msg = data.get("message", {})
            u = self._extract_usage(msg)
            if u:
                self._usage = u
            return None

        if event_type == "content_block_start":
            block = data.get("content_block", {})
            index = data.get("index", 0)
            block_type = block.get("type", "")
            self._block_type_by_index[index] = block_type
            if block_type == "text":
                initial_text = block.get("text", "")
                if initial_text:
                    self._content_parts.append(initial_text)
                    return _build_openai_chunk(
                        self.chunk_id, self.created, self.model,
                        {"role": "assistant", "content": initial_text},
                    )
            elif block_type == "tool_use":
                call_id = block.get("id", f"toolu_{_random_hex(12)}")
                if call_id in self._seen_tool_ids:
                    call_id = f"toolu_{_random_hex(12)}"
                self._seen_tool_ids.add(call_id)
                name = block.get("name", "")
                self._tool_name_by_index[index] = name
                self._tool_args_by_index[index] = ""
                self._tool_calls_by_index[index] = {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
                # Buffer the start chunk — only emit when we receive the first delta
                # to avoid emitting orphan tool_use blocks from buggy upstreams
                start_chunk = _build_openai_chunk(
                    self.chunk_id, self.created, self.model,
                    {"tool_calls": [{"index": index, "id": call_id, "type": "function",
                                     "function": {"name": name, "arguments": ""}}]},
                )
                self._pending_tool_starts[index] = start_chunk
            elif block_type == "redacted_thinking":
                enc_data = block.get("data", "")
                self._reasoning_details.append({
                    "type": "reasoning.encrypted",
                    "data": enc_data,
                })
            return None

        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            index = data.get("index", 0)
            delta_type = delta.get("type", "")
            if delta_type == "thinking_delta":
                thinking_text = delta.get("thinking", "")
                if thinking_text:
                    self._reasoning_parts.append(thinking_text)
                    return _build_openai_chunk(
                        self.chunk_id, self.created, self.model,
                        {"reasoning_details": [{"type": "reasoning.text", "text": thinking_text}]},
                    )
                return None
            text = delta.get("text", "")
            if text:
                self._content_parts.append(text)
                return _build_openai_chunk(
                    self.chunk_id, self.created, self.model,
                    {"content": text},
                )
            partial_json = delta.get("partial_json", "")
            if partial_json:
                self._tool_args_by_index[index] = (
                    self._tool_args_by_index.get(index, "") + partial_json
                )
                tc = self._tool_calls_by_index.get(index, {})
                chunk = _build_openai_chunk(
                    self.chunk_id, self.created, self.model,
                    {"tool_calls": [{"index": index,
                                     "id": tc.get("id", ""),
                                     "type": "function",
                                     "function": {"arguments": partial_json}}]},
                )
                # Flush pending start chunk for this index before emitting delta
                pending = self._pending_tool_starts.pop(index, None)
                if pending is not None:
                    self._pending_chunks.append(pending)
                    self._pending_chunks.append(chunk)
                    return self._pending_chunks.popleft()
                return chunk
            return None

        if event_type == "content_block_stop":
            index = data.get("index", 0)
            self._stopped_blocks.add(index)
            # Discard orphan pending start if block stopped without receiving delta
            self._pending_tool_starts.pop(index, None)
            tc = self._tool_calls_by_index.get(index)
            if tc:
                args = self._tool_args_by_index.get(index, "").strip()
                if not args:
                    args = "{}"
                tc["function"]["arguments"] = args
            block_type = self._block_type_by_index.get(index, "")
            if block_type == "thinking" and self._reasoning_parts:
                full_reasoning = "".join(self._reasoning_parts)
                self._reasoning_details.append({"type": "reasoning.text", "text": full_reasoning})
                self._reasoning_parts = []
            self._block_type_by_index.pop(index, None)
            return None

        if event_type == "message_delta":
            delta = data.get("delta", {})
            stop_reason = delta.get("stop_reason", "")
            mapped = _map_stop_reason(stop_reason)
            if mapped:
                self._finish_reason = mapped
            u = self._extract_usage(data)
            if u:
                self._usage = u
            return None

        if event_type == "message_stop":
            return None

        if event_type == "error":
            error = data.get("error", {})
            return {
                "error": {
                    "message": error.get("message", "Upstream error"),
                    "type": error.get("type", "upstream_error"),
                },
            }

        return None

    # ── Non-streaming: aggregate full response ─────────────────────────────

    def build_final_response(self) -> dict[str, Any]:
        """Build the complete OpenAI response from accumulated state."""
        content = "".join(self._content_parts) if self._content_parts else None
        tool_calls = None
        if self._tool_calls_by_index:
            tool_calls = []
            for idx in sorted(self._tool_calls_by_index.keys()):
                # Skip orphan tool_use blocks that never received content_block_stop
                # (some upstreams emit phantom content_block_start events)
                if idx not in self._stopped_blocks:
                    continue
                tc = self._tool_calls_by_index[idx]
                tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": tc["function"],
                })
            if not tool_calls:
                tool_calls = None

        # Determine final finish reason
        finish = self._finish_reason
        if tool_calls and not content:
            finish = "tool_calls"
        elif tool_calls and content:
            # Anthropic can return both text and tool_use
            finish = "tool_calls"

        return _build_openai_response(
            self.chunk_id, self.created, self.model,
            content, tool_calls, finish, self._usage,
            reasoning_details=self._reasoning_details or None,
        )


def sse_lines_to_openai_chunk(
    data_lines: list[str],
    decoder: AnthropicStreamDecoder,
) -> Optional[str]:
    """Process Anthropic SSE data lines → OpenAI SSE string (or None)."""
    chunk = decoder.process_sse_lines(data_lines)
    if chunk is None:
        return None
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ── Backward-compatible alias ────────────────────────────────────────────────

translate_request = translate_openai_to_anthropic


# ══════════════════════════════════════════════════════════════════════════════
# REVERSE DIRECTION: Anthropic → OpenAI
# ══════════════════════════════════════════════════════════════════════════════


# ── Anthropic → OpenAI request translation ───────────────────────────────────


def _anthropic_content_to_text(content: Any) -> str:
    """Extract plain text from Anthropic content blocks or string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content) if content else ""


def _anthropic_tool_use_to_openai_calls(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract tool_use blocks → OpenAI tool_calls array."""
    calls: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        calls.append({
            "id": block.get("id", f"toolu_{_random_hex(12)}"),
            "type": "function",
            "function": {
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
            },
        })
    return calls


def _anthropic_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions → OpenAI tool definitions."""
    result: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name", "")
        if not name:
            continue
        result.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


def _anthropic_tool_choice_to_openai(choice: Any) -> Optional[Any]:
    """Convert Anthropic tool_choice → OpenAI tool_choice."""
    if choice is None:
        return None
    if isinstance(choice, dict):
        ctype = choice.get("type", "")
        if ctype == "none":
            return "none"
        if ctype == "any":
            return "required"
        if ctype == "auto":
            return "auto"
        if ctype == "tool":
            return {"type": "function", "function": {"name": choice.get("name", "")}}
    return None


def translate_anthropic_to_openai(
    anthropic_body: dict[str, Any],
    *,
    model_prefix: str = "",
) -> tuple[dict[str, Any], str]:
    """Translate an Anthropic Messages request to OpenAI Chat Completions format.

    Returns (openai_body, model_name).
    """
    model = anthropic_body.get("model", "")
    if model_prefix:
        model = model_prefix + model

    messages_raw = anthropic_body.get("messages", [])
    if not messages_raw:
        raise ValueError("messages array is required")

    openai_messages: list[dict[str, Any]] = []

    # System blocks → system message
    system = anthropic_body.get("system")
    if system:
        text = _anthropic_content_to_text(system)
        if text.strip():
            openai_messages.append({"role": "system", "content": text})

    for msg in messages_raw:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            if isinstance(content, list):
                # Extract text and tool_use blocks separately
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                            },
                        })
                omsg: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    omsg["content"] = "\n".join(text_parts)
                if tool_calls:
                    omsg["tool_calls"] = tool_calls
                    if "content" not in omsg:
                        omsg["content"] = None
                openai_messages.append(omsg)
            else:
                openai_messages.append({"role": "assistant", "content": content})

        elif role == "user":
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        openai_messages.append({"role": "user", "content": block.get("text", "")})
                    elif block.get("type") == "tool_result":
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": _anthropic_content_to_text(block.get("content", "")),
                        })
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                        else:
                            url = source.get("url", "")
                        openai_messages.append({
                            "role": "user",
                            "content": [{"type": "image_url", "image_url": {"url": url}}],
                        })
            else:
                openai_messages.append({"role": "user", "content": content or ""})

        else:
            # Unknown role — pass through
            openai_messages.append({"role": role, "content": _anthropic_content_to_text(content)})

    if not openai_messages:
        raise ValueError("At least one message is required")

    body: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
    }

    # max_tokens / max_completion_tokens
    max_tokens = anthropic_body.get("max_tokens")
    if max_tokens and max_tokens > 0:
        body["max_tokens"] = max_tokens

    # Stream
    if anthropic_body.get("stream"):
        body["stream"] = True

    # Sampling params
    if anthropic_body.get("temperature") is not None:
        body["temperature"] = anthropic_body["temperature"]
    if anthropic_body.get("top_p") is not None:
        body["top_p"] = anthropic_body["top_p"]

    # Stop sequences
    stop_seq = anthropic_body.get("stop_sequences")
    if isinstance(stop_seq, list) and stop_seq:
        body["stop"] = stop_seq

    # Tools
    tools = anthropic_body.get("tools")
    if tools:
        body["tools"] = _anthropic_tools_to_openai(tools)
        tc = _anthropic_tool_choice_to_openai(anthropic_body.get("tool_choice"))
        if tc is not None:
            body["tool_choice"] = tc

    # Translate Anthropic thinking → OpenAI reasoning
    thinking = anthropic_body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "disabled":
            body["reasoning"] = {"enabled": False}
        elif thinking.get("type") == "enabled":
            budget = thinking.get("budget_tokens", 0)
            if budget:
                body["reasoning"] = {"max_tokens": budget}

    return body, model


# ── OpenAI SSE → Anthropic SSE translator ────────────────────────────────────


class OpenAIToAnthropicSSETranslator:
    """Converts OpenAI Chat Completions SSE chunks → Anthropic Messages SSE events.

    Used when client expects Anthropic format but upstream returns OpenAI format.
    """

    def __init__(self, model: str):
        self.model = model
        self._msg_id = f"msg_{_random_hex(12)}"
        self._started = False
        self._block_started = False
        self._block_index = 0
        self._finished = False
        self._input_tokens = 0
        self._output_tokens = 0

    def _emit_event(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def start_message(self) -> str:
        """Emit the message_start event."""
        self._started = True
        return self._emit_event("message_start", {
            "type": "message_start",
            "message": {
                "id": self._msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": self._input_tokens, "output_tokens": 0},
            },
        })

    def start_text_block(self) -> str:
        """Emit content_block_start for a text block."""
        self._block_started = True
        return self._emit_event("content_block_start", {
            "type": "content_block_start",
            "index": self._block_index,
            "content_block": {"type": "text", "text": ""},
        })

    def emit_text_delta(self, text: str) -> str:
        """Emit a text content_block_delta."""
        return self._emit_event("content_block_delta", {
            "type": "content_block_delta",
            "index": self._block_index,
            "delta": {"type": "text_delta", "text": text},
        })

    def stop_block(self) -> str:
        """Emit content_block_stop."""
        result = self._emit_event("content_block_stop", {
            "type": "content_block_stop",
            "index": self._block_index,
        })
        self._block_index += 1
        self._block_started = False
        return result

    def finish_message(self, stop_reason: str = "end_turn", usage: Optional[dict[str, Any]] = None) -> str:
        """Emit message_delta + message_stop."""
        if self._finished:
            return ""
        self._finished = True
        out_tokens = 0
        if usage:
            self._input_tokens = usage.get("prompt_tokens", self._input_tokens)
            out_tokens = usage.get("completion_tokens", 0)
        result = ""
        result += self._emit_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": out_tokens},
        })
        result += self._emit_event("message_stop", {"type": "message_stop"})
        return result

    def translate_chunk(self, openai_chunk: dict[str, Any]) -> str:
        """Translate a single OpenAI SSE chunk to Anthropic SSE events.

        Returns concatenated event strings. May return "" if no output yet.
        """
        result = ""

        # Emit start events on first chunk
        if not self._started:
            usage = openai_chunk.get("usage")
            if usage:
                self._input_tokens = usage.get("prompt_tokens", 0)
            result += self.start_message()

        choices = openai_chunk.get("choices", [])
        if not choices:
            # Usage-only chunk
            return result

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Content delta
        content = delta.get("content")
        if content:
            if not self._block_started:
                result += self.start_text_block()
            result += self.emit_text_delta(content)

        # Tool calls delta
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            # For tool calls, we'd need to emit tool_use blocks
            # This is complex for streaming; for now, handle in final response
            pass

        # Finish
        if finish_reason:
            if self._block_started:
                result += self.stop_block()
            stop_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
            anthropic_stop = stop_map.get(finish_reason, "end_turn")
            usage = openai_chunk.get("usage")
            result += self.finish_message(anthropic_stop, usage)

        return result

    def translate_complete_response(self, openai_resp: dict[str, Any]) -> dict[str, Any]:
        """Translate a complete OpenAI non-streaming response → Anthropic Messages response."""
        model = openai_resp.get("model", self.model)
        msg_id = f"msg_{_random_hex(12)}"

        content_blocks: list[dict[str, Any]] = []
        choices = openai_resp.get("choices", [])
        stop_reason = "end_turn"

        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            finish = choice.get("finish_reason", "stop")

            text = message.get("content")
            if text:
                content_blocks.append({"type": "text", "text": text})

            tool_calls = message.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    try:
                        inp = json.loads(func.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        inp = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{_random_hex(12)}"),
                        "name": func.get("name", ""),
                        "input": inp,
                    })

            finish_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
            stop_reason = finish_map.get(finish, "end_turn")

        usage_raw = openai_resp.get("usage", {})
        input_tokens = usage_raw.get("prompt_tokens", 0)
        output_tokens = usage_raw.get("completion_tokens", 0)

        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }


class AnthropicMessageAggregator:
    """Accumulates Anthropic SSE events into a complete Anthropic Messages response.

    Used for non-streaming Anthropic→Anthropic passthrough.
    """

    def __init__(self):
        self._message: dict[str, Any] = {}
        self._content_blocks: dict[int, dict[str, Any]] = {}
        self._block_order: list[int] = []
        self._stop_reason: Optional[str] = None
        self._stop_sequence: Optional[str] = None
        self._output_tokens: int = 0

    def process_sse_lines(self, lines: list[str]) -> None:
        """Process a complete SSE event (data: lines)."""
        if not lines:
            return
        data_str = "\n".join(lines)
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, TypeError):
            return

        event_type = data.get("type", "")

        if event_type == "message_start":
            self._message = data.get("message", {})

        elif event_type == "content_block_start":
            index = data.get("index", 0)
            block = data.get("content_block", {})
            self._content_blocks[index] = dict(block)
            if index not in self._block_order:
                self._block_order.append(index)

        elif event_type == "content_block_delta":
            index = data.get("index", 0)
            delta = data.get("delta", {})
            block = self._content_blocks.get(index, {})
            if delta.get("type") == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif delta.get("type") == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            elif delta.get("type") == "input_json_delta":
                block["_partial_json"] = block.get("_partial_json", "") + delta.get("partial_json", "")

        elif event_type == "content_block_stop":
            index = data.get("index", 0)
            block = self._content_blocks.get(index, {})
            # Parse accumulated JSON for tool_use blocks
            if block.get("type") == "tool_use" and "_partial_json" in block:
                try:
                    block["input"] = json.loads(block.pop("_partial_json"))
                except (json.JSONDecodeError, TypeError):
                    block["input"] = {}
                    block.pop("_partial_json", None)

        elif event_type == "message_delta":
            delta = data.get("delta", {})
            if delta.get("stop_reason"):
                self._stop_reason = delta["stop_reason"]
            if delta.get("stop_sequence"):
                self._stop_sequence = delta["stop_sequence"]
            usage = data.get("usage", {})
            if usage.get("output_tokens"):
                self._output_tokens = usage["output_tokens"]

    def build_response(self) -> dict[str, Any]:
        """Build the complete Anthropic Messages response."""
        content = []
        for idx in sorted(self._block_order):
            block = self._content_blocks.get(idx, {})
            clean_block = {k: v for k, v in block.items() if not k.startswith("_")}
            if clean_block:
                content.append(clean_block)

        result = dict(self._message)
        result["content"] = content
        if self._stop_reason:
            result["stop_reason"] = self._stop_reason
        if self._stop_sequence:
            result["stop_sequence"] = self._stop_sequence
        # Update usage
        if "usage" in result:
            result["usage"]["output_tokens"] = self._output_tokens
        return result


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI Chat ↔ OpenAI Responses API translation
# ══════════════════════════════════════════════════════════════════════════════


def _blocks_to_text_simple(content: Any) -> str:
    """Extract plain text from an OpenAI content field."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") in ("text", "input_text", "output_text"):
                    parts.append(p.get("text", ""))
                elif "text" in p:
                    parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)


def _openai_content_to_responses_input(content: list) -> list[dict[str, Any]]:
    """Convert OpenAI chat content array to Responses input content items."""
    items: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype in ("text", "input_text"):
            items.append({"type": "input_text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url_obj = part.get("image_url", {})
            url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
            detail = url_obj.get("detail", "auto") if isinstance(url_obj, dict) else "auto"
            items.append({"type": "input_image", "image_url": url, "detail": detail})
    return items


def _openai_chat_tools_to_responses(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI Chat tools to Responses API tools format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type", "function") != "function":
            continue
        func = tool.get("function", {})
        result.append({
            "type": "function",
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {"type": "object", "properties": {}}),
            "strict": func.get("strict", False),
        })
    return result


def translate_openai_chat_to_responses(chat_body: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions request to Responses API format."""
    response_body: dict[str, Any] = {}

    if chat_body.get("model"):
        response_body["model"] = chat_body["model"]

    messages = chat_body.get("messages", [])
    input_items: list[dict[str, Any]] = []
    instructions_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content")

        if role in ("system", "developer"):
            text = _blocks_to_text_simple(content)
            if text:
                instructions_parts.append(text)
        elif role == "user":
            if isinstance(content, str):
                input_items.append({"role": "user", "content": [{"type": "input_text", "text": content}]})
            elif isinstance(content, list):
                items = _openai_content_to_responses_input(content)
                if items:
                    input_items.append({"role": "user", "content": items})
                else:
                    input_items.append({"role": "user", "content": [{"type": "input_text", "text": ""}]})
            else:
                input_items.append({"role": "user", "content": [{"type": "input_text", "text": ""}]})
        elif role == "assistant":
            text = _blocks_to_text_simple(content)
            parts: list[dict[str, Any]] = []
            if text:
                parts.append({"type": "output_text", "text": text})
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    parts.append({
                        "type": "tool_call",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                    })
            if parts:
                input_items.append({"role": "assistant", "content": parts})
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            text = _blocks_to_text_simple(content)
            input_items.append({
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": text or "",
            })

    if instructions_parts:
        response_body["instructions"] = "\n\n".join(instructions_parts)

    response_body["input"] = input_items

    max_tokens = chat_body.get("max_tokens") or chat_body.get("max_completion_tokens")
    if max_tokens:
        response_body["max_output_tokens"] = max_tokens
    if chat_body.get("stream"):
        response_body["stream"] = True
    if chat_body.get("temperature") is not None:
        response_body["temperature"] = chat_body["temperature"]
    if chat_body.get("top_p") is not None:
        response_body["top_p"] = chat_body["top_p"]

    reasoning = chat_body.get("reasoning")
    if reasoning:
        response_body["reasoning"] = reasoning

    tools = chat_body.get("tools")
    if tools:
        responses_tools = _openai_chat_tools_to_responses(tools)
        if responses_tools:
            response_body["tools"] = responses_tools
        tc = chat_body.get("tool_choice")
        if tc:
            response_body["tool_choice"] = tc

    rf = chat_body.get("response_format")
    if rf:
        response_body["text"] = {"format": rf}

    return response_body


def _responses_tools_to_openai_chat(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses API tools to Chat Completions tools format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type", "function") != "function":
            continue
        result.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    return result


def translate_openai_responses_to_chat(responses_body: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI Responses API request to Chat Completions format."""
    chat_body: dict[str, Any] = {}

    if responses_body.get("model"):
        chat_body["model"] = responses_body["model"]

    messages: list[dict[str, Any]] = []

    instructions = responses_body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    input_data = responses_body.get("input", "")
    if isinstance(input_data, str):
        if input_data:
            messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type", "")
            role = item.get("role", "")

            if role == "user":
                content_parts = item.get("content", [])
                if isinstance(content_parts, str):
                    messages.append({"role": "user", "content": content_parts})
                elif isinstance(content_parts, list):
                    parts_list: list[Any] = []
                    for cp in content_parts:
                        if isinstance(cp, dict):
                            if cp.get("type") in ("input_text", "text"):
                                parts_list.append(cp.get("text", ""))
                            elif cp.get("type") == "input_image":
                                url = cp.get("image_url", "")
                                detail = cp.get("detail", "auto")
                                parts_list.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
                    if len(parts_list) == 1 and isinstance(parts_list[0], str):
                        messages.append({"role": "user", "content": parts_list[0]})
                    else:
                        messages.append({"role": "user", "content": parts_list})
                else:
                    messages.append({"role": "user", "content": ""})

            elif role == "assistant":
                content_parts = item.get("content", [])
                if isinstance(content_parts, str):
                    messages.append({"role": "assistant", "content": content_parts})
                elif isinstance(content_parts, list):
                    text_parts: list[str] = []
                    tool_calls_list: list[dict[str, Any]] = []
                    for cp in content_parts:
                        if isinstance(cp, dict):
                            if cp.get("type") in ("output_text", "text"):
                                text_parts.append(cp.get("text", ""))
                            elif cp.get("type") == "tool_call":
                                tool_calls_list.append({
                                    "id": cp.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": cp.get("name", ""),
                                        "arguments": cp.get("arguments", "{}"),
                                    },
                                })
                    omsg: dict[str, Any] = {"role": "assistant"}
                    if text_parts:
                        omsg["content"] = "\n".join(text_parts)
                    else:
                        omsg["content"] = None
                    if tool_calls_list:
                        omsg["tool_calls"] = tool_calls_list
                    messages.append(omsg)
                else:
                    messages.append({"role": "assistant", "content": ""})

            elif item_type == "function_call_output":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                })

    chat_body["messages"] = messages

    max_output = responses_body.get("max_output_tokens")
    if max_output:
        chat_body["max_tokens"] = max_output
    if responses_body.get("stream"):
        chat_body["stream"] = True
    if responses_body.get("temperature") is not None:
        chat_body["temperature"] = responses_body["temperature"]
    if responses_body.get("top_p") is not None:
        chat_body["top_p"] = responses_body["top_p"]

    reasoning = responses_body.get("reasoning")
    if reasoning:
        chat_body["reasoning"] = reasoning

    tools = responses_body.get("tools")
    if tools:
        chat_tools = _responses_tools_to_openai_chat(tools)
        if chat_tools:
            chat_body["tools"] = chat_tools
        tc = responses_body.get("tool_choice")
        if tc:
            chat_body["tool_choice"] = tc

    text_config = responses_body.get("text")
    if isinstance(text_config, dict):
        fmt = text_config.get("format")
        if fmt:
            chat_body["response_format"] = fmt

    return chat_body


# ══════════════════════════════════════════════════════════════════════════════
# Anthropic ↔ OpenAI Responses API translation
# ══════════════════════════════════════════════════════════════════════════════


def _anthropic_tools_to_responses(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tools to Responses API tools format."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name", "")
        if not name:
            continue
        result.append({
            "type": "function",
            "name": name,
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        })
    return result


def translate_anthropic_to_responses(anthropic_body: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages request to OpenAI Responses API format."""
    response_body: dict[str, Any] = {}

    model = anthropic_body.get("model", "")
    if model:
        response_body["model"] = model

    system = anthropic_body.get("system")
    if system:
        text = _anthropic_content_to_text(system)
        if text:
            response_body["instructions"] = text

    messages = anthropic_body.get("messages", [])
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, list):
                items: list[dict[str, Any]] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        items.append({"type": "input_text", "text": block.get("text", "")})
                    elif btype == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            url = f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                        else:
                            url = source.get("url", "")
                        items.append({"type": "input_image", "image_url": url})
                    elif btype == "tool_result":
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": block.get("tool_use_id", ""),
                            "output": _anthropic_content_to_text(block.get("content", "")),
                        })
                        continue
                if items:
                    input_items.append({"role": "user", "content": items})
            elif isinstance(content, str):
                input_items.append({"role": "user", "content": [{"type": "input_text", "text": content}]})

        elif role == "assistant":
            if isinstance(content, list):
                output_parts: list[dict[str, Any]] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        output_parts.append({"type": "output_text", "text": block.get("text", "")})
                    elif btype == "tool_use":
                        output_parts.append({
                            "type": "tool_call",
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        })
                    elif btype == "thinking":
                        output_parts.append({"type": "reasoning_text", "text": block.get("thinking", "")})
                if output_parts:
                    input_items.append({"role": "assistant", "content": output_parts})
            elif isinstance(content, str):
                input_items.append({"role": "assistant", "content": [{"type": "output_text", "text": content}]})

    response_body["input"] = input_items

    max_tokens = anthropic_body.get("max_tokens")
    if max_tokens:
        response_body["max_output_tokens"] = max_tokens
    if anthropic_body.get("stream"):
        response_body["stream"] = True
    if anthropic_body.get("temperature") is not None:
        response_body["temperature"] = anthropic_body["temperature"]
    if anthropic_body.get("top_p") is not None:
        response_body["top_p"] = anthropic_body["top_p"]

    thinking = anthropic_body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "enabled":
            budget = thinking.get("budget_tokens")
            if budget:
                response_body["reasoning"] = {"max_tokens": budget}
        elif thinking.get("type") == "disabled":
            response_body["reasoning"] = {"enabled": False}

    tools = anthropic_body.get("tools")
    if tools:
        responses_tools = _anthropic_tools_to_responses(tools)
        if responses_tools:
            response_body["tools"] = responses_tools

    return response_body


def _responses_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Responses API tools to Anthropic tool definitions."""
    result: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type", "function") != "function":
            continue
        name = tool.get("name", "")
        if not name:
            continue
        result.append({
            "name": name,
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def translate_responses_to_anthropic(
    responses_body: dict[str, Any],
    *,
    default_max_tokens: int = 4096,
) -> dict[str, Any]:
    """Translate an OpenAI Responses API request to Anthropic Messages format."""
    body: dict[str, Any] = {}

    model = responses_body.get("model", "")
    if model:
        body["model"] = model

    system_blocks: list[dict[str, Any]] = []
    anthropic_messages: list[dict[str, Any]] = []

    instructions = responses_body.get("instructions")
    if instructions:
        system_blocks.append({"type": "text", "text": instructions})

    input_data = responses_body.get("input", "")
    if isinstance(input_data, str):
        if input_data:
            anthropic_messages.append({"role": "user", "content": [{"type": "text", "text": input_data}]})
    elif isinstance(input_data, list):
        for item in input_data:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "")
            item_type = item.get("type", "")

            if role == "user":
                content_parts = item.get("content", [])
                if isinstance(content_parts, str):
                    anthropic_messages = _append_or_merge(
                        anthropic_messages, "user",
                        [{"type": "text", "text": content_parts}],
                    )
                elif isinstance(content_parts, list):
                    blocks: list[dict[str, Any]] = []
                    for cp in content_parts:
                        if isinstance(cp, dict):
                            if cp.get("type") in ("input_text", "text"):
                                blocks.append({"type": "text", "text": cp.get("text", "")})
                            elif cp.get("type") == "input_image":
                                url = cp.get("image_url", "")
                                if url.startswith("data:") and ";base64," in url:
                                    meta, data = url.split(";base64,", 1)
                                    mime = meta.split(":", 1)[1] if ":" in meta else "image/png"
                                    blocks.append({
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": mime, "data": data},
                                    })
                                else:
                                    blocks.append({
                                        "type": "image",
                                        "source": {"type": "url", "url": url},
                                    })
                    if blocks:
                        anthropic_messages = _append_or_merge(anthropic_messages, "user", blocks)

            elif role == "assistant":
                content_parts = item.get("content", [])
                if isinstance(content_parts, str):
                    anthropic_messages = _append_or_merge(
                        anthropic_messages, "assistant",
                        [{"type": "text", "text": content_parts}],
                    )
                elif isinstance(content_parts, list):
                    blocks = []
                    for cp in content_parts:
                        if isinstance(cp, dict):
                            if cp.get("type") in ("output_text", "text"):
                                blocks.append({"type": "text", "text": cp.get("text", "")})
                            elif cp.get("type") == "tool_call":
                                try:
                                    inp = json.loads(cp.get("arguments", "{}"))
                                except (json.JSONDecodeError, TypeError):
                                    inp = {}
                                blocks.append({
                                    "type": "tool_use",
                                    "id": cp.get("id", ""),
                                    "name": cp.get("name", ""),
                                    "input": inp,
                                })
                    if blocks:
                        anthropic_messages = _append_or_merge(anthropic_messages, "assistant", blocks)

            elif item_type == "function_call_output":
                anthropic_messages = _append_or_merge(
                    anthropic_messages, "user",
                    [{
                        "type": "tool_result",
                        "tool_use_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    }],
                )

    if not anthropic_messages:
        raise ValueError("At least one non-system input is required")

    body["messages"] = anthropic_messages
    body["max_tokens"] = default_max_tokens
    body["stream"] = True

    if system_blocks:
        body["system"] = system_blocks

    max_output = responses_body.get("max_output_tokens")
    if max_output and max_output > 0:
        body["max_tokens"] = max_output
    if responses_body.get("temperature") is not None:
        body["temperature"] = responses_body["temperature"]
    if responses_body.get("top_p") is not None:
        body["top_p"] = responses_body["top_p"]

    reasoning = responses_body.get("reasoning")
    if isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            body["thinking"] = {"type": "disabled"}
        elif reasoning.get("max_tokens"):
            body["thinking"] = {"type": "enabled", "budget_tokens": reasoning["max_tokens"]}
        elif reasoning.get("effort"):
            budget = REASONING_EFFORT_TO_BUDGET.get(reasoning["effort"], 4096)
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}

    tools = responses_body.get("tools")
    if tools:
        anthropic_tools = _responses_tools_to_anthropic(tools)
        if anthropic_tools:
            body["tools"] = anthropic_tools

    return body
