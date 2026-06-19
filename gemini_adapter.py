#!/usr/bin/env python3
"""
Bidirectional Gemini GenerateContent (v1beta) protocol adapter.

Supports the full translation matrix for Gemini:
  - OpenAI Chat → Gemini: translate_openai_to_gemini()
  - Anthropic → Gemini: translate_anthropic_to_gemini()
  - Gemini → OpenAI Chat: translate_gemini_to_openai()
  - Gemini → Anthropic: translate_gemini_to_anthropic()
  - Gemini ↔ OpenAI Responses: via two-hop through Chat
  - Streaming decoders for Gemini SSE → OpenAI/Anthropic SSE
"""

import json
import time
import re
from typing import Any, Optional

from constants import (
    GEMINI_FINISH_MAP,
    OPENAI_TO_GEMINI_FINISH,
    ANTHROPIC_TO_GEMINI_FINISH,
    DEFAULT_GEMINI_SAFETY_SETTINGS,
    REASONING_EFFORT_TO_BUDGET,
)

_PRIVATE_HOST_RE = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+|0\.0\.0\.0|::1|fc00:|fe80:)",
    re.IGNORECASE,
)


def _is_safe_url(url: str) -> bool:
    """Return True if the URL is safe to forward (blocks private/internal hosts)."""
    if not url or not url.startswith(("http://", "https://")):
        return False
    try:
        host = url.split("://", 1)[1].split("/")[0].split(":")[0]
        return not _PRIVATE_HOST_RE.match(host)
    except (IndexError, AttributeError):
        return False
from protocol_adapter import (
    _random_hex,
    _build_openai_chunk,
    _build_openai_response,
    _append_or_merge,
)


# ── Schema helpers ────────────────────────────────────────────────────────────


def _openai_schema_to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI JSON Schema type names to Gemini SCHEMA_TYPE names."""
    if not isinstance(schema, dict):
        return schema
    result: dict[str, Any] = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            result["type"] = v.upper()
        elif k == "properties" and isinstance(v, dict):
            result["properties"] = {pk: _openai_schema_to_gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            result["items"] = _openai_schema_to_gemini_schema(v)
        elif k == "anyOf" and isinstance(v, list):
            result["anyOf"] = [_openai_schema_to_gemini_schema(item) for item in v]
        elif k == "oneOf" and isinstance(v, list):
            result["oneOf"] = [_openai_schema_to_gemini_schema(item) for item in v]
        elif k == "allOf" and isinstance(v, list):
            result["allOf"] = [_openai_schema_to_gemini_schema(item) for item in v]
        elif k == "not" and isinstance(v, dict):
            result["not"] = _openai_schema_to_gemini_schema(v)
        elif k == "prefixItems" and isinstance(v, list):
            result["prefixItems"] = [_openai_schema_to_gemini_schema(item) for item in v]
        else:
            result[k] = v
    return result


def _gemini_schema_to_openai_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert Gemini schema type names back to OpenAI JSON Schema type names."""
    if not isinstance(schema, dict):
        return schema
    result: dict[str, Any] = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            result["type"] = v.lower()
        elif k == "properties" and isinstance(v, dict):
            result["properties"] = {pk: _gemini_schema_to_openai_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            result["items"] = _gemini_schema_to_openai_schema(v)
        elif k == "anyOf" and isinstance(v, list):
            result["anyOf"] = [_gemini_schema_to_openai_schema(item) for item in v]
        elif k == "oneOf" and isinstance(v, list):
            result["oneOf"] = [_gemini_schema_to_openai_schema(item) for item in v]
        elif k == "allOf" and isinstance(v, list):
            result["allOf"] = [_gemini_schema_to_openai_schema(item) for item in v]
        elif k == "not" and isinstance(v, dict):
            result["not"] = _gemini_schema_to_openai_schema(v)
        elif k == "prefixItems" and isinstance(v, list):
            result["prefixItems"] = [_gemini_schema_to_openai_schema(item) for item in v]
        else:
            result[k] = v
    return result


# ── Gemini role append/merge ─────────────────────────────────────────────────


def _gemini_append_or_merge(
    contents: list[dict[str, Any]],
    role: str,
    parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append or merge with last content entry if same role (Gemini requires alternating roles)."""
    if contents and contents[-1]["role"] == role:
        contents[-1]["parts"] = list(contents[-1]["parts"]) + parts
        return contents
    return contents + [{"role": role, "parts": parts}]


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI Chat → Gemini
# ══════════════════════════════════════════════════════════════════════════════


def _openai_content_to_parts(content: Any) -> list[dict[str, Any]]:
    """Convert OpenAI message content to Gemini parts."""
    if content is None:
        return [{"text": ""}]
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append({"text": str(part)})
                continue
            ptype = part.get("type", "")
            if ptype in ("text", "input_text", "output_text"):
                parts.append({"text": part.get("text", "")})
            elif ptype == "image_url":
                url_obj = part.get("image_url", {})
                url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
                if url.startswith("data:") and ";base64," in url:
                    meta, data = url.split(";base64,", 1)
                    mime = meta.split(":", 1)[1] if ":" in meta else "image/png"
                    parts.append({"inlineData": {"mimeType": mime, "data": data}})
                elif _is_safe_url(url):
                    parts.append({"fileData": {"mimeType": "image/png", "fileUri": url}})
                else:
                    parts.append({"text": f"[Image URL: {url[:100]}]"})
            else:
                parts.append({"text": json.dumps(part, ensure_ascii=False)})
        return parts if parts else [{"text": ""}]
    return [{"text": str(content)}]


def _openai_tool_calls_to_gemini_parts(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI assistant tool_calls to Gemini functionCall parts."""
    parts: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except (json.JSONDecodeError, TypeError):
            args = {}
        if name:
            fc_part = {"name": name, "args": args if isinstance(args, dict) else {}}
            if tc.get("id"):
                fc_part["id"] = tc["id"]
            parts.append({"functionCall": fc_part})
    return parts


def _openai_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool definitions to Gemini functionDeclarations."""
    declarations: list[dict[str, Any]] = []
    for tool in tools or []:
        if tool.get("type", "function") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        if not name:
            continue
        if name == "google_search":
            name = "google_search_proxy_renamed"
        if name == "web_search":
            name = "web_search_proxy_renamed"
        params = func.get("parameters")
        if params is None:
            params = {"type": "OBJECT", "properties": {}}
        else:
            params = _openai_schema_to_gemini_schema(params)
        declarations.append({
            "name": name,
            "description": func.get("description", ""),
            "parameters": params,
        })
    return [{"functionDeclarations": declarations}] if declarations else []


def build_gemini_tool_config(
    mode: str = "AUTO",
    allowed_names: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build clean camelCase toolConfig dictionary.

    All sub-keys inside toolConfig must be camelCase.
    """
    tc_camel: dict[str, Any] = {
        "functionCallingConfig": {
            "mode": mode
        },
        "includeServerSideToolInvocations": True
    }
    if allowed_names:
        tc_camel["functionCallingConfig"]["allowedFunctionNames"] = allowed_names

    return tc_camel


def _openai_tool_choice_to_gemini(choice: Any) -> Optional[dict[str, Any]]:
    """Convert OpenAI tool_choice to Gemini toolConfig."""
    if choice is None:
        return None
    if isinstance(choice, str):
        if choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if choice == "required":
            return {"functionCallingConfig": {"mode": "ANY"}}
        if choice == "auto":
            return {"functionCallingConfig": {"mode": "AUTO"}}
    if isinstance(choice, dict):
        func = choice.get("function", {})
        name = func.get("name") if isinstance(func, dict) else None
        if not name:
            name = choice.get("name")
        if name:
            return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    return None


def translate_openai_to_gemini(
    openai_body: dict[str, Any],
    *,
    model_prefix: str = "",
    default_max_tokens: int = 8192,
) -> tuple[dict[str, Any], str]:
    """Translate an OpenAI Chat Completions request to Gemini GenerateContent format.

    Returns (gemini_body, model_name).
    """
    model = openai_body.get("model", "")
    if model_prefix and model.startswith(model_prefix):
        model = model[len(model_prefix):]

    messages = openai_body.get("messages", [])
    if not messages:
        raise ValueError("messages array is required and must not be empty")

    # Pre-scan: build tool_call_id → function name mapping from assistant messages
    tool_id_to_name: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    tc_id = tc.get("id")
                    func = tc.get("function")
                    if isinstance(func, dict):
                        name = func.get("name")
                        if tc_id and name:
                            tool_id_to_name[tc_id] = name

    system_parts: list[dict[str, Any]] = []
    gemini_contents: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "").lower().strip()
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")

        if role in ("system", "developer"):
            system_parts.extend(_openai_content_to_parts(content))
        elif role == "assistant":
            parts = _openai_content_to_parts(content)
            if tool_calls:
                parts.extend(_openai_tool_calls_to_gemini_parts(tool_calls))
            gemini_contents = _gemini_append_or_merge(gemini_contents, "model", parts)
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            name = msg.get("name")
            if not name and tool_call_id:
                name = tool_id_to_name.get(tool_call_id)
            if not name:
                name = "unknown"
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    response_data = parsed
                else:
                    response_data = {"result": text}
            except (json.JSONDecodeError, ValueError):
                response_data = {"result": text}

            func_resp = {"name": name, "response": response_data}
            if tool_call_id:
                func_resp["id"] = tool_call_id
            gemini_contents = _gemini_append_or_merge(
                gemini_contents, "user",
                [{"functionResponse": func_resp}],
            )
        else:
            gemini_contents = _gemini_append_or_merge(
                gemini_contents, "user", _openai_content_to_parts(content),
            )

    if not gemini_contents:
        raise ValueError("At least one non-system message is required")

    body: dict[str, Any] = {"contents": gemini_contents}
    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}

    gen_config: dict[str, Any] = {}
    max_tokens = openai_body.get("max_tokens") or openai_body.get("max_completion_tokens") or default_max_tokens
    if max_tokens and max_tokens > 0:
        gen_config["maxOutputTokens"] = max_tokens
    if openai_body.get("temperature") is not None:
        gen_config["temperature"] = openai_body["temperature"]
    if openai_body.get("top_p") is not None:
        gen_config["topP"] = openai_body["top_p"]
    stop = openai_body.get("stop")
    if isinstance(stop, str) and stop:
        gen_config["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        gen_config["stopSequences"] = [s for s in stop if isinstance(s, str) and s]
    if openai_body.get("n") is not None and openai_body["n"] > 0:
        gen_config["candidateCount"] = openai_body["n"]

    reasoning = openai_body.get("reasoning")
    if isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            pass
        else:
            budget = reasoning.get("max_tokens")
            if budget and isinstance(budget, (int, float)) and budget > 0:
                gen_config["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": int(budget)}
            elif reasoning.get("effort"):
                b = REASONING_EFFORT_TO_BUDGET.get(reasoning["effort"], 4096)
                gen_config["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": b}

    if gen_config:
        body["generationConfig"] = gen_config

    tools = openai_body.get("tools")
    if tools:
        gemini_tools = _openai_tools_to_gemini(tools)
        if gemini_tools:
            body["tools"] = gemini_tools
        
        mode = "AUTO"
        allowed_names = None
        
        choice = openai_body.get("tool_choice")
        if choice is not None:
            if isinstance(choice, str):
                if choice == "none":
                    mode = "NONE"
                elif choice == "required":
                    mode = "ANY"
                elif choice == "auto":
                    mode = "AUTO"
            elif isinstance(choice, dict):
                func = choice.get("function", {})
                name = func.get("name") if isinstance(func, dict) else None
                if not name:
                    name = choice.get("name")
                if name:
                    mode = "ANY"
                    allowed_names = [name]
                    
        tc_camel = build_gemini_tool_config(mode, allowed_names)
        body["toolConfig"] = tc_camel

    body["safetySettings"] = DEFAULT_GEMINI_SAFETY_SETTINGS
    return body, model


# ══════════════════════════════════════════════════════════════════════════════
# Gemini → OpenAI Chat
# ══════════════════════════════════════════════════════════════════════════════


def _gemini_parts_to_openai_content(
    parts: list[dict[str, Any]],
    func_name_to_tool_id: Optional[dict[str, str]] = None,
) -> tuple[Optional[str], Optional[list[dict[str, Any]]], Optional[list[dict[str, Any]]]]:
    """Convert Gemini parts to (content_text, tool_calls, reasoning_details)."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_details: list[dict[str, Any]] = []

    for part in parts or []:
        if not isinstance(part, dict):
            continue
        if part.get("thought"):
            text = part.get("text", "")
            if text:
                reasoning_details.append({"type": "reasoning.text", "text": text})
            continue
        if "text" in part:
            text_parts.append(part["text"])
        if "functionCall" in part:
            fc = part["functionCall"]
            name = fc.get("name", "")
            if name == "google_search_proxy_renamed":
                name = "google_search"
            elif name == "web_search_proxy_renamed":
                name = "web_search"
            args = fc.get("args", {})
            tool_call_id = fc.get("id")
            if not tool_call_id:
                if func_name_to_tool_id and name in func_name_to_tool_id:
                    tool_call_id = func_name_to_tool_id[name]
                else:
                    tool_call_id = f"call_{_random_hex(12)}"
                    if func_name_to_tool_id is not None and name:
                        func_name_to_tool_id[name] = tool_call_id
            tool_calls.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                },
            })

    content = "\n".join(text_parts) if text_parts else None
    return (
        content,
        tool_calls if tool_calls else None,
        reasoning_details if reasoning_details else None,
    )


def _gemini_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Gemini tools to OpenAI tool definitions."""
    result: list[dict[str, Any]] = []
    for tool in tools or []:
        decls = tool.get("functionDeclarations", [])
        for decl in decls:
            name = decl.get("name", "")
            if not name:
                continue
            if name == "google_search_proxy_renamed":
                name = "google_search"
            params = decl.get("parameters")
            if params:
                params = _gemini_schema_to_openai_schema(params)
            result.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": decl.get("description", ""),
                    "parameters": params or {"type": "object", "properties": {}},
                },
            })
    return result


def _gemini_tool_config_to_openai(tool_config: Any) -> Optional[Any]:
    """Convert Gemini toolConfig to OpenAI tool_choice."""
    if not tool_config or not isinstance(tool_config, dict):
        return None
    fcc = tool_config.get("functionCallingConfig", {})
    mode = fcc.get("mode", "")
    if mode == "NONE":
        return "none"
    if mode == "AUTO":
        return "auto"
    if mode == "ANY":
        allowed = fcc.get("allowedFunctionNames", [])
        if len(allowed) == 1:
            name = allowed[0]
            if name == "google_search_proxy_renamed":
                name = "google_search"
            return {"type": "function", "function": {"name": name}}
        return "required"
    return None


def translate_gemini_to_openai(
    gemini_body: dict[str, Any],
    *,
    model_prefix: str = "",
) -> tuple[dict[str, Any], str]:
    """Translate a Gemini GenerateContent request to OpenAI Chat Completions format.

    Returns (openai_body, model_name).
    """
    model = gemini_body.get("model", "")
    if model_prefix:
        model = model_prefix + model

    contents = gemini_body.get("contents", [])
    if not contents:
        raise ValueError("contents array is required")

    # Pre-scan: build function name → tool_call_id mapping for consistent IDs
    func_name_to_tool_id: dict[str, str] = {}
    for content in contents:
        for part in content.get("parts", []):
            if isinstance(part, dict) and "functionCall" in part:
                fc = part["functionCall"]
                name = fc.get("name", "")
                if name and name not in func_name_to_tool_id:
                    func_name_to_tool_id[name] = fc.get("id") or f"call_{_random_hex(12)}"

    openai_messages: list[dict[str, Any]] = []

    sys_inst = gemini_body.get("systemInstruction")
    if sys_inst:
        text_parts = [
            p.get("text", "") for p in sys_inst.get("parts", [])
            if isinstance(p, dict) and "text" in p
        ]
        if text_parts:
            openai_messages.append({"role": "system", "content": "\n".join(text_parts)})

    for content in contents:
        role = content.get("role", "")
        parts = content.get("parts", [])

        if role == "user":
            openai_parts: list[dict[str, Any]] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if "functionResponse" in part:
                    fr = part["functionResponse"]
                    func_name = fr.get("name", "")
                    if func_name == "google_search_proxy_renamed":
                        func_name = "google_search"
                    resp = fr.get("response", {})
                    if "result" in resp and len(resp) == 1:
                        result_val = resp["result"]
                        if isinstance(result_val, (dict, list)):
                            result_text = json.dumps(result_val, ensure_ascii=False)
                        else:
                            result_text = str(result_val)
                    else:
                        result_text = json.dumps(resp, ensure_ascii=False)
                    tool_call_id = fr.get("id") or func_name_to_tool_id.get(func_name)
                    if not tool_call_id:
                        tool_call_id = f"call_{_random_hex(12)}"
                        func_name_to_tool_id[func_name] = tool_call_id
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": func_name,
                        "content": str(result_text),
                    })
                elif "text" in part:
                    openai_parts.append({"type": "text", "text": part["text"]})
                elif "inlineData" in part:
                    data = part["inlineData"]
                    mime = data.get("mimeType", "image/png")
                    b64 = data.get("data", "")
                    if b64:
                        openai_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        })
                elif "fileData" in part:
                    fd = part["fileData"]
                    openai_parts.append({
                        "type": "image_url",
                        "image_url": {"url": fd.get("fileUri", "")},
                    })
            if openai_parts:
                if len(openai_parts) == 1 and openai_parts[0].get("type") == "text":
                    openai_messages.append({"role": "user", "content": openai_parts[0]["text"]})
                else:
                    openai_messages.append({"role": "user", "content": openai_parts})

        elif role == "model":
            content_text, tool_calls, reasoning_details = _gemini_parts_to_openai_content(parts, func_name_to_tool_id=func_name_to_tool_id)
            omsg: dict[str, Any] = {"role": "assistant"}
            omsg["content"] = content_text
            if tool_calls:
                omsg["tool_calls"] = tool_calls
            if reasoning_details:
                omsg["reasoning_details"] = reasoning_details
            openai_messages.append(omsg)

        else:
            text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
            openai_messages.append({"role": "user", "content": "\n".join(text_parts) or ""})

    if not openai_messages:
        raise ValueError("At least one message is required")

    body: dict[str, Any] = {"model": model, "messages": openai_messages}

    gen_config = gemini_body.get("generationConfig", {})
    if gen_config:
        max_tokens = gen_config.get("maxOutputTokens")
        if max_tokens and max_tokens > 0:
            body["max_tokens"] = max_tokens
        if gen_config.get("temperature") is not None:
            body["temperature"] = gen_config["temperature"]
        if gen_config.get("topP") is not None:
            body["top_p"] = gen_config["topP"]
        stop = gen_config.get("stopSequences")
        if isinstance(stop, list) and stop:
            body["stop"] = stop
        if gen_config.get("candidateCount") and gen_config["candidateCount"] > 1:
            body["n"] = gen_config["candidateCount"]

        thinking_config = gen_config.get("thinkingConfig")
        if isinstance(thinking_config, dict) and thinking_config.get("includeThoughts"):
            budget = thinking_config.get("thinkingBudget")
            if budget and isinstance(budget, (int, float)) and budget > 0:
                body["reasoning"] = {"max_tokens": int(budget)}

    tools = gemini_body.get("tools")
    if tools:
        openai_tools = _gemini_tools_to_openai(tools)
        if openai_tools:
            body["tools"] = openai_tools
        tc = _gemini_tool_config_to_openai(gemini_body.get("toolConfig"))
        if tc is not None:
            body["tool_choice"] = tc

    return body, model


# ══════════════════════════════════════════════════════════════════════════════
# Anthropic → Gemini
# ══════════════════════════════════════════════════════════════════════════════


def _anthropic_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tools to Gemini functionDeclarations."""
    declarations: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name", "")
        if not name:
            continue
        if name == "google_search":
            name = "google_search_proxy_renamed"
        params = tool.get("input_schema")
        if params:
            params = _openai_schema_to_gemini_schema(params)
        declarations.append({
            "name": name,
            "description": tool.get("description", ""),
            "parameters": params or {"type": "OBJECT", "properties": {}},
        })
    return [{"functionDeclarations": declarations}] if declarations else []


def _anthropic_tool_choice_to_gemini(choice: Any) -> Optional[dict[str, Any]]:
    """Convert Anthropic tool_choice to Gemini toolConfig."""
    if choice is None:
        return None
    if isinstance(choice, dict):
        ctype = choice.get("type", "")
        if ctype == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if ctype == "any":
            return {"functionCallingConfig": {"mode": "ANY"}}
        if ctype == "auto":
            return {"functionCallingConfig": {"mode": "AUTO"}}
        if ctype == "tool":
            name = choice.get("name", "")
            if name == "google_search":
                name = "google_search_proxy_renamed"
            if name:
                return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    return None


def translate_anthropic_to_gemini(
    anthropic_body: dict[str, Any],
    *,
    model_prefix: str = "",
    default_max_tokens: int = 8192,
) -> tuple[dict[str, Any], str]:
    """Translate an Anthropic Messages request to Gemini GenerateContent format.

    Returns (gemini_body, model_name).
    """
    model = anthropic_body.get("model", "")
    if model_prefix and model.startswith(model_prefix):
        model = model[len(model_prefix):]

    messages = anthropic_body.get("messages", [])
    if not messages:
        raise ValueError("messages array is required")

    system_parts: list[dict[str, Any]] = []
    gemini_contents: list[dict[str, Any]] = []

    system = anthropic_body.get("system")
    if system:
        if isinstance(system, str):
            system_parts.append({"text": system})
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_parts.append({"text": block.get("text", "")})

    # Build a mapping of tool_use_id → function name from assistant messages
    tool_name_by_id: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_name_by_id[block.get("id", "")] = block.get("name", "")

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            if isinstance(content, list):
                parts: list[dict[str, Any]] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append({"text": block.get("text", "")})
                    elif btype == "thinking":
                        parts.append({"thought": True, "text": block.get("thinking", "")})
                    elif btype == "tool_use":
                        fc_part = {
                            "name": block.get("name", ""),
                            "args": block.get("input", {}),
                        }
                        if block.get("id"):
                            fc_part["id"] = block["id"]
                        parts.append({"functionCall": fc_part})
                if parts:
                    gemini_contents = _gemini_append_or_merge(gemini_contents, "model", parts)
            elif isinstance(content, str) and content:
                gemini_contents = _gemini_append_or_merge(
                    gemini_contents, "model", [{"text": content}],
                )

        elif role == "user":
            if isinstance(content, list):
                parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append({"text": block.get("text", "")})
                    elif btype == "tool_result":
                        result_content = block.get("content", "")
                        if isinstance(result_content, list):
                            result_content = "\n".join(
                                b.get("text", "") for b in result_content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        text = result_content or ""
                        try:
                            parsed = json.loads(text)
                            if isinstance(parsed, dict):
                                response_data = parsed
                            else:
                                response_data = {"result": text}
                        except (json.JSONDecodeError, ValueError):
                            response_data = {"result": text}

                        tool_use_id = block.get("tool_use_id")
                        func_name = tool_name_by_id.get(tool_use_id or "") or tool_use_id or "unknown"
                        func_resp = {
                            "name": func_name,
                            "response": response_data,
                        }
                        if tool_use_id:
                            func_resp["id"] = tool_use_id
                        parts.append({
                            "functionResponse": func_resp
                        })
                    elif btype == "image":
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            parts.append({
                                "inlineData": {
                                    "mimeType": source.get("media_type", "image/png"),
                                    "data": source.get("data", ""),
                                },
                            })
                        elif source.get("url"):
                            parts.append({
                                "fileData": {
                                    "mimeType": "image/png",
                                    "fileUri": source["url"],
                                },
                            })
                if parts:
                    gemini_contents = _gemini_append_or_merge(gemini_contents, "user", parts)
            elif isinstance(content, str):
                gemini_contents = _gemini_append_or_merge(
                    gemini_contents, "user", [{"text": content}],
                )

    if not gemini_contents:
        raise ValueError("At least one message is required")

    body: dict[str, Any] = {"contents": gemini_contents}
    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}

    gen_config: dict[str, Any] = {}
    max_tokens = anthropic_body.get("max_tokens") or default_max_tokens
    if max_tokens and max_tokens > 0:
        gen_config["maxOutputTokens"] = max_tokens
    if anthropic_body.get("temperature") is not None:
        gen_config["temperature"] = anthropic_body["temperature"]
    if anthropic_body.get("top_p") is not None:
        gen_config["topP"] = anthropic_body["top_p"]
    stop_seq = anthropic_body.get("stop_sequences")
    if isinstance(stop_seq, list) and stop_seq:
        gen_config["stopSequences"] = stop_seq

    thinking = anthropic_body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "enabled":
            budget = thinking.get("budget_tokens", 4096)
            gen_config["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": budget}

    if gen_config:
        body["generationConfig"] = gen_config

    tools = anthropic_body.get("tools")
    if tools:
        gemini_tools = _anthropic_tools_to_gemini(tools)
        if gemini_tools:
            body["tools"] = gemini_tools
        
        mode = "AUTO"
        allowed_names = None
        
        choice = anthropic_body.get("tool_choice")
        if choice is not None:
            if isinstance(choice, dict):
                ctype = choice.get("type", "")
                if ctype == "none":
                    mode = "NONE"
                elif ctype == "any":
                    mode = "ANY"
                elif ctype == "auto":
                    mode = "AUTO"
                elif ctype == "tool":
                    name = choice.get("name", "")
                    if name == "google_search":
                        name = "google_search_proxy_renamed"
                    if name:
                        mode = "ANY"
                        allowed_names = [name]
                        
        tc_camel = build_gemini_tool_config(mode, allowed_names)
        body["toolConfig"] = tc_camel

    body["safetySettings"] = DEFAULT_GEMINI_SAFETY_SETTINGS
    return body, model


# ══════════════════════════════════════════════════════════════════════════════
# Gemini → Anthropic
# ══════════════════════════════════════════════════════════════════════════════


def _gemini_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Gemini tools to Anthropic tool definitions."""
    result: list[dict[str, Any]] = []
    for tool in tools or []:
        decls = tool.get("functionDeclarations", [])
        for decl in decls:
            name = decl.get("name", "")
            if not name:
                continue
            if name == "google_search_proxy_renamed":
                name = "google_search"
            params = decl.get("parameters")
            if params:
                params = _gemini_schema_to_openai_schema(params)
            result.append({
                "name": name,
                "description": decl.get("description", ""),
                "input_schema": params or {"type": "object", "properties": {}},
            })
    return result


def translate_gemini_to_anthropic(
    gemini_body: dict[str, Any],
    *,
    model_prefix: str = "",
    default_max_tokens: int = 8192,
) -> tuple[dict[str, Any], str]:
    """Translate a Gemini GenerateContent request to Anthropic Messages format.

    Returns (anthropic_body, model_name).
    """
    model = gemini_body.get("model", "")
    if model_prefix and model.startswith(model_prefix):
        model = model[len(model_prefix):]

    contents = gemini_body.get("contents", [])
    if not contents:
        raise ValueError("contents array is required")

    system_blocks: list[dict[str, Any]] = []
    anthropic_messages: list[dict[str, Any]] = []

    sys_inst = gemini_body.get("systemInstruction")
    if sys_inst:
        for part in sys_inst.get("parts", []):
            if isinstance(part, dict) and "text" in part:
                system_blocks.append({"type": "text", "text": part["text"]})

    # Pre-scan: build function name → tool_use_id mapping for consistent IDs
    _func_name_to_tool_id: dict[str, str] = {}
    for content in contents:
        for part in content.get("parts", []):
            if isinstance(part, dict) and "functionCall" in part:
                fc = part["functionCall"]
                name = fc.get("name", "")
                if name and name not in _func_name_to_tool_id:
                    _func_name_to_tool_id[name] = fc.get("id") or f"toolu_{_random_hex(12)}"

    for content in contents:
        role = content.get("role", "")
        parts = content.get("parts", [])

        if role == "user":
            blocks: list[dict[str, Any]] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if "functionResponse" in part:
                    fr = part["functionResponse"]
                    func_name = fr.get("name", "")
                    if func_name == "google_search_proxy_renamed":
                        func_name = "google_search"
                    resp = fr.get("response", {})
                    if "result" in resp and len(resp) == 1:
                        result_val = resp["result"]
                        if isinstance(result_val, (dict, list)):
                            result_text = json.dumps(result_val, ensure_ascii=False)
                        else:
                            result_text = str(result_val)
                    else:
                        result_text = json.dumps(resp, ensure_ascii=False)
                    tool_use_id = fr.get("id") or _func_name_to_tool_id.get(func_name)
                    if not tool_use_id:
                        tool_use_id = f"toolu_{_random_hex(12)}"
                        _func_name_to_tool_id[func_name] = tool_use_id
                    anthropic_messages = _append_or_merge(
                        anthropic_messages, "user",
                        [{"type": "tool_result", "tool_use_id": tool_use_id, "content": str(result_text)}],
                    )
                elif "text" in part:
                    blocks.append({"type": "text", "text": part["text"]})
                elif "inlineData" in part:
                    data = part["inlineData"]
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": data.get("mimeType", "image/png"),
                            "data": data.get("data", ""),
                        },
                    })
            if blocks:
                anthropic_messages = _append_or_merge(anthropic_messages, "user", blocks)

        elif role == "model":
            blocks = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("thought"):
                    blocks.append({"type": "thinking", "thinking": part.get("text", "")})
                elif "text" in part:
                    blocks.append({"type": "text", "text": part["text"]})
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    func_name = fc.get("name", "")
                    if func_name == "google_search_proxy_renamed":
                        func_name = "google_search"
                    tool_use_id = fc.get("id") or _func_name_to_tool_id.get(func_name)
                    if not tool_use_id:
                        tool_use_id = f"toolu_{_random_hex(12)}"
                        _func_name_to_tool_id[func_name] = tool_use_id
                    blocks.append({
                        "type": "tool_use",
                        "id": tool_use_id,
                        "name": func_name,
                        "input": fc.get("args", {}),
                    })
            if blocks:
                anthropic_messages = _append_or_merge(anthropic_messages, "assistant", blocks)

    if not anthropic_messages:
        raise ValueError("At least one message is required")

    body: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": default_max_tokens,
        "stream": True,
    }
    if system_blocks:
        body["system"] = system_blocks

    gen_config = gemini_body.get("generationConfig", {})
    if gen_config:
        max_tokens = gen_config.get("maxOutputTokens")
        if max_tokens and max_tokens > 0:
            body["max_tokens"] = max_tokens
        if gen_config.get("temperature") is not None:
            body["temperature"] = gen_config["temperature"]
        if gen_config.get("topP") is not None:
            body["top_p"] = gen_config["topP"]
        stop = gen_config.get("stopSequences")
        if isinstance(stop, list) and stop:
            body["stop_sequences"] = stop

        thinking_config = gen_config.get("thinkingConfig")
        if isinstance(thinking_config, dict) and thinking_config.get("includeThoughts"):
            budget = thinking_config.get("thinkingBudget", 4096)
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}

    tools = gemini_body.get("tools")
    if tools:
        anthropic_tools = _gemini_tools_to_anthropic(tools)
        if anthropic_tools:
            body["tools"] = anthropic_tools

    return body, model


# ══════════════════════════════════════════════════════════════════════════════
# Gemini ↔ OpenAI Responses (two-hop via Chat)
# ══════════════════════════════════════════════════════════════════════════════


def translate_gemini_to_responses(
    gemini_body: dict[str, Any],
    *,
    model_prefix: str = "",
) -> dict[str, Any]:
    """Translate a Gemini request to OpenAI Responses API format (via Chat)."""
    from protocol_adapter import translate_openai_chat_to_responses
    openai_body, _model = translate_gemini_to_openai(gemini_body, model_prefix=model_prefix)
    return translate_openai_chat_to_responses(openai_body)


def translate_responses_to_gemini(
    responses_body: dict[str, Any],
    *,
    model_prefix: str = "",
    default_max_tokens: int = 8192,
) -> tuple[dict[str, Any], str]:
    """Translate an OpenAI Responses API request to Gemini format (via Chat)."""
    from protocol_adapter import translate_openai_responses_to_chat
    chat_body = translate_openai_responses_to_chat(responses_body)
    return translate_openai_to_gemini(chat_body, model_prefix=model_prefix, default_max_tokens=default_max_tokens)


# ══════════════════════════════════════════════════════════════════════════════
# Gemini → OpenAI response translation (for non-streaming proxy)
# ══════════════════════════════════════════════════════════════════════════════


def translate_gemini_response_to_openai(gemini_data: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate a complete Gemini response to OpenAI Chat Completions format."""
    candidates = gemini_data.get("candidates", [])
    usage_meta = gemini_data.get("usageMetadata", {})

    usage = {
        "prompt_tokens": usage_meta.get("promptTokenCount", 0),
        "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
        "total_tokens": usage_meta.get("totalTokenCount", 0),
    }

    if not candidates:
        return {
            "id": f"chatcmpl-{_random_hex(12)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [],
            "usage": usage,
        }

    candidate = candidates[0]
    content_parts = candidate.get("content", {}).get("parts", [])
    finish_reason_raw = candidate.get("finishReason", "STOP")

    content_text, tool_calls, reasoning_details = _gemini_parts_to_openai_content(content_parts)
    finish_reason = GEMINI_FINISH_MAP.get(finish_reason_raw, "stop")
    if tool_calls and not content_text:
        finish_reason = "tool_calls"

    message: dict[str, Any] = {"role": "assistant"}
    message["content"] = content_text
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_details:
        message["reasoning_details"] = reasoning_details

    return {
        "id": f"chatcmpl-{_random_hex(12)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Gemini → Anthropic response translation (for non-streaming proxy)
# ══════════════════════════════════════════════════════════════════════════════


def translate_gemini_response_to_anthropic(gemini_data: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate a complete Gemini response to Anthropic Messages format."""
    candidates = gemini_data.get("candidates", [])
    usage_meta = gemini_data.get("usageMetadata", {})

    input_tokens = usage_meta.get("promptTokenCount", 0)
    output_tokens = usage_meta.get("candidatesTokenCount", 0)

    content_blocks: list[dict[str, Any]] = []
    stop_reason = "end_turn"

    if candidates:
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        finish_raw = candidate.get("finishReason", "STOP")

        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("thought"):
                content_blocks.append({"type": "thinking", "thinking": part.get("text", "")})
            elif "text" in part:
                content_blocks.append({"type": "text", "text": part["text"]})
            elif "functionCall" in part:
                fc = part["functionCall"]
                func_name = fc.get("name", "")
                if func_name == "google_search_proxy_renamed":
                    func_name = "google_search"
                elif func_name == "web_search_proxy_renamed":
                    func_name = "web_search"
                content_blocks.append({
                    "type": "tool_use",
                    "id": fc.get("id") or f"toolu_{_random_hex(12)}",
                    "name": func_name,
                    "input": fc.get("args", {}),
                })

        stop_map = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens", "SAFETY": "end_turn"}
        stop_reason = stop_map.get(finish_raw, "end_turn")

    msg_id = f"msg_{_random_hex(12)}"
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


# ══════════════════════════════════════════════════════════════════════════════
# Streaming decoders: Gemini SSE → OpenAI chunks
# ══════════════════════════════════════════════════════════════════════════════


class GeminiStreamDecoder:
    """Parses Gemini SSE response objects and yields OpenAI-format chunks.

    Gemini streaming returns full JSON objects per SSE event.
    """

    def __init__(self, model: str, *, request_id: Optional[str] = None):
        self.model = model
        self.chunk_id = request_id or f"chatcmpl-{_random_hex(12)}"
        self.created = int(time.time())
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._reasoning_details: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._finish_reason: str = ""
        self._usage: Optional[dict[str, Any]] = None

    def process_sse_data(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Process a single Gemini SSE data object. Returns list of OpenAI chunks."""
        chunks: list[dict[str, Any]] = []

        usage_meta = data.get("usageMetadata")
        if usage_meta:
            self._usage = {
                "prompt_tokens": usage_meta.get("promptTokenCount", 0),
                "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
                "total_tokens": usage_meta.get("totalTokenCount", 0),
            }

        candidates = data.get("candidates", [])
        if not candidates:
            return chunks

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        finish_reason = candidate.get("finishReason", "")

        for part in parts:
            if not isinstance(part, dict):
                continue

            if part.get("thought"):
                text = part.get("text", "")
                if text:
                    self._reasoning_parts.append(text)
                    chunks.append(_build_openai_chunk(
                        self.chunk_id, self.created, self.model,
                        {"reasoning_details": [{"type": "reasoning.text", "text": text}]},
                    ))
                continue

            if "text" in part:
                text = part["text"]
                if text:
                    self._content_parts.append(text)
                    chunks.append(_build_openai_chunk(
                        self.chunk_id, self.created, self.model,
                        {"content": text},
                    ))

            if "functionCall" in part:
                fc = part["functionCall"]
                call_id = fc.get("id") or f"call_{_random_hex(12)}"
                name = fc.get("name", "")
                if name == "google_search_proxy_renamed":
                    name = "google_search"
                args = fc.get("args", {})
                tc = {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                    },
                }
                self._tool_calls.append(tc)
                chunks.append(_build_openai_chunk(
                    self.chunk_id, self.created, self.model,
                    {"tool_calls": [{"index": len(self._tool_calls) - 1, **tc}]},
                ))

        if finish_reason:
            mapped = GEMINI_FINISH_MAP.get(finish_reason, "stop")
            self._finish_reason = mapped

        return chunks

    def build_final_response(self) -> dict[str, Any]:
        """Build complete OpenAI response from accumulated state."""
        content = "".join(self._content_parts) if self._content_parts else None
        tool_calls = self._tool_calls if self._tool_calls else None

        # Populate reasoning_details from accumulated reasoning_parts
        if self._reasoning_parts and not self._reasoning_details:
            self._reasoning_details = [{"type": "reasoning.text", "text": "".join(self._reasoning_parts)}]

        finish = self._finish_reason or "stop"
        if tool_calls and not content:
            finish = "tool_calls"

        return _build_openai_response(
            self.chunk_id, self.created, self.model,
            content, tool_calls, finish, self._usage,
            reasoning_details=self._reasoning_details or None,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Streaming decoders: Gemini SSE → Anthropic SSE
# ══════════════════════════════════════════════════════════════════════════════


class GeminiAnthropicStreamDecoder:
    """Parses Gemini SSE events and yields Anthropic Messages SSE events."""

    def __init__(self, model: str):
        self.model = model
        self._msg_id = f"msg_{_random_hex(12)}"
        self._started = False
        self._block_index = 0
        self._block_started = False
        self._finished = False
        self._input_tokens = 0
        self._output_tokens = 0
        self._stop_reason = "end_turn"

    def _emit_event(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def start_message(self) -> str:
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

    def _close_current_block(self) -> str:
        if not self._block_started:
            return ""
        result = self._emit_event("content_block_stop", {
            "type": "content_block_stop",
            "index": self._block_index,
        })
        self._block_index += 1
        self._block_started = False
        return result

    def process_sse_data(self, data: dict[str, Any]) -> str:
        """Process a Gemini SSE data object. Returns Anthropic SSE events string."""
        result = ""

        if not self._started:
            usage_meta = data.get("usageMetadata")
            if usage_meta:
                self._input_tokens = usage_meta.get("promptTokenCount", 0)
            result += self.start_message()

        candidates = data.get("candidates", [])
        if not candidates:
            return result

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        finish_reason = candidate.get("finishReason", "")

        for part in parts:
            if not isinstance(part, dict):
                continue

            if part.get("thought"):
                if not self._block_started:
                    result += self._emit_event("content_block_start", {
                        "type": "content_block_start",
                        "index": self._block_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    })
                    self._block_started = True
                text = part.get("text", "")
                if text:
                    result += self._emit_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {"type": "thinking_delta", "thinking": text},
                    })
                continue

            if "text" in part:
                text = part["text"]
                if text:
                    if not self._block_started:
                        result += self._emit_event("content_block_start", {
                            "type": "content_block_start",
                            "index": self._block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                        self._block_started = True
                    result += self._emit_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {"type": "text_delta", "text": text},
                    })

            if "functionCall" in part:
                fc = part["functionCall"]
                call_id = fc.get("id") or f"toolu_{_random_hex(12)}"
                name = fc.get("name", "")
                if name == "google_search_proxy_renamed":
                    name = "google_search"
                elif name == "web_search_proxy_renamed":
                    name = "web_search"
                args = fc.get("args", {})
                result += self._close_current_block()
                result += self._emit_event("content_block_start", {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {"type": "tool_use", "id": call_id, "name": name, "input": {}},
                })
                result += self._emit_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self._block_index,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(args, ensure_ascii=False)},
                })
                result += self._emit_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self._block_index,
                })
                self._block_index += 1

        if finish_reason:
            result += self._close_current_block()
            stop_map = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens", "SAFETY": "end_turn"}
            self._stop_reason = stop_map.get(finish_reason, "end_turn")
            usage_meta = data.get("usageMetadata")
            if usage_meta:
                self._output_tokens = usage_meta.get("candidatesTokenCount", 0)
                self._input_tokens = usage_meta.get("promptTokenCount", self._input_tokens)

        return result

    def finish_message(self) -> str:
        if self._finished:
            return ""
        self._finished = True
        result = self._emit_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": self._stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": self._output_tokens},
        })
        result += self._emit_event("message_stop", {"type": "message_stop"})
        return result


# ══════════════════════════════════════════════════════════════════════════════
# SSE translators: OpenAI/Anthropic chunks → Gemini SSE response objects
# ══════════════════════════════════════════════════════════════════════════════


class OpenAIToGeminiSSETranslator:
    """Translates OpenAI Chat Completions SSE chunks → Gemini SSE response objects."""

    def __init__(self, model: str):
        self.model = model
        self._text_parts: list[str] = []
        self._thought_parts: list[str] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._pending_tool_calls: dict[int, dict[str, Any]] = {}
        self._finished = False
        self._finish_reason = "STOP"
        self._input_tokens = 0
        self._output_tokens = 0

    def translate_chunk(self, openai_chunk: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Translate an OpenAI SSE chunk to a Gemini response object (or None)."""
        choices = openai_chunk.get("choices", [])
        usage = openai_chunk.get("usage")
        if usage:
            self._input_tokens = usage.get("prompt_tokens", self._input_tokens)
            self._output_tokens = usage.get("completion_tokens", self._output_tokens)

        if not choices:
            return None

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        parts: list[dict[str, Any]] = []

        content = delta.get("content")
        if content:
            self._text_parts.append(content)
            parts.append({"text": content})

        reasoning_details = delta.get("reasoning_details")
        if reasoning_details:
            for rd in reasoning_details:
                if isinstance(rd, dict) and rd.get("type") == "reasoning.text":
                    text = rd.get("text", "")
                    if text:
                        self._thought_parts.append(text)
                        parts.append({"thought": True, "text": text})

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                idx = tc.get("index", 0)
                func = tc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "")

                if idx not in self._pending_tool_calls:
                    self._pending_tool_calls[idx] = {"name": "", "args_buf": "", "id": tc.get("id", "")}

                if name:
                    self._pending_tool_calls[idx]["name"] = name
                if tc.get("id"):
                    self._pending_tool_calls[idx]["id"] = tc.get("id")
                if args_str:
                    self._pending_tool_calls[idx]["args_buf"] += args_str

        if finish_reason:
            finish_map = {"stop": "STOP", "length": "MAX_TOKENS", "tool_calls": "STOP", "content_filter": "SAFETY"}
            self._finish_reason = finish_map.get(finish_reason, "STOP")
            self._finished = True
            # Flush pending tool calls
            for idx in sorted(self._pending_tool_calls.keys()):
                pending = self._pending_tool_calls[idx]
                name = pending["name"]
                if name:
                    try:
                        args = json.loads(pending["args_buf"]) if pending["args_buf"] else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tc_obj: dict[str, Any] = {"name": name, "args": args}
                    if pending.get("id"):
                        tc_obj["id"] = pending["id"]
                    self._tool_calls.append(tc_obj)
                    parts.append({"functionCall": tc_obj})
            self._pending_tool_calls.clear()

        if not parts:
            if finish_reason:
                # Emit a minimal response with just the finish reason
                return {
                    "candidates": [{
                        "content": {"role": "model", "parts": [{"text": ""}]},
                        "finishReason": self._finish_reason,
                        "index": 0,
                    }],
                    "usageMetadata": {
                        "promptTokenCount": self._input_tokens,
                        "candidatesTokenCount": self._output_tokens,
                        "totalTokenCount": self._input_tokens + self._output_tokens,
                    },
                    "modelVersion": self.model,
                }
            return None

        return {
            "candidates": [{
                "content": {"role": "model", "parts": parts},
                "finishReason": self._finish_reason if self._finished else None,
                "index": 0,
            }],
            "usageMetadata": {
                "promptTokenCount": self._input_tokens,
                "candidatesTokenCount": self._output_tokens,
                "totalTokenCount": self._input_tokens + self._output_tokens,
            },
            "modelVersion": self.model,
        }

    def build_final_response(self) -> dict[str, Any]:
        """Build complete Gemini response from accumulated state."""
        parts: list[dict[str, Any]] = []
        for text in self._thought_parts:
            parts.append({"thought": True, "text": text})
        if self._text_parts:
            parts.append({"text": "".join(self._text_parts)})
        for tc in self._tool_calls:
            parts.append({"functionCall": tc})

        return {
            "candidates": [{
                "content": {"role": "model", "parts": parts or [{"text": ""}]},
                "finishReason": self._finish_reason,
                "index": 0,
            }],
            "usageMetadata": {
                "promptTokenCount": self._input_tokens,
                "candidatesTokenCount": self._output_tokens,
                "totalTokenCount": self._input_tokens + self._output_tokens,
            },
            "modelVersion": self.model,
        }


class AnthropicToGeminiSSETranslator:
    """Translates Anthropic SSE events → Gemini SSE response objects."""

    def __init__(self, model: str):
        self.model = model
        self._text_parts: list[str] = []
        self._thought_parts: list[str] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._pending_tool_name: str = ""
        self._pending_tool_args_buf: str = ""
        self._pending_tool_id: str = ""
        self._finished = False
        self._finish_reason = "STOP"
        self._input_tokens = 0
        self._output_tokens = 0

    def translate_event(self, event_type: str, data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Translate a single Anthropic SSE event to a Gemini response object (or None)."""
        if event_type == "message_start":
            msg = data.get("message", {})
            usage = msg.get("usage", {})
            self._input_tokens = usage.get("input_tokens", 0)
            return None

        if event_type == "content_block_start":
            block = data.get("content_block", {})
            if block.get("type") == "tool_use":
                self._pending_tool_name = block.get("name", "")
                self._pending_tool_id = block.get("id", "")
                self._pending_tool_args_buf = ""
            return None

        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "thinking_delta":
                text = delta.get("thinking", "")
                if text:
                    self._thought_parts.append(text)
                    return {
                        "candidates": [{"content": {"role": "model", "parts": [{"thought": True, "text": text}]}, "index": 0}],
                        "usageMetadata": {
                            "promptTokenCount": self._input_tokens,
                            "candidatesTokenCount": self._output_tokens,
                            "totalTokenCount": self._input_tokens + self._output_tokens,
                        },
                        "modelVersion": self.model,
                    }
            elif delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    self._text_parts.append(text)
                    return {
                        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "index": 0}],
                        "usageMetadata": {
                            "promptTokenCount": self._input_tokens,
                            "candidatesTokenCount": self._output_tokens,
                            "totalTokenCount": self._input_tokens + self._output_tokens,
                        },
                        "modelVersion": self.model,
                    }
            elif delta_type == "input_json_delta":
                self._pending_tool_args_buf += delta.get("partial_json", "")
            return None

        if event_type == "content_block_stop":
            if self._pending_tool_name:
                try:
                    args = json.loads(self._pending_tool_args_buf) if self._pending_tool_args_buf else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tc: dict[str, Any] = {"name": self._pending_tool_name, "args": args}
                if self._pending_tool_id:
                    tc["id"] = self._pending_tool_id
                self._tool_calls.append(tc)
                self._pending_tool_name = ""
                self._pending_tool_args_buf = ""
                self._pending_tool_id = ""
            return None

        if event_type == "message_delta":
            delta = data.get("delta", {})
            stop_reason = delta.get("stop_reason", "")
            stop_map = {"end_turn": "STOP", "max_tokens": "MAX_TOKENS", "tool_use": "STOP"}
            self._finish_reason = stop_map.get(stop_reason, "STOP")
            usage = data.get("usage", {})
            self._output_tokens = usage.get("output_tokens", self._output_tokens)
            return None

        if event_type == "message_stop":
            self._finished = True
            return None

        return None

    def build_final_response(self) -> dict[str, Any]:
        """Build complete Gemini response from accumulated state."""
        parts: list[dict[str, Any]] = []
        for text in self._thought_parts:
            parts.append({"thought": True, "text": text})
        if self._text_parts:
            parts.append({"text": "".join(self._text_parts)})
        for tc in self._tool_calls:
            parts.append({"functionCall": tc})

        return {
            "candidates": [{
                "content": {"role": "model", "parts": parts or [{"text": ""}]},
                "finishReason": self._finish_reason,
                "index": 0,
            }],
            "usageMetadata": {
                "promptTokenCount": self._input_tokens,
                "candidatesTokenCount": self._output_tokens,
                "totalTokenCount": self._input_tokens + self._output_tokens,
            },
            "modelVersion": self.model,
        }
