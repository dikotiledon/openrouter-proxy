# Multi-Protocol Translation Matrix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full bidirectional protocol translation between OpenAI Chat Completions, OpenAI Responses API, Anthropic Messages API, and Gemini GenerateContent API (v1beta), enabling any client format to talk to any upstream format through the proxy.

**Architecture:** A 4×4 translation matrix where each client-facing protocol can connect to each upstream-facing protocol. The existing `protocol_adapter.py` handles OpenAI↔Anthropic. New `gemini_adapter.py` handles Gemini translations. `routes.py` gets Gemini client endpoints and upstream Gemini proxy logic. Each direction is a standalone translation function (pure, testable).

**Tech Stack:** Python 3.11+, FastAPI, httpx, PyYAML. No new dependencies.

---

## Translation Matrix (16 cells)

| Client ↓ \ Upstream → | OpenAI Chat | OpenAI Responses | Anthropic | Gemini |
|---|---|---|---|---|
| **OpenAI Chat** | passthrough | chat→responses ✗ | chat→anthropic ✓ | chat→gemini ✗ |
| **OpenAI Responses** | responses→chat ✗ | passthrough ✓ | responses→anthropic ✗ | responses→gemini ✗ |
| **Anthropic** | anthropic→chat ✓ | anthropic→responses ✗ | passthrough ✓ | anthropic→gemini ✗ |
| **Gemini** | gemini→chat ✗ | gemini→responses ✗ | gemini→anthropic ✗ | passthrough ✗ |

✓ = already implemented, ✗ = new work needed (12 new translation paths)

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `gemini_adapter.py` | **Create** | Gemini↔OpenAI and Gemini↔Anthropic translation functions + Gemini stream decoder |
| `protocol_adapter.py` | **Modify** | Add OpenAI Chat↔Responses translation, Anthropic↔Responses translation |
| `routes.py` | **Modify** | Add Gemini client endpoints, Gemini upstream proxy, Responses↔other protocol routing |
| `provider_registry.py` | **Modify** | Add `"gemini"` as valid `upstream_format`, Gemini-specific config fields |
| `config.yml.example` | **Modify** | Add Gemini provider example |
| `constants.py` | **Modify** | Add Gemini-related constants |
| `tests/test_gemini_adapter.py` | **Create** | Unit tests for all Gemini translation functions |
| `tests/test_chat_responses_translation.py` | **Create** | Unit tests for Chat↔Responses translation |
| `tests/test_full_matrix.py` | **Create** | Integration tests covering all 16 matrix cells |

---

## Gemini API Reference (v1beta)

### Request Format
```json
{
  "contents": [
    {"role": "user", "parts": [{"text": "Hello"}]},
    {"role": "model", "parts": [{"text": "Hi there!"}]}
  ],
  "systemInstruction": {"parts": [{"text": "You are a helpful assistant."}]},
  "generationConfig": {
    "temperature": 1.0,
    "topP": 0.95,
    "topK": 40,
    "candidateCount": 1,
    "maxOutputTokens": 8192,
    "stopSequences": ["END"],
    "responseMimeType": "application/json",
    "thinkingConfig": {"includeThoughts": true, "thinkingBudget": 4096}
  },
  "tools": [{"functionDeclarations": [{"name": "fn", "description": "desc", "parameters": {...}}]}],
  "toolConfig": {"functionCallingConfig": {"mode": "AUTO", "allowedFunctionNames": ["fn"]}},
  "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
}
```

### Response Format (non-streaming)
```json
{
  "candidates": [{
    "content": {"role": "model", "parts": [
      {"text": "Hello!"},
      {"functionCall": {"name": "fn", "args": {"key": "val"}}}
    ]},
    "finishReason": "STOP",
    "index": 0
  }],
  "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
  "modelVersion": "gemini-2.0-flash"
}
```

### Streaming (SSE)
Endpoint: `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`
Returns SSE events where each `data:` line contains a full Gemini response JSON object (same schema as non-streaming, but partial content deltas).

### Thinking Parts
When `thinkingConfig.includeThoughts: true`, parts can include:
```json
{"thought": true, "text": "Let me reason about this..."}
```

### Function Calling
```json
// Request tools:
{"functionDeclarations": [{"name": "get_weather", "description": "...", "parameters": {"type": "OBJECT", "properties": {"loc": {"type": "STRING"}}}}]}
// Response function call:
{"functionCall": {"name": "get_weather", "args": {"loc": "NYC"}}}
// Function response (in contents):
{"role": "function", "parts": [{"functionResponse": {"name": "get_weather", "response": {"result": "sunny"}}}]}
```

---

## Task 1: Gemini Constants and Schema Helpers

**Files:**
- Create: `gemini_adapter.py`
- Modify: `constants.py`

- [ ] **Step 1: Add Gemini constants to constants.py**

```python
# Add to constants.py

# Gemini API
GEMINI_GENERATE_ENDPOINT = ":generateContent"
GEMINI_STREAM_ENDPOINT = ":streamGenerateContent"
GEMINI_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
}
```

- [ ] **Step 2: Create gemini_adapter.py with constants and pure helpers**

```python
#!/usr/bin/env python3
"""
Bidirectional Gemini ↔ OpenAI/Anthropic protocol adapter.

Supports the full translation matrix for Gemini GenerateContent API (v1beta):
  - OpenAI Chat → Gemini: translate_openai_to_gemini()
  - Anthropic → Gemini: translate_anthropic_to_gemini()
  - Gemini → OpenAI Chat: translate_gemini_to_openai()
  - Gemini → Anthropic: translate_gemini_to_anthropic()
  - Both directions support streaming and non-streaming.
"""

import json
import time
from typing import Any, Optional

# Gemini finish_reason → OpenAI finish_reason mapping
GEMINI_FINISH_MAP: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
    "FINISH_REASON_STOP": "stop",
    "FINISH_REASON_UNSPECIFIED": "stop",
}

# OpenAI finish_reason → Gemini finishReason mapping
OPENAI_TO_GEMINI_FINISH: dict[str, str] = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "tool_calls": "STOP",
    "content_filter": "SAFETY",
}

# Anthropic stop_reason → Gemini finishReason
ANTHROPIC_TO_GEMINI_FINISH: dict[str, str] = {
    "end_turn": "STOP",
    "stop_sequence": "STOP",
    "max_tokens": "MAX_TOKENS",
    "tool_use": "STOP",
}

# Gemini safety category constants
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
```

- [ ] **Step 3: Run syntax check**

Run: `python -c "import gemini_adapter"`
Expected: No error

---

## Task 2: OpenAI Chat → Gemini Translation

**Files:**
- Modify: `gemini_adapter.py`

- [ ] **Step 1: Implement translate_openai_to_gemini()**

```python
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
                else:
                    parts.append({"fileData": {"mimeType": "image/png", "fileUri": url}})
            else:
                parts.append({"text": json.dumps(part, ensure_ascii=False)})
        return parts if parts else [{"text": ""}]
    return [{"text": str(content)}]


def _openai_tool_calls_to_gemini(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI tool_calls from assistant message to Gemini functionCall parts."""
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
            parts.append({"functionCall": {"name": name, "args": args if isinstance(args, dict) else {}}})
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


def _openai_schema_to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI JSON Schema type names to Gemini schema type names."""
    if not isinstance(schema, dict):
        return schema
    result = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            result["type"] = v.upper()
        elif k == "properties" and isinstance(v, dict):
            result["properties"] = {pk: _openai_schema_to_gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            result["items"] = _openai_schema_to_gemini_schema(v)
        elif k == "anyOf" and isinstance(v, list):
            result["anyOf"] = [_openai_schema_to_gemini_schema(item) for item in v]
        else:
            result[k] = v
    return result


def _gemini_schema_to_openai_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert Gemini schema type names back to OpenAI JSON Schema type names."""
    if not isinstance(schema, dict):
        return schema
    result = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            result["type"] = v.lower()
        elif k == "properties" and isinstance(v, dict):
            result["properties"] = {pk: _gemini_schema_to_openai_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            result["items"] = _gemini_schema_to_openai_schema(v)
        elif k == "anyOf" and isinstance(v, list):
            result["anyOf"] = [_gemini_schema_to_openai_schema(item) for item in v]
        else:
            result[k] = v
    return result


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

    # Separate system messages from conversation
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
                parts.extend(_openai_tool_calls_to_gemini(tool_calls))
            gemini_contents = _gemini_append_or_merge(gemini_contents, "model", parts)
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
            response_data = {"result": text}
            if tool_call_id:
                response_data["tool_call_id"] = tool_call_id
            gemini_contents = _gemini_append_or_merge(
                gemini_contents, "user",
                [{"functionResponse": {"name": msg.get("name", "unknown"), "response": response_data}}],
            )
        else:
            # user
            gemini_contents = _gemini_append_or_merge(gemini_contents, "user", _openai_content_to_parts(content))

    if not gemini_contents:
        raise ValueError("At least one non-system message is required")

    body: dict[str, Any] = {
        "contents": gemini_contents,
    }

    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}

    # Generation config
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

    # Reasoning / thinking
    reasoning = openai_body.get("reasoning")
    if isinstance(reasoning, dict):
        budget = reasoning.get("max_tokens")
        if reasoning.get("enabled") is False:
            pass  # no thinking config needed
        elif budget and isinstance(budget, (int, float)) and budget > 0:
            gen_config["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": int(budget)}
        elif reasoning.get("effort"):
            from protocol_adapter import REASONING_EFFORT_TO_BUDGET
            effort = reasoning["effort"]
            b = REASONING_EFFORT_TO_BUDGET.get(effort, 4096)
            gen_config["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": b}

    if gen_config:
        body["generationConfig"] = gen_config

    # Tools
    tools = openai_body.get("tools")
    if tools:
        gemini_tools = _openai_tools_to_gemini(tools)
        if gemini_tools:
            body["tools"] = gemini_tools
        tc = _openai_tool_choice_to_gemini(openai_body.get("tool_choice"))
        if tc is not None:
            body["toolConfig"] = tc

    # Safety settings (permissive defaults for proxy usage)
    body["safetySettings"] = DEFAULT_SAFETY_SETTINGS

    return body, model


def _gemini_append_or_merge(
    contents: list[dict[str, Any]],
    role: str,
    parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append a content entry or merge with last if same role (Gemini requires alternating roles)."""
    if contents and contents[-1]["role"] == role:
        contents[-1]["parts"] = list(contents[-1]["parts"]) + parts
        return contents
    return contents + [{"role": role, "parts": parts}]
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_gemini_adapter.py`:

```python
"""Tests for Gemini protocol adapter."""
import json
import pytest
from gemini_adapter import (
    translate_openai_to_gemini,
    translate_gemini_to_openai,
    translate_anthropic_to_gemini,
    translate_gemini_to_anthropic,
    GeminiStreamDecoder,
    OpenAIToGeminiSSETranslator,
    AnthropicToGeminiSSETranslator,
)


class TestOpenAIToGemini:
    def test_basic_translation(self):
        body = {
            "model": "provider/gemini-2.0-flash",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        result, model = translate_openai_to_gemini(body, model_prefix="provider/")
        assert model == "gemini-2.0-flash"
        assert result["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
        assert result["systemInstruction"] == {"parts": [{"text": "You are helpful."}]}
        assert result["generationConfig"]["maxOutputTokens"] == 1024
        assert result["generationConfig"]["temperature"] == 0.7

    def test_multimodal_content(self):
        body = {
            "model": "test/model",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                ],
            }],
        }
        result, _ = translate_openai_to_gemini(body, inject_billing=False)
        parts = result["contents"][0]["parts"]
        assert parts[0] == {"text": "What is this?"}
        assert parts[1]["inlineData"]["mimeType"] == "image/png"
        assert parts[1]["inlineData"]["data"] == "abc123"

    def test_tool_calls_in_assistant_message(self):
        body = {
            "model": "test/model",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location":"NYC"}'},
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"temperature": 72}',
                    "name": "get_weather",
                },
            ],
        }
        result, _ = translate_openai_to_gemini(body)
        # Assistant message should have functionCall
        assert result["contents"][1]["role"] == "model"
        assert "functionCall" in result["contents"][1]["parts"][0]
        # Tool response should have functionResponse
        assert result["contents"][2]["role"] == "user"
        assert "functionResponse" in result["contents"][2]["parts"][0]

    def test_tools_translation(self):
        body = {
            "model": "test/model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }],
            "tool_choice": "auto",
        }
        result, _ = translate_openai_to_gemini(body)
        assert "tools" in result
        decl = result["tools"][0]["functionDeclarations"][0]
        assert decl["name"] == "get_weather"
        assert decl["parameters"]["type"] == "OBJECT"
        assert result["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"

    def test_reasoning_translation(self):
        body = {
            "model": "test/model",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "high"},
        }
        result, _ = translate_openai_to_gemini(body)
        tc = result["generationConfig"]["thinkingConfig"]
        assert tc["includeThoughts"] is True
        assert tc["thinkingBudget"] == 16384

    def test_empty_messages_raises(self):
        with pytest.raises(ValueError):
            translate_openai_to_gemini({"model": "test", "messages": []})
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_gemini_adapter.py::TestOpenAIToGemini -v`
Expected: FAIL (module not fully implemented)

- [ ] **Step 4: Implement remaining helpers until tests pass**

Run: `pytest tests/test_gemini_adapter.py::TestOpenAIToGemini -v`
Expected: PASS

---

## Task 3: Gemini → OpenAI Chat Translation

**Files:**
- Modify: `gemini_adapter.py`

- [ ] **Step 1: Implement translate_gemini_to_openai()**

```python
def _gemini_parts_to_openai_content(parts: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
    """Convert Gemini parts to OpenAI (content_text, tool_calls, reasoning_details)."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_details: list[dict[str, Any]] = []
    call_index = 0

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
            args = fc.get("args", {})
            tool_calls.append({
                "id": f"call_{_random_hex(12)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                },
            })
            call_index += 1

    content = "\n".join(text_parts) if text_parts else None
    return content, tool_calls if tool_calls else None, reasoning_details if reasoning_details else None


def _gemini_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Gemini tools to OpenAI tool definitions."""
    result: list[dict[str, Any]] = []
    for tool in tools or []:
        decls = tool.get("functionDeclarations", [])
        for decl in decls:
            name = decl.get("name", "")
            if not name:
                continue
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


def _gemini_tool_config_to_openai(tool_config: dict[str, Any]) -> Optional[str | dict[str, Any]]:
    """Convert Gemini toolConfig to OpenAI tool_choice."""
    if not tool_config:
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
            return {"type": "function", "function": {"name": allowed[0]}}
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

    openai_messages: list[dict[str, Any]] = []

    # System instruction → system message
    sys_inst = gemini_body.get("systemInstruction")
    if sys_inst:
        parts = sys_inst.get("parts", [])
        text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
        if text_parts:
            openai_messages.append({"role": "system", "content": "\n".join(text_parts)})

    for content in contents:
        role = content.get("role", "")
        parts = content.get("parts", [])

        if role == "user":
            # Check for functionResponse parts
            has_func_response = any(isinstance(p, dict) and "functionResponse" in p for p in parts)
            if has_func_response:
                for part in parts:
                    if isinstance(part, dict) and "functionResponse" in part:
                        fr = part["functionResponse"]
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": fr.get("response", {}).get("tool_call_id", ""),
                            "name": fr.get("name", "unknown"),
                            "content": json.dumps(fr.get("response", {}), ensure_ascii=False),
                        })
            else:
                openai_parts: list[dict[str, Any]] = []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if "text" in part:
                        openai_parts.append({"type": "text", "text": part["text"]})
                    elif "inlineData" in part:
                        data = part["inlineData"]
                        mime = data.get("mimeType", "image/png")
                        b64 = data.get("data", "")
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
                if len(openai_parts) == 1 and openai_parts[0].get("type") == "text":
                    openai_messages.append({"role": "user", "content": openai_parts[0]["text"]})
                else:
                    openai_messages.append({"role": "user", "content": openai_parts or [{"type": "text", "text": ""}]})

        elif role == "model":
            content_text, tool_calls, reasoning_details = _gemini_parts_to_openai_content(parts)
            omsg: dict[str, Any] = {"role": "assistant"}
            if content_text:
                omsg["content"] = content_text
            else:
                omsg["content"] = None
            if tool_calls:
                omsg["tool_calls"] = tool_calls
            if reasoning_details:
                omsg["reasoning_details"] = reasoning_details
            openai_messages.append(omsg)

        else:
            # "function" role or unknown
            text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p]
            openai_messages.append({"role": "user", "content": "\n".join(text_parts) or ""})

    if not openai_messages:
        raise ValueError("At least one message is required")

    body: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
    }

    # Generation config → OpenAI params
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

        # Thinking config → reasoning
        thinking_config = gen_config.get("thinkingConfig")
        if isinstance(thinking_config, dict):
            if thinking_config.get("includeThoughts"):
                budget = thinking_config.get("thinkingBudget")
                if budget and isinstance(budget, (int, float)) and budget > 0:
                    body["reasoning"] = {"max_tokens": int(budget)}

    # Tools
    tools = gemini_body.get("tools")
    if tools:
        openai_tools = _gemini_tools_to_openai(tools)
        if openai_tools:
            body["tools"] = openai_tools
        tc = _gemini_tool_config_to_openai(gemini_body.get("toolConfig"))
        if tc is not None:
            body["tool_choice"] = tc

    return body, model
```

- [ ] **Step 2: Add tests for Gemini → OpenAI translation**

```python
class TestGeminiToOpenAI:
    def test_basic_translation(self):
        body = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "Hi!"}]},
            ],
            "systemInstruction": {"parts": [{"text": "Be helpful"}]},
            "generationConfig": {"temperature": 0.5, "maxOutputTokens": 2048},
        }
        result, model = translate_gemini_to_openai(body, model_prefix="google/")
        assert model == "google/gemini-2.0-flash"
        assert result["messages"][0] == {"role": "system", "content": "Be helpful"}
        assert result["messages"][1] == {"role": "user", "content": "Hello"}
        assert result["messages"][2] == {"role": "assistant", "content": "Hi!"}
        assert result["temperature"] == 0.5
        assert result["max_tokens"] == 2048

    def test_function_call_translation(self):
        body = {
            "model": "test",
            "contents": [
                {"role": "user", "parts": [{"text": "Weather?"}]},
                {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"loc": "NYC"}}}]},
                {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"result": "sunny"}}}]},
                {"role": "model", "parts": [{"text": "It's sunny in NYC!"}]},
            ],
        }
        result, _ = translate_gemini_to_openai(body)
        msgs = result["messages"]
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert msgs[2]["role"] == "tool"

    def test_thinking_parts(self):
        body = {
            "model": "test",
            "contents": [{
                "role": "model",
                "parts": [
                    {"thought": True, "text": "Let me think..."},
                    {"text": "The answer is 42."},
                ],
            }],
            "generationConfig": {"thinkingConfig": {"includeThoughts": True, "thinkingBudget": 8192}},
        }
        result, _ = translate_gemini_to_openai(body)
        msg = result["messages"][0]
        assert msg["content"] == "The answer is 42."
        assert msg["reasoning_details"][0]["type"] == "reasoning.text"
        assert msg["reasoning_details"][0]["text"] == "Let me think..."
        assert result["reasoning"] == {"max_tokens": 8192}
```

- [ ] **Step 3: Run tests, implement, verify**

Run: `pytest tests/test_gemini_adapter.py -v`
Expected: All PASS

---

## Task 4: Gemini → Anthropic and Anthropic → Gemini Translation

**Files:**
- Modify: `gemini_adapter.py`

- [ ] **Step 1: Implement translate_anthropic_to_gemini()**

```python
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

    # System
    system = anthropic_body.get("system")
    if system:
        if isinstance(system, str):
            system_parts.append({"text": system})
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_parts.append({"text": block.get("text", "")})

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
                        parts.append({
                            "functionCall": {
                                "name": block.get("name", ""),
                                "args": block.get("input", {}),
                            },
                        })
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
                                b.get("text", "") for b in result_content if isinstance(b, dict) and b.get("type") == "text"
                            )
                        response_data = {"result": result_content or ""}
                        tool_use_id = block.get("tool_use_id")
                        if tool_use_id:
                            response_data["tool_call_id"] = tool_use_id
                        parts.append({
                            "functionResponse": {
                                "name": block.get("tool_use_id", "unknown"),
                                "response": response_data,
                            },
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

    # Generation config
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

    # Thinking → thinkingConfig
    thinking = anthropic_body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "enabled":
            budget = thinking.get("budget_tokens", 4096)
            gen_config["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": budget}

    if gen_config:
        body["generationConfig"] = gen_config

    # Tools
    tools = anthropic_body.get("tools")
    if tools:
        gemini_tools = _anthropic_tools_to_gemini(tools)
        if gemini_tools:
            body["tools"] = gemini_tools
        tc = _anthropic_tool_choice_to_gemini(anthropic_body.get("tool_choice"))
        if tc is not None:
            body["toolConfig"] = tc

    body["safetySettings"] = DEFAULT_SAFETY_SETTINGS
    return body, model


def _anthropic_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tools to Gemini functionDeclarations."""
    declarations: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name", "")
        if not name:
            continue
        params = tool.get("input_schema")
        if params:
            params = _openai_schema_to_gemini_schema(params)  # same conversion works
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
            if name:
                return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    return None
```

- [ ] **Step 2: Implement translate_gemini_to_anthropic()**

```python
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

    # System instruction
    sys_inst = gemini_body.get("systemInstruction")
    if sys_inst:
        for part in sys_inst.get("parts", []):
            if isinstance(part, dict) and "text" in part:
                system_blocks.append({"type": "text", "text": part["text"]})

    for content in contents:
        role = content.get("role", "")
        parts = content.get("parts", [])

        if role == "user":
            has_func_response = any(isinstance(p, dict) and "functionResponse" in p for p in parts)
            if has_func_response:
                for part in parts:
                    if isinstance(part, dict) and "functionResponse" in part:
                        fr = part["functionResponse"]
                        response = fr.get("response", {})
                        result_text = response.get("result", "")
                        if isinstance(result_text, dict):
                            result_text = json.dumps(result_text, ensure_ascii=False)
                        anthropic_messages = _anthropic_append_or_merge(
                            anthropic_messages, "user",
                            [{"type": "tool_result", "tool_use_id": fr.get("name", ""), "content": str(result_text)}],
                        )
            else:
                blocks: list[dict[str, Any]] = []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if "text" in part:
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
                    anthropic_messages = _anthropic_append_or_merge(anthropic_messages, "user", blocks)

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
                    blocks.append({
                        "type": "tool_use",
                        "id": f"toolu_{_random_hex(12)}",
                        "name": fc.get("name", ""),
                        "input": fc.get("args", {}),
                    })
            if blocks:
                anthropic_messages = _anthropic_append_or_merge(anthropic_messages, "assistant", blocks)

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

    # Tools
    tools = gemini_body.get("tools")
    if tools:
        anthropic_tools = _gemini_tools_to_anthropic(tools)
        if anthropic_tools:
            body["tools"] = anthropic_tools

    return body, model


def _gemini_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Gemini tools to Anthropic tool definitions."""
    result: list[dict[str, Any]] = []
    for tool in tools or []:
        decls = tool.get("functionDeclarations", [])
        for decl in decls:
            name = decl.get("name", "")
            if not name:
                continue
            params = decl.get("parameters")
            if params:
                params = _gemini_schema_to_openai_schema(params)
            result.append({
                "name": name,
                "description": decl.get("description", ""),
                "input_schema": params or {"type": "object", "properties": {}},
            })
    return result


def _anthropic_append_or_merge(
    messages: list[dict[str, Any]],
    role: str,
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if messages and messages[-1]["role"] == role:
        messages[-1]["content"] = list(messages[-1]["content"]) + blocks
        return messages
    return messages + [{"role": role, "content": blocks}]
```

- [ ] **Step 3: Add tests and verify**

```python
class TestAnthropicToGemini:
    def test_basic_translation(self):
        body = {
            "model": "test/claude-sonnet",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "system": "Be helpful",
            "max_tokens": 4096,
        }
        result, model = translate_anthropic_to_gemini(body, model_prefix="test/")
        assert model == "claude-sonnet"
        assert result["systemInstruction"]["parts"][0]["text"] == "Be helpful"
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][1]["role"] == "model"

    def test_tool_use_translation(self):
        body = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"loc": "NYC"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "sunny"},
                ]},
            ],
        }
        result, _ = translate_anthropic_to_gemini(body)
        assert "functionCall" in result["contents"][1]["parts"][0]
        assert "functionResponse" in result["contents"][2]["parts"][0]


class TestGeminiToAnthropic:
    def test_basic_translation(self):
        body = {
            "model": "gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "Hi!"}]},
            ],
            "systemInstruction": {"parts": [{"text": "Be helpful"}]},
        }
        result, model = translate_gemini_to_anthropic(body, model_prefix="google/")
        assert model == "google/gemini-2.0-flash"
        assert result["system"][0]["text"] == "Be helpful"
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_thinking_translation(self):
        body = {
            "model": "test",
            "contents": [{
                "role": "model",
                "parts": [
                    {"thought": True, "text": "Reasoning..."},
                    {"text": "Answer."},
                ],
            }],
            "generationConfig": {"thinkingConfig": {"includeThoughts": True, "thinkingBudget": 8192}},
        }
        result, _ = translate_gemini_to_anthropic(body)
        msg = result["messages"][0]
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][0]["thinking"] == "Reasoning..."
        assert msg["content"][1]["type"] == "text"
        assert result["thinking"]["budget_tokens"] == 8192
```

Run: `pytest tests/test_gemini_adapter.py -v`
Expected: All PASS

---

## Task 5: Gemini Streaming Decoders

**Files:**
- Modify: `gemini_adapter.py`

- [ ] **Step 1: Implement GeminiStreamDecoder (Gemini SSE → OpenAI chunks)**

```python
import time as _time
from protocol_adapter import _random_hex


class GeminiStreamDecoder:
    """Parses Gemini SSE response events and yields OpenAI-format chunks.

    Gemini streaming returns full JSON objects per SSE event (not deltas).
    Each event may have partial text in candidates[].content.parts[].
    """

    def __init__(self, model: str, *, request_id: Optional[str] = None):
        self.model = model
        self.chunk_id = request_id or f"chatcmpl-{_random_hex(12)}"
        self.created = int(_time.time())
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._reasoning_details: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._finish_reason: str = ""
        self._usage: Optional[dict[str, Any]] = None

    def process_sse_data(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Process a single Gemini SSE data object. Returns list of OpenAI chunks."""
        chunks: list[dict[str, Any]] = []

        # Extract usage
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
                call_id = f"call_{_random_hex(12)}"
                name = fc.get("name", "")
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
        tool_calls = None
        if self._tool_calls:
            tool_calls = self._tool_calls

        finish = self._finish_reason or "stop"
        if tool_calls and not content:
            finish = "tool_calls"

        from protocol_adapter import _build_openai_response
        return _build_openai_response(
            self.chunk_id, self.created, self.model,
            content, tool_calls, finish, self._usage,
            reasoning_details=self._reasoning_details or None,
        )


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
                # Emit thinking block
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
                call_id = f"toolu_{_random_hex(12)}"
                name = fc.get("name", "")
                args = fc.get("args", {})
                if self._block_started:
                    result += self._emit_event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self._block_index,
                    })
                    self._block_index += 1
                    self._block_started = False
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
            if self._block_started:
                result += self._emit_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self._block_index,
                })
                self._block_index += 1
                self._block_started = False
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
        result = ""
        result += self._emit_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": self._stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": self._output_tokens},
        })
        result += self._emit_event("message_stop", {"type": "message_stop"})
        return result
```

- [ ] **Step 2: Implement SSE translators (OpenAI/Anthropic → Gemini SSE)**

```python
class OpenAIToGeminiSSETranslator:
    """Translates OpenAI SSE chunks → Gemini SSE response events.

    Gemini streaming returns full JSON objects per SSE event.
    """

    def __init__(self, model: str):
        self.model = model
        self._text_parts: list[str] = []
        self._thought_parts: list[str] = []
        self._tool_calls: list[dict[str, Any]] = []
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
                func = tc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "")
                if name:
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    self._tool_calls.append({"name": name, "args": args})
                    parts.append({"functionCall": {"name": name, "args": args}})

        if finish_reason:
            finish_map = {"stop": "STOP", "length": "MAX_TOKENS", "tool_calls": "STOP", "content_filter": "SAFETY"}
            self._finish_reason = finish_map.get(finish_reason, "STOP")
            self._finished = True

        if not parts:
            return None

        gemini_response: dict[str, Any] = {
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
        return gemini_response

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

        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "thinking_delta":
                text = delta.get("thinking", "")
                if text:
                    self._thought_parts.append(text)
                    return {
                        "candidates": [{"content": {"role": "model", "parts": [{"thought": True, "text": text}]}, "index": 0}],
                        "usageMetadata": {"promptTokenCount": self._input_tokens, "candidatesTokenCount": self._output_tokens, "totalTokenCount": self._input_tokens + self._output_tokens},
                        "modelVersion": self.model,
                    }
            elif delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    self._text_parts.append(text)
                    return {
                        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "index": 0}],
                        "usageMetadata": {"promptTokenCount": self._input_tokens, "candidatesTokenCount": self._output_tokens, "totalTokenCount": self._input_tokens + self._output_tokens},
                        "modelVersion": self.model,
                    }
            elif delta_type == "input_json_delta":
                # Partial JSON for tool use — for simplicity, ignore partial
                pass
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
```

- [ ] **Step 3: Add streaming tests and verify**

```python
class TestGeminiStreamDecoder:
    def test_text_streaming(self):
        decoder = GeminiStreamDecoder("gemini-2.0-flash")
        chunks1 = decoder.process_sse_data({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}, "index": 0}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6},
        })
        assert len(chunks1) == 1
        assert chunks1[0]["choices"][0]["delta"]["content"] == "Hello"

        chunks2 = decoder.process_sse_data({
            "candidates": [{"content": {"role": "model", "parts": [{"text": " world!"}]}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
        })
        assert len(chunks2) == 1
        assert chunks2[0]["choices"][0]["delta"]["content"] == " world!"

        final = decoder.build_final_response()
        assert final["choices"][0]["message"]["content"] == "Hello world!"
        assert final["choices"][0]["finish_reason"] == "stop"

    def test_thinking_streaming(self):
        decoder = GeminiStreamDecoder("gemini-2.0-flash")
        chunks = decoder.process_sse_data({
            "candidates": [{"content": {"role": "model", "parts": [
                {"thought": True, "text": "Thinking..."},
                {"text": "Answer"},
            ]}, "finishReason": "STOP", "index": 0}],
        })
        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["reasoning_details"][0]["text"] == "Thinking..."
        assert chunks[1]["choices"][0]["delta"]["content"] == "Answer"
```

Run: `pytest tests/test_gemini_adapter.py -v`
Expected: All PASS

---

## Task 6: OpenAI Chat ↔ OpenAI Responses Translation

**Files:**
- Modify: `protocol_adapter.py`

- [ ] **Step 1: Implement translate_openai_chat_to_responses()**

```python
def translate_openai_chat_to_responses(
    chat_body: dict[str, Any],
) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions request to Responses API format."""
    response_body: dict[str, Any] = {}

    if chat_body.get("model"):
        response_body["model"] = chat_body["model"]

    # Convert messages to input
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
            text = _blocks_to_text_simple(content)
            if text:
                input_items.append({"role": "user", "content": [{"type": "input_text", "text": text}]})
            elif isinstance(content, list):
                items = _openai_content_to_responses_input(content)
                if items:
                    input_items.append({"role": "user", "content": items})
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

    # Params
    max_tokens = chat_body.get("max_tokens") or chat_body.get("max_completion_tokens")
    if max_tokens:
        response_body["max_output_tokens"] = max_tokens
    if chat_body.get("stream"):
        response_body["stream"] = True
    if chat_body.get("temperature") is not None:
        response_body["temperature"] = chat_body["temperature"]
    if chat_body.get("top_p") is not None:
        response_body["top_p"] = chat_body["top_p"]

    # Reasoning
    reasoning = chat_body.get("reasoning")
    if reasoning:
        response_body["reasoning"] = reasoning

    # Tools
    tools = chat_body.get("tools")
    if tools:
        responses_tools = _openai_chat_tools_to_responses(tools)
        if responses_tools:
            response_body["tools"] = responses_tools
        tc = chat_body.get("tool_choice")
        if tc:
            response_body["tool_choice"] = tc

    # Response format
    rf = chat_body.get("response_format")
    if rf:
        response_body["text"] = {"format": rf}

    return response_body


def _blocks_to_text_simple(content: Any) -> str:
    """Extract text from OpenAI content field."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
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
```

- [ ] **Step 2: Implement translate_openai_responses_to_chat()**

```python
def translate_openai_responses_to_chat(
    responses_body: dict[str, Any],
) -> dict[str, Any]:
    """Translate an OpenAI Responses API request to Chat Completions format."""
    chat_body: dict[str, Any] = {}

    if responses_body.get("model"):
        chat_body["model"] = responses_body["model"]

    messages: list[dict[str, Any]] = []

    # Instructions → system message
    instructions = responses_body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # Input → messages
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
                    text_parts = []
                    for cp in content_parts:
                        if isinstance(cp, dict):
                            if cp.get("type") in ("input_text", "text"):
                                text_parts.append(cp.get("text", ""))
                            elif cp.get("type") == "input_image":
                                url = cp.get("image_url", "")
                                detail = cp.get("detail", "auto")
                                text_parts.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
                    if len(text_parts) == 1 and isinstance(text_parts[0], str):
                        messages.append({"role": "user", "content": text_parts[0]})
                    else:
                        messages.append({"role": "user", "content": text_parts})

            elif role == "assistant":
                content_parts = item.get("content", [])
                if isinstance(content_parts, str):
                    messages.append({"role": "assistant", "content": content_parts})
                elif isinstance(content_parts, list):
                    text_parts = []
                    tool_calls = []
                    for cp in content_parts:
                        if isinstance(cp, dict):
                            if cp.get("type") in ("output_text", "text"):
                                text_parts.append(cp.get("text", ""))
                            elif cp.get("type") == "tool_call":
                                tool_calls.append({
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
                    if tool_calls:
                        omsg["tool_calls"] = tool_calls
                    messages.append(omsg)

            elif item_type == "function_call_output":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                })

    chat_body["messages"] = messages

    # Params
    max_output = responses_body.get("max_output_tokens")
    if max_output:
        chat_body["max_tokens"] = max_output
    if responses_body.get("stream"):
        chat_body["stream"] = True
    if responses_body.get("temperature") is not None:
        chat_body["temperature"] = responses_body["temperature"]
    if responses_body.get("top_p") is not None:
        chat_body["top_p"] = responses_body["top_p"]

    # Reasoning
    reasoning = responses_body.get("reasoning")
    if reasoning:
        chat_body["reasoning"] = reasoning

    # Tools
    tools = responses_body.get("tools")
    if tools:
        chat_tools = _responses_tools_to_openai_chat(tools)
        if chat_tools:
            chat_body["tools"] = chat_tools
        tc = responses_body.get("tool_choice")
        if tc:
            chat_body["tool_choice"] = tc

    # Text format → response_format
    text_config = responses_body.get("text")
    if isinstance(text_config, dict):
        fmt = text_config.get("format")
        if fmt:
            chat_body["response_format"] = fmt

    return chat_body


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
```

- [ ] **Step 3: Write and run tests**

Create `tests/test_chat_responses_translation.py`:

```python
"""Tests for Chat Completions ↔ Responses API translation."""
import json
from protocol_adapter import (
    translate_openai_chat_to_responses,
    translate_openai_responses_to_chat,
)


class TestChatToResponses:
    def test_basic_translation(self):
        chat = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        result = translate_openai_chat_to_responses(chat)
        assert result["model"] == "gpt-4o"
        assert result["instructions"] == "You are helpful."
        assert result["input"][0]["role"] == "user"
        assert result["input"][1]["role"] == "assistant"
        assert result["max_output_tokens"] == 1024
        assert result["temperature"] == 0.7

    def test_tool_calls_translation(self):
        chat = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location":"NYC"}'},
                    }],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": '{"temp":72}'},
            ],
        }
        result = translate_openai_chat_to_responses(chat)
        assert result["input"][1]["content"][0]["type"] == "tool_call"
        assert result["input"][2]["type"] == "function_call_output"


class TestResponsesToChat:
    def test_basic_translation(self):
        responses = {
            "model": "gpt-4o",
            "instructions": "Be helpful.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "output_text", "text": "Hi!"}]},
            ],
            "max_output_tokens": 1024,
        }
        result = translate_openai_responses_to_chat(responses)
        assert result["messages"][0] == {"role": "system", "content": "Be helpful."}
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "Hello"
        assert result["max_tokens"] == 1024

    def test_function_call_output(self):
        responses = {
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Weather?"}]},
                {"role": "assistant", "content": [{"type": "tool_call", "id": "c1", "name": "get_weather", "arguments": "{}"}]},
                {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            ],
        }
        result = translate_openai_responses_to_chat(responses)
        assert result["messages"][2]["role"] == "tool"
        assert result["messages"][2]["tool_call_id"] == "c1"
```

Run: `pytest tests/test_chat_responses_translation.py -v`
Expected: All PASS

---

## Task 7: Anthropic ↔ OpenAI Responses Translation

**Files:**
- Modify: `protocol_adapter.py`

- [ ] **Step 1: Implement translate_anthropic_to_responses()**

```python
def translate_anthropic_to_responses(
    anthropic_body: dict[str, Any],
) -> dict[str, Any]:
    """Translate an Anthropic Messages request to OpenAI Responses API format."""
    response_body: dict[str, Any] = {}

    model = anthropic_body.get("model", "")
    if model:
        response_body["model"] = model

    # System → instructions
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

    # Params
    max_tokens = anthropic_body.get("max_tokens")
    if max_tokens:
        response_body["max_output_tokens"] = max_tokens
    if anthropic_body.get("stream"):
        response_body["stream"] = True
    if anthropic_body.get("temperature") is not None:
        response_body["temperature"] = anthropic_body["temperature"]
    if anthropic_body.get("top_p") is not None:
        response_body["top_p"] = anthropic_body["top_p"]

    # Thinking → reasoning
    thinking = anthropic_body.get("thinking")
    if isinstance(thinking, dict):
        if thinking.get("type") == "enabled":
            budget = thinking.get("budget_tokens")
            if budget:
                response_body["reasoning"] = {"max_tokens": budget}
        elif thinking.get("type") == "disabled":
            response_body["reasoning"] = {"enabled": False}

    # Tools
    tools = anthropic_body.get("tools")
    if tools:
        responses_tools = _anthropic_tools_to_responses(tools)
        if responses_tools:
            response_body["tools"] = responses_tools

    return response_body


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
```

- [ ] **Step 2: Implement translate_responses_to_anthropic()**

```python
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

    # Instructions → system
    instructions = responses_body.get("instructions")
    if instructions:
        system_blocks.append({"type": "text", "text": instructions})

    # Input → messages
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
                                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}})
                                else:
                                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
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
                    [{"type": "tool_result", "tool_use_id": item.get("call_id", ""), "content": item.get("output", "")}],
                )

    if not anthropic_messages:
        raise ValueError("At least one non-system input is required")

    body["messages"] = anthropic_messages
    body["max_tokens"] = default_max_tokens
    body["stream"] = True

    if system_blocks:
        body["system"] = system_blocks

    # Params
    max_output = responses_body.get("max_output_tokens")
    if max_output and max_output > 0:
        body["max_tokens"] = max_output
    if responses_body.get("temperature") is not None:
        body["temperature"] = responses_body["temperature"]
    if responses_body.get("top_p") is not None:
        body["top_p"] = responses_body["top_p"]

    # Reasoning → thinking
    reasoning = responses_body.get("reasoning")
    if isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            body["thinking"] = {"type": "disabled"}
        elif reasoning.get("max_tokens"):
            body["thinking"] = {"type": "enabled", "budget_tokens": reasoning["max_tokens"]}
        elif reasoning.get("effort"):
            budget = REASONING_EFFORT_TO_BUDGET.get(reasoning["effort"], 4096)
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}

    # Tools
    tools = responses_body.get("tools")
    if tools:
        anthropic_tools = _responses_tools_to_anthropic(tools)
        if anthropic_tools:
            body["tools"] = anthropic_tools

    return body


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
```

- [ ] **Step 3: Write tests and verify**

Add to `tests/test_chat_responses_translation.py`:

```python
class TestAnthropicToResponses:
    def test_basic(self):
        body = {
            "model": "claude-sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "Be helpful",
            "max_tokens": 4096,
        }
        result = translate_anthropic_to_responses(body)
        assert result["instructions"] == "Be helpful"
        assert result["input"][0]["role"] == "user"
        assert result["max_output_tokens"] == 4096


class TestResponsesToAnthropic:
    def test_basic(self):
        body = {
            "model": "gpt-4o",
            "instructions": "Be helpful",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "max_output_tokens": 4096,
        }
        result = translate_responses_to_anthropic(body)
        assert result["system"][0]["text"] == "Be helpful"
        assert result["messages"][0]["role"] == "user"
        assert result["max_tokens"] == 4096
```

Run: `pytest tests/test_chat_responses_translation.py -v`
Expected: All PASS

---

## Task 8: Gemini ↔ OpenAI Responses Translation

**Files:**
- Modify: `gemini_adapter.py`

- [ ] **Step 1: Implement translate_gemini_to_responses() and translate_responses_to_gemini()**

```python
def translate_gemini_to_responses(
    gemini_body: dict[str, Any],
    *,
    model_prefix: str = "",
) -> dict[str, Any]:
    """Translate a Gemini request to OpenAI Responses API format."""
    # First translate to OpenAI Chat, then to Responses
    from protocol_adapter import translate_openai_chat_to_responses
    openai_body, model = translate_gemini_to_openai(gemini_body, model_prefix=model_prefix)
    responses_body = translate_openai_chat_to_responses(openai_body)
    return responses_body


def translate_responses_to_gemini(
    responses_body: dict[str, Any],
    *,
    model_prefix: str = "",
    default_max_tokens: int = 8192,
) -> tuple[dict[str, Any], str]:
    """Translate an OpenAI Responses API request to Gemini format."""
    from protocol_adapter import translate_openai_responses_to_chat
    chat_body = translate_openai_responses_to_chat(responses_body)
    return translate_openai_to_gemini(chat_body, model_prefix=model_prefix, default_max_tokens=default_max_tokens)
```

- [ ] **Step 2: Add tests**

```python
class TestGeminiResponsesTranslation:
    def test_gemini_to_responses(self):
        body = {
            "model": "test/gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "systemInstruction": {"parts": [{"text": "Be helpful"}]},
        }
        result = translate_gemini_to_responses(body, model_prefix="test/")
        assert result["instructions"] == "Be helpful"
        assert result["input"][0]["role"] == "user"

    def test_responses_to_gemini(self):
        body = {
            "model": "test/model",
            "instructions": "Be helpful",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        }
        result, model = translate_responses_to_gemini(body, model_prefix="test/")
        assert result["systemInstruction"]["parts"][0]["text"] == "Be helpful"
        assert result["contents"][0]["role"] == "user"
```

Run: `pytest tests/test_gemini_adapter.py -v`
Expected: All PASS

---

## Task 9: Provider Registry — Add Gemini Format Support

**Files:**
- Modify: `provider_registry.py`

- [ ] **Step 1: Extend upstream_format validation to accept "gemini"**

In `_normalize_provider_block()`, change:

```python
# Line ~399
upstream_format = normalized.get("upstream_format") or normalized.get("upstream_mode", "openai")
if not isinstance(upstream_format, str) or upstream_format not in ("openai", "anthropic", "gemini"):
    upstream_format = "openai"
```

- [ ] **Step 2: Add Gemini-specific config fields**

```python
# After the model_prefix normalization block (~line 419), add:

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
```

- [ ] **Step 3: Add fields to ProviderRuntime dataclass**

```python
# In provider_registry.py ProviderRuntime dataclass, add:
upstream_generate_path: str = ""
gemini_api_version: str = "v1beta"
```

- [ ] **Step 4: Pass new fields in build_provider_registry()**

In `build_provider_registry()`, add to the `ProviderRuntime(...)` constructor:

```python
upstream_generate_path=str(normalized.get("upstream_generate_path", "")),
gemini_api_version=str(normalized.get("gemini_api_version", "v1beta")),
```

- [ ] **Step 5: Add Gemini model list handling in routes.py**

In `fetch_provider_models()` within `handle_models_endpoint()`, add Gemini check:

```python
# After the anthropic check:
if provider.upstream_format == "gemini":
    return _build_local_models_payload(provider)
```

---

## Task 10: Client-Facing Gemini Endpoints

**Files:**
- Modify: `routes.py`

- [ ] **Step 1: Add Gemini endpoint routes**

```python
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
    # ... (implementation in next step)
```

- [ ] **Step 2: Implement the Gemini endpoint handler**

```python
async def gemini_generate_endpoint(
    request: Request,
    model_path: str,
    authorization: Optional[str] = Header(None),
):
    try:
        body_bytes = await request.body()
        try:
            request_body = json.loads(body_bytes) if body_bytes else {}
        except (json.JSONDecodeError, ValueError):
            return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON body", "code": 400}})

        # Determine streaming from URL path
        is_stream = ":streamGenerateContent" in request.url.path

        # Extract model name from path (e.g., "gemini-2.0-flash" from "models/gemini-2.0-flash")
        model_name = model_path
        if request_body.get("model"):
            model_name = request_body["model"]
        else:
            request_body["model"] = model_name

        # Resolve provider from model prefix
        provider_name = model_name.split("/")[0] if "/" in model_name else None
        if not provider_name:
            return JSONResponse(status_code=400, content={
                "error": {"message": "Model must be provider-qualified as <provider>/<model>", "code": 400}
            })

        provider = get_provider_by_hint(provider_name)
        if provider is None:
            return JSONResponse(status_code=400, content={
                "error": {"message": f"Unknown provider '{provider_name}'", "code": 400}
            })

        await verify_client_access(provider, "/generateContent", authorization)
        api_key = await provider.key_manager.get_next_key()

        return await _handle_gemini_upstream(
            request, provider, request_body, api_key, is_stream,
        )

    except HTTPException as exc:
        detail = exc.detail
        if isinstance(detail, dict):
            msg = detail.get("message", str(detail))
        else:
            msg = str(detail)
        return JSONResponse(status_code=exc.status_code, content={"error": {"message": msg, "code": exc.status_code}})
    except Exception:
        logger.exception("Gemini endpoint error")
        return JSONResponse(status_code=500, content={"error": {"message": "Internal Proxy Error", "code": 500}})
```

- [ ] **Step 3: Implement _handle_gemini_upstream() router**

```python
async def _handle_gemini_upstream(
    request: Request,
    provider: ProviderRuntime,
    request_body: dict[str, Any],
    api_key: str,
    is_stream: bool,
) -> Response:
    """Route a Gemini client request to the appropriate upstream format."""

    # ── Upstream is Gemini → passthrough ──────────────────────────────
    if provider.upstream_format == "gemini":
        return await _proxy_gemini_passthrough(request, provider, request_body, api_key, is_stream)

    # ── Upstream is OpenAI → translate Gemini→OpenAI → send → translate back
    if provider.upstream_format == "openai":
        return await _proxy_gemini_to_openai_upstream(request, provider, request_body, api_key, is_stream)

    # ── Upstream is Anthropic → translate Gemini→Anthropic → send → translate back
    if provider.upstream_format == "anthropic":
        return await _proxy_gemini_to_anthropic_upstream(request, provider, request_body, api_key, is_stream)

    return JSONResponse(status_code=500, content={"error": {"message": f"Unknown upstream format: {provider.upstream_format}", "code": 500}})
```

- [ ] **Step 4: Implement the three upstream proxy functions**

These follow the same patterns as `proxy_anthropic_with_httpx()` and `handle_anthropic_messages()`:

```python
async def _proxy_gemini_passthrough(
    request: Request,
    provider: ProviderRuntime,
    request_body: dict[str, Any],
    api_key: str,
    is_stream: bool,
) -> Response:
    """Passthrough to a Gemini upstream."""
    from gemini_adapter import GeminiStreamDecoder, GeminiAnthropicStreamDecoder

    model = request_body.get("model", "")
    if provider.model_prefix and model.startswith(provider.model_prefix):
        request_body["model"] = model[len(provider.model_prefix):]

    api_version = provider.gemini_api_version or "v1beta"
    generate_path = provider.upstream_generate_path
    if not generate_path:
        endpoint = ":streamGenerateContent?alt=sse" if is_stream else ":generateContent"
        generate_path = f"/{api_version}/models/{request_body.get('model', 'unknown')}{endpoint}"

    body_bytes = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    upstream_url = f"{provider.base_url}{generate_path}"

    await provider.request_pacer.wait()
    client = await get_async_client(request)
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

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
```

For `_proxy_gemini_to_openai_upstream()` and `_proxy_gemini_to_anthropic_upstream()`, follow the same pattern as the existing Anthropic↔OpenAI translation in `handle_anthropic_messages()` but using the Gemini adapter functions.

- [ ] **Step 5: Run existing tests to verify no regressions**

Run: `pytest tests/ -v`
Expected: All PASS

---

## Task 11: Upstream Gemini Support for Existing Client Formats

**Files:**
- Modify: `routes.py`

When an OpenAI Chat, OpenAI Responses, or Anthropic client sends to a Gemini upstream, we need to translate.

- [ ] **Step 1: Add Gemini upstream handling in proxy_endpoint()**

In `proxy_endpoint()`, add before the existing anthropic check (~line 637):

```python
# Gemini-mode providers need request translation
if provider.upstream_format == "gemini" and normalized_path == "/chat/completions":
    return await proxy_openai_to_gemini(
        request, provider, normalized_path, api_key, is_stream,
        body_bytes=body_bytes, request_body=request_body,
    )
```

- [ ] **Step 2: Implement proxy_openai_to_gemini()**

```python
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
    """Proxy OpenAI Chat request to Gemini upstream."""
    from gemini_adapter import (
        translate_openai_to_gemini,
        translate_gemini_to_openai,
        GeminiStreamDecoder,
    )

    if request_body is None:
        if body_bytes is None:
            body_bytes = await request.body()
        try:
            request_body = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        gemini_body, model_name = translate_openai_to_gemini(
            request_body,
            model_prefix=provider.model_prefix or provider.provider_id + "/",
            default_max_tokens=provider.default_max_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build upstream URL
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
        # Non-streaming: aggregate SSE → full response → translate to OpenAI
        resp_data = None
        try:
            async for line in upstream_resp.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    try:
                        resp_data = json.loads(line[6:])
                    except (json.JSONDecodeError, TypeError):
                        pass
        except Exception:
            pass
        finally:
            await upstream_resp.aclose()

        if resp_data is None:
            # Try parsing as plain JSON
            try:
                resp_data = json.loads(upstream_resp.content) if hasattr(upstream_resp, 'content') else None
            except Exception:
                pass

        if resp_data:
            openai_resp = translate_gemini_response_to_openai(resp_data, model_name)
            return JSONResponse(content=openai_resp)
        raise HTTPException(502, "No response from upstream")

    # Streaming: translate Gemini SSE → OpenAI SSE
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
        except Exception:
            logger.exception("[%s] Gemini→OpenAI SSE error", provider.provider_id)
        finally:
            yield b"data: [DONE]\n\n"
            await upstream_resp.aclose()

    return StreamingResponse(translated_sse(), status_code=200, media_type="text/event-stream")
```

- [ ] **Step 3: Implement translate_gemini_response_to_openai()**

```python
def translate_gemini_response_to_openai(gemini_data: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate a complete Gemini response to OpenAI Chat Completions format."""
    from gemini_adapter import _gemini_parts_to_openai_content, GEMINI_FINISH_MAP
    import time as _time

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
            "created": int(_time.time()),
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
    if content_text:
        message["content"] = content_text
    else:
        message["content"] = None
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_details:
        message["reasoning_details"] = reasoning_details

    return {
        "id": f"chatcmpl-{_random_hex(12)}",
        "object": "chat.completion",
        "created": int(_time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }
```

- [ ] **Step 4: Add Anthropic client → Gemini upstream routing**

In `handle_anthropic_messages()`, add a Gemini branch:

```python
# After the openai upstream branch, add:
if provider.upstream_format == "gemini":
    return await _proxy_anthropic_to_gemini_upstream(request, provider, request_body, authorization)
```

Implement `_proxy_anthropic_to_gemini_upstream()` using `translate_anthropic_to_gemini()` from `gemini_adapter.py`, following the same pattern.

- [ ] **Step 5: Add Responses client → Gemini upstream routing**

In `handle_responses_endpoint()`, add translation when upstream is not OpenAI Responses.

---

## Task 12: Update Config Example

**Files:**
- Modify: `config.yml.example`

- [ ] **Step 1: Add Gemini upstream provider example**

```yaml
  # ─── Gemini upstream (native GenerateContent API) ─────────────────────────
  gemini:
    base_url: "https://generativelanguage.googleapis.com"
    upstream_format: "gemini"
    gemini_api_version: "v1beta"
    model_prefix: "gemini/"
    default_max_tokens: 8192
    keys:
      - "your-gemini-api-key"
    allowed_models:
      - "gemini/gemini-2.0-flash"
      - "gemini/gemini-2.5-pro"
    key_selection_strategy: "round-robin"
    key_selection_opts: []
    public_endpoints:
      - "/models"
    rate_limit_cooldown: 14400
    global_rate_delay: 5
```

- [ ] **Step 2: Update translation matrix comment at top**

```yaml
# Translation matrix:
#   Client OpenAI    → upstream_format: "openai"     → passthrough
#   Client OpenAI    → upstream_format: "anthropic"  → translate + spoof Claude Code headers
#   Client OpenAI    → upstream_format: "gemini"     → translate to Gemini
#   Client Anthropic → upstream_format: "openai"     → translate to OpenAI, send, translate back
#   Client Anthropic → upstream_format: "anthropic"  → strip prefix, spoof headers, passthrough
#   Client Anthropic → upstream_format: "gemini"     → translate to Gemini
#   Client Gemini    → upstream_format: "openai"     → translate to OpenAI, send, translate back
#   Client Gemini    → upstream_format: "anthropic"  → translate to Anthropic, send, translate back
#   Client Gemini    → upstream_format: "gemini"     → passthrough
#   Client Responses → upstream_format: "openai"     → passthrough (if upstream supports Responses)
#   Client Responses → upstream_format: "anthropic"  → translate to Anthropic
#   Client Responses → upstream_format: "gemini"     → translate to Gemini
```

---

## Task 13: Full Matrix Integration Tests

**Files:**
- Create: `tests/test_full_matrix.py`

- [ ] **Step 1: Write comprehensive matrix tests**

```python
"""Tests for the full 4x4 translation matrix."""
import json
import pytest
from protocol_adapter import (
    translate_openai_to_anthropic,
    translate_anthropic_to_openai,
    translate_openai_chat_to_responses,
    translate_openai_responses_to_chat,
    translate_anthropic_to_responses,
    translate_responses_to_anthropic,
)
from gemini_adapter import (
    translate_openai_to_gemini,
    translate_gemini_to_openai,
    translate_anthropic_to_gemini,
    translate_gemini_to_anthropic,
    translate_gemini_to_responses,
    translate_responses_to_gemini,
)


def _make_openai_chat():
    return {
        "model": "test/model",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
    }


def _make_anthropic():
    return {
        "model": "test/claude",
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ],
        "max_tokens": 1024,
    }


def _make_gemini():
    return {
        "model": "test/gemini",
        "systemInstruction": {"parts": [{"text": "You are helpful."}]},
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi!"}]},
            {"role": "user", "parts": [{"text": "How are you?"}]},
        ],
        "generationConfig": {"maxOutputTokens": 1024},
    }


def _make_responses():
    return {
        "model": "test/model",
        "instructions": "You are helpful.",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Hi!"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "How are you?"}]},
        ],
        "max_output_tokens": 1024,
    }


class TestOpenAIChatToAll:
    def test_to_anthropic(self):
        result, model = translate_openai_to_anthropic(_make_openai_chat(), inject_billing=False)
        assert result["messages"][-1]["role"] == "user"
        assert result["max_tokens"] == 1024

    def test_to_responses(self):
        result = translate_openai_chat_to_responses(_make_openai_chat())
        assert result["instructions"] == "You are helpful."
        assert result["input"][-1]["role"] == "user"

    def test_to_gemini(self):
        result, model = translate_openai_to_gemini(_make_openai_chat())
        assert result["contents"][-1]["role"] == "user"
        assert result["generationConfig"]["maxOutputTokens"] == 1024


class TestAnthropicToAll:
    def test_to_openai(self):
        result, model = translate_anthropic_to_openai(_make_anthropic(), model_prefix="test/")
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][-1]["role"] == "user"

    def test_to_responses(self):
        result = translate_anthropic_to_responses(_make_anthropic())
        assert result["instructions"] == "You are helpful."
        assert result["input"][-1]["role"] == "user"

    def test_to_gemini(self):
        result, model = translate_anthropic_to_gemini(_make_anthropic(), model_prefix="test/")
        assert result["systemInstruction"]["parts"][0]["text"] == "You are helpful."
        assert result["contents"][-1]["role"] == "user"


class TestGeminiToAll:
    def test_to_openai(self):
        result, model = translate_gemini_to_openai(_make_gemini(), model_prefix="test/")
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][-1]["role"] == "user"

    def test_to_anthropic(self):
        result, model = translate_gemini_to_anthropic(_make_gemini(), model_prefix="test/")
        assert result["system"][0]["text"] == "You are helpful."
        assert result["messages"][-1]["role"] == "user"

    def test_to_responses(self):
        result = translate_gemini_to_responses(_make_gemini(), model_prefix="test/")
        assert result["instructions"] == "You are helpful."


class TestResponsesToAll:
    def test_to_openai(self):
        result = translate_openai_responses_to_chat(_make_responses())
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][-1]["role"] == "user"

    def test_to_anthropic(self):
        result = translate_responses_to_anthropic(_make_responses())
        assert result["system"][0]["text"] == "You are helpful."
        assert result["messages"][-1]["role"] == "user"

    def test_to_gemini(self):
        result, model = translate_responses_to_gemini(_make_responses(), model_prefix="test/")
        assert result["systemInstruction"]["parts"][0]["text"] == "You are helpful."


class TestRoundTripPreservation:
    """Test that translating A→B→A preserves the essential content."""

    def test_openai_anthropic_roundtrip(self):
        original = _make_openai_chat()
        anthropic, _ = translate_openai_to_anthropic(original, inject_billing=False)
        restored, _ = translate_anthropic_to_openai(anthropic, model_prefix="")
        # System preserved
        assert any(m.get("role") == "system" for m in restored["messages"])
        # User messages preserved
        user_msgs = [m for m in restored["messages"] if m.get("role") == "user"]
        assert len(user_msgs) >= 1

    def test_openai_gemini_roundtrip(self):
        original = _make_openai_chat()
        gemini, _ = translate_openai_to_gemini(original)
        restored, _ = translate_gemini_to_openai(gemini, model_prefix="")
        assert any(m.get("role") == "system" for m in restored["messages"])

    def test_anthropic_gemini_roundtrip(self):
        original = _make_anthropic()
        gemini, _ = translate_anthropic_to_gemini(original, model_prefix="test/")
        restored, _ = translate_gemini_to_anthropic(gemini, model_prefix="test/")
        assert "system" in restored
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

---

## Task 14: Verification and Cleanup

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Run lint/typecheck if configured**

Run: `python -m py_compile gemini_adapter.py && python -m py_compile protocol_adapter.py && python -m py_compile routes.py && python -m py_compile provider_registry.py`
Expected: No errors

- [ ] **Step 3: Verify imports resolve**

Run: `python -c "from gemini_adapter import translate_openai_to_gemini, translate_gemini_to_openai, translate_anthropic_to_gemini, translate_gemini_to_anthropic, GeminiStreamDecoder, OpenAIToGeminiSSETranslator, AnthropicToGeminiSSETranslator; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Update agentrouter_adapter.py if needed**

Verify `agentrouter_adapter.py` still works (it re-exports from `protocol_adapter.py`).

---

## Summary: Translation Matrix Complete

| Client ↓ \ Upstream → | OpenAI Chat | OpenAI Responses | Anthropic | Gemini |
|---|---|---|---|---|
| **OpenAI Chat** | passthrough | `translate_openai_chat_to_responses()` | `translate_openai_to_anthropic()` ✓ | `translate_openai_to_gemini()` |
| **OpenAI Responses** | `translate_openai_responses_to_chat()` | passthrough | `translate_responses_to_anthropic()` | `translate_responses_to_gemini()` |
| **Anthropic** | `translate_anthropic_to_openai()` ✓ | `translate_anthropic_to_responses()` | passthrough ✓ | `translate_anthropic_to_gemini()` |
| **Gemini** | `translate_gemini_to_openai()` | `translate_gemini_to_responses()` | `translate_gemini_to_anthropic()` | passthrough |

All 16 cells covered. Each translation function is pure and independently testable.
