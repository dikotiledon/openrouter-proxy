"""Tests for reasoning/thinking translation."""
import json

from protocol_adapter import (
    REASONING_EFFORT_TO_BUDGET,
    AnthropicStreamDecoder,
    translate_anthropic_to_openai,
    translate_openai_to_anthropic,
)


def test_reasoning_effort_to_budget():
    body = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "high"},
        "max_tokens": 100,
    }
    result, model = translate_openai_to_anthropic(body, inject_billing=False)
    assert result.get("thinking") == {"type": "enabled", "budget_tokens": 16384}

    body["reasoning"] = {"effort": "low"}
    result, model = translate_openai_to_anthropic(body, inject_billing=False)
    assert result.get("thinking") == {"type": "enabled", "budget_tokens": 1024}

    body["reasoning"] = {"effort": "xhigh"}
    result, model = translate_openai_to_anthropic(body, inject_billing=False)
    assert result.get("thinking") == {"type": "enabled", "budget_tokens": 32768}

    body["reasoning"] = {"effort": "medium"}
    result, model = translate_openai_to_anthropic(body, inject_billing=False)
    assert result.get("thinking") == {"type": "enabled", "budget_tokens": 4096}

    body["reasoning"] = {"effort": "minimal"}
    result, model = translate_openai_to_anthropic(body, inject_billing=False)
    assert result.get("thinking") == {"type": "enabled", "budget_tokens": 512}


def test_reasoning_exclude_disabled():
    body = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 100,
    }
    body["reasoning"] = {"exclude": True}
    result, model = translate_openai_to_anthropic(body, inject_billing=False)
    assert result.get("thinking") == {"type": "disabled"}

    body["reasoning"] = {"enabled": False}
    result, model = translate_openai_to_anthropic(body, inject_billing=False)
    assert result.get("thinking") == {"type": "disabled"}


def test_reasoning_max_tokens_direct():
    body = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"max_tokens": 8000},
        "max_tokens": 100,
    }
    result, model = translate_openai_to_anthropic(body, inject_billing=False)
    assert result.get("thinking") == {"type": "enabled", "budget_tokens": 8000}


def test_no_reasoning_no_thinking():
    body = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 100,
    }
    result, model = translate_openai_to_anthropic(body, inject_billing=False)
    assert "thinking" not in result


def test_unknown_effort_falls_back_to_default():
    body = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "unknown_value"},
        "max_tokens": 100,
    }
    result, model = translate_openai_to_anthropic(body, inject_billing=False)
    assert result.get("thinking") == {"type": "enabled", "budget_tokens": 4096}


def test_anthropic_stream_decoder_thinking_delta():
    decoder = AnthropicStreamDecoder("test-model")
    decoder.process_sse_lines(
        [json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}})]
    )
    chunk = decoder.process_sse_lines(
        [json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think about this."}})]
    )
    assert chunk is not None
    delta = chunk["choices"][0]["delta"]
    assert "reasoning_details" in delta
    assert delta["reasoning_details"][0]["type"] == "reasoning.text"
    assert delta["reasoning_details"][0]["text"] == "Let me think about this."


def test_anthropic_stream_decoder_redacted_thinking():
    decoder = AnthropicStreamDecoder("test-model")
    decoder.process_sse_lines(
        [json.dumps({"type": "content_block_start", "index": 1, "content_block": {"type": "redacted_thinking", "data": "enc123"}})]
    )
    resp = decoder.build_final_response()
    msg = resp["choices"][0]["message"]
    assert "reasoning_details" in msg
    assert msg["reasoning_details"] == [{"type": "reasoning.encrypted", "data": "enc123"}]


def test_anthropic_thinking_to_openai_reasoning():
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "thinking": {"type": "enabled", "budget_tokens": 4096},
    }
    openai_body, model = translate_anthropic_to_openai(body, model_prefix="prefix/")
    assert openai_body.get("reasoning") == {"max_tokens": 4096}

    body["thinking"] = {"type": "disabled"}
    openai_body, model = translate_anthropic_to_openai(body, model_prefix="prefix/")
    assert openai_body.get("reasoning") == {"enabled": False}


if __name__ == "__main__":
    test_reasoning_effort_to_budget()
    test_reasoning_exclude_disabled()
    test_reasoning_max_tokens_direct()
    test_no_reasoning_no_thinking()
    test_unknown_effort_falls_back_to_default()
    test_anthropic_stream_decoder_thinking_delta()
    test_anthropic_stream_decoder_redacted_thinking()
    test_anthropic_thinking_to_openai_reasoning()
    print("All tests passed!")