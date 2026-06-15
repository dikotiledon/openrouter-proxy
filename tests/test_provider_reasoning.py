"""Tests for provider registry reasoning integration."""
from provider_registry import (
    REASONING_MODE_OPENAI,
    REASONING_MODE_ANTHROPIC,
    REASONING_MODE_ENABLE_FLAG,
    REASONING_MODE_CHAT_TEMPLATE,
    _inject_reasoning,
    build_provider_registry,
    normalize_request_body_for_provider,
)
from config import config


def test_reasoning_effort_injection():
    registry = build_provider_registry(config)
    ar = registry.get("agentrouter")
    assert ar.reasoning_effort == "medium"
    assert ar.reasoning_mode == "anthropic_thinking"

    body = {
        "model": "agentrouter/deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
    }
    normalized = normalize_request_body_for_provider(ar, body)
    assert normalized.get("reasoning") == {"effort": "medium"}


def test_client_reasoning_not_overridden():
    registry = build_provider_registry(config)
    ar = registry.get("agentrouter")

    body = {
        "model": "agentrouter/deepseek-v4-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning": {"effort": "high"},
    }
    normalized = normalize_request_body_for_provider(ar, body)
    assert normalized.get("reasoning") == {"effort": "high"}


def test_provider_without_reasoning():
    registry = build_provider_registry(config)
    for provider_id, provider in registry.items():
        if provider_id != "agentrouter":
            body = {
                "model": f"{provider_id}/test-model",
                "messages": [{"role": "user", "content": "hi"}],
            }
            if provider_id not in ("mimo", "kiroman", "openrouter"):
                continue
            normalized = normalize_request_body_for_provider(provider, body)
            assert "reasoning" not in normalized, f"Provider {provider_id} should not inject reasoning"


def test_inject_openai_reasoning():
    body = {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]}
    result = _inject_reasoning(body, REASONING_MODE_OPENAI, "high")
    assert result.get("reasoning") == {"effort": "high"}
    assert "enable_thinking" not in result


def test_inject_anthropic_thinking():
    body = {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]}
    result = _inject_reasoning(body, REASONING_MODE_ANTHROPIC, "xhigh")
    assert result.get("reasoning") == {"effort": "xhigh"}


def test_inject_enable_thinking_flag():
    body = {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]}
    result = _inject_reasoning(body, REASONING_MODE_ENABLE_FLAG, "medium")
    assert result.get("enable_thinking") is True
    assert "reasoning" not in result


def test_inject_chat_template():
    body = {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]}
    result = _inject_reasoning(body, REASONING_MODE_CHAT_TEMPLATE, "low")
    assert result["chat_template_kwargs"]["enable_thinking"] is True
    assert result["chat_template_kwargs"]["thinking_budget"] == 1024
    assert "reasoning" not in result


if __name__ == "__main__":
    test_reasoning_effort_injection()
    test_client_reasoning_not_overridden()
    test_provider_without_reasoning()
    test_inject_openai_reasoning()
    test_inject_anthropic_thinking()
    test_inject_enable_thinking_flag()
    test_inject_chat_template()
    print("All provider integration tests passed!")