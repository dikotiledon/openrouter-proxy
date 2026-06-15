"""Tests for the full 4x4 translation matrix."""
import json
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
        assert any(m.get("role") == "system" for m in restored["messages"])
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

    def test_openai_responses_roundtrip(self):
        original = _make_openai_chat()
        responses = translate_openai_chat_to_responses(original)
        restored = translate_openai_responses_to_chat(responses)
        assert restored["messages"][0]["role"] == "system"
        user_msgs = [m for m in restored["messages"] if m.get("role") == "user"]
        assert len(user_msgs) >= 1
