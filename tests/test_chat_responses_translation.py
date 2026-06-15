"""Tests for Chat Completions ↔ Responses API and Anthropic ↔ Responses translation."""
import json
from protocol_adapter import (
    translate_openai_chat_to_responses,
    translate_openai_responses_to_chat,
    translate_anthropic_to_responses,
    translate_responses_to_anthropic,
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

    def test_multimodal_content(self):
        chat = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png", "detail": "high"}},
                ],
            }],
        }
        result = translate_openai_chat_to_responses(chat)
        items = result["input"][0]["content"]
        assert items[0]["type"] == "input_text"
        assert items[1]["type"] == "input_image"
        assert items[1]["detail"] == "high"


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

    def test_string_input(self):
        responses = {"model": "gpt-4o", "input": "Hello"}
        result = translate_openai_responses_to_chat(responses)
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"


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

    def test_tool_use(self):
        body = {
            "model": "claude",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"loc": "NYC"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "sunny"},
                ]},
            ],
        }
        result = translate_anthropic_to_responses(body)
        assert result["input"][1]["content"][0]["type"] == "tool_call"
        assert result["input"][2]["type"] == "function_call_output"


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

    def test_with_reasoning(self):
        body = {
            "model": "gpt-4o",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "reasoning": {"max_tokens": 8192},
        }
        result = translate_responses_to_anthropic(body)
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 8192
