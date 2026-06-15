"""Tests for Gemini protocol adapter."""
import json
import pytest
from gemini_adapter import (
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
        result, _ = translate_openai_to_gemini(body)
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
                },
            ],
        }
        result, _ = translate_openai_to_gemini(body)
        # Assistant message should have functionCall (after the empty text part)
        assert result["contents"][1]["role"] == "model"
        func_parts = [p for p in result["contents"][1]["parts"] if "functionCall" in p]
        assert len(func_parts) == 1
        assert func_parts[0]["functionCall"]["name"] == "get_weather"
        assert result["contents"][2]["role"] == "user"
        assert "functionResponse" in result["contents"][2]["parts"][0]
        fr = result["contents"][2]["parts"][0]["functionResponse"]
        assert fr["name"] == "get_weather"
        assert fr["response"] == {"temperature": 72}
        assert fr["id"] == "call_1"

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
        assert result["toolConfig"]["includeServerSideToolInvocations"] is True
        assert "function_calling_config" not in result["toolConfig"]
        assert "include_server_side_tool_invocations" not in result["toolConfig"]
        assert "tool_config" in result
        assert "function_calling_config" in result["tool_config"]
        assert "include_server_side_tool_invocations" in result["tool_config"]
        assert "functionCallingConfig" not in result["tool_config"]
        assert "includeServerSideToolInvocations" not in result["tool_config"]

    def test_tools_without_tool_choice_gets_default_config(self):
        body = {
            "model": "test/model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "do_thing",
                    "description": "Does a thing",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
        }
        result, _ = translate_openai_to_gemini(body)
        assert result["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
        assert result["toolConfig"]["includeServerSideToolInvocations"] is True
        assert "function_calling_config" not in result["toolConfig"]
        assert "include_server_side_tool_invocations" not in result["toolConfig"]
        assert "tool_config" in result
        assert "function_calling_config" in result["tool_config"]
        assert "include_server_side_tool_invocations" in result["tool_config"]
        assert "functionCallingConfig" not in result["tool_config"]
        assert "includeServerSideToolInvocations" not in result["tool_config"]

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
        assert msgs[2]["tool_call_id"] == msgs[1]["tool_calls"][0]["id"]
        assert msgs[2]["tool_call_id"] != ""

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

    def test_empty_contents_raises(self):
        with pytest.raises(ValueError):
            translate_gemini_to_openai({"model": "test", "contents": []})


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

    def test_tool_config_clean_casing(self):
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"loc": {"type": "string"}},
                },
            }],
            "tool_choice": {"type": "auto"},
        }
        result, _ = translate_anthropic_to_gemini(body)
        assert result["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
        assert result["toolConfig"]["includeServerSideToolInvocations"] is True
        assert "function_calling_config" not in result["toolConfig"]
        assert "include_server_side_tool_invocations" not in result["toolConfig"]
        assert "tool_config" in result
        assert "function_calling_config" in result["tool_config"]
        assert "include_server_side_tool_invocations" in result["tool_config"]
        assert "functionCallingConfig" not in result["tool_config"]
        assert "includeServerSideToolInvocations" not in result["tool_config"]


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
        assert model == "gemini-2.0-flash"
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

    def test_gemini_to_anthropic_tool_id_alignment(self):
        body = {
            "model": "google/gemini-2.0-flash",
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "write_file", "args": {"content": "abc"}, "id": "call_1"}}],
                },
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"name": "write_file", "response": {"result": "success"}, "id": "call_1"}}],
                },
            ],
        }
        result, _ = translate_gemini_to_anthropic(body)
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["content"][0]["type"] == "tool_use"
        assert result["messages"][1]["content"][0]["id"] == "call_1"
        assert result["messages"][2]["role"] == "user"
        assert result["messages"][2]["content"][0]["type"] == "tool_result"
        assert result["messages"][2]["content"][0]["tool_use_id"] == "call_1"


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

    def test_function_call_streaming(self):
        decoder = GeminiStreamDecoder("gemini-2.0-flash")
        chunks = decoder.process_sse_data({
            "candidates": [{"content": {"role": "model", "parts": [
                {"functionCall": {"name": "get_weather", "args": {"loc": "NYC"}}},
            ]}, "finishReason": "STOP", "index": 0}],
        })
        assert len(chunks) == 1
        tc = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"

    def test_empty_candidates(self):
        decoder = GeminiStreamDecoder("test")
        chunks = decoder.process_sse_data({"candidates": []})
        assert chunks == []


class TestGeminiAnthropicStreamDecoder:
    def test_text_streaming(self):
        decoder = GeminiAnthropicStreamDecoder("gemini-2.0-flash")
        events = decoder.process_sse_data({
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1, "totalTokenCount": 6},
        })
        assert "event: message_start" in events
        assert "event: content_block_start" in events
        assert "Hello" in events

        finish = decoder.finish_message()
        assert "event: message_delta" in finish
        assert "event: message_stop" in finish


class TestOpenAIToGeminiSSETranslator:
    def test_basic_translation(self):
        translator = OpenAIToGeminiSSETranslator("test-model")
        chunk = {
            "id": "chatcmpl-123",
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        }
        result = translator.translate_chunk(chunk)
        assert result is not None
        assert result["candidates"][0]["content"]["parts"][0]["text"] == "Hello"

    def test_finish_translation(self):
        translator = OpenAIToGeminiSSETranslator("test-model")
        translator.translate_chunk({
            "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}],
        })
        result = translator.translate_chunk({
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        })
        assert result is not None
        assert result["candidates"][0]["finishReason"] == "STOP"


class TestGeminiResponseTranslation:
    def test_to_openai(self):
        gemini_data = {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello!"}]},
                "finishReason": "STOP",
                "index": 0,
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
        }
        result = translate_gemini_response_to_openai(gemini_data, "gemini-2.0-flash")
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 5

    def test_to_anthropic(self):
        gemini_data = {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello!"}]},
                "finishReason": "STOP",
                "index": 0,
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2, "totalTokenCount": 7},
        }
        result = translate_gemini_response_to_anthropic(gemini_data, "gemini-2.0-flash")
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 5
