#!/usr/bin/env python3
"""
Test script for the OpenRouter proxy using an OpenAI-compatible client path.
Tests the proxy using configuration from config.yml.
"""

import asyncio
import json
import os

import httpx
import yaml

API_MODE = os.environ.get("API_MODE", "responses").strip().lower()
DEFAULT_MODELS = {
    "responses": "deepseek/deepseek-r1:free",
    "chat": "deepseek/deepseek-r1:free",
    "embeddings": "openai/text-embedding-3-small",
}
MODEL = os.environ.get("MODEL", DEFAULT_MODELS.get(API_MODE, "deepseek/deepseek-r1:free"))
STREAM = False if API_MODE == "embeddings" else True
MAX_TOKENS = 600


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}

def load_config():
    """
    Load configuration from config.yml
    """
    with open("config.yml", encoding="utf-8") as file:
        return yaml.safe_load(file)


# Get configuration
config = load_config()
server_config = config["server"]

# Configure proxy settings from config
host = server_config["host"]
# Replace 0.0.0.0 with 127.0.0.1 for client connections
if host == "0.0.0.0":
    host = "127.0.0.1"
port = server_config["port"]
PROXY_URL = f"http://{host}:{port}"
ACCESS_KEY = server_config["access_key"]
PROXY_BASE_PATH = os.environ.get("PROXY_BASE_PATH", "/v1").strip() or "/v1"
if not PROXY_BASE_PATH.startswith("/"):
    PROXY_BASE_PATH = "/" + PROXY_BASE_PATH
PROXY_BASE_PATH = PROXY_BASE_PATH.rstrip("/") or "/v1"

# Override with environment variable if set
if os.environ.get("ACCESS_KEY"):
    ACCESS_KEY = os.environ.get("ACCESS_KEY")

INCLUDE_REASONING = env_flag("INCLUDE_REASONING", False)


async def test_proxy_streaming():
    """
    Test the proxy with a standard OpenAI-style request.
    """
    if API_MODE == "embeddings":
        endpoint_path = f"{PROXY_BASE_PATH}/embeddings"
    elif API_MODE == "chat":
        endpoint_path = f"{PROXY_BASE_PATH}/chat/completions"
    else:
        endpoint_path = f"{PROXY_BASE_PATH}/responses"
    print(f"Testing proxy at {PROXY_URL}{endpoint_path} with model {MODEL}")

    url = f"{PROXY_URL}{endpoint_path}"
    headers = {"Authorization": f"Bearer {ACCESS_KEY or 'dummy'}"}
    if not ACCESS_KEY:
        print("No valid access key found. Request may fail if server requires authentication.")
    else:
        print(f"Using access key: {ACCESS_KEY[:5]}...{ACCESS_KEY[-5:]}")

    prompt = "Write a short poem about AI and humanity working together"
    if API_MODE == "embeddings":
        request_data = {
            "model": MODEL,
            "input": "The quick brown fox jumps over the lazy dog",
            "encoding_format": "float",
        }
    elif API_MODE == "responses":
        request_data = {
            "model": MODEL,
            "input": prompt,
            "stream": STREAM,
            "max_output_tokens": MAX_TOKENS,
        }
        if INCLUDE_REASONING:
            request_data["reasoning"] = {"effort": "low"}
    else:
        request_data = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": STREAM,
            "max_tokens": MAX_TOKENS,
        }
        if INCLUDE_REASONING:
            request_data["include_reasoning"] = True

    client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, read=600.0))
    req = client.build_request("POST", url, headers=headers, json=request_data)

    print(f"\nStarting to receive data streaming: {STREAM}...\n")
    print("-" * 50)

    resp = await client.send(req, stream=STREAM)
    try:
        resp.raise_for_status()
        if STREAM:
            reasoning_phase = False
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                if API_MODE == "responses":
                    event_type = data.get("type")
                    if event_type == "response.output_text.delta":
                        if content := data.get("delta"):
                            print(content, end="", flush=True)
                    elif event_type == "response.reasoning_text.delta":
                        if reasoning := data.get("delta"):
                            if not reasoning_phase:
                                reasoning_phase = True
                                print("<reasoning>")
                            print(reasoning, end="", flush=True)
                    elif event_type in {"response.completed", "response.incomplete", "response.failed"}:
                        if reasoning_phase:
                            reasoning_phase = False
                            print("</reasoning>")
                        print(f"\n[{event_type}]")
                else:
                    if "error" in data:
                        raise ValueError(str(data))
                    choice = data["choices"][0]["delta"]
                    if content := choice.get("content"):
                        if reasoning_phase:
                            reasoning_phase = False
                            print("</reasoning>\n")
                        print(content, end='', flush=True)
                    elif reasoning := choice.get("reasoning"):
                        if not reasoning_phase:
                            reasoning_phase = True
                            print("<reasoning>")
                        print(reasoning, end='', flush=True)
        else:
            data = resp.json()
            if API_MODE == "embeddings":
                if "error" in data:
                    raise ValueError(str(data))
                vectors = data.get("data", [])
                print(f"embedding_count={len(vectors)}")
                if vectors:
                    print(f"embedding_dimensions={len(vectors[0].get('embedding', []))}")
                    print(f"model={data.get('model')}")
            elif API_MODE == "responses":
                if "error" in data:
                    raise ValueError(str(data))
                if output_text := data.get("output_text"):
                    print(output_text, end='')
                else:
                    for item in data.get("output", []):
                        if item.get("type") == "message":
                            for part in item.get("content", []):
                                if part.get("type") == "output_text":
                                    print(part.get("text", ""), end='')
                                elif part.get("type") == "output_refusal":
                                    print(part.get("refusal", ""), end='')
                        elif item.get("type") == "reasoning":
                            for part in item.get("summary", []):
                                if part.get("type") == "summary_text":
                                    print(f"<reasoning>\n{part.get('text', '')}</reasoning>\n")
            else:
                if "error" in data:
                    raise ValueError(str(data))
                choice = data["choices"][0]["message"]
                if reasoning := choice.get("reasoning"):
                    print(f"<reasoning>\n{reasoning}</reasoning>\n")
                if content := choice.get("content"):
                    print(content, end='')
    except Exception as e:
        print(f"Error occurred during test: {str(e)}")
    finally:
        if STREAM:
            await resp.aclose()
    print("\n" + "-" * 50)
    if STREAM:
        print("\nStream completed!")
    else:
        print("\nNon-streaming response completed!")


if __name__ == "__main__":
    asyncio.run(test_proxy_streaming())
