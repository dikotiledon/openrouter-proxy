# OpenRouter Proxy

A provider-routed OpenAI-compatible proxy for OpenRouter, Ollama, and OpenAI.
It forwards client requests to a configured upstream provider registry and rotates
through provider-specific API keys when applicable.

## Features

- Proxies OpenAI-compatible client requests through a provider registry
- Supports both `/v1` and `/api/v1` client prefixes
- Requires provider-qualified model IDs such as `openrouter/...`, `ollama/...`, and `openai/...`
- Aggregates `/models` from configured providers and prefixes returned model IDs
- Proxies OpenAI Responses and embeddings requests to the selected provider
- Enforces optional provider-scoped `allowed_models` lists and free-only policies
- Passes provider-supported prompt-caching, reasoning, sampling, and tool-calling fields through unchanged
- Rotates multiple API keys per provider to bypass rate limits
- Automatically disables provider API keys temporarily when rate limits are reached
- Streams responses chunk by chunk for efficient data transfer
- Simple authentication for accessing the proxy
- Preserves OpenAI-style error envelopes for client-facing compatibility
- Supports any OpenAI-compatible upstream by adding a provider block in `config.yml`

## Setup

1. Clone the repository
2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Create a configuration file:
   ```
   cp config.yml.example config.yml
   ```
4. Edit `config.yml` to add one or more provider blocks and configure the server

## Configuration

The `config.yml` file supports the following settings:

```yaml
# Server settings
server:
  host: "0.0.0.0" # Interface to bind to
  port: 5555 # Port to listen on
  access_key: "your_local_access_key_here" # Authentication key
  log_level: "INFO" # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  http_log_level: "INFO" # HTTP access logs level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Provider registry
providers:
  openrouter:
    keys:
      - "sk-or-v1-your-first-api-key"
      - "sk-or-v1-your-second-api-key"

    # Optional list of model IDs that clients are allowed to use.
    # Every entry must be provider-qualified.
    allowed_models:
      - "openrouter/google/gemma-4-31b-it:free"
      - "openrouter/openai/text-embedding-3-small"

    # Key selection strategy: "round-robin" (default), "first" or "random".
    key_selection_strategy: "round-robin"
    key_selection_opts: []
    base_url: "https://openrouter.ai/api/v1"
    public_endpoints:
      - "/models"
    rate_limit_cooldown: 14400
    free_only: true
    # OpenRouter can return a 429 error if a model is overloaded.
    # Additionally, Google sometimes returns 429 RESOURCE_EXHAUSTED errors repeatedly,
    # which can cause Roo Code to stop.
    # This option spaces out upstream requests for the provider.
    # Set it to 0 to disable pacing.
    # global_rate_delay: 10 # in seconds
    global_rate_delay: 30

  ollama:
    base_url: "http://127.0.0.1:11434/v1"
    keys: []
    allowed_models:
      - "ollama/llama3.1"
      - "ollama/qwen2.5:14b"
    key_selection_strategy: "round-robin"
    key_selection_opts: []
    public_endpoints:
      - "/models"
    rate_limit_cooldown: 14400
    global_rate_delay: 0
    supports_stateful_responses: false

  openai:
    base_url: "https://api.openai.com/v1"
    keys:
      - "sk-proj-your-openai-key"
    allowed_models:
      - "openai/o3-mini"
      - "openai/text-embedding-3-small"
    key_selection_strategy: "round-robin"
    key_selection_opts: []
    public_endpoints:
      - "/models"
    rate_limit_cooldown: 14400
    global_rate_delay: 0

# Proxy settings for outgoing requests
requestProxy:
  enabled: false # Set to true to enable proxy
  url: "socks5://username:password@example.com:1080" # Proxy URL with optional credentials embedded
```

### Provider Routing

Requests must choose a provider explicitly.

- Use provider-qualified model IDs such as `openrouter/google/gemma-4-31b-it:free`, `ollama/llama3.1`, and `openai/o3-mini`
- Use `?provider=openrouter`, `?provider=ollama`, or `?provider=openai` to pin model-list requests to one provider
- OpenRouter-only request fields such as `models`, `provider`, and `plugins` are preserved only when the selected provider is `openrouter`
- Providers with no upstream key can use `keys: []` in `config.yml`

## Usage

### Running Manually

Start the server:

```
python main.py
```

The proxy will be available at `http://localhost:5555/v1` and `http://localhost:5555/api/v1` (or the host/port configured in your config file).

OpenAI SDK example:

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:5555/v1",
  api_key="your_local_access_key_here",
)

response = client.chat.completions.create(
  model="openrouter/google/gemma-4-31b-it:free",
  messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

Responses API example:

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:5555/v1",
  api_key="your_local_access_key_here",
)

response = client.responses.create(
  model="openai/o3-mini",
  input="Write one short sentence about the ocean.",
)
print(response.output_text)
```

Embeddings example:

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:5555/v1",
  api_key="your_local_access_key_here",
)

response = client.embeddings.create(
  model="openai/text-embedding-3-small",
  input="The quick brown fox jumps over the lazy dog",
)
print(len(response.data[0].embedding))
```

### Tool Calling

The proxy does not emulate tool calling locally. It forwards provider-compatible request bodies to the selected upstream provider, so tool behavior is determined by the selected model and upstream support.

| Surface                | What is forwarded                                                                                                                                                                                               | What is not done locally                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `/v1/responses`        | `tools`, `tool_choice`, `parallel_tool_calls`, `previous_response_id`, `input` items including `function_call_output`, `stream`, `max_output_tokens`, `reasoning`, `response_format`, and streaming tool events | No translation layer, no local tool execution, no conversation state machine |
| `/v1/chat/completions` | OpenAI-style `tools`, `tool_choice`, `tool_calls`, and `tool_call_id` handling                                                                                                                                  | No conversion from Responses objects into chat messages                      |

Notes:

- `tool_choice` is passed through as the upstream provider accepts it (`auto`, `none`, or a forced function tool).
- OpenRouter-specific fields such as `cache_control`, `models`, `provider`, `plugins`, `reasoning`, and `reasoning_details` flow through on the OpenRouter provider and are stripped for other providers.
- The proxy does not run tools. Your application still needs to execute the function and send the result back in a follow-up request when the flow requires it.
- If the chosen model does not support tools, the upstream provider may ignore or reject the request.

### Prompt Caching

The proxy does not implement caching locally. It forwards OpenRouter prompt-caching controls unchanged when the selected provider is OpenRouter, including top-level `cache_control` on supported requests and per-block `cache_control` breakpoints inside message content.

That means OpenRouter can apply provider sticky routing upstream, and the proxy will preserve the request body needed for cache hits. Other providers receive only the fields they support.

Example with an OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:5555/v1",
  api_key="your_local_access_key_here",
)

response = client.chat.completions.create(
  model="openrouter/anthropic/claude-sonnet-4.6",
  messages=[
    {
      "role": "system",
      "content": [
        {"type": "text", "text": "Use the reference below when answering."},
        {
          "type": "text",
          "text": "HUGE STABLE REFERENCE TEXT",
          "cache_control": {"type": "ephemeral", "ttl": "1h"},
        },
      ],
    },
    {"role": "user", "content": "Summarize the reference."},
  ],
  extra_body={"cache_control": {"type": "ephemeral", "ttl": "1h"}},
)
```

### Reasoning Tokens

The proxy also forwards OpenRouter reasoning controls unchanged when the selected provider is OpenRouter. Use the unified `reasoning` object for modern requests, and the proxy will preserve returned `reasoning` and `reasoning_details` blocks in both streaming and non-streaming responses.

Legacy `include_reasoning` still passes through, but `reasoning` is the preferred interface.

Example with reasoning enabled:

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:5555/v1",
  api_key="your_local_access_key_here",
)

response = client.chat.completions.create(
  model="openrouter/openai/o3-mini",
  messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}],
  extra_body={"reasoning": {"effort": "high", "exclude": False}},
)

message = response.choices[0].message
print(getattr(message, "reasoning", None))
print(getattr(message, "reasoning_details", None))
print(getattr(message, "content", None))
```

If you replay an assistant turn that contains `reasoning_details`, keep that field unchanged so the reasoning context remains valid for follow-up tool calls.

### Installing as a Systemd Service

For Linux systems with systemd, you can install the proxy as a system service:

1. Make sure you've created and configured your `config.yml` file
2. Run the installation script:

`sudo ./service_install.sh` or `sudo ./service_install_venv.sh` for venv.

This will create a systemd service that starts automatically on boot.

To check the service status:

```
sudo systemctl status openrouter-proxy
```

To view logs:

```
sudo journalctl -u openrouter-proxy -f
```

To uninstall the service:

```
sudo ./service_uninstall.sh
```

### Authentication

Add your local access key to requests:

```
Authorization: Bearer your_local_access_key_here
```

## API Endpoints

The proxy supports provider-routed OpenAI-compatible client endpoints through the following client-facing endpoints:

- `/v1/{path}` - Recommended OpenAI-compatible client surface
- `/api/v1/{path}` - Backward-compatible OpenRouter-style surface
- `/v1/models` or `/api/v1/models` - Aggregated provider model list
- `/v1/models?provider=openrouter` - Provider-scoped model list
- `/v1/responses` - OpenAI Responses API passthrough
- `/v1/responses/{response_id}` - Retrieve, delete, or cancel a response
- `/v1/responses/{response_id}/input_items` - List the stored input items for a response
- `/v1/embeddings` - OpenAI-compatible embeddings passthrough
- `/v1/chat/completions` - OpenAI-compatible chat completions passthrough

It also provides a health check endpoint:

- `/health` - Health check endpoint that returns `{"status": "ok"}`
