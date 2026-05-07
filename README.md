# OpenRouter Proxy

A simple proxy server for OpenRouter API that helps bypass rate limits on free API keys
by rotating through multiple API keys in a round-robin fashion.

## Features

- Proxies OpenAI-compatible client requests to OpenRouter API v1
- Supports both `/v1` and `/api/v1` client prefixes
- Proxies OpenAI Responses API requests directly to OpenRouter's Responses endpoints
- Proxies OpenAI embeddings requests directly to OpenRouter's embeddings endpoint
- Enforces an optional `allowed_models` list so clients can only use configured models
- Filters `/models` to the configured allowlist and free-only policy
- Passes OpenRouter prompt-caching controls, reasoning settings, sampling parameters, and tool-calling fields through unchanged
- Rotates multiple API keys to bypass rate limits
- Automatically disables API keys temporarily when rate limits are reached
- Streams responses chunk by chunk for efficient data transfer
- Simple authentication for accessing the proxy
- Preserves OpenAI-style error envelopes for client-facing compatibility
- Theoretically compatible with any OpenAI-compatible API by changing the `base_url` and `public_endpoints` in `config.yml`

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
4. Edit `config.yml` to add your OpenRouter API keys and configure the server

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

# OpenRouter API keys
openrouter:
  keys:
    - "sk-or-v1-your-first-api-key"
    - "sk-or-v1-your-second-api-key"
    - "sk-or-v1-your-third-api-key"

  # Optional list of model IDs that clients are allowed to use.
  # When set, the proxy rejects any other model and defaults to the first
  # entry when a request omits the model field.
  allowed_models:
    - "deepseek/deepseek-r1:free"
    - "openai/text-embedding-3-small"

  # Key selection strategy: "round-robin" (default), "first" or "random".
  key_selection_strategy: "round-robin"
  # List of key selection options:
  #   "same": Always use the last used key as long as it is possible.
  key_selection_opts: []

  # OpenRouter API base URL
  base_url: "https://openrouter.ai/api/v1"

  # Public endpoints that don't require authentication
  public_endpoints:
    - "/api/v1/models"

  # Time in seconds to temporarily disable a key when rate limit is reached by default
  rate_limit_cooldown: 14400 # 4 hours
  free_only: false # try to show only free models
  # OpenRouter can return a 429 error if a model is overloaded.
  # Additionally, Google sometimes returns 429 RESOURCE_EXHAUSTED errors repeatedly,
  # which can cause Roo Code to stop.
  # This option prevents repeated failures by introducing a delay before retrying.
  # global_rate_delay: 10 # in seconds
  global_rate_delay: 0

# Proxy settings for outgoing requests to OpenRouter
requestProxy:
  enabled: false # Set to true to enable proxy
  url: "socks5://username:password@example.com:1080" # Proxy URL with optional credentials embedded
```

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
  model="deepseek/deepseek-r1:free",
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
  model="deepseek/deepseek-r1:free",
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

The proxy does not emulate tool calling locally. It forwards OpenRouter-compatible request bodies to OpenRouter, so tool behavior is determined by the selected model and upstream support.

| Surface                | What is forwarded                                                                                                                                                                                               | What is not done locally                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `/v1/responses`        | `tools`, `tool_choice`, `parallel_tool_calls`, `previous_response_id`, `input` items including `function_call_output`, `stream`, `max_output_tokens`, `reasoning`, `response_format`, and streaming tool events | No translation layer, no local tool execution, no conversation state machine |
| `/v1/chat/completions` | OpenAI-style `tools`, `tool_choice`, `tool_calls`, and `tool_call_id` handling                                                                                                                                  | No conversion from Responses objects into chat messages                      |

Notes:

- `tool_choice` is passed through as OpenRouter accepts it (`auto`, `none`, or a forced function tool).
- OpenRouter-specific fields such as `cache_control`, `models`, `provider`, `plugins`, `reasoning`, and `reasoning_details` also flow through because the proxy forwards the JSON body unchanged.
- The proxy does not run tools. Your application still needs to execute the function and send the result back in a follow-up request when the flow requires it.
- If the chosen model does not support tools, OpenRouter may ignore or reject the request.

### Prompt Caching

The proxy does not implement caching locally. It forwards OpenRouter prompt-caching controls unchanged, including top-level `cache_control` on supported requests and per-block `cache_control` breakpoints inside message content.

That means OpenRouter can apply provider sticky routing upstream, and the proxy will preserve the request body needed for cache hits.

Example with an OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:5555/v1",
  api_key="your_local_access_key_here",
)

response = client.chat.completions.create(
  model="anthropic/claude-sonnet-4.6",
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

The proxy also forwards OpenRouter reasoning controls unchanged. Use the unified `reasoning` object for modern requests, and the proxy will preserve returned `reasoning` and `reasoning_details` blocks in both streaming and non-streaming responses.

Legacy `include_reasoning` still passes through, but `reasoning` is the preferred interface.

Example with reasoning enabled:

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:5555/v1",
  api_key="your_local_access_key_here",
)

response = client.chat.completions.create(
  model="openai/o3-mini",
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

The proxy supports OpenRouter passthrough and OpenAI-compatible client endpoints through the following client-facing endpoints:

- `/v1/{path}` - Recommended OpenAI-compatible client surface
- `/api/v1/{path}` - Backward-compatible OpenRouter-style surface
- `/v1/responses` - OpenAI Responses API passthrough
- `/v1/responses/{response_id}` - Retrieve, delete, or cancel a response
- `/v1/responses/{response_id}/input_items` - List the stored input items for a response
- `/v1/embeddings` - OpenAI-compatible embeddings passthrough
- `/v1/chat/completions` - OpenAI-compatible chat completions passthrough

It also provides a health check endpoint:

- `/health` - Health check endpoint that returns `{"status": "ok"}`
