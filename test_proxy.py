import json
import httpx

# This mimics what Hermes sends: OpenAI chat/completions with duplicate tool_call IDs
payload = {
    "model": "kirochina/auto",
    "messages": [
        {"role": "system", "content": "You are Hermes Agent."},
        {"role": "user", "content": "Conduct adversarial review"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "tooluse_UZNkIhzewxeYvHKvxUqVZX", "type": "function", "function": {"name": "readFile", "arguments": "{}"}},
                {"id": "tooluse_UZNkIhzewxeYvHKvxUqVZX", "type": "function", "function": {"name": "readFile", "arguments": '{"path": "D:/Code/HANGING-ORDER-FIX-PLAN.md"}'}},
                {"id": "tooluse_kj8mRGZhSjGH11iwovRY8Z", "type": "function", "function": {"name": "searchFiles", "arguments": "{}"}},
                {"id": "tooluse_kj8mRGZhSjGH11iwovRY8Z", "type": "function", "function": {"name": "searchFiles", "arguments": '{"path": "D:/Code/audit", "pattern": "*.md"}'}},
            ]
        },
        {"role": "tool", "tool_call_id": "tooluse_UZNkIhzewxeYvHKvxUqVZX", "content": '{"error": "File not found"}'},
        {"role": "tool", "tool_call_id": "tooluse_UZNkIhzewxeYvHKvxUqVZX", "content": '{"content": "# Fix plan"}'},
        {"role": "tool", "tool_call_id": "tooluse_kj8mRGZhSjGH11iwovRY8Z", "content": '{"total_count": 50}'},
        {"role": "tool", "tool_call_id": "tooluse_kj8mRGZhSjGH11iwovRY8Z", "content": '{"total_count": 36}'},
    ],
    "max_tokens": 256,
    "stream": False,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "readFile",
                "description": "Read a text file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "searchFiles",
                "description": "Search file contents",
                "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}
            }
        }
    ]
}

print("Sending request to proxy...")
resp = httpx.post(
    "http://127.0.0.1:5556/v1/chat/completions",
    headers={"Authorization": "Bearer sk-lm-Ql7o5B5H:H36USx7eyg6qoUA1f0bn", "Content-Type": "application/json"},
    json=payload,
    timeout=60,
)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.text[:2000]}")
