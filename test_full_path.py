import json
from protocol_adapter import (
    deduplicate_openai_tool_call_ids,
    translate_openai_to_anthropic,
    _openai_tools_to_anthropic,
)

# Exact payload structure from the user's error
openai_body = {
    "model": "auto",
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
    "stream": True,
    "tools": [
        {"type": "function", "function": {"name": "readFile", "description": "Read file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "searchFiles", "description": "Search", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    ]
}

print("=== STEP 1: deduplicate_openai_tool_call_ids ===")
deduped = deduplicate_openai_tool_call_ids(openai_body)
for msg in deduped["messages"]:
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            print(f"  tool_call id={tc['id']} name={tc['function']['name']}")
    elif msg.get("role") == "tool":
        print(f"  tool_result tool_call_id={msg['tool_call_id']}")

print("\n=== STEP 2: translate_openai_to_anthropic ===")
anthropic_body, model_name = translate_openai_to_anthropic(
    deduped,
    model_prefix="kirochina/",
    default_max_tokens=8192,
    inject_billing=False,
    billing_header="",
)

# Print the full Anthropic body as JSON
print(json.dumps(anthropic_body, indent=2))

print("\n=== STEP 3: UNIQUENESS CHECK ===")
all_ids = []
for msg in anthropic_body["messages"]:
    for block in msg.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            all_ids.append(block["id"])
print(f"Tool use IDs: {all_ids}")
print(f"ALL UNIQUE: {len(all_ids) == len(set(all_ids))}")

# Now check what Kiro actually receives
print("\n=== STEP 4: Final payload bytes ===")
payload_bytes = json.dumps(anthropic_body, ensure_ascii=False).encode("utf-8")
# Parse back and verify
check = json.loads(payload_bytes)
check_ids = []
for msg in check["messages"]:
    for block in msg.get("content", []):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            check_ids.append(block["id"])
print(f"Payload tool_use IDs: {check_ids}")
print(f"PAYLOAD UNIQUE: {len(check_ids) == len(set(check_ids))}")
