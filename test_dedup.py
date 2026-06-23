import json
from protocol_adapter import deduplicate_openai_tool_call_ids, translate_openai_to_anthropic

# Simulate the exact OpenAI request with duplicate tool_call IDs from the user's error
openai_body = {
    "model": "kirochina/auto",
    "messages": [
        {"role": "system", "content": "You are Hermes Agent..."},
        {"role": "user", "content": "Conduct adversarial review..."},
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
        {"role": "tool", "tool_call_id": "tooluse_UZNkIhzewxeYvHKvxUqVZX", "content": '{"content": "# Hanging Order Fix"}'},
        {"role": "tool", "tool_call_id": "tooluse_kj8mRGZhSjGH11iwovRY8Z", "content": '{"total_count": 50}'},
        {"role": "tool", "tool_call_id": "tooluse_kj8mRGZhSjGH11iwovRY8Z", "content": '{"total_count": 36}'},
    ]
}

print("=== BEFORE DEDUP ===")
for msg in openai_body["messages"]:
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            print(f"  tool_call: id={tc['id']}, name={tc['function']['name']}")
    elif msg.get("role") == "tool":
        print(f"  tool_result: tool_call_id={msg['tool_call_id']}")

print("\n=== AFTER OPENAI DEDUP ===")
deduped = deduplicate_openai_tool_call_ids(openai_body)
for msg in deduped["messages"]:
    if msg.get("role") == "assistant" and msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            print(f"  tool_call: id={tc['id']}, name={tc['function']['name']}")
    elif msg.get("role") == "tool":
        print(f"  tool_result: tool_call_id={msg['tool_call_id']}")

print("\n=== AFTER TRANSLATE TO ANTHROPIC ===")
anthropic_body, model = translate_openai_to_anthropic(deduped, model_prefix="kirochina/")
for msg in anthropic_body["messages"]:
    if msg.get("role") == "assistant":
        for block in msg.get("content", []):
            if block.get("type") == "tool_use":
                print(f"  tool_use: id={block['id']}, name={block['name']}, input={json.dumps(block['input'])[:80]}")
    elif msg.get("role") == "user":
        for block in msg.get("content", []):
            if block.get("type") == "tool_result":
                print(f"  tool_result: tool_use_id={block['tool_use_id']}")

# Check uniqueness
tool_use_ids = []
for msg in anthropic_body["messages"]:
    if msg.get("role") == "assistant":
        for block in msg.get("content", []):
            if block.get("type") == "tool_use":
                tool_use_ids.append(block["id"])

print(f"\n=== UNIQUENESS CHECK ===")
print(f"Tool use IDs: {tool_use_ids}")
print(f"Unique: {len(tool_use_ids) == len(set(tool_use_ids))}")
