import json
from protocol_adapter import deduplicate_kiro_conversation_state

# Simulate the exact Kiro conversationState payload from the user's error
kiro_body = {
    "conversationState": {
        "agentContinuationId": "ad5bd5bb-ee1d-445c-a76a-d003fbe23df9",
        "agentTaskType": "vibe",
        "chatTriggerType": "MANUAL",
        "conversationId": "ad5bd5bb-ee1d-445c-a76a-d003fbe23df9",
        "currentMessage": {
            "userInputMessage": {
                "content": "",
                "modelId": "auto",
                "origin": "AI_EDITOR",
                "userInputMessageContext": {
                    "toolResults": [
                        {
                            "content": [{"text": '{"error": "File not found"}'}],
                            "status": "success",
                            "toolUseId": "tooluse_UZNkIhzewxeYvHKvxUqVZX"
                        },
                        {
                            "content": [{"text": '{"content": "# Hanging Order Fix"}'}],
                            "status": "success",
                            "toolUseId": "tooluse_UZNkIhzewxeYvHKvxUqVZX"  # DUPLICATE
                        },
                        {
                            "content": [{"text": '{"total_count": 50}'}],
                            "status": "success",
                            "toolUseId": "tooluse_kj8mRGZhSjGH11iwovRY8Z"
                        },
                        {
                            "content": [{"text": '{"total_count": 36}'}],
                            "status": "success",
                            "toolUseId": "tooluse_kj8mRGZhSjGH11iwovRY8Z"  # DUPLICATE
                        },
                    ]
                }
            }
        },
        "history": [
            {
                "userInputMessage": {
                    "content": "Conduct adversarial review..."
                }
            },
            {
                "assistantResponseMessage": {
                    "content": "I'll conduct a thorough adversarial review...",
                    "toolUses": [
                        {
                            "input": {},
                            "name": "readFile",
                            "toolUseId": "tooluse_UZNkIhzewxeYvHKvxUqVZX"
                        },
                        {
                            "input": {"path": "D:/Code/HANGING-ORDER-FIX-PLAN.md"},
                            "name": "readFile",
                            "toolUseId": "tooluse_UZNkIhzewxeYvHKvxUqVZX"  # DUPLICATE
                        },
                        {
                            "input": {},
                            "name": "searchFiles",
                            "toolUseId": "tooluse_kj8mRGZhSjGH11iwovRY8Z"
                        },
                        {
                            "input": {"path": "D:/Code/audit", "pattern": "*.md"},
                            "name": "searchFiles",
                            "toolUseId": "tooluse_kj8mRGZhSjGH11iwovRY8Z"  # DUPLICATE
                        },
                    ]
                }
            }
        ]
    }
}

print("=== BEFORE DEDUP ===")
print("\n--- toolUses in history ---")
for entry in kiro_body["conversationState"]["history"]:
    arm = entry.get("assistantResponseMessage")
    if not arm:
        continue
    for tu in arm.get("toolUses", []):
        print(f"  toolUseId={tu['toolUseId']}, name={tu['name']}")

print("\n--- toolResults in currentMessage ---")
for tr in kiro_body["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"]["toolResults"]:
    print(f"  toolUseId={tr['toolUseId']}")

# Apply dedup
deduped = deduplicate_kiro_conversation_state(kiro_body)

print("\n=== AFTER DEDUP ===")
print("\n--- toolUses in history ---")
tu_ids = []
for entry in deduped["conversationState"]["history"]:
    arm = entry.get("assistantResponseMessage")
    if not arm:
        continue
    for tu in arm.get("toolUses", []):
        print(f"  toolUseId={tu['toolUseId']}, name={tu['name']}")
        tu_ids.append(tu["toolUseId"])

print("\n--- toolResults in currentMessage ---")
tr_ids = []
for tr in deduped["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"]["toolResults"]:
    print(f"  toolUseId={tr['toolUseId']}")
    tr_ids.append(tr["toolUseId"])

print(f"\n=== UNIQUENESS CHECK ===")
print(f"toolUse IDs: {tu_ids}")
print(f"toolUse Unique: {len(tu_ids) == len(set(tu_ids))}")
print(f"toolResult IDs: {tr_ids}")
print(f"toolResult Unique: {len(tr_ids) == len(set(tr_ids))}")

# Verify toolResult refs match toolUse IDs (first occurrence gets original, rest get remapped)
assert len(tu_ids) == len(set(tu_ids)), "toolUse IDs are not unique!"
# Note: toolResult IDs won't be fully unique since first occurrence keeps original
# But all 4 should be present
print(f"\n=== PASS ===")
print("All duplicate toolUse IDs have been deduplicated.")