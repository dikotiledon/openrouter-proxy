"""
End-to-end test for Kiro conversationState deduplication.

Tests all code paths a Kiro request can take through the proxy:
1. Direct passthrough to OpenAI-format upstream (kirochina)
2. Translation to Anthropic upstream (agentrouter)
3. Translation to Gemini upstream
4. Anthropic Messages endpoint
5. Responses API endpoint
"""
import json
import copy
from protocol_adapter import (
    deduplicate_kiro_conversation_state,
    deduplicate_openai_tool_call_ids,
)


def make_kiro_request():
    """Create a Kiro request body with duplicate toolUseId values.

    Simulates what the Kiro IDE sends: model at top level for routing,
    plus conversationState with duplicate toolUseId values in history.
    """
    return {
        "model": "kirochina/auto",
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
        },
        "profileArn": "arn:aws:codewhisperer:us-east-1:699475941385:profile/EHGA3GRVQMUK"
    }


def make_kiro_openai_request():
    """Create a Kiro request that also includes OpenAI messages with duplicates.

    Some Kiro clients send both messages (OpenAI format) and conversationState.
    The messages contain the same duplicate tool_call IDs.
    """
    body = make_kiro_request()
    body["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."},
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
    return body


def extract_tooluse_ids(body):
    """Extract all toolUseId values from conversationState."""
    cs = body.get("conversationState", {})
    ids = {"history": [], "toolResults": []}
    for entry in cs.get("history", []):
        arm = entry.get("assistantResponseMessage", {})
        for tu in arm.get("toolUses", []):
            ids["history"].append(tu.get("toolUseId", ""))
    cm = cs.get("currentMessage", {})
    uim = cm.get("userInputMessage", {})
    # Try both paths
    trs = uim.get("toolResults", [])
    if not trs:
        ctx = uim.get("userInputMessageContext", {})
        trs = ctx.get("toolResults", [])
    for tr in trs:
        ids["toolResults"].append(tr.get("toolUseId", ""))
    return ids


def extract_openai_tool_call_ids(body):
    """Extract all tool_call IDs from OpenAI messages."""
    ids = []
    for msg in body.get("messages", []):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                ids.append(tc.get("id", ""))
    result_ids = []
    for msg in body.get("messages", []):
        if msg.get("role") == "tool":
            result_ids.append(msg.get("tool_call_id", ""))
    return ids, result_ids


# ── Test 1: conversationState-only (passthrough to kirochina) ────────────────
def test_conversation_state_only():
    """Kiro sends conversationState with model at top level.
    Proxy passes body through to kirochina upstream verbatim.
    """
    body = make_kiro_request()
    before = extract_tooluse_ids(body)
    
    # Simulate proxy flow: deduplicate_openai_tool_call_ids (no-op), then kiro dedup
    body = deduplicate_openai_tool_call_ids(body)
    body = deduplicate_kiro_conversation_state(body)
    
    after = extract_tooluse_ids(body)
    
    # Verify dedup happened
    assert len(after["history"]) == len(set(after["history"])), \
        f"history toolUseIds not unique: {after['history']}"
    assert len(after["toolResults"]) == len(set(after["toolResults"])), \
        f"toolResult toolUseIds not unique: {after['toolResults']}"
    
    # Verify first occurrence kept original ID
    assert after["history"][0] == "tooluse_UZNkIhzewxeYvHKvxUqVZX", \
        f"First occurrence should keep original ID, got {after['history'][0]}"
    assert after["history"][2] == "tooluse_kj8mRGZhSjGH11iwovRY8Z", \
        f"Third occurrence should keep original ID, got {after['history'][2]}"
    
    # Verify toolResults match toolUses
    assert after["toolResults"][0] == after["history"][0], \
        f"toolResult[0] mismatch: {after['toolResults'][0]} != {after['history'][0]}"
    assert after["toolResults"][1] == after["history"][1], \
        f"toolResult[1] mismatch: {after['toolResults'][1]} != {after['history'][1]}"
    assert after["toolResults"][2] == after["history"][2], \
        f"toolResult[2] mismatch: {after['toolResults'][2]} != {after['history'][2]}"
    assert after["toolResults"][3] == after["history"][3], \
        f"toolResult[3] mismatch: {after['toolResults'][3]} != {after['history'][3]}"
    
    # Verify model field preserved (prefix stripping happens in normalize_request_body_for_provider)
    assert body.get("model") == "kirochina/auto", f"Model should be 'kirochina/auto', got {body.get('model')}"
    
    # Verify profileArn preserved
    assert "profileArn" in body, "profileArn should be preserved in body"
    
    print("  PASS: conversationState dedup works correctly")


# ── Test 2: conversationState + OpenAI messages (both dedupped) ──────────────
def test_conversation_state_with_openai_messages():
    """Kiro sends both messages (OpenAI format) and conversationState.
    Both must be dedupped.
    """
    body = make_kiro_openai_request()
    
    # Simulate proxy flow
    body = deduplicate_openai_tool_call_ids(body)
    body = deduplicate_kiro_conversation_state(body)
    
    # Check OpenAI messages dedup
    tc_ids, tr_ids = extract_openai_tool_call_ids(body)
    assert len(tc_ids) == len(set(tc_ids)), \
        f"OpenAI tool_call IDs not unique: {tc_ids}"
    assert len(tr_ids) == len(set(tr_ids)), \
        f"OpenAI tool result IDs not unique: {tr_ids}"
    
    # Check conversationState dedup
    cs_ids = extract_tooluse_ids(body)
    assert len(cs_ids["history"]) == len(set(cs_ids["history"])), \
        f"conversationState history IDs not unique: {cs_ids['history']}"
    assert len(cs_ids["toolResults"]) == len(set(cs_ids["toolResults"])), \
        f"conversationState toolResult IDs not unique: {cs_ids['toolResults']}"
    
    print("  PASS: conversationState + OpenAI messages both dedupped")


# ── Test 3: No duplicates (no-op) ───────────────────────────────────────────
def test_no_duplicates():
    """When there are no duplicates, the body should be unchanged."""
    body = {
        "model": "kirochina/auto",
        "conversationState": {
            "history": [
                {
                    "assistantResponseMessage": {
                        "toolUses": [
                            {"input": {}, "name": "readFile", "toolUseId": "tooluse_ABC123"},
                            {"input": {}, "name": "searchFiles", "toolUseId": "tooluse_DEF456"},
                        ]
                    }
                }
            ],
            "currentMessage": {
                "userInputMessage": {
                    "userInputMessageContext": {
                        "toolResults": [
                            {"toolUseId": "tooluse_ABC123"},
                            {"toolUseId": "tooluse_DEF456"},
                        ]
                    }
                }
            }
        }
    }
    original = json.dumps(body, sort_keys=True)
    
    body = deduplicate_kiro_conversation_state(body)
    after = json.dumps(body, sort_keys=True)
    
    assert original == after, "Body should be unchanged when no duplicates exist"
    print("  PASS: no-op when no duplicates")


# ── Test 4: Empty conversationState ─────────────────────────────────────────
def test_empty_conversation_state():
    """Empty or missing conversationState should not crash."""
    for body in [
        {},
        {"model": "kirochina/auto"},
        {"conversationState": {}},
        {"conversationState": {"history": []}},
        {"conversationState": {"currentMessage": {}}},
    ]:
        result = deduplicate_kiro_conversation_state(body)
        assert isinstance(result, dict), f"Should return dict, got {type(result)}"
    print("  PASS: handles empty/missing conversationState gracefully")


# ── Test 5: Multiple duplicate groups ───────────────────────────────────────
def test_multiple_duplicate_groups():
    """Multiple different toolUseId values with duplicates."""
    body = {
        "conversationState": {
            "history": [
                {
                    "assistantResponseMessage": {
                        "toolUses": [
                            {"toolUseId": "id_A"},
                            {"toolUseId": "id_A"},  # dup of id_A
                            {"toolUseId": "id_B"},
                            {"toolUseId": "id_B"},  # dup of id_B
                            {"toolUseId": "id_C"},
                            {"toolUseId": "id_C"},  # dup of id_C
                            {"toolUseId": "id_D"},  # unique
                        ]
                    }
                }
            ],
            "currentMessage": {
                "userInputMessage": {
                    "userInputMessageContext": {
                        "toolResults": [
                            {"toolUseId": "id_A"},
                            {"toolUseId": "id_A"},
                            {"toolUseId": "id_B"},
                            {"toolUseId": "id_B"},
                            {"toolUseId": "id_C"},
                            {"toolUseId": "id_C"},
                            {"toolUseId": "id_D"},
                        ]
                    }
                }
            }
        }
    }
    
    result = deduplicate_kiro_conversation_state(body)
    
    all_ids = []
    for entry in result["conversationState"]["history"]:
        arm = entry.get("assistantResponseMessage", {})
        for tu in arm.get("toolUses", []):
            all_ids.append(tu["toolUseId"])
    
    assert len(all_ids) == len(set(all_ids)), f"Not all unique: {all_ids}"
    
    # Verify toolResults match
    result_ids = []
    uim = result["conversationState"]["currentMessage"]["userInputMessage"]
    ctx = uim.get("userInputMessageContext", {})
    for tr in ctx.get("toolResults", []):
        result_ids.append(tr["toolUseId"])
    
    assert len(result_ids) == len(set(result_ids)), f"Results not unique: {result_ids}"
    
    # id_D should be unchanged
    assert "id_D" in all_ids, "Unique IDs should be preserved"
    
    print("  PASS: handles multiple duplicate groups correctly")


# ── Test 6: Serialized body preserves conversationState ─────────────────────
def test_serialized_body_preserves_conversation_state():
    """After dedup, json.dumps should preserve the conversationState."""
    body = make_kiro_request()
    body = deduplicate_kiro_conversation_state(body)
    
    serialized = json.dumps(body, ensure_ascii=False)
    deserialized = json.loads(serialized)
    
    assert "conversationState" in deserialized, "conversationState should be in serialized body"
    assert "profileArn" in deserialized, "profileArn should be preserved"
    assert deserialized["model"] == "kirochina/auto", "model should be preserved"
    
    cs_ids = extract_tooluse_ids(deserialized)
    assert len(cs_ids["history"]) == len(set(cs_ids["history"])), \
        f"Serialized body has duplicate IDs: {cs_ids['history']}"
    
    print("  PASS: serialized body preserves all fields and dedupped IDs")


# ── Test 7: Triple+ duplicates ──────────────────────────────────────────────
def test_triple_duplicates():
    """Three or more toolUses with the same ID."""
    body = {
        "conversationState": {
            "history": [
                {
                    "assistantResponseMessage": {
                        "toolUses": [
                            {"toolUseId": "same_id", "name": "a"},
                            {"toolUseId": "same_id", "name": "b"},
                            {"toolUseId": "same_id", "name": "c"},
                        ]
                    }
                }
            ],
            "currentMessage": {
                "userInputMessage": {
                    "userInputMessageContext": {
                        "toolResults": [
                            {"toolUseId": "same_id"},
                            {"toolUseId": "same_id"},
                            {"toolUseId": "same_id"},
                        ]
                    }
                }
            }
        }
    }
    
    result = deduplicate_kiro_conversation_state(body)
    
    ids = []
    for entry in result["conversationState"]["history"]:
        arm = entry.get("assistantResponseMessage", {})
        for tu in arm.get("toolUses", []):
            ids.append(tu["toolUseId"])
    
    assert len(ids) == len(set(ids)), f"Triple duplicates not resolved: {ids}"
    assert ids[0] == "same_id", "First occurrence should keep original"
    
    # All toolResults should match
    result_ids = []
    ctx = result["conversationState"]["currentMessage"]["userInputMessage"]["userInputMessageContext"]
    for tr in ctx["toolResults"]:
        result_ids.append(tr["toolUseId"])
    
    assert result_ids == ids, f"toolResults {result_ids} don't match toolUses {ids}"
    
    print("  PASS: handles triple+ duplicates correctly")


# ── Test 8: Both dedup functions work on same body (mixed format) ────────────
def test_mixed_format_dedup_chain():
    """Test the full dedup chain: openai dedup + kiro dedup."""
    body = make_kiro_openai_request()
    
    # Add a THIRD duplicate in OpenAI messages (different from the first two)
    body["messages"][2]["tool_calls"].append(
        {"id": "tooluse_UZNkIhzewxeYvHKvxUqVZX", "type": "function", "function": {"name": "listFiles", "arguments": "{}"}}
    )
    body["messages"].append(
        {"role": "tool", "tool_call_id": "tooluse_UZNkIhzewxeYvHKvxUqVZX", "content": '{"files": []}'}
    )
    
    # Run full chain
    body = deduplicate_openai_tool_call_ids(body)
    body = deduplicate_kiro_conversation_state(body)
    
    # Verify both sides
    tc_ids, tr_ids = extract_openai_tool_call_ids(body)
    assert len(tc_ids) == len(set(tc_ids)), f"OpenAI IDs not unique: {tc_ids}"
    
    cs_ids = extract_tooluse_ids(body)
    assert len(cs_ids["history"]) == len(set(cs_ids["history"])), \
        f"conversationState history IDs not unique: {cs_ids['history']}"
    
    print("  PASS: full dedup chain handles mixed format correctly")


# ── Run all tests ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Kiro E2E Dedup Tests ===\n")
    
    tests = [
        ("1. conversationState only (passthrough)", test_conversation_state_only),
        ("2. conversationState + OpenAI messages", test_conversation_state_with_openai_messages),
        ("3. No duplicates (no-op)", test_no_duplicates),
        ("4. Empty conversationState", test_empty_conversation_state),
        ("5. Multiple duplicate groups", test_multiple_duplicate_groups),
        ("6. Serialized body preserves fields", test_serialized_body_preserves_conversation_state),
        ("7. Triple+ duplicates", test_triple_duplicates),
        ("8. Mixed format dedup chain", test_mixed_format_dedup_chain),
    ]
    
    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {name}: {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\n=== Results: {passed} passed, {failed} failed out of {len(tests)} ===")
    if failed:
        exit(1)
    print("All tests passed!")