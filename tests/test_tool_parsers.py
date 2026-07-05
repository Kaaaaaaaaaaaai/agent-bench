from agent_bench.tool_parsers import parse_tool_calls


def test_openai_native_tool_parser_extracts_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"ticker":"MSFT"}'},
                        }
                    ]
                }
            }
        ]
    }

    parsed = parse_tool_calls("openai-native", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "lookup"
    assert parsed.tool_calls[0].arguments == {"ticker": "MSFT"}


def test_json_in_content_tool_parser_extracts_fallback_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "content": '{"tool_calls":[{"name":"search","arguments":{"query":"revenue"}}]}'
                }
            }
        ]
    }

    parsed = parse_tool_calls("json-in-content", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "search"
    assert parsed.tool_calls[0].arguments == {"query": "revenue"}


def test_json_in_content_tool_parser_ignores_final_answer_json():
    payload = {"choices": [{"message": {"content": '{"answer":"B","confidence":0.9}'}}]}

    parsed = parse_tool_calls("json-in-content", payload)

    assert parsed.status == "no_tool_calls"
    assert parsed.error is None
    assert parsed.tool_calls == []


def test_vllm_compatible_parser_handles_function_call_shape():
    payload = {
        "choices": [
            {
                "message": {
                    "function_call": {"name": "final_answer", "arguments": '{"answer":"A"}'}
                }
            }
        ]
    }

    parsed = parse_tool_calls("vllm-compatible", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "final_answer"
    assert parsed.tool_calls[0].arguments == {"answer": "A"}


def test_none_tool_parser_disables_parsing():
    payload = {"choices": [{"message": {"tool_calls": [{"function": {"name": "x"}}]}}]}

    parsed = parse_tool_calls("none", payload)

    assert parsed.status == "no_tool_calls"
    assert parsed.tool_calls == []
