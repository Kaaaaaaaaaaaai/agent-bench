from agent_bench.tool_parsers import normalize_parser_name, parse_tool_calls


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


def test_qwen35_tool_parser_extracts_tagged_json_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '<tool_call>\n'
                        '{"name":"read_file","arguments":{"path":"TASK.md"}}\n'
                        "</tool_call>"
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("qwen3.5", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "read_file"
    assert parsed.tool_calls[0].arguments == {"path": "TASK.md"}


def test_qwen35_tool_parser_extracts_multiple_tagged_json_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '<tool_call>{"name":"read_file","arguments":{"path":"a.txt"}}</tool_call>\n'
                        '<tool_call>{"name":"search_files","arguments":{"query":"needle"}}</tool_call>'
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("qwen", payload)

    assert parsed.status == "parsed"
    assert [call.name for call in parsed.tool_calls] == ["read_file", "search_files"]
    assert parsed.tool_calls[1].arguments == {"query": "needle"}


def test_qwen35_parser_aliases_normalize_to_canonical_name():
    assert normalize_parser_name("qwen3_5") == "qwen3.5"
    assert normalize_parser_name("qwen3.6") == "qwen3"
    assert normalize_parser_name("qwen35") == "qwen3.5"
    assert normalize_parser_name("qwen3_xml") == "qwen3"
    assert normalize_parser_name("seed_oss") == "seed-oss"
    assert normalize_parser_name("minimax_m2") == "minimax-m2"
    assert normalize_parser_name("glm47_moe") == "glm47-moe"
    assert normalize_parser_name("llama4_pythonic") == "pythonic"
    assert normalize_parser_name("granite20b_fc") == "json-in-content"
    assert normalize_parser_name("deepseek_v31") == "json-in-content"


def test_qwen3_tool_parser_extracts_vllm_xml_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "<think>checking</think>\n"
                        "<tool_call>\n"
                        "<function=read_file>\n"
                        "<parameter=path>TASK.md</parameter>\n"
                        "</function>\n"
                        "</tool_call>"
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("qwen3", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "read_file"
    assert parsed.tool_calls[0].arguments == {"path": "TASK.md"}


def test_seed_oss_tool_parser_uses_qwen3_xml_body_with_seed_wrappers():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "<seed:think>checking</seed:think>\n"
                        "<seed:tool_call>"
                        "<function=search_files><parameter=query>needle</parameter></function>"
                        "</seed:tool_call>"
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("seed_oss", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "search_files"
    assert parsed.tool_calls[0].arguments == {"query": "needle"}


def test_auto_tool_parser_detects_qwen3_xml_before_tagged_json():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "<tool_call>"
                        "<function=search_files><parameter=query>needle</parameter></function>"
                        "</tool_call>"
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("auto", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "search_files"
    assert parsed.tool_calls[0].arguments == {"query": "needle"}


def test_minimax_m2_tool_parser_extracts_invoke_blocks():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '<minimax:tool_call><invoke name="get_weather">'
                        '<parameter name="city">Seattle</parameter>'
                        "<parameter name='unit'>celsius</parameter>"
                        "</invoke></minimax:tool_call>"
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("minimax_m2", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "get_weather"
    assert parsed.tool_calls[0].arguments == {"city": "Seattle", "unit": "celsius"}


def test_glm47_moe_tool_parser_extracts_arg_key_value_pairs():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "<tool_call>search_files"
                        "<arg_key>query</arg_key><arg_value>needle</arg_value>"
                        "<arg_key>path</arg_key><arg_value>.</arg_value>"
                        "</tool_call>"
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("glm47_moe", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "search_files"
    assert parsed.tool_calls[0].arguments == {"query": "needle", "path": "."}


def test_gemma4_tool_parser_extracts_custom_call_arguments():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '<|channel>thought\nok<channel|>\n'
                        '<|tool_call>call:get_weather{'
                        'location:<|"|>San Francisco<|"|>,'
                        'options:{unit:<|"|>celsius<|"|>},'
                        'days:[1,2]'
                        '}<tool_call|>'
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("gemma4", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "get_weather"
    assert parsed.tool_calls[0].arguments == {
        "location": "San Francisco",
        "options": {"unit": "celsius"},
        "days": ["1", "2"],
    }


def test_kimi_k2_tool_parser_extracts_section_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "<|tool_calls_section_begin|>"
                        "<|tool_call_begin|>functions.get_weather:0\n"
                        '<|tool_call_argument_begin|>{"city":"Tokyo"}<|tool_call_end|>'
                        "<|tool_calls_section_end|>"
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("kimi_k2", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "get_weather"
    assert parsed.tool_calls[0].call_id == "functions.get_weather:0"
    assert parsed.tool_calls[0].arguments == {"city": "Tokyo"}


def test_harmony_tool_parser_extracts_raw_message_call():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "<|start|>assistant to=functions.search_files "
                        "<|channel|>commentary <|constrain|>json"
                        '<|message|>{"query":"needle"}<|call|>'
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("gpt_oss", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "search_files"
    assert parsed.tool_calls[0].arguments == {"query": "needle"}


def test_longcat_tool_parser_extracts_tagged_json_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '<longcat_tool_call>{"name":"read_file","arguments":{"path":"TASK.md"}}'
                        "</longcat_tool_call>"
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("longcat", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "read_file"
    assert parsed.tool_calls[0].arguments == {"path": "TASK.md"}


def test_longcat_tool_parser_extracts_tagged_json_array_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '<longcat_tool_call>[{"name":"read_file","arguments":{"path":"a.txt"}},'
                        '{"name":"search_files","arguments":{"query":"needle"}}]</longcat_tool_call>'
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("longcat", payload)

    assert parsed.status == "parsed"
    assert [call.name for call in parsed.tool_calls] == ["read_file", "search_files"]


def test_auto_tool_parser_detects_longcat_tagged_json_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '<longcat_tool_call>{"name":"read_file","arguments":{"path":"TASK.md"}}'
                        "</longcat_tool_call>"
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("auto", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "read_file"


def test_xlam_tool_parser_extracts_json_array_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "content": '[{"name":"search_files","arguments":{"query":"needle"}}]'
                }
            }
        ]
    }

    parsed = parse_tool_calls("xlam", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "search_files"
    assert parsed.tool_calls[0].arguments == {"query": "needle"}


def test_functiongemma_tool_parser_extracts_escaped_arguments():
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "<start_function_call>call:search_files{"
                        "query:<escape>needle<escape>,path:<escape>.<escape>"
                        "}<end_function_call>"
                    )
                }
            }
        ]
    }

    parsed = parse_tool_calls("functiongemma", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "search_files"
    assert parsed.tool_calls[0].arguments == {"query": "needle", "path": "."}


def test_pythonic_tool_parser_extracts_call_list():
    payload = {
        "choices": [
            {
                "message": {
                    "content": "[read_file(path='TASK.md'), search_files(query='needle', path='.')]"
                }
            }
        ]
    }

    parsed = parse_tool_calls("pythonic", payload)

    assert parsed.status == "parsed"
    assert [call.name for call in parsed.tool_calls] == ["read_file", "search_files"]
    assert parsed.tool_calls[1].arguments == {"query": "needle", "path": "."}


def test_olmo3_tool_parser_extracts_xml_wrapped_pythonic_calls():
    payload = {
        "choices": [
            {
                "message": {
                    "content": "<function_calls>\nsearch_files(query='needle', recursive=true)\n</function_calls>"
                }
            }
        ]
    }

    parsed = parse_tool_calls("olmo3", payload)

    assert parsed.status == "parsed"
    assert parsed.tool_calls[0].name == "search_files"
    assert parsed.tool_calls[0].arguments == {"query": "needle", "recursive": True}


def test_none_tool_parser_disables_parsing():
    payload = {"choices": [{"message": {"tool_calls": [{"function": {"name": "x"}}]}}]}

    parsed = parse_tool_calls("none", payload)

    assert parsed.status == "no_tool_calls"
    assert parsed.tool_calls == []
