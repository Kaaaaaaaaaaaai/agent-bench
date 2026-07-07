from agent_bench.cli import build_parser


def test_run_parser_accepts_tool_call_parser_alias():
    parser = build_parser()

    args = parser.parse_args(["run", "--tool-call-parser", "longcat"])

    assert args.tool_parser == "longcat"


def test_run_parser_normalizes_vllm_parser_alias():
    parser = build_parser()

    args = parser.parse_args(["run", "--tool-call-parser", "deepseek_v31"])

    assert args.tool_parser == "json-in-content"


def test_run_parser_normalizes_vllm_engine_parser_alias():
    parser = build_parser()

    args = parser.parse_args(["run", "--tool-call-parser", "minimax_m2"])

    assert args.tool_parser == "minimax-m2"
