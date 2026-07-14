import pytest

from agent_bench.cli import _validate_cli_arguments, build_parser


def test_run_parser_accepts_tool_call_parser_alias():
    parser = build_parser()

    args = parser.parse_args(["run", "--tool-call-parser", "longcat"])

    assert args.tool_parser == "longcat"


def test_run_parser_normalizes_vllm_parser_alias():
    parser = build_parser()

    args = parser.parse_args(["run", "--tool-call-parser", "deepseek_v31"])

    assert args.tool_parser == "json-in-content"


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--request-concurrency", "0"],
        ["run", "--timeout", "-1"],
        ["run", "--max-tokens", "0"],
        ["run", "--max-retries", "-1"],
        ["run", "--top-p", "1.5"],
    ],
)
def test_run_parser_rejects_invalid_runtime_numbers(arguments):
    parser = build_parser()
    args = parser.parse_args(arguments)

    with pytest.raises(SystemExit):
        _validate_cli_arguments(parser, args)
