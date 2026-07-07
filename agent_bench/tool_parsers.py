from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


TOOL_PARSER_AUTO = "auto"
TOOL_PARSER_OPENAI_NATIVE = "openai-native"
TOOL_PARSER_VLLM_COMPATIBLE = "vllm-compatible"
TOOL_PARSER_HERMES = "hermes"
TOOL_PARSER_QWEN3 = "qwen3"
TOOL_PARSER_QWEN35 = "qwen3.5"
TOOL_PARSER_SEED_OSS = "seed-oss"
TOOL_PARSER_NEMOTRON_V3 = "nemotron-v3"
TOOL_PARSER_MINIMAX_M2 = "minimax-m2"
TOOL_PARSER_GLM47_MOE = "glm47-moe"
TOOL_PARSER_GEMMA4 = "gemma4"
TOOL_PARSER_KIMI_K2 = "kimi-k2"
TOOL_PARSER_HARMONY = "harmony"
TOOL_PARSER_LONGCAT = "longcat"
TOOL_PARSER_XLAM = "xlam"
TOOL_PARSER_FUNCTIONGEMMA = "functiongemma"
TOOL_PARSER_PYTHONIC = "pythonic"
TOOL_PARSER_OLMO3 = "olmo3"
TOOL_PARSER_JSON_IN_CONTENT = "json-in-content"
TOOL_PARSER_NONE = "none"

TOOL_PARSER_CANONICAL_NAMES = (
    TOOL_PARSER_AUTO,
    TOOL_PARSER_OPENAI_NATIVE,
    TOOL_PARSER_VLLM_COMPATIBLE,
    TOOL_PARSER_HERMES,
    TOOL_PARSER_QWEN3,
    TOOL_PARSER_QWEN35,
    TOOL_PARSER_SEED_OSS,
    TOOL_PARSER_NEMOTRON_V3,
    TOOL_PARSER_MINIMAX_M2,
    TOOL_PARSER_GLM47_MOE,
    TOOL_PARSER_GEMMA4,
    TOOL_PARSER_KIMI_K2,
    TOOL_PARSER_HARMONY,
    TOOL_PARSER_LONGCAT,
    TOOL_PARSER_XLAM,
    TOOL_PARSER_FUNCTIONGEMMA,
    TOOL_PARSER_PYTHONIC,
    TOOL_PARSER_OLMO3,
    TOOL_PARSER_JSON_IN_CONTENT,
    TOOL_PARSER_NONE,
)

TOOL_PARSER_ALIASES = {
    "openai": TOOL_PARSER_OPENAI_NATIVE,
    "native": TOOL_PARSER_OPENAI_NATIVE,
    "vllm": TOOL_PARSER_VLLM_COMPATIBLE,
    "json": TOOL_PARSER_JSON_IN_CONTENT,
    "json-content": TOOL_PARSER_JSON_IN_CONTENT,
    "disabled": TOOL_PARSER_NONE,
    "qwen": TOOL_PARSER_QWEN35,
    "qwen-2.5": TOOL_PARSER_QWEN35,
    "qwen-3.5": TOOL_PARSER_QWEN35,
    "qwen-3.6": TOOL_PARSER_QWEN3,
    "qwen2.5": TOOL_PARSER_QWEN35,
    "qwen25": TOOL_PARSER_QWEN35,
    "qwen3": TOOL_PARSER_QWEN3,
    "qwen3-5": TOOL_PARSER_QWEN35,
    "qwen3-6": TOOL_PARSER_QWEN3,
    "qwen3.6": TOOL_PARSER_QWEN3,
    "qwen35": TOOL_PARSER_QWEN35,
    "qwen36": TOOL_PARSER_QWEN3,
    "qwq": TOOL_PARSER_QWEN35,
    "qwen3-xml": TOOL_PARSER_QWEN3,
    "qwen-3-xml": TOOL_PARSER_QWEN3,
    "seed": TOOL_PARSER_SEED_OSS,
    "seed-oss": TOOL_PARSER_SEED_OSS,
    "minimax": TOOL_PARSER_MINIMAX_M2,
    "minimax-m2": TOOL_PARSER_MINIMAX_M2,
    "llama3-json": TOOL_PARSER_JSON_IN_CONTENT,
    "llama4-pythonic": TOOL_PARSER_PYTHONIC,
    "toolace": TOOL_PARSER_PYTHONIC,
    "nemotron": TOOL_PARSER_NEMOTRON_V3,
    "nemotron3": TOOL_PARSER_NEMOTRON_V3,
    "nemotron-v3": TOOL_PARSER_NEMOTRON_V3,
    "granite": TOOL_PARSER_JSON_IN_CONTENT,
    "granite4": TOOL_PARSER_JSON_IN_CONTENT,
    "granite-20b-fc": TOOL_PARSER_JSON_IN_CONTENT,
    "granite20b-fc": TOOL_PARSER_JSON_IN_CONTENT,
    "granite20bfc": TOOL_PARSER_JSON_IN_CONTENT,
    "internlm": TOOL_PARSER_JSON_IN_CONTENT,
    "jamba": TOOL_PARSER_JSON_IN_CONTENT,
    "mistral": TOOL_PARSER_JSON_IN_CONTENT,
    "deepseek-v3": TOOL_PARSER_JSON_IN_CONTENT,
    "deepseek-v31": TOOL_PARSER_JSON_IN_CONTENT,
    "deepseek-v3.1": TOOL_PARSER_JSON_IN_CONTENT,
    "kimi-k2": TOOL_PARSER_KIMI_K2,
    "hunyuan-a13b": TOOL_PARSER_JSON_IN_CONTENT,
    "cohere-command3": TOOL_PARSER_JSON_IN_CONTENT,
    "glm45": TOOL_PARSER_JSON_IN_CONTENT,
    "glm47": TOOL_PARSER_GLM47_MOE,
    "glm-47": TOOL_PARSER_GLM47_MOE,
    "glm4.7": TOOL_PARSER_GLM47_MOE,
    "glm-4.7": TOOL_PARSER_GLM47_MOE,
    "glm47-moe": TOOL_PARSER_GLM47_MOE,
    "gigachat3": TOOL_PARSER_JSON_IN_CONTENT,
    "apertus": TOOL_PARSER_JSON_IN_CONTENT,
    "gpt-oss": TOOL_PARSER_HARMONY,
    "gptoss": TOOL_PARSER_HARMONY,
}

TOOL_PARSER_NAMES = tuple(dict.fromkeys((*TOOL_PARSER_CANONICAL_NAMES, *TOOL_PARSER_ALIASES)))


@dataclass(slots=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolParseResult:
    parser: str
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    status: str = "no_tool_calls"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return payload


ParserFn = Callable[[dict[str, Any]], ToolParseResult]


def parse_tool_calls(parser: str, payload: dict[str, Any]) -> ToolParseResult:
    """Parse OpenAI-compatible tool calls from a response payload.

    The parser is intentionally benchmark-agnostic. It normalizes common
    OpenAI-compatible server variants so benchmark adapters can depend on one
    core representation instead of hard-coding server quirks.
    """

    parser = normalize_parser_name(parser)
    if parser == TOOL_PARSER_AUTO:
        for candidate in (
            TOOL_PARSER_OPENAI_NATIVE,
            TOOL_PARSER_VLLM_COMPATIBLE,
            TOOL_PARSER_QWEN3,
            TOOL_PARSER_SEED_OSS,
            TOOL_PARSER_NEMOTRON_V3,
            TOOL_PARSER_MINIMAX_M2,
            TOOL_PARSER_GLM47_MOE,
            TOOL_PARSER_GEMMA4,
            TOOL_PARSER_KIMI_K2,
            TOOL_PARSER_HARMONY,
            TOOL_PARSER_HERMES,
            TOOL_PARSER_QWEN35,
            TOOL_PARSER_LONGCAT,
            TOOL_PARSER_XLAM,
            TOOL_PARSER_FUNCTIONGEMMA,
            TOOL_PARSER_OLMO3,
            TOOL_PARSER_PYTHONIC,
            TOOL_PARSER_JSON_IN_CONTENT,
        ):
            parsed = parse_tool_calls(candidate, payload)
            if parsed.tool_calls or parsed.error:
                parsed.parser = TOOL_PARSER_AUTO
                return parsed
        return ToolParseResult(parser=TOOL_PARSER_AUTO)
    return _PARSERS[parser](payload)


def normalize_parser_name(parser: str | None) -> str:
    value = (parser or TOOL_PARSER_AUTO).strip().lower().replace("_", "-")
    value = TOOL_PARSER_ALIASES.get(value, value)
    if value not in TOOL_PARSER_CANONICAL_NAMES:
        raise ValueError(f"Unsupported tool parser: {parser}")
    return value


def _parse_none(payload: dict[str, Any]) -> ToolParseResult:
    return ToolParseResult(parser=TOOL_PARSER_NONE)


def _parse_openai_native(payload: dict[str, Any]) -> ToolParseResult:
    tool_calls = _tool_calls_from_messages(payload)
    if not tool_calls:
        return ToolParseResult(parser=TOOL_PARSER_OPENAI_NATIVE)
    parsed, error = _normalize_tool_calls(tool_calls)
    return ToolParseResult(
        parser=TOOL_PARSER_OPENAI_NATIVE,
        tool_calls=parsed,
        status="parsed" if parsed else "parse_error",
        error=error,
    )


def _parse_vllm_compatible(payload: dict[str, Any]) -> ToolParseResult:
    native = _parse_openai_native(payload)
    if native.tool_calls or native.error:
        native.parser = TOOL_PARSER_VLLM_COMPATIBLE
        return native

    messages = _candidate_messages(payload)
    for message in messages:
        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            parsed, error = _normalize_tool_calls([{"type": "function", "function": function_call}])
            return ToolParseResult(
                parser=TOOL_PARSER_VLLM_COMPATIBLE,
                tool_calls=parsed,
                status="parsed" if parsed else "parse_error",
                error=error,
            )
    return ToolParseResult(parser=TOOL_PARSER_VLLM_COMPATIBLE)


def _parse_json_in_content(payload: dict[str, Any]) -> ToolParseResult:
    for message in _candidate_messages(payload):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        parsed_json = _loads_json_value(content)
        if parsed_json is None:
            continue
        raw_tool_calls, error = _raw_tool_calls_from_json_value(parsed_json)
        if error is not None:
            return ToolParseResult(
                parser=TOOL_PARSER_JSON_IN_CONTENT,
                status="parse_error",
                error=error,
            )
        if raw_tool_calls is None:
            continue
        parsed, error = _normalize_tool_calls(raw_tool_calls)
        return ToolParseResult(
            parser=TOOL_PARSER_JSON_IN_CONTENT,
            tool_calls=parsed,
            status="parsed" if parsed else "parse_error",
            error=error,
        )
    return ToolParseResult(parser=TOOL_PARSER_JSON_IN_CONTENT)


def _parse_hermes(payload: dict[str, Any]) -> ToolParseResult:
    return _parse_tagged_json(
        payload,
        parser=TOOL_PARSER_HERMES,
        patterns=[
            r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>",
            r"<\|tool_call\|?>\s*(?P<body>.*?)\s*(?:<\|/tool_call\|?>|<\|tool_call_end\|?>)",
        ],
    )


def _parse_qwen3(payload: dict[str, Any]) -> ToolParseResult:
    return _parse_qwen3_xml_payload(
        payload,
        parser=TOOL_PARSER_QWEN3,
        tool_start="tool_call",
        tool_end="tool_call",
    )


def _parse_seed_oss(payload: dict[str, Any]) -> ToolParseResult:
    return _parse_qwen3_xml_payload(
        payload,
        parser=TOOL_PARSER_SEED_OSS,
        tool_start="seed:tool_call",
        tool_end="seed:tool_call",
    )


def _parse_nemotron_v3(payload: dict[str, Any]) -> ToolParseResult:
    return _parse_qwen3_xml_payload(
        payload,
        parser=TOOL_PARSER_NEMOTRON_V3,
        tool_start="tool_call",
        tool_end="tool_call",
    )


def _parse_qwen3_xml_payload(
    payload: dict[str, Any],
    *,
    parser: str,
    tool_start: str,
    tool_end: str,
) -> ToolParseResult:
    native = _parse_openai_native(payload)
    if native.tool_calls or native.error:
        native.parser = parser
        return native

    raw_calls: list[Any] = []
    for message in _candidate_messages(payload):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        raw_calls.extend(
            _qwen3_xml_tool_calls_from_content(content, tool_start, tool_end)
        )

    if not raw_calls:
        return ToolParseResult(parser=parser)
    parsed, error = _normalize_tool_calls(raw_calls)
    return ToolParseResult(
        parser=parser,
        tool_calls=parsed,
        status="parsed" if parsed else "parse_error",
        error=error,
    )


def _qwen3_xml_tool_calls_from_content(
    content: str,
    tool_start: str,
    tool_end: str,
) -> list[Any]:
    sections = [
        match.group("body")
        for match in re.finditer(
            _xml_tag_pattern(tool_start, tool_end),
            content,
            re.DOTALL | re.IGNORECASE,
        )
    ]
    if not sections:
        sections = [content]

    raw_calls: list[Any] = []
    function_pattern = re.compile(
        r"<\s*function\s*=\s*(?P<name>[^>]*)\s*>(?P<body>.*?)<\s*/\s*function\s*>",
        re.DOTALL | re.IGNORECASE,
    )
    param_pattern = re.compile(
        r"<\s*parameter\s*=\s*(?P<name>[^>]*)\s*>"
        r"(?P<value>.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=)|$)",
        re.DOTALL | re.IGNORECASE,
    )
    for section in sections:
        for function_match in function_pattern.finditer(section):
            name = _clean_xml_name(function_match.group("name"))
            if not _is_probable_tool_name(name):
                continue
            arguments: dict[str, Any] = {}
            for param_match in param_pattern.finditer(function_match.group("body")):
                param_name = _clean_xml_name(param_match.group("name"))
                if param_name:
                    arguments[param_name] = param_match.group("value").strip()
            raw_calls.append({"name": name, "arguments": arguments})
    return raw_calls


def _parse_minimax_m2(payload: dict[str, Any]) -> ToolParseResult:
    native = _parse_openai_native(payload)
    if native.tool_calls or native.error:
        native.parser = TOOL_PARSER_MINIMAX_M2
        return native

    raw_calls: list[Any] = []
    for message in _candidate_messages(payload):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        raw_calls.extend(_minimax_m2_tool_calls_from_content(content))

    if not raw_calls:
        return ToolParseResult(parser=TOOL_PARSER_MINIMAX_M2)
    parsed, error = _normalize_tool_calls(raw_calls)
    return ToolParseResult(
        parser=TOOL_PARSER_MINIMAX_M2,
        tool_calls=parsed,
        status="parsed" if parsed else "parse_error",
        error=error,
    )


def _minimax_m2_tool_calls_from_content(content: str) -> list[Any]:
    invoke_pattern = re.compile(
        r"<\s*invoke\s+name\s*=\s*"
        r"(?:\"(?P<dq_name>[^\"]*)\"|'(?P<sq_name>[^']*)'|(?P<bare_name>[^>\s]+))"
        r"\s*>(?P<body>.*?)<\s*/\s*invoke\s*>",
        re.DOTALL | re.IGNORECASE,
    )
    param_pattern = re.compile(
        r"<\s*parameter\s+name\s*=\s*"
        r"(?:\"(?P<dq_name>[^\"]*)\"|'(?P<sq_name>[^']*)'|(?P<bare_name>[^>\s]+))"
        r"\s*>"
        r"(?P<value>.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s+name\s*=)|$)",
        re.DOTALL | re.IGNORECASE,
    )
    raw_calls: list[Any] = []
    for invoke_match in invoke_pattern.finditer(content):
        name = _first_regex_group(
            invoke_match,
            "dq_name",
            "sq_name",
            "bare_name",
        ).strip()
        if not _is_probable_tool_name(name):
            continue
        arguments: dict[str, Any] = {}
        for param_match in param_pattern.finditer(invoke_match.group("body")):
            param_name = _first_regex_group(
                param_match,
                "dq_name",
                "sq_name",
                "bare_name",
            ).strip()
            if param_name:
                arguments[param_name] = param_match.group("value").strip()
        raw_calls.append({"name": name, "arguments": arguments})
    return raw_calls


def _parse_glm47_moe(payload: dict[str, Any]) -> ToolParseResult:
    native = _parse_openai_native(payload)
    if native.tool_calls or native.error:
        native.parser = TOOL_PARSER_GLM47_MOE
        return native

    raw_calls: list[Any] = []
    for message in _candidate_messages(payload):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        raw_calls.extend(_glm47_moe_tool_calls_from_content(content))

    if not raw_calls:
        return ToolParseResult(parser=TOOL_PARSER_GLM47_MOE)
    parsed, error = _normalize_tool_calls(raw_calls)
    return ToolParseResult(
        parser=TOOL_PARSER_GLM47_MOE,
        tool_calls=parsed,
        status="parsed" if parsed else "parse_error",
        error=error,
    )


def _glm47_moe_tool_calls_from_content(content: str) -> list[Any]:
    arg_pattern = re.compile(
        r"<\s*arg_key\s*>(?P<key>.*?)<\s*/\s*arg_key\s*>\s*"
        r"<\s*arg_value\s*>(?P<value>.*?)<\s*/\s*arg_value\s*>",
        re.DOTALL | re.IGNORECASE,
    )
    raw_calls: list[Any] = []
    for match in re.finditer(
        _xml_tag_pattern("tool_call", "tool_call"),
        content,
        re.DOTALL | re.IGNORECASE,
    ):
        body = match.group("body")
        first_arg = re.search(r"<\s*arg_key\s*>", body, re.IGNORECASE)
        name = body[: first_arg.start()].strip() if first_arg else body.strip()
        if not _is_probable_tool_name(name):
            continue
        arguments = {
            arg_match.group("key").strip(): arg_match.group("value")
            for arg_match in arg_pattern.finditer(body)
            if arg_match.group("key").strip()
        }
        raw_calls.append({"name": name, "arguments": arguments})
    return raw_calls


_GEMMA4_STRING_DELIM = '<|"|>'


def _parse_gemma4(payload: dict[str, Any]) -> ToolParseResult:
    native = _parse_openai_native(payload)
    if native.tool_calls or native.error:
        native.parser = TOOL_PARSER_GEMMA4
        return native

    raw_calls: list[Any] = []
    for message in _candidate_messages(payload):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        raw_calls.extend(_gemma4_tool_calls_from_content(content))

    if not raw_calls:
        return ToolParseResult(parser=TOOL_PARSER_GEMMA4)
    parsed, error = _normalize_tool_calls(raw_calls)
    return ToolParseResult(
        parser=TOOL_PARSER_GEMMA4,
        tool_calls=parsed,
        status="parsed" if parsed else "parse_error",
        error=error,
    )


def _gemma4_tool_calls_from_content(content: str) -> list[Any]:
    pattern = re.compile(
        r"<\|tool_call>\s*call:(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)\{"
        r"(?P<body>.*?)\}\s*<tool_call\|>",
        re.DOTALL,
    )
    raw_calls: list[Any] = []
    for match in pattern.finditer(content):
        name = match.group("name").strip()
        if not _is_probable_tool_name(name):
            continue
        raw_calls.append(
            {"name": name, "arguments": _parse_gemma4_arguments(match.group("body"))}
        )
    return raw_calls


def _parse_gemma4_arguments(text: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for part in _split_gemma4_top_level(text, ","):
        item = part.strip()
        if not item:
            continue
        colon = _find_gemma4_top_level(item, ":")
        if colon <= 0:
            continue
        key = _parse_gemma4_key(item[:colon])
        if not key:
            continue
        arguments[key] = _parse_gemma4_value(item[colon + 1 :])
    return arguments


def _parse_gemma4_key(text: str) -> str:
    key = text.strip()
    if key.startswith(_GEMMA4_STRING_DELIM) and key.endswith(_GEMMA4_STRING_DELIM):
        key = key[len(_GEMMA4_STRING_DELIM) : -len(_GEMMA4_STRING_DELIM)]
    return key.strip()


def _parse_gemma4_value(text: str) -> Any:
    value = text.strip()
    if value.startswith(_GEMMA4_STRING_DELIM) and value.endswith(_GEMMA4_STRING_DELIM):
        return value[len(_GEMMA4_STRING_DELIM) : -len(_GEMMA4_STRING_DELIM)]
    if value.startswith("{") and value.endswith("}"):
        return _parse_gemma4_arguments(value[1:-1])
    if value.startswith("[") and value.endswith("]"):
        return [
            _parse_gemma4_value(item)
            for item in _split_gemma4_top_level(value[1:-1], ",")
            if item.strip()
        ]
    return value


def _split_gemma4_top_level(text: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    index = 0
    while index < len(text):
        if text.startswith(_GEMMA4_STRING_DELIM, index):
            index = _skip_gemma4_string(text, index)
            continue
        char = text[index]
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth = max(0, depth - 1)
        elif char == delimiter and depth == 0:
            parts.append(text[start:index])
            start = index + 1
        index += 1
    parts.append(text[start:])
    return parts


def _find_gemma4_top_level(text: str, target: str) -> int:
    depth = 0
    index = 0
    while index < len(text):
        if text.startswith(_GEMMA4_STRING_DELIM, index):
            index = _skip_gemma4_string(text, index)
            continue
        char = text[index]
        if char in "[{":
            depth += 1
        elif char in "]}":
            depth = max(0, depth - 1)
        elif char == target and depth == 0:
            return index
        index += 1
    return -1


def _skip_gemma4_string(text: str, start: int) -> int:
    end = text.find(_GEMMA4_STRING_DELIM, start + len(_GEMMA4_STRING_DELIM))
    return len(text) if end == -1 else end + len(_GEMMA4_STRING_DELIM)


def _parse_kimi_k2(payload: dict[str, Any]) -> ToolParseResult:
    native = _parse_openai_native(payload)
    if native.tool_calls or native.error:
        native.parser = TOOL_PARSER_KIMI_K2
        return native

    raw_calls: list[Any] = []
    for message in _candidate_messages(payload):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        raw_calls.extend(_kimi_k2_tool_calls_from_content(content))

    if not raw_calls:
        return ToolParseResult(parser=TOOL_PARSER_KIMI_K2)
    parsed, error = _normalize_tool_calls(raw_calls)
    return ToolParseResult(
        parser=TOOL_PARSER_KIMI_K2,
        tool_calls=parsed,
        status="parsed" if parsed else "parse_error",
        error=error,
    )


def _kimi_k2_tool_calls_from_content(content: str) -> list[Any]:
    pattern = re.compile(
        r"<\|tool_call_begin\|>(?P<header>.*?)"
        r"<\|tool_call_argument_begin\|>(?P<arguments>.*?)<\|tool_call_end\|>",
        re.DOTALL,
    )
    raw_calls: list[Any] = []
    for match in pattern.finditer(content):
        header = match.group("header").strip()
        header_match = re.match(r"(?P<id>.+:\d+)\s*$", header)
        if not header_match:
            continue
        call_id = header_match.group("id").strip()
        name = call_id.split(":", 1)[0].removeprefix("functions.")
        if not _is_probable_tool_name(name):
            continue
        raw_calls.append(
            {
                "id": call_id,
                "name": name,
                "arguments": match.group("arguments").strip() or "{}",
            }
        )
    return raw_calls


def _parse_harmony(payload: dict[str, Any]) -> ToolParseResult:
    native = _parse_vllm_compatible(payload)
    if native.tool_calls or native.error:
        native.parser = TOOL_PARSER_HARMONY
        return native

    raw_calls: list[Any] = []
    for message in _candidate_messages(payload):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        raw_calls.extend(_harmony_tool_calls_from_content(content))

    if not raw_calls:
        return ToolParseResult(parser=TOOL_PARSER_HARMONY)
    parsed, error = _normalize_tool_calls(raw_calls)
    return ToolParseResult(
        parser=TOOL_PARSER_HARMONY,
        tool_calls=parsed,
        status="parsed" if parsed else "parse_error",
        error=error,
    )


def _harmony_tool_calls_from_content(content: str) -> list[Any]:
    if "<|message|>" not in content:
        return []
    pattern = re.compile(
        r"(?:to=|recipient=)\s*(?:functions\.)?(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)"
        r"(?:(?!<\|message\|>).)*<\|message\|>(?P<arguments>.*?)"
        r"(?=<\|call\|>|<\|end\|>|<\|start\|>|$)",
        re.DOTALL,
    )
    raw_calls: list[Any] = []
    for match in pattern.finditer(content):
        name = match.group("name").strip()
        if name == "assistant" or not _is_probable_tool_name(name):
            continue
        raw_calls.append(
            {"name": name, "arguments": match.group("arguments").strip() or "{}"}
        )
    return raw_calls


def _parse_qwen35(payload: dict[str, Any]) -> ToolParseResult:
    return _parse_tagged_json(
        payload,
        parser=TOOL_PARSER_QWEN35,
        patterns=[
            r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>",
            r"<\|?tool_call\|?>\s*(?P<body>.*?)\s*(?:</tool_call>|<\|/tool_call\|?>|<\|tool_call_end\|?>)",
        ],
    )


def _parse_longcat(payload: dict[str, Any]) -> ToolParseResult:
    return _parse_tagged_json(
        payload,
        parser=TOOL_PARSER_LONGCAT,
        patterns=[r"<longcat_tool_call>\s*(?P<body>.*?)\s*</longcat_tool_call>"],
    )


def _parse_tagged_json(
    payload: dict[str, Any],
    *,
    parser: str,
    patterns: list[str],
) -> ToolParseResult:
    native = _parse_openai_native(payload)
    if native.tool_calls or native.error:
        native.parser = parser
        return native

    for message in _candidate_messages(payload):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        parsed, error, found_tags = _tagged_json_tool_calls_from_content(content, patterns)
        if found_tags:
            return ToolParseResult(
                parser=parser,
                tool_calls=parsed,
                status="parsed" if parsed else "parse_error",
                error=error,
            )

    json_fallback = _parse_json_in_content(payload)
    if json_fallback.tool_calls or json_fallback.error:
        json_fallback.parser = parser
        return json_fallback
    return ToolParseResult(parser=parser)


def _tagged_json_tool_calls_from_content(
    content: str,
    patterns: list[str],
) -> tuple[list[ParsedToolCall], str | None, bool]:
    raw_calls: list[Any] = []
    errors: list[str] = []
    found_tags = False
    seen_spans: set[tuple[int, int]] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, content, re.DOTALL):
            if match.span() in seen_spans:
                continue
            seen_spans.add(match.span())
            found_tags = True
            body = match.group("body").strip()
            try:
                value = json.loads(body)
            except json.JSONDecodeError as exc:
                errors.append(f"tagged tool call has invalid JSON: {exc.msg}")
                continue
            if isinstance(value, list):
                raw_calls.extend(value)
            else:
                raw_calls.append(value)

    if not raw_calls:
        error = "; ".join(errors) if errors else None
        return [], error, found_tags

    parsed, normalize_error = _normalize_tool_calls(raw_calls)
    if normalize_error:
        errors.append(normalize_error)
    error = "; ".join(errors) if errors else None
    return parsed, error, found_tags


def _parse_xlam(payload: dict[str, Any]) -> ToolParseResult:
    native = _parse_openai_native(payload)
    if native.tool_calls or native.error:
        native.parser = TOOL_PARSER_XLAM
        return native

    for message in _candidate_messages(payload):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        raw_calls, error, found = _xlam_tool_calls_from_content(content)
        if found:
            if error is not None:
                return ToolParseResult(parser=TOOL_PARSER_XLAM, status="parse_error", error=error)
            parsed, normalize_error = _normalize_tool_calls(raw_calls or [])
            return ToolParseResult(
                parser=TOOL_PARSER_XLAM,
                tool_calls=parsed,
                status="parsed" if parsed else "parse_error",
                error=normalize_error,
            )
    return ToolParseResult(parser=TOOL_PARSER_XLAM)


def _xlam_tool_calls_from_content(content: str) -> tuple[list[Any] | None, str | None, bool]:
    for candidate in _json_candidates_from_content(content):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        raw_calls, error = _raw_tool_calls_from_json_value(value)
        if error is not None:
            return None, error, True
        if raw_calls is not None:
            return raw_calls, None, True
    return None, None, False


def _json_candidates_from_content(content: str) -> list[str]:
    stripped = content.strip()
    candidates: list[str] = [stripped]
    after_think = re.search(r"</think>\s*(?P<body>[\s\S]+)$", stripped)
    if after_think:
        candidates.append(after_think.group("body").strip())
    for match in re.finditer(r"```(?:json)?\s*(?P<body>[\s\S]*?)```", stripped, re.IGNORECASE):
        candidates.append(match.group("body").strip())
    for match in re.finditer(r"\[TOOL_CALLS\]\s*(?P<body>[\s\S]+)", stripped):
        candidates.append(match.group("body").strip())
    for match in re.finditer(r"<tool_call>\s*(?P<body>[\s\S]*?)\s*</tool_call>", stripped):
        candidates.append(match.group("body").strip())
    array = _extract_balanced_json(stripped, "[", "]")
    if array:
        candidates.append(array)
    obj = _extract_balanced_json(stripped, "{", "}")
    if obj:
        candidates.append(obj)
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _parse_functiongemma(payload: dict[str, Any]) -> ToolParseResult:
    native = _parse_openai_native(payload)
    if native.tool_calls or native.error:
        native.parser = TOOL_PARSER_FUNCTIONGEMMA
        return native

    raw_calls: list[Any] = []
    found = False
    for message in _candidate_messages(payload):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        for match in re.finditer(
            r"<start_function_call>\s*call:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\{(?P<body>.*?)\}\s*<end_function_call>",
            content,
            re.DOTALL,
        ):
            found = True
            raw_calls.append(
                {
                    "name": match.group("name"),
                    "arguments": _parse_functiongemma_arguments(match.group("body")),
                }
            )
    if not found:
        return ToolParseResult(parser=TOOL_PARSER_FUNCTIONGEMMA)
    parsed, error = _normalize_tool_calls(raw_calls)
    return ToolParseResult(
        parser=TOOL_PARSER_FUNCTIONGEMMA,
        tool_calls=parsed,
        status="parsed" if parsed else "parse_error",
        error=error,
    )


def _parse_functiongemma_arguments(body: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {}
    for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*<escape>(.*?)<escape>", body, re.DOTALL):
        try:
            arguments[key] = json.loads(value)
        except json.JSONDecodeError:
            arguments[key] = value
    return arguments


def _parse_pythonic(payload: dict[str, Any]) -> ToolParseResult:
    return _parse_pythonic_payload(payload, parser=TOOL_PARSER_PYTHONIC)


def _parse_olmo3(payload: dict[str, Any]) -> ToolParseResult:
    return _parse_pythonic_payload(payload, parser=TOOL_PARSER_OLMO3, olmo3=True)


def _parse_pythonic_payload(
    payload: dict[str, Any],
    *,
    parser: str,
    olmo3: bool = False,
) -> ToolParseResult:
    native = _parse_openai_native(payload)
    if native.tool_calls or native.error:
        native.parser = parser
        return native

    for message in _candidate_messages(payload):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        candidate = _olmo3_pythonic_candidate(content) if olmo3 else content.strip()
        if not candidate:
            continue
        raw_calls, error, found = _python_tool_calls_from_content(candidate)
        if found:
            if error is not None:
                return ToolParseResult(parser=parser, status="parse_error", error=error)
            parsed, normalize_error = _normalize_tool_calls(raw_calls or [])
            return ToolParseResult(
                parser=parser,
                tool_calls=parsed,
                status="parsed" if parsed else "parse_error",
                error=normalize_error,
            )
    return ToolParseResult(parser=parser)


def _olmo3_pythonic_candidate(content: str) -> str | None:
    match = re.search(r"<function_calls>\s*(?P<body>.*?)\s*</function_calls>", content, re.DOTALL)
    if not match:
        return None
    rows = [line.strip().rstrip(",") for line in match.group("body").splitlines() if line.strip()]
    if not rows:
        return None
    return "[" + ", ".join(rows) + "]"


def _python_tool_calls_from_content(content: str) -> tuple[list[Any] | None, str | None, bool]:
    stripped = content.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return None, None, False
    try:
        expression = ast.parse(stripped, mode="eval").body
    except SyntaxError as exc:
        return None, f"invalid pythonic tool calls: {exc.msg}", True
    if not isinstance(expression, ast.List):
        return None, None, False
    raw_calls: list[Any] = []
    errors: list[str] = []
    for index, item in enumerate(expression.elts):
        if not isinstance(item, ast.Call):
            errors.append(f"tool call {index} must be a function call")
            continue
        name = _python_call_name(item.func)
        if not name:
            errors.append(f"tool call {index} missing function name")
            continue
        arguments: dict[str, Any] = {}
        for keyword in item.keywords:
            if keyword.arg is None:
                errors.append(f"tool call {index} has unsupported **kwargs")
                continue
            try:
                arguments[keyword.arg] = _literal_from_ast(keyword.value)
            except ValueError as exc:
                errors.append(f"tool call {index} argument {keyword.arg}: {exc}")
        if item.args:
            errors.append(f"tool call {index} positional arguments are unsupported")
        raw_calls.append({"name": name, "arguments": arguments})
    error = "; ".join(errors) if errors else None
    return raw_calls, error, True


def _python_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _python_call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        aliases = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}
        if node.id in aliases:
            return aliases[node.id]
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as exc:
        raise ValueError("value must be a literal") from exc


def _xml_tag_pattern(start_tag: str, end_tag: str) -> str:
    return (
        rf"<\s*{re.escape(start_tag)}\s*>"
        rf"(?P<body>.*?)"
        rf"<\s*/\s*{re.escape(end_tag)}\s*>"
    )


def _clean_xml_name(value: str) -> str:
    return value.strip().strip("\"'")


def _first_regex_group(match: re.Match[str], *names: str) -> str:
    for name in names:
        value = match.group(name)
        if value is not None:
            return value
    return ""


def _is_probable_tool_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", value.strip()))


def _looks_like_final_answer_json(parsed_json: dict[str, Any]) -> bool:
    return "answer" in parsed_json and not any(
        key in parsed_json for key in ("tool", "tool_name", "tool_call", "tool_calls", "function_call")
    )


def _tool_calls_from_messages(payload: dict[str, Any]) -> list[Any]:
    calls: list[Any] = []
    for message in _candidate_messages(payload):
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            calls.extend(tool_calls)
    return calls


def _candidate_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if isinstance(payload.get("message"), dict):
        messages.append(payload["message"])
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            for key in ("message", "delta"):
                value = choice.get(key)
                if isinstance(value, dict):
                    messages.append(value)
    if any(key in payload for key in ("tool_calls", "function_call", "content")):
        messages.append(payload)
    return messages


def _normalize_tool_calls(raw_calls: list[Any]) -> tuple[list[ParsedToolCall], str | None]:
    parsed: list[ParsedToolCall] = []
    errors: list[str] = []
    for index, raw in enumerate(raw_calls):
        if not isinstance(raw, dict):
            errors.append(f"tool call {index} must be an object")
            continue
        function = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        name = function.get("name") or function.get("tool") or function.get("tool_name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"tool call {index} missing function name")
            continue
        arguments = _first_present(function, ("arguments", "parameters", "params", "args", "input"), {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                errors.append(f"tool call {index} has invalid JSON arguments: {exc.msg}")
                continue
        if not isinstance(arguments, dict):
            errors.append(f"tool call {index} arguments must be an object")
            continue
        call_id = raw.get("id")
        parsed.append(
            ParsedToolCall(
                name=name.strip(),
                arguments=arguments,
                call_id=call_id if isinstance(call_id, str) else None,
                raw=raw,
            )
        )
    error = "; ".join(errors) if errors else None
    return parsed, error


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _raw_tool_calls_from_json_value(value: Any) -> tuple[list[Any] | None, str | None]:
    if isinstance(value, dict):
        if _looks_like_final_answer_json(value):
            return None, None
        raw_tool_calls = value.get("tool_calls")
        if raw_tool_calls is None and "tool_call" in value:
            raw_tool_calls = [value["tool_call"]]
        if raw_tool_calls is None and _looks_like_raw_tool_call(value):
            raw_tool_calls = [value]
        if raw_tool_calls is None:
            return None, None
        if not isinstance(raw_tool_calls, list):
            return None, "content JSON field 'tool_calls' must be a list"
        return raw_tool_calls, None
    if isinstance(value, list):
        if value and all(_looks_like_raw_tool_call(item) for item in value):
            return value, None
    return None, None


def _looks_like_raw_tool_call(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("function"), dict):
        return _looks_like_raw_tool_call(value["function"])
    return any(isinstance(value.get(key), str) for key in ("name", "tool", "tool_name"))


def _loads_json_value(text: str) -> Any | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    candidates = [stripped]
    array = _extract_balanced_json(stripped, "[", "]")
    if array:
        candidates.append(array)
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def _extract_balanced_json(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


_PARSERS: dict[str, ParserFn] = {
    TOOL_PARSER_OPENAI_NATIVE: _parse_openai_native,
    TOOL_PARSER_VLLM_COMPATIBLE: _parse_vllm_compatible,
    TOOL_PARSER_HERMES: _parse_hermes,
    TOOL_PARSER_QWEN3: _parse_qwen3,
    TOOL_PARSER_QWEN35: _parse_qwen35,
    TOOL_PARSER_SEED_OSS: _parse_seed_oss,
    TOOL_PARSER_NEMOTRON_V3: _parse_nemotron_v3,
    TOOL_PARSER_MINIMAX_M2: _parse_minimax_m2,
    TOOL_PARSER_GLM47_MOE: _parse_glm47_moe,
    TOOL_PARSER_GEMMA4: _parse_gemma4,
    TOOL_PARSER_KIMI_K2: _parse_kimi_k2,
    TOOL_PARSER_HARMONY: _parse_harmony,
    TOOL_PARSER_LONGCAT: _parse_longcat,
    TOOL_PARSER_XLAM: _parse_xlam,
    TOOL_PARSER_FUNCTIONGEMMA: _parse_functiongemma,
    TOOL_PARSER_PYTHONIC: _parse_pythonic,
    TOOL_PARSER_OLMO3: _parse_olmo3,
    TOOL_PARSER_JSON_IN_CONTENT: _parse_json_in_content,
    TOOL_PARSER_NONE: _parse_none,
}
