from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


TOOL_PARSER_AUTO = "auto"
TOOL_PARSER_OPENAI_NATIVE = "openai-native"
TOOL_PARSER_VLLM_COMPATIBLE = "vllm-compatible"
TOOL_PARSER_JSON_IN_CONTENT = "json-in-content"
TOOL_PARSER_NONE = "none"

TOOL_PARSER_NAMES = (
    TOOL_PARSER_AUTO,
    TOOL_PARSER_OPENAI_NATIVE,
    TOOL_PARSER_VLLM_COMPATIBLE,
    TOOL_PARSER_JSON_IN_CONTENT,
    TOOL_PARSER_NONE,
)


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
    aliases = {
        "openai": TOOL_PARSER_OPENAI_NATIVE,
        "native": TOOL_PARSER_OPENAI_NATIVE,
        "vllm": TOOL_PARSER_VLLM_COMPATIBLE,
        "json": TOOL_PARSER_JSON_IN_CONTENT,
        "disabled": TOOL_PARSER_NONE,
    }
    value = aliases.get(value, value)
    if value not in TOOL_PARSER_NAMES:
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
        parsed_json = _loads_json_object(content)
        if parsed_json is None:
            continue
        if _looks_like_final_answer_json(parsed_json):
            continue
        raw_tool_calls = parsed_json.get("tool_calls")
        if raw_tool_calls is None and "tool_call" in parsed_json:
            raw_tool_calls = [parsed_json["tool_call"]]
        if raw_tool_calls is None and "name" in parsed_json:
            raw_tool_calls = [parsed_json]
        if not isinstance(raw_tool_calls, list):
            return ToolParseResult(
                parser=TOOL_PARSER_JSON_IN_CONTENT,
                status="parse_error",
                error="content JSON field 'tool_calls' must be a list",
            )
        parsed, error = _normalize_tool_calls(raw_tool_calls)
        return ToolParseResult(
            parser=TOOL_PARSER_JSON_IN_CONTENT,
            tool_calls=parsed,
            status="parsed" if parsed else "parse_error",
            error=error,
        )
    return ToolParseResult(parser=TOOL_PARSER_JSON_IN_CONTENT)


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
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"tool call {index} missing function name")
            continue
        arguments = function.get("arguments", {})
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


def _loads_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


_PARSERS: dict[str, ParserFn] = {
    TOOL_PARSER_OPENAI_NATIVE: _parse_openai_native,
    TOOL_PARSER_VLLM_COMPATIBLE: _parse_vllm_compatible,
    TOOL_PARSER_JSON_IN_CONTENT: _parse_json_in_content,
    TOOL_PARSER_NONE: _parse_none,
}
