import ast
import json
import re
from typing import Any

_ACTION_PATTERN = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.IGNORECASE)
_ACTION_INPUT_PATTERN = re.compile(r"^\s*Action\s*Input\s*:\s*(.*)$", re.IGNORECASE)


def _brace_balance(text: str) -> int:
    return text.count("{") + text.count("[") - text.count("}") - text.count("]")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _normalize_action_name(name: str) -> str:
    return " ".join(name.strip().split())


def _parse_action_input(raw_text: str) -> tuple[bool, Any]:
    text = _strip_code_fences(raw_text.strip())

    for loader in (json.loads, ast.literal_eval):
        try:
            return True, _canonicalize(loader(text))
        except Exception:
            continue

    return False, text


def _extract_action_input_payload(lines: list[str], index: int, first_payload_line: str) -> tuple[str, int]:
    payload_lines = [first_payload_line]
    stripped_first = first_payload_line.strip()
    balance = _brace_balance(first_payload_line)
    cursor = index + 1

    if stripped_first.startswith("```"):
        while cursor < len(lines):
            payload_lines.append(lines[cursor])
            if lines[cursor].strip().startswith("```"):
                cursor += 1
                break
            cursor += 1
        return "\n".join(payload_lines).strip(), cursor

    # Continue multiline JSON blocks until braces/brackets are balanced.
    while balance > 0 and cursor < len(lines):
        payload_lines.append(lines[cursor])
        balance += _brace_balance(lines[cursor])
        cursor += 1

    return "\n".join(payload_lines).strip(), cursor


def _parse_predicted_tool_calls(prediction: str) -> tuple[list[dict[str, Any]], bool]:
    lines = prediction.splitlines()
    calls: list[dict[str, Any]] = []
    pending_action: str | None = None
    parse_success = True

    i = 0
    while i < len(lines):
        line = lines[i]

        action_match = _ACTION_PATTERN.match(line)
        if action_match:
            pending_action = _normalize_action_name(action_match.group(1))
            i += 1
            continue

        action_input_match = _ACTION_INPUT_PATTERN.match(line)
        if action_input_match and pending_action is not None:
            payload, next_index = _extract_action_input_payload(lines, i, action_input_match.group(1).strip())
            input_ok, action_input = _parse_action_input(payload)
            parse_success = parse_success and input_ok
            calls.append({"Action": pending_action, "Action_Input": action_input})
            pending_action = None
            i = next_index
            continue

        i += 1

    if pending_action is not None:
        parse_success = False

    if not calls:
        parse_success = False

    return calls, parse_success


def _normalize_expected_calls(reference: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for step in reference:
        action_name = _normalize_action_name(str(step.get("Action", "")))
        action_input_raw = str(step.get("Action_Input", ""))
        _, action_input = _parse_action_input(action_input_raw)
        normalized.append({"Action": action_name, "Action_Input": action_input})
    return normalized


def score_tooluse_predictions(
    predictions: list[str],
    references: list[list[dict[str, Any]]],
) -> dict[str, list[float]]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    strict_scores: list[float] = []
    parse_scores: list[float] = []
    action_name_scores: list[float] = []

    for prediction, reference in zip(predictions, references):
        predicted_calls, parse_success = _parse_predicted_tool_calls(prediction)
        expected_calls = _normalize_expected_calls(reference)

        parse_scores.append(1.0 if parse_success else 0.0)

        predicted_action_names = [call["Action"] for call in predicted_calls]
        expected_action_names = [call["Action"] for call in expected_calls]
        action_name_scores.append(1.0 if predicted_action_names == expected_action_names else 0.0)

        strict_scores.append(1.0 if predicted_calls == expected_calls else 0.0)

    return {
        "strict_match": strict_scores,
        "parse_success": parse_scores,
        "action_name_match": action_name_scores,
    }
