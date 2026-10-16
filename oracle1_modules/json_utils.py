"""Robust parsing helpers for model-generated JSON."""
from __future__ import annotations

from ._shared import *

# ══════════════════════════════════════════════════════════════════
#  Утилиты
# ══════════════════════════════════════════════════════════════════

def _parse_json_safe(raw: str) -> Dict:
    if not isinstance(raw, str):
        raise json.JSONDecodeError("LLM response is not text", str(raw), 0)
    clean = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    clean = clean.replace("```", "").lstrip("\ufeff").strip()
    brace = clean.find("{")
    if brace >= 0:
        clean = clean[brace:]

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        parsed = json.loads(_repair_common_json_syntax(clean))
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("Expected a JSON object", clean, 0)
    return parsed


def _repair_common_json_syntax(text: str) -> str:
    """Conservatively repair common local-LLM punctuation mistakes.

    It only inserts missing commas between obvious object fields/array
    elements, removes trailing commas, and closes containers when the
    response ended outside a string. It never rewrites quoted content.
    """
    output: List[str] = []
    stack: List[str] = []
    in_string = False
    escaped = False

    def last_significant() -> str:
        for char in reversed(output):
            if not char.isspace():
                return char
        return ""

    def value_ended(previous: str) -> bool:
        if previous in {'"', "}", "]"} or previous.isdigit():
            return True
        if previous in {"e", "l"}:
            token = "".join(output).rstrip().split()[-1:] 
            return bool(token and token[0] in {"true", "false", "null"})
        return False

    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            end = index + 1
            string_escape = False
            while end < len(text):
                if string_escape:
                    string_escape = False
                elif text[end] == "\\":
                    string_escape = True
                elif text[end] == '"':
                    break
                end += 1
            after = end + 1
            while after < len(text) and text[after].isspace():
                after += 1
            is_object_key = after < len(text) and text[after] == ":"
            previous = last_significant()
            if value_ended(previous) and (is_object_key and stack and stack[-1] == "}"
                                          or not is_object_key and stack and stack[-1] == "]"):
                output.append(",")
            output.append(char)
            in_string = True
            index += 1
            continue

        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue

        if char in "{[":
            previous = last_significant()
            if (stack and stack[-1] == "[" and value_ended(previous)):
                output.append(",")
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack and stack[-1] == char:
                stack.pop()

        output.append(char)
        index += 1

    if not in_string:
        while output and output[-1].isspace():
            output.pop()
        if output and output[-1] == ",":
            output.pop()
        output.extend(reversed(stack))
    return "".join(output)


def _recover_partial_array(raw: str, array_key: str) -> List[Dict]:
    """Best-effort recovery of complete elements of a top-level JSON array
    from a response that failed to parse as a whole — typically a
    generation truncated mid-object by a token/length limit, or a single
    malformed element blocking json.loads() on an otherwise-good response.
    Only called as a fallback when _parse_json_safe(raw) already raised;
    the normal path never reaches this. Never raises; returns [] if
    nothing could be recovered.

    Used for two array shapes: the unified extraction prompt's "nodes"
    (ordered first in that schema specifically so a truncation loses the
    cheaper, re-derivable "edges"/"temporal_zones" tail rather than the
    nodes themselves) and the batched Pass D prompt's "results".

    Scans char-by-char tracking brace depth and string/escape state (so a
    "}" inside a quoted evidence string is never mistaken for the end of
    an object), collects each top-level element as its own substring, and
    parses it independently — one malformed element is skipped without
    losing every element around it.
    """
    clean = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "")
    key_pos = clean.find(f'"{array_key}"')
    if key_pos < 0:
        return []
    arr_start = clean.find("[", key_pos)
    if arr_start < 0:
        return []

    recovered: List[Dict] = []
    depth = 0
    in_string = False
    escape = False
    obj_start = None
    i = arr_start + 1
    n = len(clean)
    while i < n:
        ch = clean[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and obj_start is not None:
                    candidate = clean[obj_start:i + 1]
                    try:
                        recovered.append(json.loads(candidate))
                    except json.JSONDecodeError:
                        try:
                            recovered.append(json.loads(_repair_common_json_syntax(candidate)))
                        except json.JSONDecodeError:
                            pass  # keep scanning after an unrecoverable element
                    obj_start = None
            elif ch == "]" and depth == 0:
                break  # reached the end of the array cleanly
        i += 1
    return recovered
def _parse_llm_json(raw: str) -> Dict:
    clean = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    brace = clean.find("{")
    if brace > 0:
        clean = clean[brace:]
    rbrace = clean.rfind("}")
    if rbrace >= 0:
        clean = clean[:rbrace + 1]
    return json.loads(clean)


# ══════════════════════════════════════════════════════════════════
#  ИДЕНТИЧНОСТЬ УЗЛА: единственная точка входа (TODO #19)
#  MD5(source+text) (NodeEdgeAssembler.make_stable_id) удалён из кода —
#  это единственный способ получить id узла.

__all__ = ['_parse_json_safe', '_repair_common_json_syntax', '_recover_partial_array', '_parse_llm_json']
