import json

def extract_first_json(text: str) -> str:
    """
    Returns the first complete, balanced JSON object from a string.
    If the object is truncated (model hit token limit mid-generation),
    attempts to repair it by closing open braces/brackets.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output:\n{text}")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    # Reached end without closing — attempt repair on truncated output
    candidate = text[start:]
    # Close an unterminated string
    if in_string:
        candidate += '"'
    # Close any open arrays then objects
    # Count unclosed brackets/braces
    open_braces = candidate.count("{") - candidate.count("}")
    open_brackets = candidate.count("[") - candidate.count("]")
    candidate += "]" * open_brackets
    candidate += "}" * open_braces

    try:
        json.loads(candidate)   # validate the repair worked
        print("[utils] Warning: repaired truncated JSON output")
        return candidate
    except json.JSONDecodeError:
        raise ValueError(
            f"No complete JSON object found and repair failed "
            f"(model likely hit token limit):\n{text}"
        )