"""
postprocess.py
Shared JSON extraction, recovery, and validation utilities
for the JD Generator + Editor combined model.
"""

import json


REQUIRED_KEYS = [
    "job_title",
    "location",
    "industry",
    "responsibilities",
    "requirements",
    "qualifications",
    "experience",
    "other_requirements",
]


def extract_json(text: str) -> dict | None:
    """
    Extract the first complete JSON object from model output text.
    If truncated mid-token, trims back to last complete list item and closes structure.
    Returns a dict or None if unrecoverable.
    """
    start = text.find("{")
    if start == -1:
        return None

    # Try clean parse first
    brace_count = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        if brace_count == 0:
            end = i
            break

    if end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Truncation recovery — trim to last safely closed string item
    raw = text[start:]
    last_safe = max(raw.rfind('",'), raw.rfind('"\n'))
    if last_safe != -1:
        raw = raw[:last_safe + 1]

    open_brackets = raw.count("[") - raw.count("]")
    open_braces   = raw.count("{") - raw.count("}")
    raw += "]" * open_brackets + "}" * open_braces

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def safe_parse(text: str) -> dict | None:
    """
    Parse a JSON string safely. Returns dict or None.
    Handles accidental markdown fences (```json ... ```) around output.
    """
    if isinstance(text, dict):
        return text
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return extract_json(text)


def validate_json(data: dict | None) -> dict:
    """
    Ensure all required keys exist in the output dict.
    Fills missing keys with empty string or empty list as appropriate.
    """
    if data is None:
        data = {}

    list_keys = {"responsibilities", "requirements", "qualifications",
                 "experience", "other_requirements"}

    for key in REQUIRED_KEYS:
        if key not in data:
            data[key] = [] if key in list_keys else ""

    return data


def enforce_input_constraints(data: dict, raw_input: str) -> dict:
    """
    Strip hallucinated fields not grounded in the original raw input.
    Only applied during generation, not editing.
    """
    lowered = raw_input.lower()

    if "industry" not in lowered:
        data["industry"] = ""

    if not any(w in lowered for w in ["degree", "bachelor", "master", "phd"]):
        data["qualifications"] = ""

    return data