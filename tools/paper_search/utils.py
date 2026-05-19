from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | list | None:
    """Extract the first JSON object or array from *text*.

    Handles raw JSON, markdown code fences, and surrounding prose.
    """
    text = text.strip()
    if not text:
        return None

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text)
    stripped = re.sub(r"\n?```\s*$", "", stripped).strip()

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: find the first JSON array or object
    for pattern in (r"\[.*\]", r"\{.*\}"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, (dict, list)):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())
