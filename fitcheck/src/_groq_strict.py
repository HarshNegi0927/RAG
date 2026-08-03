"""
_groq_strict.py
CORRECTION (see FIXES_AND_INSIGHTS notes): this originally added "strict": true
inside a `tools[].function` definition, forced via `tool_choice`. That had no
effect -- Groq's docs state plainly that tool use is not supported with
Structured Outputs at all, so the "strict" flag there was silently ignored
and the model fell back to ordinary best-effort tool-calling, which is why
the exact same class of bug (missing/renamed fields) kept recurring even
after that "fix".

Groq's actual guaranteed-schema mechanism is a SEPARATE, incompatible code
path: `response_format: {"type": "json_schema", "json_schema": {"strict":
true, "schema": ...}}`, with no `tools`/`tool_choice` involved at all. That
uses real constrained decoding -- the model is restricted at the token level,
so a missing or renamed field becomes structurally impossible, not just
prompted against. This module now builds that `response_format` payload
instead of a tool definition. Response text comes back in
`message.content` as a JSON string, not `message.tool_calls[...]`.

Requirements for strict mode (per Groq's docs) -- every object node in the
schema, including nested ones under $defs, must have:
  - "additionalProperties": false
  - every one of its properties listed in "required" (fields that are
    conceptually optional must instead allow null in their type, e.g.
    `anyOf: [{"type": "array", ...}, {"type": "null"}]` -- "required but
    nullable" is how strict mode represents "optional").
"""


def make_strict(schema: dict) -> dict:
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            schema["additionalProperties"] = False
            schema["required"] = list(schema["properties"].keys())
        for v in schema.values():
            make_strict(v)
    elif isinstance(schema, list):
        for item in schema:
            make_strict(item)
    return schema


def strict_response_format(name: str, schema: dict) -> dict:
    """Build a Groq/OpenAI `response_format` payload with strict mode enabled."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": make_strict(schema),
        },
    }
