def build_system_prompt(use_tools: bool, use_rag: bool) -> str:
    base = (
        "You are a flood classification agent. Given a location, a time window, "
        "and any available crisis-text reports for that window, determine whether a "
        "flood was occurring.\n\n"
    )
    steps = []
    if use_tools:
        steps.append(
            "Call get_site_baseline and get_gauge_time_series to see how far the current "
            "reading deviates from the site's pre-event baseline."
        )
    else:
        base += (
            "Structured USGS gauge data for this example is provided directly in the "
            "user message (already fetched) -- read it carefully rather than calling a tool.\n\n"
        )
    if use_rag:
        steps.append(
            "Call retrieve_flood_knowledge at least once to ground your classification in "
            "the documented flood-stage criteria and to check for a matching historical analog."
        )
    steps.append(
        "Weigh the text reports (if any) as corroborating or contradicting evidence -- text "
        "alone, without a structured signal, is weaker evidence of an active flood."
    )
    if steps:
        base += "Before answering:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) + "\n\n"
    base += (
        "When you are done gathering evidence, call submit_classification exactly once with "
        "your final answer. Your rationale must cite the specific evidence (tool outputs, "
        "retrieved passages, and/or structured data provided) you grounded on -- do not state "
        "a classification you cannot trace back to evidence."
    )
    return base

SUBMIT_CLASSIFICATION_TOOL_SCHEMA = {
    "name": "submit_classification",
    "description": "Submit the final flood classification for this example. Call exactly once, after gathering evidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": ["no_flood", "flood_watch", "active_flood"],
                "description": "no_flood: readings near baseline, no credible flood evidence. "
                "flood_watch: elevated/rising readings or credible reports, but not clearly at flood stage. "
                "active_flood: readings consistent with NWS minor/moderate/major flood stage and/or corroborated reports.",
            },
            "confidence": {"type": "number", "description": "0.0-1.0 confidence in the classification."},
            "rationale": {"type": "string", "description": "Explanation citing specific tool outputs / retrieved passages."},
            "cited_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short quotes/refs to the specific tool results or retrieved passage titles used.",
            },
        },
        "required": ["classification", "confidence", "rationale", "cited_evidence"],
    },
}


def build_user_prompt(example: dict, structured_dump: dict | None = None) -> str:
    """structured_dump: when the agent runs in no-tools ablation mode, the same
    structured data the tool-calling agent would have fetched is dumped here as text
    instead, so the ablation isolates "agentic tool orchestration" vs. "single-shot
    prompt with the same information available" -- not "has data" vs. "has no data"."""
    text_reports = example.get("text_reports") or []
    reports_block = (
        "\n".join(f"- {t}" for t in text_reports) if text_reports else "(no text reports available for this window)"
    )
    prompt = f"""Location/event: {example['event']} -- site {example['site_id']} ({example.get('site_name', '')})
Time window: {example['start_date']} to {example['end_date']}
Target date to classify: {example['target_date']}

Text reports for this window:
{reports_block}
"""
    if structured_dump is not None:
        prompt += f"""
Structured USGS gauge data (already fetched for you -- no tool call needed):
{structured_dump}
"""
    prompt += "\nDetermine the flood classification for the target date."
    return prompt
