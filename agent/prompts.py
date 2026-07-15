_SCENE = (
    "You are a water-level classification agent for a fixed river camera at "
    "Mineirinho Creek (Sao Carlos, Brazil). You will be shown one photograph from "
    "the camera, looking down into a concrete drainage canal from a road bridge, "
    "with a grass embankment above the canal wall on the far bank.\n\n"
)

_STEP_JUDGE = (
    "Judge severity primarily by how much concrete wall is visible between the "
    "water surface and the grass/wall-top line -- a large exposed wall margin means "
    "low, little or no margin means high, and water actually touching the grass "
    "above the wall means flood. This is a specific physical threshold, not a vague "
    "impression of how much water is visible."
)

_STEP_NIGHT = (
    "If the image is a nighttime shot with heavy glare from the streetlight in "
    "frame, say so explicitly in your rationale and lower your confidence rather "
    "than guessing with false certainty."
)

_RARITY_NOTE = (
    "Given how rare anything other than 'low' is at this site, do not let that "
    "rarity bias you toward under-calling a higher category when the visual "
    "evidence actually supports it."
)

LABEL_CRITERIA = (
    "low: large exposed concrete wall margin between water and grass/wall-top -- normal/dry-weather level. "
    "medium: water has risen but a visible wall margin remains. "
    "high: little or no wall margin remains, but water has not yet reached the grass. "
    "flood: water is above the top of the wall and touching the grass bank -- a specific physical threshold, not just \"a lot of water\"."
)


def build_system_prompt(use_rag: bool, use_image_rag: bool = False) -> str:
    base = _SCENE
    steps = []
    if use_rag:
        steps.append(
            "Call retrieve_flood_knowledge at least once to ground your classification in "
            "the documented low/medium/high/flood criteria for this specific site."
        )
    if use_image_rag:
        steps.append(
            "Call retrieve_similar_examples to see labeled reference photos of this site "
            "that are visually similar to the current frame, and compare the current "
            "water level against those references."
        )
    steps.append(_STEP_JUDGE)
    steps.append(_STEP_NIGHT)
    base += "Before answering:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) + "\n\n"
    base += (
        "When you are done gathering evidence, call submit_classification exactly once with "
        "your final answer. Your rationale must cite specific visual details you observed "
        "(and retrieved passages, if you used them) -- do not state a classification you "
        "cannot trace back to something visible in the image or explicitly retrieved. "
        + _RARITY_NOTE
    )
    return base


def build_baseline_system_prompt() -> str:
    """Single-shot ablation baseline: same scene context and judgment guidance as the
    agent, but no tools/retrieval -- the model answers in one turn with a bare JSON
    object instead of a submit_classification tool call."""
    return (
        _SCENE
        + "Before answering:\n1. " + _STEP_JUDGE + "\n2. " + _STEP_NIGHT + "\n\n"
        + "Level definitions: " + LABEL_CRITERIA + "\n\n"
        + "Your rationale must cite specific visual details you observed -- do not state "
        "a classification you cannot trace back to something visible in the image. "
        + _RARITY_NOTE + "\n\n"
        + "Respond with ONLY a single JSON object and no other text, in exactly this form:\n"
        '{"classification": "low"|"medium"|"high"|"flood", "confidence": <number 0.0-1.0>, '
        '"rationale": "<explanation citing specific visual details>", '
        '"cited_evidence": ["<short description of each visual detail used>"]}'
    )


SUBMIT_CLASSIFICATION_TOOL_SCHEMA = {
    "name": "submit_classification",
    "description": "Submit the final water-level classification for this image. Call exactly once, after gathering evidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": ["low", "medium", "high", "flood"],
                "description": LABEL_CRITERIA,
            },
            "confidence": {"type": "number", "description": "0.0-1.0 confidence in the classification."},
            "rationale": {"type": "string", "description": "Explanation citing specific visual details and/or retrieved passages."},
            "cited_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short descriptions of the specific visual details and/or retrieved passage titles used.",
            },
        },
        "required": ["classification", "confidence", "rationale", "cited_evidence"],
    },
}


def build_user_text_block(example: dict) -> str:
    text = (
        f"Timestamp: {example['datetime']} (season {example['season']}, "
        f"{'nighttime' if example['is_night'] else 'daytime'} shot)\n"
        f"Camera: {example['place']}\n"
        "\nClassify the water level visible in this image (low / medium / high / flood)."
    )
    return text
