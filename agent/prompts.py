def build_system_prompt(use_rag: bool) -> str:
    base = (
        "You are a water-level classification agent for a fixed river camera at "
        "Mineirinho Creek (Sao Carlos, Brazil). You will be shown one photograph from "
        "the camera, looking down into a concrete drainage canal from a road bridge, "
        "with a grass embankment above the canal wall on the far bank.\n\n"
    )
    steps = []
    if use_rag:
        steps.append(
            "Call retrieve_flood_knowledge at least once to ground your classification in "
            "the documented low/medium/high/flood criteria for this specific site."
        )
    steps.append(
        "Judge severity primarily by how much concrete wall is visible between the "
        "water surface and the grass/wall-top line -- a large exposed wall margin means "
        "low, little or no margin means high, and water actually touching the grass "
        "above the wall means flood. This is a specific physical threshold, not a vague "
        "impression of how much water is visible."
    )
    steps.append(
        "If the image is a nighttime shot with heavy glare from the streetlight in "
        "frame, say so explicitly in your rationale and lower your confidence rather "
        "than guessing with false certainty."
    )
    base += "Before answering:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)) + "\n\n"
    base += (
        "When you are done gathering evidence, call submit_classification exactly once with "
        "your final answer. Your rationale must cite specific visual details you observed "
        "(and retrieved passages, if you used them) -- do not state a classification you "
        "cannot trace back to something visible in the image or explicitly retrieved. Given "
        "how rare anything other than 'low' is at this site, do not let that rarity bias you "
        "toward under-calling a higher category when the visual evidence actually supports it."
    )
    return base


SUBMIT_CLASSIFICATION_TOOL_SCHEMA = {
    "name": "submit_classification",
    "description": "Submit the final water-level classification for this image. Call exactly once, after gathering evidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "classification": {
                "type": "string",
                "enum": ["low", "medium", "high", "flood"],
                "description": "low: large exposed concrete wall margin between water and grass/wall-top -- normal/dry-weather level. "
                "medium: water has risen but a visible wall margin remains. "
                "high: little or no wall margin remains, but water has not yet reached the grass. "
                "flood: water is above the top of the wall and touching the grass bank -- a specific physical threshold, not just \"a lot of water\".",
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
