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

_FLOOD_BIAS = (
    "A missed flood (false negative) is far more costly than a false alarm in this "
    "application. When you are genuinely torn between two adjacent categories, "
    "resolve the tie toward the MORE severe one, not the less severe one."
)

LABEL_CRITERIA = (
    "low: large exposed concrete wall margin between water and grass/wall-top -- normal/dry-weather level. "
    "medium: water has risen but a visible wall margin remains. "
    "high: little or no wall margin remains, but water has not yet reached the grass. "
    "flood: water is above the top of the wall and touching the grass bank -- a specific physical threshold, not just \"a lot of water\"."
)

# --- Binary ablation: collapses low/medium/high into a single not_flood class, testing
# whether task framing itself (not just RAG) affects performance. Isolates the one
# threshold our own criteria doc calls "a specific physical threshold, not a vague
# impression" from the three fuzzier, proportional low/medium/high boundaries.
_STEP_JUDGE_BINARY = (
    "Judge strictly by a single physical threshold: has the water risen above the top "
    "of the concrete wall and made contact with the grass embankment on the far bank? "
    "If yes, classify as flood. If no -- regardless of how high or close the water is, "
    "even if the wall margin is very small -- classify as not_flood. This is a specific "
    "physical threshold, not a vague impression of how much water is visible."
)

_RARITY_NOTE_BINARY = (
    "Flood is rare at this site (occurs in a small minority of images). Do not let that "
    "rarity bias you toward under-calling flood when the water is genuinely touching the "
    "grass -- but also do not over-call flood just because the water level looks high; "
    "the wall-contact threshold is what matters, not overall water volume or how close "
    "the water appears to the top of the wall."
)

_FLOOD_BIAS_BINARY = (
    "A missed flood (false negative) is far more costly than a false alarm in this "
    "application. When you are genuinely uncertain whether the water has crossed the "
    "wall-top/grass-contact threshold, lean toward classifying as flood."
)

BINARY_LABEL_CRITERIA = (
    "flood: water is above the top of the wall and touching the grass bank -- a specific "
    "physical threshold, not just \"a lot of water\". "
    "not_flood: water has not crossed that threshold -- this covers everything from normal "
    "dry-weather conditions up to water very close to the top of the wall but not yet "
    "touching the grass."
)

# Fixed, non-model-chosen query used to retrieve text-knowledge context whenever
# text RAG is enabled -- retrieval always fires (see agent/classifier.py); there is no
# tool-calling decision involved, so this query has to be generically good rather than
# tailored per example.
TEXT_RAG_QUERY = (
    "physical criteria and thresholds for classifying water level as low, medium, high, "
    "or flood at this site, including nighttime-glare guidance and documented flood events"
)
TEXT_RAG_N_RESULTS = 5

IMAGE_RAG_N_RESULTS = 3


def format_retrieved_passages(passages: list[dict]) -> str:
    """Render retrieve_flood_knowledge's passages list as an inline prompt section."""
    if not passages:
        return ""
    blocks = [f"- {p['title']}: {p['text']}" for p in passages]
    return "Reference knowledge (retrieved for this task):\n" + "\n".join(blocks) + "\n\n"


def format_retrieved_exemplars(examples: list[dict]) -> str:
    """Render retrieve_similar_examples' metadata (season/date deliberately omitted --
    see build_user_text_block for why) as an inline prompt section describing the
    attached reference images, in the same order the images are attached."""
    if not examples:
        return ""
    lines = []
    for i, ex in enumerate(examples, 1):
        if not ex.get("image_available_locally", True):
            continue
        night = "nighttime" if ex["is_night"] else "daytime"
        lines.append(f"- Reference image {i}: labeled '{ex['label']}', {night} shot.")
    if not lines:
        return ""
    return (
        "Attached below are labeled reference photos of this site (visually similar to "
        "the current frame, from other points in time) for comparison:\n" + "\n".join(lines) + "\n\n"
    )


def build_single_shot_prompt(
    use_rag: bool,
    use_image_rag: bool,
    retrieved_text: str = "",
    retrieved_exemplars: str = "",
    binary: bool = False,
) -> str:
    """The one prompt builder for every arm (baseline = use_rag=False, use_image_rag=False).
    Retrieval, when enabled, is not a tool the model may or may not invoke -- it always
    runs before this prompt is built (see agent/classifier.py), and its results are
    injected directly here. There is no multi-turn loop and no tool schema.

    binary=True switches to the flood/not_flood ablation (see BINARY_LABEL_CRITERIA)."""
    step_judge = _STEP_JUDGE_BINARY if binary else _STEP_JUDGE
    rarity_note = _RARITY_NOTE_BINARY if binary else _RARITY_NOTE
    flood_bias = _FLOOD_BIAS_BINARY if binary else _FLOOD_BIAS
    label_criteria = BINARY_LABEL_CRITERIA if binary else LABEL_CRITERIA
    labels_json = '"flood"|"not_flood"' if binary else '"low"|"medium"|"high"|"flood"'

    base = _SCENE
    if retrieved_text:
        base += retrieved_text
    if retrieved_exemplars:
        base += retrieved_exemplars
    base += "Before answering:\n1. " + step_judge + "\n2. " + _STEP_NIGHT + "\n\n"
    base += "Level definitions: " + label_criteria + "\n\n"
    base += (
        "Your rationale must cite specific visual details you observed"
        + (" and/or the reference knowledge/images above" if (use_rag or use_image_rag) else "")
        + " -- do not state a classification you cannot trace back to something visible "
        "in the image" + (" or explicitly provided above" if (use_rag or use_image_rag) else "") + ". "
        + rarity_note + " " + flood_bias + "\n\n"
    )
    base += (
        "Respond with ONLY a single JSON object and no other text, in exactly this form:\n"
        '{"classification": ' + labels_json + ', "confidence": <number 0.0-1.0>, '
        '"rationale": "<explanation citing specific visual details>", '
        '"cited_evidence": ["<short description of each visual detail used>"]}'
    )
    return base


def build_user_text_block(example: dict, binary: bool = False) -> str:
    # Deliberately excludes datetime/season/place: giving the model the exact date lets
    # it (a) correlate with same-period text-RAG event chunks, and (b) potentially
    # recall real-world news about this exact date from its own pretraining data --
    # both are answer leaks unrelated to the actual visual judgment being tested.
    # is_night is kept because it's legitimately needed for the night-glare guidance.
    task = "Classify whether the water level visible in this image is flood or not_flood." if binary \
        else "Classify the water level visible in this image (low / medium / high / flood)."
    return f"{'Nighttime' if example['is_night'] else 'Daytime'} shot.\n\n{task}"
