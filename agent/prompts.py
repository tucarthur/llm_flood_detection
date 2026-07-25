"""Prompt construction for the water-level classifier.

Three things about this module are load-bearing methodology, not style:

1. **Response field order is reasoning order.** The JSON schema asks for
   `observations` -> `rationale` -> `classification` -> `confidence`, in that order,
   because under constrained decoding (JSON mode with a strict schema) the schema's
   key order *is* the generation order. The previous classification-first schema made
   the model commit to a label before writing any justification, which (a) reduced the
   rationale to post-hoc narration and (b) pushed models into emitting an unrequested
   `<thought>` preamble instead of JSON -- the single cause of all 1,101 parse failures
   observed in the earlier runs. Reasoning fields must stay first.

2. **Physical criteria are held fixed across prompt variants; only incidental framing
   is paraphrased.** `PROMPT_VARIANTS` exists for prompt-sensitivity reporting (report
   the distribution over paraphrases rather than tuning a single prompt, which is
   indefensible on 10 flood events). The variants therefore paraphrase the scene
   description, night guidance and evidence instruction, while each taxonomy's
   `judge`/`criteria`/`rarity`/`cost_asymmetry` text -- the operative definitions and
   the cost-asymmetry instruction -- are identical across variants. Sensitivity to
   wording is the quantity of interest; sensitivity to changed definitions is not.

3. **Criteria can be withheld** (`include_criteria=False`). This is the naive
   zero-shot arm: no verbal specification of the thresholds at all. With it enabled you
   are measuring *verbally-specified* zero-shot, which is a different and much stronger
   condition, and the difference between the two is how much labeled data a paragraph
   of expert text is worth.

Text retrieval has been removed: the retrieved corpus restated criteria that every arm
already carried inline, and measurably hurt (it raised false alarms without buying
recall). See agent/retrieval.py's docstring.
"""
from __future__ import annotations

from dataclasses import dataclass

from data.taxonomy import labels_for, map_label, ordinal_index


# --------------------------------------------------------------------------------------
# Taxonomy-specific text: operative definitions. Identical across prompt variants.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TaxonomyPrompt:
    judge: str
    criteria: str
    rarity: str
    cost_asymmetry: str
    task: str


TAXONOMY_PROMPTS: dict[str, TaxonomyPrompt] = {
    "3class": TaxonomyPrompt(
        judge=(
            "Judge by how much bare concrete wall is visible between the water surface "
            "and the grass/wall-top line on the far bank. A large exposed wall margin "
            "means no_risk. A reduced margin -- anywhere from partially covered down to "
            "no margin left at all -- means risky, as long as the water is still below "
            "the top of the wall. Water that has risen above the wall top and is "
            "touching the grass means flood. These are physical thresholds, not a vague "
            "impression of how much water is visible."
        ),
        criteria=(
            "no_risk: a large exposed concrete wall margin separates the water from the "
            "grass/wall-top line -- normal or dry-weather level. "
            "risky: the water has risen far enough to reduce that margin, from partially "
            "covered through to no margin remaining, but has not reached the grass. "
            "flood: the water is above the top of the wall and touching the grass bank -- "
            "a specific physical threshold, not just \"a lot of water\"."
        ),
        rarity=(
            "Anything other than no_risk is rare at this site. Do not let that rarity bias "
            "you toward under-calling a more severe category when the visual evidence "
            "actually supports it."
        ),
        cost_asymmetry=(
            "A missed flood (false negative) is far more costly than a false alarm in this "
            "application. When you are genuinely torn between two adjacent categories, "
            "resolve the tie toward the MORE severe one."
        ),
        task="Classify the water level visible in this image (no_risk / risky / flood).",
    ),
    "4class": TaxonomyPrompt(
        judge=(
            "Judge severity primarily by how much concrete wall is visible between the "
            "water surface and the grass/wall-top line -- a large exposed wall margin "
            "means low, little or no margin means high, and water actually touching the "
            "grass above the wall means flood. This is a specific physical threshold, not "
            "a vague impression of how much water is visible."
        ),
        criteria=(
            "low: large exposed concrete wall margin between water and grass/wall-top -- "
            "normal/dry-weather level. "
            "medium: water has risen but a visible wall margin remains. "
            "high: little or no wall margin remains, but water has not yet reached the grass. "
            "flood: water is above the top of the wall and touching the grass bank -- a "
            "specific physical threshold, not just \"a lot of water\"."
        ),
        rarity=(
            "Given how rare anything other than 'low' is at this site, do not let that "
            "rarity bias you toward under-calling a higher category when the visual "
            "evidence actually supports it."
        ),
        cost_asymmetry=(
            "A missed flood (false negative) is far more costly than a false alarm in this "
            "application. When you are genuinely torn between two adjacent categories, "
            "resolve the tie toward the MORE severe one, not the less severe one."
        ),
        task="Classify the water level visible in this image (low / medium / high / flood).",
    ),
    "binary": TaxonomyPrompt(
        judge=(
            "Judge strictly by a single physical threshold: has the water risen above the "
            "top of the concrete wall and made contact with the grass embankment on the far "
            "bank? If yes, classify as flood. If no -- regardless of how high or close the "
            "water is, even if the wall margin is very small -- classify as not_flood. This "
            "is a specific physical threshold, not a vague impression of how much water is "
            "visible."
        ),
        criteria=(
            "flood: water is above the top of the wall and touching the grass bank -- a "
            "specific physical threshold, not just \"a lot of water\". "
            "not_flood: water has not crossed that threshold -- this covers everything from "
            "normal dry-weather conditions up to water very close to the top of the wall but "
            "not yet touching the grass."
        ),
        rarity=(
            "Flood is rare at this site (it occurs in a small minority of images). Do not let "
            "that rarity bias you toward under-calling flood when the water is genuinely "
            "touching the grass -- but also do not over-call flood just because the water "
            "level looks high; the wall-contact threshold is what matters, not overall water "
            "volume or how close the water appears to the top of the wall."
        ),
        cost_asymmetry=(
            "A missed flood (false negative) is far more costly than a false alarm in this "
            "application. When you are genuinely uncertain whether the water has crossed the "
            "wall-top/grass-contact threshold, lean toward classifying as flood."
        ),
        task="Classify whether the water level visible in this image is flood or not_flood.",
    ),
}


# --------------------------------------------------------------------------------------
# Prompt variants: paraphrases of incidental framing only, for sensitivity reporting.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PromptVariant:
    scene: str
    night: str
    evidence: str  # "{sources}" is filled with what the rationale may cite


PROMPT_VARIANTS: dict[str, PromptVariant] = {
    # v1 is the wording used by every run recorded before the 3-class rework, kept
    # verbatim so those runs remain comparable to the v1 cells of the new matrix.
    "v1": PromptVariant(
        scene=(
            "You are a water-level classification agent for a fixed river camera at "
            "Mineirinho Creek (Sao Carlos, Brazil). You will be shown one photograph from "
            "the camera, looking down into a concrete drainage canal from a road bridge, "
            "with a grass embankment above the canal wall on the far bank."
        ),
        night=(
            "If the image is a nighttime shot with heavy glare from the streetlight in "
            "frame, say so explicitly in your rationale and lower your confidence rather "
            "than guessing with false certainty."
        ),
        evidence=(
            "Your rationale must cite specific visual details you observed{sources} -- do "
            "not state a classification you cannot trace back to something visible in the "
            "image{sources_tail}."
        ),
    ),
    "v2": PromptVariant(
        scene=(
            "You assess water level from a fixed monitoring camera on Mineirinho Creek in "
            "Sao Carlos, Brazil. Each photograph is taken from a road bridge looking down "
            "into a concrete drainage canal; above the canal wall on the far side of the "
            "channel there is a grass embankment."
        ),
        night=(
            "Where a frame was taken at night and streetlight glare obscures the water "
            "line, state that limitation in your rationale and reduce your confidence "
            "accordingly instead of asserting a level you cannot actually see."
        ),
        evidence=(
            "Ground your rationale in concrete visual detail{sources}; every "
            "classification you give must be traceable to something actually visible in "
            "the photograph{sources_tail}."
        ),
    ),
    "v3": PromptVariant(
        scene=(
            "Task: read the water level in a photograph from a stationary river camera at "
            "Mineirinho Creek, Sao Carlos, Brazil. The camera sits on a road bridge and "
            "points down at a concrete drainage canal, with grass embankment visible above "
            "the canal wall on the opposite bank."
        ),
        night=(
            "Night frames often carry strong streetlight glare that can hide the waterline. "
            "When that is the case, name the problem in your rationale and lower your "
            "confidence rather than committing to a level with false certainty."
        ),
        evidence=(
            "Point to specific things you can see{sources}. Do not report a classification "
            "that cannot be traced back to visible evidence in the image{sources_tail}."
        ),
    ),
}

DEFAULT_VARIANT = "v1"


# --------------------------------------------------------------------------------------
# Few-shot exemplars
# --------------------------------------------------------------------------------------
def severity_sort_key(exemplar: dict, taxonomy: str) -> int:
    """Ascending-severity position of an exemplar's (gold-labeled) class.

    In-context learning is order-sensitive, so exemplar order is fixed by construction
    rather than left to whatever order the sampler or the vector store returned. Callers
    must sort the attached image payloads with the same key so text descriptions and
    images stay aligned.
    """
    return ordinal_index(taxonomy)[map_label(exemplar["label"], taxonomy)]


def format_exemplars(examples: list[dict], taxonomy: str, kind: str = "reference") -> str:
    """Render labeled exemplars as a prompt section describing the attached images, in
    the same order the images are attached.

    `kind` selects the framing: "reference" for similarity-retrieved neighbours of the
    current frame, "calibration" for a support set that is not similar to the current
    frame and is there only to locate the class boundaries. Season/date are deliberately
    omitted -- see build_user_text_block.
    """
    available = [ex for ex in examples if ex.get("image_available_locally", True)]
    if not available:
        return ""

    noun = "Reference" if kind == "reference" else "Calibration"
    lines = []
    for i, ex in enumerate(available, 1):
        night = "nighttime" if ex["is_night"] else "daytime"
        label = map_label(ex["label"], taxonomy)
        lines.append(f"- {noun} image {i}: labeled '{label}', {night} shot.")

    if kind == "reference":
        preamble = (
            "Attached below are labeled reference photos of this site (visually similar to "
            "the current frame, from other points in time), ordered from least to most "
            "severe, for comparison:"
        )
    else:
        preamble = (
            "Attached below are labeled example photos of this exact site, ordered from "
            "least to most severe, showing what each level actually looks like here. They "
            "are NOT similar to the current frame -- use them only to calibrate where the "
            "thresholds fall, especially the wall-margin and grass-contact distinctions:"
        )
    return preamble + "\n" + "\n".join(lines) + "\n\n"


# --------------------------------------------------------------------------------------
# Response schema
# --------------------------------------------------------------------------------------
# Field order == generation order under constrained decoding. See module docstring.
RESPONSE_FIELD_ORDER = ["observations", "rationale", "classification", "confidence", "cited_evidence"]


def response_json_schema(taxonomy: str) -> dict:
    """Strict JSON schema for structured-output (JSON) mode.

    `classification` is an enum over the taxonomy's labels, which is what removes the
    out-of-vocabulary label failure mode; the two free-text reasoning fields come first
    so the model still reasons before committing to a label.
    """
    return {
        "type": "object",
        "properties": {
            "observations": {
                "type": "string",
                "description": "What is visible in the image, especially the wall margin between the water surface and the grass/wall-top line. Reason here before deciding.",
            },
            "rationale": {
                "type": "string",
                "description": "How those observations map onto the level definitions.",
            },
            "classification": {"type": "string", "enum": labels_for(taxonomy)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "cited_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short description of each visual detail used.",
            },
        },
        "required": RESPONSE_FIELD_ORDER,
        "additionalProperties": False,
    }


def _response_format_block(taxonomy: str) -> str:
    labels = "|".join(f'"{label}"' for label in labels_for(taxonomy))
    return (
        "Respond with ONLY a single JSON object and no other text -- no preamble, no "
        "thinking block, no markdown fence. Fill the fields in the order given; the two "
        "reasoning fields come first so that you work out what you see before naming a "
        "level:\n"
        '{"observations": "<what you can see, especially the wall margin between the '
        'water surface and the grass/wall-top line>", '
        '"rationale": "<how those observations map onto the level definitions>", '
        '"classification": ' + labels + ", "
        '"confidence": <number 0.0-1.0>, '
        '"cited_evidence": ["<short description of each visual detail used>"]}'
    )


# --------------------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------------------
def build_system_prompt(
    taxonomy: str = "3class",
    variant: str = DEFAULT_VARIANT,
    exemplars_text: str = "",
    include_criteria: bool = True,
) -> str:
    """Assemble the system prompt.

    taxonomy         -- which prompted task (see data/taxonomy.py)
    variant          -- which framing paraphrase (prompt-sensitivity axis)
    exemplars_text   -- rendered few-shot section, or "" for the zero-shot arm
    include_criteria -- False gives the naive zero-shot arm: no verbal thresholds
    """
    tax = TAXONOMY_PROMPTS[taxonomy] if taxonomy in TAXONOMY_PROMPTS else None
    if tax is None:
        raise ValueError(f"unknown taxonomy {taxonomy!r} (expected one of {list(TAXONOMY_PROMPTS)})")
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r} (expected one of {list(PROMPT_VARIANTS)})")
    var = PROMPT_VARIANTS[variant]

    has_exemplars = bool(exemplars_text)
    sources = " and/or the reference images above" if has_exemplars else ""
    sources_tail = " or explicitly provided above" if has_exemplars else ""

    parts = [var.scene + "\n\n"]
    if exemplars_text:
        parts.append(exemplars_text)

    # The criteria block is what makes this "verbally-specified" rather than naive
    # zero-shot; withholding it also withholds the judging instruction, since that
    # restates the same thresholds in imperative form.
    if include_criteria:
        parts.append("Before answering:\n1. " + tax.judge + "\n2. " + var.night + "\n\n")
        parts.append("Level definitions: " + tax.criteria + "\n\n")
    else:
        parts.append("Before answering:\n1. " + var.night + "\n\n")

    parts.append(var.evidence.format(sources=sources, sources_tail=sources_tail) + " ")
    if include_criteria:
        parts.append(tax.rarity + " " + tax.cost_asymmetry)
    else:
        parts.append(tax.cost_asymmetry)
    parts.append("\n\n")

    parts.append(_response_format_block(taxonomy))
    return "".join(parts)


def build_user_text_block(example: dict, taxonomy: str = "3class") -> str:
    """The per-example user turn.

    Deliberately excludes datetime/season/place: the exact date would let the model
    recall real-world news about this specific date from its own pretraining, which is
    an answer leak unrelated to the visual judgment being tested. `is_night` is kept
    because the night-glare guidance legitimately needs it.
    """
    task = TAXONOMY_PROMPTS[taxonomy].task
    return f"{'Nighttime' if example['is_night'] else 'Daytime'} shot.\n\n{task}"
