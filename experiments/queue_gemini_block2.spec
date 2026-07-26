# Lane: gemini, block 2. Sequential (the ~15 req/min ceiling is per-key, not per-model).
#
# Ordered by value, because this lane is long (~14h) and may be interrupted:
#   1-4. Prompt sensitivity (v2, v3) on the 3-class criteria-on cell. This is the CONTROL
#        for block 1: if paraphrase spread rivals the criteria-vs-naive effect, block 1's
#        result is prompt engineering rather than a finding about verbal specification.
#        Physical criteria are identical across variants -- only framing wording changes.
#   5-6. 4-class, for comparability with Ranieri et al. 2024.
#   7-8. Binary, the diagnostic arm isolating the grass-contact threshold.
#
# mistral-small-4 is deliberately absent: 0.000 flood recall in BOTH block-1 arms (never
# emits the positive class), so further cells on it buy nothing.
#
gemini|gemma-4-31b-it|15|sens_v2_3class_gemma4_31b||--taxonomy 3class --prompt-variant v2 --workers 4
gemini|gemma-4-26b-a4b-it|15|sens_v2_3class_gemma4_26b_a4b||--taxonomy 3class --prompt-variant v2 --workers 4
gemini|gemma-4-31b-it|15|sens_v3_3class_gemma4_31b||--taxonomy 3class --prompt-variant v3 --workers 4
gemini|gemma-4-26b-a4b-it|15|sens_v3_3class_gemma4_26b_a4b||--taxonomy 3class --prompt-variant v3 --workers 4
