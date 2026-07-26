# Lane: nvidia NIM, block 2. Sequential (the ~40 req/min ceiling is per-key).
#
# Same ordering logic as the gemini block-2 spec: prompt-sensitivity control first, then
# the comparability (4class) and diagnostic (binary) taxonomies. Including nemotron in the
# sensitivity sweep matters because it is the one model whose block-1 behaviour differed in
# sign from the gemma family -- a claim about prompt sensitivity cannot rest on one family.
#
# mistral-small-4 is deliberately absent: 0.000 flood recall in BOTH block-1 arms.
# provider|model|rpm|out_stem|extra flags
nvidia|nvidia/llama-3.1-nemotron-nano-vl-8b-v1|30|sens_v2_3class_nemotron_nano_vl_8b||--taxonomy 3class --prompt-variant v2 --workers 12
nvidia|nvidia/llama-3.1-nemotron-nano-vl-8b-v1|30|sens_v3_3class_nemotron_nano_vl_8b||--taxonomy 3class --prompt-variant v3 --workers 12
