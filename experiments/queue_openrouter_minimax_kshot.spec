# Lane: openrouter, minimax/minimax-m3 -- the K-shot cells NVIDIA could not serve.
#
# NVIDIA NIM rejected K>=2 with "Request payload is too large for this deployment" (~6.3 MB at
# K=2). OpenRouter accepts the same content: a K=4 probe (13 images, ~11.6 MB, 8,542 prompt
# tokens) returned a schema-valid response on the first attempt. The limit was the deployment's,
# not the model's, which is what the paper's Section 4.2 says.
#
# Budget note. OpenRouter lists minimax/minimax-m3 at $0.30/M input and $1.20/M output, not the
# $0.24/$0.96 a promotional rate suggested. Measured from the probe, a K=4 call costs ~$0.0029
# and a K=2 call ~$0.0018, so these six cells are ~$5.34 of the $6.00 available. Re-running K=1
# here for endpoint consistency would add ~$1.44 and does not fit; K=1 therefore stays on the
# NVIDIA cells and the shot ladder mixes serving stacks. That is a disclosed limitation, not an
# oversight -- see the note added to Section 4.2.
#
# Ordered K=2 before K=4 so the cheaper, faster cells land first if the budget runs short.
# provider|model|rpm|out_stem|input_jsonl|extra flags
openrouter|minimax/minimax-m3|20|k2_e0_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 0 --workers 4
openrouter|minimax/minimax-m3|20|k2_e1_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 1 --workers 4
openrouter|minimax/minimax-m3|20|k2_e2_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 2 --workers 4
openrouter|minimax/minimax-m3|20|k4_e0_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k4_e3.json --episode 0 --workers 4
openrouter|minimax/minimax-m3|20|k4_e1_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k4_e3.json --episode 1 --workers 4
openrouter|minimax/minimax-m3|20|k4_e2_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k4_e3.json --episode 2 --workers 4
