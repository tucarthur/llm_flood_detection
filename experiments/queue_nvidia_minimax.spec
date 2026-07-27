# Lane: nvidia NIM, minimaxai/minimax-m3 -- the fourth model, added late.
#
# Why this model: of six candidates piloted it is the only one that clears all three gates
# -- strict JSON schema works, multi-image prompts are accepted (so it can appear in the
# K-shot arm), and it emits the positive class (2/2 floods in a 30-frame pilot, zero false
# alarms on dry frames). It is also from a different family than the two Gemma models, which
# is what the model-dependence claims need. Rejected candidates: llama-3.2-90b-vision
# (hard 400 on multi-image, 0 flood predictions in 24 frames), mistral-small-4 (0.000 flood
# recall), deepseek-v4-pro (404), qwen3.5-122b (410 mid-study), llama-4-scout (404).
#
# Rate settings are measured, not guessed. Concurrency triggers 429s hard on this endpoint:
# 8 workers gave 47% RateLimitError, 16 workers 84%. At rpm 10 with 4 workers the error rate
# is zero. Mean latency is ~63s per call (6x gemma-4-31b), so throughput is latency-bound at
# roughly 4 rows/min regardless of the cap: ~6.5h per full-n cell, ~1.5h per subsample cell.
#
# Ordered by value: the two full-n zero-shot cells feed the paper's main comparison, then the
# sensitivity sweep (needed before any of its criteria effects can be interpreted), then
# K-shot last since K=1 and K=2 already degrade on both Gemma models.
# provider|model|rpm|out_stem|input_jsonl|extra flags
nvidia|minimaxai/minimax-m3|10|zs_3class_minimax_m3||--taxonomy 3class --workers 4
nvidia|minimaxai/minimax-m3|10|zsnaive_3class_minimax_m3||--taxonomy 3class --workers 4 --no-criteria
nvidia|minimaxai/minimax-m3|10|sens_v2_sub_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --prompt-variant v2 --workers 4
nvidia|minimaxai/minimax-m3|10|sens_v3_sub_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --prompt-variant v3 --workers 4
nvidia|minimaxai/minimax-m3|10|sens_v4_sub_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --prompt-variant v4 --workers 4
nvidia|minimaxai/minimax-m3|10|sens_v5_sub_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --prompt-variant v5 --workers 4
nvidia|minimaxai/minimax-m3|10|k1_e0_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k1_e3.json --episode 0 --workers 4
nvidia|minimaxai/minimax-m3|10|k1_e1_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k1_e3.json --episode 1 --workers 4
nvidia|minimaxai/minimax-m3|10|k1_e2_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k1_e3.json --episode 2 --workers 4
nvidia|minimaxai/minimax-m3|10|k2_e0_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 0 --workers 4
nvidia|minimaxai/minimax-m3|10|k2_e1_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 1 --workers 4
nvidia|minimaxai/minimax-m3|10|k2_e2_minimax_m3|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 2 --workers 4
