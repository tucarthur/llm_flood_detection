# Lane: nvidia NIM, minimaxai/minimax-m3 -- the four prompt-sensitivity variants at FULL n.
#
# Why: minimax previously ran v2-v5 only on the 375-row subsample, while both Gemma models ran
# v2/v3 at the full 1,592. That left the non-Gemma model with no full-set sensitivity data, so
# a per-model paraphrase spread at n=1592 was impossible for it.
#
# Note on comparability: the CROSS-model spread still has to be computed on rows common to
# every variant of every model, which remains the 375-row subsample (Gemma has no v4/v5 at full
# n). These cells buy minimax its own full-n spread, not a wider cross-model comparison.
#
# Rate: rpm 3 / 2 workers. Measured over ~1,900 repaired rows at a 1-8% first-pass residual;
# the earlier rpm 10 / 4 workers produced 22-40% placeholders on cells of this length.
# Throughput ~2.5 rows/min, so ~10.5 h per full-n cell.
#
# Ordered by value: v2 and v3 first, since those are the variants Gemma also has at full n.
# provider|model|rpm|out_stem|input_jsonl|extra flags
nvidia|minimaxai/minimax-m3|3|sens_v2_3class_minimax_m3||--taxonomy 3class --prompt-variant v2 --workers 2
nvidia|minimaxai/minimax-m3|3|sens_v3_3class_minimax_m3||--taxonomy 3class --prompt-variant v3 --workers 2
nvidia|minimaxai/minimax-m3|3|sens_v4_3class_minimax_m3||--taxonomy 3class --prompt-variant v4 --workers 2
nvidia|minimaxai/minimax-m3|3|sens_v5_3class_minimax_m3||--taxonomy 3class --prompt-variant v5 --workers 2
