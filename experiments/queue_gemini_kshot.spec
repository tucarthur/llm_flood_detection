# Lane: gemini, K-shot sweep. K-shot cells FIRST (headline experiment), then the
# remaining prompt-sensitivity cells from block 2, which resume from partial progress.
#
# rpm is set per K from the measured token cost, NOT from the request-rate cap. Measured
# 4,163 prompt tokens at K=4 (12 exemplars + query); base is ~1,000 tokens and each
# exemplar image ~270. Against the ~16k input-tokens/min free-tier ceiling that allows
# ~8/min at K=1, ~6 at K=2, ~4 at K=4. Setting rpm above that just earns 429s.
#
# Ordered K-major so an interruption leaves complete K levels for both models rather than
# one model's whole curve and none of the other's.
# provider|model|rpm|out_stem|input_jsonl|extra flags
# Sensitivity cells MOVED AHEAD of the K=4 block: they gate the specification finding and
# the uncertainty bands on the zero-shot reference line, whereas K=4 is a third point on a
# curve that K=1 and K=2 already show degrading. Control before confirmation.
gemini|gemma-4-31b-it|8|k1_e0_gemma4_31b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k1_e3.json --episode 0 --workers 4
gemini|gemma-4-26b-a4b-it|8|k1_e0_gemma4_26b_a4b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k1_e3.json --episode 0 --workers 4
gemini|gemma-4-31b-it|8|k1_e1_gemma4_31b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k1_e3.json --episode 1 --workers 4
gemini|gemma-4-26b-a4b-it|8|k1_e1_gemma4_26b_a4b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k1_e3.json --episode 1 --workers 4
gemini|gemma-4-31b-it|8|k1_e2_gemma4_31b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k1_e3.json --episode 2 --workers 4
gemini|gemma-4-26b-a4b-it|8|k1_e2_gemma4_26b_a4b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k1_e3.json --episode 2 --workers 4
gemini|gemma-4-31b-it|6|k2_e0_gemma4_31b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 0 --workers 4
gemini|gemma-4-26b-a4b-it|6|k2_e0_gemma4_26b_a4b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 0 --workers 4
gemini|gemma-4-31b-it|6|k2_e1_gemma4_31b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 1 --workers 4
gemini|gemma-4-26b-a4b-it|6|k2_e1_gemma4_26b_a4b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 1 --workers 4
gemini|gemma-4-31b-it|6|k2_e2_gemma4_31b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 2 --workers 4
gemini|gemma-4-26b-a4b-it|6|k2_e2_gemma4_26b_a4b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 2 --workers 4
gemini|gemma-4-31b-it|15|sens_v2_3class_gemma4_31b||--taxonomy 3class --prompt-variant v2 --workers 4
gemini|gemma-4-26b-a4b-it|15|sens_v2_3class_gemma4_26b_a4b||--taxonomy 3class --prompt-variant v2 --workers 4
gemini|gemma-4-31b-it|15|sens_v3_3class_gemma4_31b||--taxonomy 3class --prompt-variant v3 --workers 4
gemini|gemma-4-26b-a4b-it|15|sens_v3_3class_gemma4_26b_a4b||--taxonomy 3class --prompt-variant v3 --workers 4
# v4/v5 run on the 375-row SUBSAMPLE, not the full set. The 5-variant spread is then
# computed with v1/v2/v3 restricted to those same 375 rows, so all five are compared on
# identical examples -- 4x cheaper with no loss of comparability. The subsample keeps every
# flood frame and all 10 flood events, so flood recall and event detection are unaffected
# by the restriction; only the macro-F1 class prior shifts, and it shifts identically for
# all five variants. The 3-variant spread at full n stays available separately.
gemini|gemma-4-31b-it|15|sens_v4_sub_gemma4_31b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --prompt-variant v4 --workers 4
gemini|gemma-4-26b-a4b-it|15|sens_v4_sub_gemma4_26b_a4b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --prompt-variant v4 --workers 4
gemini|gemma-4-31b-it|15|sens_v5_sub_gemma4_31b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --prompt-variant v5 --workers 4
gemini|gemma-4-26b-a4b-it|15|sens_v5_sub_gemma4_26b_a4b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --prompt-variant v5 --workers 4
gemini|gemma-4-31b-it|4|k4_e0_gemma4_31b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k4_e3.json --episode 0 --workers 4
gemini|gemma-4-26b-a4b-it|4|k4_e0_gemma4_26b_a4b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k4_e3.json --episode 0 --workers 4
gemini|gemma-4-31b-it|4|k4_e1_gemma4_31b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k4_e3.json --episode 1 --workers 4
gemini|gemma-4-26b-a4b-it|4|k4_e1_gemma4_26b_a4b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k4_e3.json --episode 1 --workers 4
gemini|gemma-4-31b-it|4|k4_e2_gemma4_31b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k4_e3.json --episode 2 --workers 4
gemini|gemma-4-26b-a4b-it|4|k4_e2_gemma4_26b_a4b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --episodes experiments/episodes_3class_k4_e3.json --episode 2 --workers 4
