# Lane: nvidia. Nemotron v4/v5 on the 375-row subsample, to take its paraphrase spread
# from a 3-point range to a 5-point standard error. This is the model where paraphrase
# already exceeded the substantive criteria effect, so it is the one that most needs a
# real spread estimate rather than a range.
# provider|model|rpm|out_stem|input_jsonl|extra flags
nvidia|nvidia/llama-3.1-nemotron-nano-vl-8b-v1|30|sens_v4_sub_nemotron_nano_vl_8b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --prompt-variant v4 --workers 12
nvidia|nvidia/llama-3.1-nemotron-nano-vl-8b-v1|30|sens_v5_sub_nemotron_nano_vl_8b|data/processed/kshot_subsample.jsonl|--taxonomy 3class --prompt-variant v5 --workers 12
