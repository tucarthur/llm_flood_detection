# Lane: gemini. Sequential -- the ~15 req/min sustained ceiling is per-key, not per-model.
# Block 1 (criteria on) first, then the naive zero-shot arm, per priority.
#
# --workers matters more than --rpm here: rpm is only a CEILING enforced globally across
# worker threads, so with a single worker throughput collapses to 1/latency (measured 5.2
# rows/min at ~11.5s mean latency, i.e. ~5h/cell). 4 workers saturate the 15 rpm cap and
# cannot breach it, because the throttle is shared across threads.
# provider|model|rpm|out_stem|extra flags
gemini|gemma-4-31b-it|15|zs_3class_gemma4_31b||--taxonomy 3class --workers 4
gemini|gemma-4-26b-a4b-it|15|zs_3class_gemma4_26b_a4b||--taxonomy 3class --workers 4
gemini|gemma-4-31b-it|15|zsnaive_3class_gemma4_31b||--taxonomy 3class --workers 4 --no-criteria
gemini|gemma-4-26b-a4b-it|15|zsnaive_3class_gemma4_26b_a4b||--taxonomy 3class --workers 4 --no-criteria
