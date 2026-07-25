# Lane: nvidia NIM. Sequential -- the ~40 req/min ceiling is per-key, not per-model.
# qwen/qwen3.5-122b-a10b is deliberately absent: the endpoint now returns 410 Gone, so
# that model cannot be rerun and mistral-small-4 takes the fourth model slot.
#
# --workers matters more than --rpm here: rpm is only a CEILING enforced globally across
# worker threads, so with a single worker throughput collapses to 1/latency. Measured mean
# latency on nemotron is 34s (median 9.3s -- a heavy tail), giving 1.8 rows/min and ~15h
# per cell single-threaded. 12 workers bring that to roughly 20 rows/min while staying
# under the 30 rpm cap, which the shared throttle enforces regardless of worker count.
# provider|model|rpm|out_stem|extra flags
nvidia|nvidia/llama-3.1-nemotron-nano-vl-8b-v1|30|zs_3class_nemotron_nano_vl_8b|--workers 12
nvidia|mistralai/mistral-small-4-119b-2603|30|zs_3class_mistral_small_4|--workers 12
nvidia|nvidia/llama-3.1-nemotron-nano-vl-8b-v1|30|zsnaive_3class_nemotron_nano_vl_8b|--workers 12 --no-criteria
nvidia|mistralai/mistral-small-4-119b-2603|30|zsnaive_3class_mistral_small_4|--workers 12 --no-criteria
