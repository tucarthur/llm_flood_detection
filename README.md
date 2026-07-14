# Multimodal LLM Agent for River Water-Level Classification

Research benchmark for a vision-capable LLM agent that classifies water level at a
fixed river camera (low / medium / high / flood), grounding its decisions via
retrieval-augmented generation (RAG) over a curated site-criteria corpus, and
comparing against the segmentation/optical-flow CNN baselines already published on
this exact dataset.

**Scope (Phase 1):** water-level classification from a single camera image only.
Warning-message generation and risk-prevention recommendations are out of scope. This
is an offline, retrospective benchmark on historical camera imagery; it does not
process live images and is not a deployable system.

Note: this project has pivoted twice. It started as a structured-sensor + text-tweet
benchmark (USGS gauge data + CrisisNLP tweets), then moved to a general crisis-photo
damage-severity benchmark (CrisisMMD), before settling on this track: the user's own
established river-camera dataset and label taxonomy, directly comparable to their
published prior work. The RAG knowledge-base and evaluation-harness infrastructure
carried over across pivots; the gauge-data agent, structured baselines, and CrisisMMD
loader did not.

## Layout

```
data/            # annotation CSV + loader; produces (image_path, season, place, is_night) -> label examples
knowledge_base/  # curated RAG corpus (site-specific water-level criteria) + vector index
agent/           # retrieval module, vision classifier agent, prompts, token/cost estimator
baselines/       # (not yet built -- likely CLIP/embedding + logreg, GPU-side on the SSH server)
eval/            # metrics (ordinal-aware), grounding-faithfulness checker, comparison report
experiments/     # run outputs + results aggregation
```

## Data source

**E-Noe river-camera dataset** (Mineirinho Creek, Sao Carlos, Brazil) — Ranieri, Souza,
Nishijima, Krishnamachari & Ueyama, "A deep learning workflow enhanced with optical flow
fields for flood risk estimation," Applied Intelligence, 2024
(DOI: 10.1007/s10489-024-05466-2). Public on Kaggle:
`caetanoranieri/river-images-at-sao-carlos` (CC-BY-NC-SA-4.0, ~73GB full dataset).

Labels are the paper's own marker-based, physically-grounded criteria: `flood` means
water has risen above the top of the concrete canal wall and is touching the grass
bank -- see `knowledge_base/corpus/mineirinho_creek_criteria.md`. Labeled subset used:
SHOP/SHOP2 camera only, Nov-Feb rainy season across 4 years (2018-19 through 2021-22) --
68,599 images total, **severely imbalanced** (low 98.84% / medium 0.78% / high 0.27% /
flood 0.11%). See `data/README.md` for fetch instructions and full details.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

The classifier agent talks to a vLLM-served model, not Anthropic's API directly --
see "Serving the model with vLLM" below. `.env`'s `ANTHROPIC_API_KEY` is only needed
for `agent/estimate_tokens.py`'s Claude-pricing estimate.

## Serving the model with vLLM

The classifier needs a running vLLM server (OpenAI-compatible API, vision-capable
model). This needs a real GPU -- not viable on a weak/no-GPU dev machine, so run it on
whatever GPU box you have (rented cloud instance, lab server, etc.), then point
`VLLM_BASE_URL` in `.env` at it:

```bash
# On the GPU machine:
docker compose up -d
# First launch downloads the model (Qwen2.5-VL-7B-Instruct by default, ~16GB) --
# check progress with `docker compose logs -f`.

# Confirm it's actually working end-to-end (not just that the container is up):
python scripts/check_vllm.py
```

Then, wherever `agent.run` runs (same machine or not):
```bash
VLLM_BASE_URL=http://<gpu-host>:8000/v1  # localhost if running there directly
```

Override the model via `VLLM_MODEL` in `.env` before `docker compose up` -- the
`--tool-call-parser hermes` flag in `docker-compose.yml` is specific to
Qwen-family models, so a different model family will likely need a different parser
(see vLLM's tool-calling docs for the right one).

## Quickstart (dev sample; full dataset fetch is a separate, much larger step -- see data/README.md)

```bash
python -m data.enoe_images          # build examples.jsonl from data/raw/sample_images
python -m knowledge_base.build_index

# Before running the agent for real: estimate token cost across candidate models
python -m agent.estimate_tokens --n-examples 68599 --sample-image data/raw/sample_images/<file>.jpg

# Agent (needs a running vLLM server -- see "Serving the model with vLLM" above) -- one run per ablation cell
python -m agent.run --input data/processed/examples.jsonl --out experiments/results/agent_full.jsonl
python -m agent.run --input data/processed/examples.jsonl --out experiments/results/agent_no_rag.jsonl --no-rag

python -m eval.report
```
