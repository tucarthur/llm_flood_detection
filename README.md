# LLM Agent Framework for Flood Classification (RAG-grounded)

Research benchmark for a classification-focused LLM agent that reasons over structured
hydrological data and unstructured crisis-text reports, grounding its decisions via
retrieval-augmented generation (RAG) over a curated flood-knowledge corpus.

**Scope (Phase 1):** classification only. Warning-message generation and risk-prevention
recommendations are explicitly out of scope for this phase — see `docs` / plan notes for
rationale. This is an offline, retrospective benchmark; it does not emit live warnings and
is not a deployable system.

## Layout

```
data/            # download + preprocessing scripts; produces (location, time_window) -> label examples
knowledge_base/  # curated RAG corpus (flood definitions, protocols, historical summaries) + vector index
agent/           # tool functions, retrieval module, classifier agent, prompts
baselines/       # threshold model, GBT/logreg model, LLM ablation configs
eval/            # metrics, significance tests, grounding-faithfulness checker
experiments/     # run configs + results aggregation
```

## Data sources

- **Joint (structured + text) track**: Hurricane Sandy (2012) / Harvey (2017) / Irma (2017)
  crisis-tweet datasets (CrisisNLP) joined with live-pulled USGS NWIS gauge/discharge/
  precipitation data for affected sites during each event window.
- **Text-only generalization track**: CrisisLexT6 2013 Alberta floods / 2013 Queensland
  floods subsets (out-of-domain, non-US — not joinable with US gauge data).
- **Structured-only auxiliary**: USGS annual peak flood classification dataset.

See `data/README.md` for fetch instructions per source.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # add ANTHROPIC_API_KEY
```

## Quickstart (small sample, end-to-end)

```bash
python -m data.build_sample        # fetches + aligns a small Sandy-region sample
python -m knowledge_base.build_index
python -m agent.run --input data/processed/sample.jsonl --out experiments/results/sample_agent.jsonl
python -m baselines.run_all --input data/processed/sample.jsonl --out experiments/results/sample_baselines.jsonl
python -m eval.report --agent experiments/results/sample_agent.jsonl --baselines experiments/results/sample_baselines.jsonl
```
