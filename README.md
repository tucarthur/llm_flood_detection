# Multimodal LLMs for River Water-Level Classification under Flood-Event Scarcity

Research benchmark comparing **vision-language models given no labelled examples** against
**supervised specialists trained on the site's labelled history**, for water-level
classification at a fixed river camera. The question is what a newly instrumented site can
actually do: when floods are rare, how much labelled data must you collect before training a
specialist beats describing the physical threshold to a general-purpose model?

**Scope.** Water-level classification from a single camera image. Offline, retrospective
benchmark on historical imagery — it does not process live images and is not a deployable
system. Warning-message generation and risk recommendations are out of scope.

**Status.** Data collection and analysis complete; the manuscript is written. Everything below
describes the final protocol.

## The finding

Not a ranking. Prompting does not beat supervision, and the honest result is more specific
than that:

| arm | labels used | macro-F1 | flood events caught | dry false alarms |
|---|---|---|---|---|
| ResNet-50, lr 3e-5 | ~900 | **0.849** | **10/10** | 0.0113 |
| ResNet-50, lr 1e-4 *(the default you'd pick)* | ~900 | 0.846 | 8/10 | 0.0038 |
| DINOv2 + logreg probe | ~900 | 0.808 | 8/10 | 0.0000 |
| gemma-4-31b, criteria in prompt | **0** | 0.595 | **10/10** | 0.0050 |

A supervised CNN reaches the same event-detection operating point as the zero-shot model —
and beats it on aggregate correctness by 0.255. But only at one learning rate out of three
tried, and nothing observable before the first flood tells you to pick it. Re-thresholding a
probe to the same recall needs labelled floods for the same reason. **The asymmetry is in what
each approach requires, not in what it achieves.**

Supporting results, all surviving a paired event-level bootstrap:

- Writing the physical criteria into the prompt raises flood recall by **0.160–0.227** on
  three of four models — larger than the spread from merely rewording the prompt.
- **K-shot prompting is worse than zero-shot** at every shot level tested. On one model, K=4
  degenerated into predicting `flood` for 87% of all frames (a constant always-flood
  classifier would score 1.000 recall).
- Against a linear probe trained on the *same* exemplars, prompting is statistically
  indistinguishable at every K.
- Class balance beats volume: 120 deliberately balanced frames give 0.821 flood recall against
  0.680 for the full ~900-frame history at its natural prior.
- **No comparison in the study resolves a difference in event detection.** With ten events,
  that is the ceiling the data imposes, and it is reported rather than worked around.

## Data source

**E-Noé river-camera dataset** (Mineirinho Creek, São Carlos, Brazil) — Ranieri, Souza,
Nishijima, Krishnamachari & Ueyama, *A deep learning workflow enhanced with optical flow
fields for flood risk estimation*, Applied Intelligence, 2024
(DOI: 10.1007/s10489-024-05466-2). Public on Kaggle:
[`caetanoranieri/river-images-at-sao-carlos`](https://www.kaggle.com/datasets/caetanoranieri/river-images-at-sao-carlos)
(CC-BY-NC-SA-4.0, ~73 GB full dataset).

Labelled subset: SHOP/SHOP2 camera only, Nov–Feb rainy season across four seasons
(2018-19 through 2021-22) — **68,599 images**, severely imbalanced (low 98.84% / medium
0.78% / high 0.27% / flood 0.11%). See `data/README.md` for fetch instructions.

Two properties of this data drive most of the design:

**The flood class is ten events, not 75 frames.** The camera fires every few minutes, so the
75 flood frames come from 10 distinct days; one (2020-01-12) supplies 22 of them, and 64 of
74 consecutive flood frames fall within 15 minutes of the previous one. Frames are not
independent observations. Events — `(date, camera)` — are therefore the unit for bootstrap
resampling, for few-shot support-set sampling, and for detection reporting. Resampling frames
instead would give intervals roughly 3× too narrow on the flood class.

**Severity correlates with poor visibility.** 46.8% of frames are night shots, but 64% of
flood frames are, often with streetlight glare heavy enough to obscure the waterline.
Illumination is controlled for in how the evaluation set is built, not by stratifying the
reported metrics.

## Task framing

The source annotations are a 4-level ordinal scale (`low < medium < high < flood`) where
`flood` is a specific physical threshold: water above the top of the concrete canal wall,
touching the grass bank. **Everything here is prompted and scored as 3-class** —
`no_risk < risky < flood`, merging `medium` and `high` — because that boundary is where
essentially all error concentrates, a probe trained on those very labels cannot reproduce it
(it scatters the 535 true `medium` frames 223/152/159/1), and the distinction maps to no
operational decision.

`data/taxonomy.py` still implements `4class` and `binary`, and cells exist for them, but they
predate the JSON-schema rework and are **not** part of the reported study.

## Layout

```
data/            annotation CSV + loader; taxonomy definitions; episodic support-set sampler
knowledge_base/  exemplar image bank (DINOv2 embeddings + metadata); retired text corpus
agent/           prompt construction, classifier, provider resolution, CLI runner
baselines/       supervised comparators: frozen-feature probes, budget curves, ResNet-50
eval/            metrics (event-level, cluster bootstrap, stratified), comparison report
scripts/         builders, provider pilots, queue/repair/watchdog drivers, analysis, figures
experiments/     queue specs, episode manifests, run outputs (results/ is gitignored)
paper/           LaTeX sources (gitignored)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # then fill in the provider keys you intend to use
```

Inference runs against **hosted OpenAI-compatible APIs**: Gemini, NVIDIA NIM, Groq and
OpenRouter, selected with `--provider`. A local vLLM path exists (`--provider vllm`,
`docker-compose.yml`) but no result in this study came from it — a 4 GB card cannot serve a 7B
VLM. `ANTHROPIC_API_KEY` is only used by `agent/estimate_tokens.py`.

GPU work (the ResNet-50 comparator) uses a **separate** environment so that installing CUDA
torch cannot disturb long-running API evaluation lanes:

```bash
python3 -m venv .venv-gpu && .venv-gpu/bin/pip install torch torchvision  # CUDA build
```

## Workflow

### 1. Build the evaluation data

```bash
python -m scripts.build_test_set          # 1,592-frame evaluation set + Kaggle image fetch
python -m scripts.build_kshot_subsample   # 375-frame subsample for the K-shot sweep
python -m data.episodes --show-limits     # print the event table and the K ceiling
python -m data.episodes --k 2 --n-episodes 3   # write a support-set manifest
```

The evaluation set keeps **every** `medium`/`high`/`flood` frame plus an equal number of
sampled `low` frames. Consequence to remember when reporting: per-class recall and conditional
rates stay unbiased, but accuracy, precision and macro-F1 depend on the class prior and are
**not** comparable to full-distribution numbers.

### 2. Verify provider support, then run cells

```bash
# Which structured-output mode works on this model? Do this before committing hours.
python -m scripts.check_json_mode --provider gemini --model gemma-4-31b-it

# One cell = one point in the matrix
python -m agent.run --input data/processed/test_examples.jsonl \
  --out experiments/results/zs_3class_gemma4_31b.jsonl \
  --taxonomy 3class --json-mode schema --provider gemini --model gemma-4-31b-it --rpm 15 --workers 4

# K-shot cell (episode manifest supplies the support set)
python -m agent.run --input data/processed/kshot_subsample.jsonl \
  --out experiments/results/k2_e0_gemma4_31b.jsonl \
  --taxonomy 3class --episodes experiments/episodes_3class_k2_e3.json --episode 0 \
  --provider gemini --model gemma-4-31b-it --rpm 6 --workers 4
```

Each run writes a `.meta.json` sidecar with the resolved model, decoding config, prompt
variant, support-set contents, timestamps, git commit and parse-failure rate.

### 3. Run a lane unattended, then repair

```bash
bash scripts/run_queue.sh geminiK experiments/queue_gemini_kshot.spec
bash scripts/repair_queue.sh "zs_3class_minimax_m3|3|2|all"    # retry placeholder rows
bash scripts/watch_run.sh <run-log> <done-marker> <driver> "<results-glob>"   # detached watchdog
```

One lane per provider, cells sequential within a lane (the rate limit is per key, not per
model), lanes parallel across providers. Idempotent: complete cells are skipped, partial ones
resumed.

`repair_queue.sh` retries each cell up to three passes and stops early on a clean exit with no
progress, which correctly identifies failures retrying cannot fix. It counts placeholders **on
disk** rather than parsing its own log — see the rate note below for why that matters.

### 4. Supervised comparators

```bash
python -m baselines.kshot_probe                    # matched-K probes on the same episodes
python -m baselines.supervised_budget_curve        # balanced + natural annotation-budget axes
.venv-gpu/bin/python -m baselines.resnet50         # fine-tuned CNN (needs the GPU venv)
.venv-gpu/bin/python -m baselines.resnet50_budget --job natural --budgets 200 --draws 0 1 2 \
    --epochs 6 --precision fp32                    # one budget-curve point
```

Probes train on **exactly** the support sets the prompted arms received, so the arms differ
only in the learner. Full per-class probabilities are stored, which is what allows a
recall-matched operating point.

### 5. Score, and regenerate every number in the paper

```bash
python -m eval.report --resamples 2000                    # per-cell metrics + intervals
python -m scripts.bootstrap_supervised --verify <cell>    # check against eval.metrics
python -m scripts.bootstrap_supervised                    # CIs for every supervised table
python -m scripts.bootstrap_supervised --paired           # the paired comparisons
python -m scripts.sensitivity_spread --scope sub          # five-variant paraphrase spread
python -m scripts.plot_supervised_curve                   # figures
python -m scripts.plot_confusion_progression
python -m scripts.make_class_examples
```

`bootstrap_supervised.py` resamples precomputed per-event confusion matrices rather than
re-scoring DataFrames, which is what makes 2000 resamples over 240+ draw-files take seconds
instead of hours. `eval.metrics.bootstrap_ci` remains the definition of the estimator;
`--verify` asserts agreement to 1e-12 and is the reason the speedup is safe to trust.

## Design decisions worth knowing before you change anything

**Response field order is reasoning order.** The JSON schema asks for `observations` →
`rationale` → `classification`, because under constrained decoding the schema's key order *is*
the generation order. Putting the label first turns the rationale into post-hoc narration. Do
not reorder these fields.

**In K-shot prompts, exemplars come first and the query goes last**, named explicitly as the
frame to classify. With the query first, models resolve "this image" to the most recent
attachment and answer about the final exemplar — which ascending-severity ordering makes the
flood example. One model classified 134/150 dry frames as flood, describing water on the grass
embankment for a nearly empty canal. Query position is load-bearing.

**JSON mode is not optional.** Unconstrained runs lost up to 28.6% of a cell to models emitting
a reasoning preamble that never reached the JSON body; those rows fell back to the majority
class and silently deflated flood recall. Under a strict schema the parse-failure rate is
0.00–0.19%. Hold the mode fixed across the whole matrix rather than enabling it per-model.

**K is capped at 6 by the data.** Flood has 1/2/3/4 distinct events across the four seasons, so
leave-one-season-out leaves at most six, and support sets take one frame per event. The sampler
refuses larger K rather than reusing an event.

**Prompt wording is a measured uncertainty, not a tuned parameter.** Five variants paraphrase
incidental framing while holding the operative criteria byte-identical. Measured spread: up to
0.187 flood recall and three of ten events — as much as varying the CNN's learning rate over an
order of magnitude. An effect is interpretable only if it exceeds the paraphrase spread for that
model and metric.

**Concurrency, not `--rpm`, is what draws rate limits — and a short pilot will lie to you.**
A pilot measured a zero error rate for minimax at `rpm 10 / 4 workers`. Over full-length cells
that setting produced 22–40% placeholder rows. Mean latency was ~63s, so four workers
self-limit to ~3.8 req/min and the rpm cap never bound: the concurrency was the whole problem.
`rpm 3 / 2 workers` gave 0.1–0.9% on the same cells. Measure on a full cell, not a pilot.

**Progress means rows on disk.** Repair rewrites rows in place and never appends, so row counts
are flat during a repair phase by design. A watchdog that counts rows will report a false stall;
`watch_run.sh` uses file mtime instead. Equally, never take a script's own log as evidence of
success — an earlier repair driver read a stale success line from a cell that had crashed
without printing one, and three cells were lost while the log looked healthy.

**Not every model can run every arm.** `llama-3.1-nemotron-nano-vl-8b-v1` cannot attend to the
query frame when reference images are attached, and `minimax-m3` on NVIDIA NIM rejects K≥2 with
a payload-size error (~6.3 MB at K=2 against ~3.6 MB at K=1). The same model accepts K=4
through OpenRouter, so that limit is the deployment's, not the model's. All four models run the
zero-shot conditions at full resolution without difficulty — zero-shot prompting is more
portable across endpoints than K-shot.

**Frames are sent unmodified.** The source JPEG is base64-encoded byte-for-byte: 1280×720,
averaging 647 kB. No resizing or re-encoding, which keeps the prompted arm free of a
preprocessing choice we would otherwise have to justify — and is why payload grows to ~11.6 MB
at K=4.

## Retired

- **Text RAG.** A fixed query over a 15-chunk corpus returned an identical passage block on
  every example — static prompt augmentation, not retrieval — and it restated criteria every
  arm already carried inline, raising false alarms without buying recall. `agent/retrieval.py`
  is kept only so the superseded cells can be rebuilt.
- **vLLM as the primary serving path.** Retained as an option; unused for all results here.
- **`mistral-small-4`** — 0.000 flood recall with CI [0.000, 0.000] in both zero-shot arms:
  structurally unable to emit the positive class, not merely poor at it.
- **`qwen3.5-122b-a10b` and `llama-4-scout-17b-16e-instruct`** — endpoints returned 410 and 404
  after complete runs had been recorded. Unreproducible, excluded. A model identifier is not a
  version, which is why every run records what it actually executed.
- **4-class and binary conditions.** Runs exist but predate the JSON-schema rework, so mixing
  them with current cells would mix decoding protocols. Excluded from the reported study.

## Notes

`experiments/results/` and `paper/` are gitignored, so results and the manuscript live only on
disk — back them up separately. Episode manifests *are* committed, since they define which
images each cell saw. `experiments/results/INVALID_*/` holds cells kept as evidence but unfit
to score; each carries a README explaining why.
