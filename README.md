# Multimodal LLMs for River Water-Level Classification under Flood-Event Scarcity

Research benchmark comparing **vision-language models given no labelled examples** against
**supervised specialists trained on the site's labelled history**, for water-level
classification at a fixed river camera. The central question is a practical one: at a site
where flood events are rare, how much labelled data must you collect before training a
specialist beats simply describing the physical threshold to a general-purpose model?

**Scope.** Water-level classification from a single camera image. Offline, retrospective
benchmark on historical imagery — it does not process live images and is not a deployable
system. Warning-message generation and risk recommendations are out of scope.

## The headline finding so far

Supervision buys aggregate correctness and then stops buying flood-event detection:

| arm | labels used | macro-F1 | flood events caught |
|---|---|---|---|
| DINOv2 + logreg probe | ~900 | **0.808** | 8/10 |
| DINOv2 + logreg probe | 120 (class-balanced) | 0.798 | 8.3/10 |
| gemma-4-31b, criteria in prompt | **0** | 0.595 | **10/10** |

The macro-F1 crossover sits at roughly 72–100 labelled images. Event detection never
crosses: threshold-tuned probes reach 9–9.4/10, and 120 well-chosen labels beat 900
naturally-distributed ones on flood recall (0.821 vs 0.680). Results are preliminary and
several controls are still running — see `experiments/fewshot_vs_llm_study.md`.

## Data source

**E-Noé river-camera dataset** (Mineirinho Creek, São Carlos, Brazil) — Ranieri, Souza,
Nishijima, Krishnamachari & Ueyama, *A deep learning workflow enhanced with optical flow
fields for flood risk estimation*, Applied Intelligence, 2024
(DOI: 10.1007/s10489-024-05466-2). Public on Kaggle:
`caetanoranieri/river-images-at-sao-carlos` (CC-BY-NC-SA-4.0, ~73 GB full dataset).

Labelled subset: SHOP/SHOP2 camera only, Nov–Feb rainy season across four seasons
(2018-19 through 2021-22) — **68,599 images**, severely imbalanced (low 98.84% / medium
0.78% / high 0.27% / flood 0.11%). See `data/README.md` for fetch instructions.

Two properties of this data drive most of the design:

**The flood class is ten events, not 75 frames.** The camera fires every few minutes, so the
75 flood frames come from 10 distinct days; one (2020-01-12) supplies 22 of them, and 64 of
74 consecutive flood frames fall within 15 minutes of the previous one. Frames are not
independent observations. Events — `(date, camera)` — are therefore the unit for bootstrap
resampling, for few-shot support-set sampling, and for detection reporting.

**Severity correlates with poor visibility.** 46.8% of frames are night shots, but 64% of
flood frames are. Night frames often carry streetlight glare heavy enough to obscure the
waterline, so metrics are reported stratified by day/night.

## Task framing

The source annotations are a 4-level ordinal scale (`low < medium < high < flood`) where
`flood` is a specific physical threshold: water above the top of the concrete canal wall,
touching the grass bank. The **primary task here is 3-class** —
`no_risk < risky < flood`, merging `medium` and `high` — because that boundary is where
essentially all error concentrates, a probe trained on those very labels cannot reproduce it
(it scatters true `medium` 223/152/159), and the distinction maps to no operational decision.
`4class` is retained for comparability with the source study and `binary` as a diagnostic.

All three are **separately prompted tasks**, not post-hoc relabellings: asking for fewer
categories moves a model's operating point by up to 0.11 macro-F1, in a model-dependent
direction. See `data/taxonomy.py`.

## Layout

```
data/            annotation CSV + loader; taxonomy definitions; episodic support-set sampler
knowledge_base/  exemplar image bank (DINOv2 embeddings + metadata); retired text corpus
agent/           prompt construction, classifier, CLI runner
baselines/       supervised comparators: frozen-feature probes, budget curves, ResNet-50
eval/            metrics (event-level, cluster bootstrap, stratified), comparison report
scripts/         provider pilots, queue runner, eval-set + subsample builders, repair tool
experiments/     queue specs, episode manifests, run outputs (results/ is gitignored)
paper/           LaTeX sources (gitignored)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # then fill in GEMINI_API_KEY / NVIDIA_API_KEY
```

Inference runs against **hosted OpenAI-compatible APIs** (Gemini, NVIDIA NIM). A local
vLLM path still exists (`--provider vllm`, `docker-compose.yml`) and is useful if you have a
capable GPU, but no result in this study came from it — a 4 GB card cannot serve a 7B VLM.
`ANTHROPIC_API_KEY` is only used by `agent/estimate_tokens.py`.

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
sampled `low` frames. Consequence to remember when reporting: per-class recall and
conditional rates stay unbiased, but accuracy, precision and macro-F1 depend on the class
prior and are **not** comparable to full-distribution numbers.

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

**`--workers` matters more than `--rpm`.** `--rpm` is only a ceiling; with a single worker
throughput collapses to `1/latency`. Measured: 5.2 rows/min at one worker versus 14.7 at
four. And with exemplars attached, throughput is bound by *input tokens per minute*, not
request rate — 4,163 prompt tokens at K=4 means ~4 requests/min against a 16k TPM ceiling.

### 3. Run a whole lane unattended

```bash
bash scripts/run_queue.sh geminiK experiments/queue_gemini_kshot.spec
```

One lane per provider, cells sequential within a lane (the rate limit is per key, not per
model), lanes parallel across providers. Idempotent: complete cells are skipped, partial ones
resumed, so relaunching after a crash costs only the backoff.

### 4. Repair, then score

```bash
python -m scripts.rerun_failed --results experiments/results/<cell>.jsonl   # patch api_error rows
python -m eval.report --resamples 2000
```

`rerun_failed.py` rebuilds the agent from the configuration recorded in each row, so a repair
cannot silently apply different settings than the original run.

### 5. Supervised comparators

```bash
python -m baselines.kshot_probe                    # matched-K probes on the same episodes
python -m baselines.supervised_budget_curve        # balanced + natural annotation-budget axes
.venv-gpu/bin/python -m baselines.resnet50         # fine-tuned CNN (needs the GPU venv)
```

Probes train on **exactly** the support sets the prompted arms received, so the arms differ
only in the learner. Full per-class probabilities are stored, which is what allows a
recall-matched operating point — comparing a cost-asymmetric prompt against an untuned
`argmax` would attribute a decision-rule difference to the choice of learner.

## Design decisions worth knowing before you change anything

**Response field order is reasoning order.** The JSON schema asks for `observations` →
`rationale` → `classification`, because under constrained decoding the schema's key order
*is* the generation order. Putting the label first turns the rationale into post-hoc
narration. Do not reorder these fields.

**In K-shot prompts, exemplars come first and the query goes last**, named explicitly as the
frame to classify. With the query first, models resolve "this image" to the most recent
attachment and answer about the final exemplar — which ascending-severity ordering makes the
flood example. One model classified 134/150 dry frames as flood, describing water on the
grass embankment for a nearly empty canal. Query position is load-bearing.

**JSON mode is not optional.** Unconstrained runs lost up to 28.6% of a cell to models
emitting a reasoning preamble that never reached the JSON body; those rows fell back to the
majority class and silently deflated flood recall. Under a strict schema the parse-failure
rate is 0.00–0.19%. Hold the mode fixed across the whole matrix rather than enabling it
per-model.

**K is capped at 6 by the data.** Flood has 1/2/3/4 distinct events across the four seasons,
so leave-one-season-out leaves at most six, and support sets take one frame per event. The
sampler refuses larger K rather than reusing an event.

**Prompt wording is a measured uncertainty, not a tuned parameter.** Five variants paraphrase
incidental framing while holding the operative criteria byte-identical. An effect is only
interpretable if it exceeds the paraphrase spread for that model and metric — a bootstrap
interval cannot tell you this, because it answers a different question.

**Model endpoints disappear.** `qwen/qwen3.5-122b-a10b` now returns 410 and
`llama-4-scout-17b-16e-instruct` returns 404, both after complete runs had been recorded.
Those results are unreproducible and excluded. A model identifier is not a version, which is
why every run records what it actually executed.

## Retired

- **Text RAG.** A fixed query over a 15-chunk corpus returned an identical passage block on
  every example — static prompt augmentation, not retrieval — and it restated criteria every
  arm already carried inline, raising false alarms without buying recall. `agent/retrieval.py`
  is kept only so the superseded cells can be rebuilt.
- **vLLM as the primary serving path.** Retained as an option; unused for all results here.
- **`mistral-small-4`** — 0.000 flood recall with CI [0.000, 0.000] in both zero-shot arms:
  structurally unable to emit the positive class, not merely poor at it.

## Notes

`experiments/results/` (~164 MB) and `paper/` are gitignored, so results and the manuscript
live only on disk — back them up separately. Episode manifests *are* committed, since they
define which images each cell saw.
