# Research note: "normal" few-shot methods vs. multimodal LLMs under flood-event data scarcity

## 1. The idea

Frame a study around one question: when a safety-critical class (`flood`) is
extremely rare, does a classical low-data supervised approach (a probe trained on
frozen embeddings) or a zero/few-shot multimodal LLM make the better classifier —
and on what axis does "better" mean?

The justification is concrete, not rhetorical. In the curated image bank this
project already builds (`knowledge_base/image_bank_backup`, the same pool the
image-RAG arm retrieves from and the DINOv2 probes train on):

| season | flood | high | low | medium |
|---|---|---|---|---|
| 2018-2019 | 2 | 13 | 54 | 39 |
| 2019-2020 | 24 | 18 | 99 | 57 |
| 2020-2021 | 29 | 79 | 273 | 165 |
| 2021-2022 | 20 | 76 | 370 | 274 |

75 flood images total, out of 68,599 raw frames (flood = 0.11% of the full
dataset). Under the leave-one-season-out protocol already in use, any
supervised model trains on **46-73 flood examples** depending on which season is
held out. That is a genuine few-shot regime for the class that matters most —
not an argument of convenience.

## 2. This comparison is not hypothetical — it's already half-built

The repo already contains both arms of exactly this comparison, run on the same
1,592-example leave-one-season-out test set:

- **Classical / few-shot-scale supervised side**: `baselines/embedding_probe.py`
  — logistic-regression and 2-layer MLP probes on frozen DINOv2 ViT-B/14
  embeddings, trained per-fold on the bank above (i.e., on the ~46-73 flood
  images), evaluated on the held-out season.
- **Zero/few-shot multimodal LLM side**: `agent/classifier.py` — a single vision
  API call, ablated across four context conditions (no context / text-RAG over a
  hand-written site-criteria corpus / image-RAG that retrieves DINOv2-nearest
  labeled exemplars from the same bank / both combined), across four model
  providers (Gemini/Gemma, NVIDIA Nemotron, Qwen3.5, Groq Llama-4-Scout).

So "normal few-shot" here is best read as **supervised learning on a
small, curated, class-balanced pool** (a linear/shallow probe on top of a
frozen self-supervised backbone), and "multimodal LLM" as **a large pretrained
VLM classifying zero-shot, optionally with retrieval-augmented in-context
exemplars** — not classical N-way-K-shot meta-learning (no episodic
meta-training set exists here to support that framing honestly; see §4).

## 3. What the existing runs already show

Computed directly from `experiments/results/*.jsonl` via `eval/metrics.py`
(full n=1,592 runs only):

| arm | acc | macro-F1 | missed-flood rate | false-alarm rate | flood recall |
|---|---|---|---|---|---|
| DINOv2 + logreg probe | **0.710** | **0.643** | 0.387 | **0.0000** | 0.613 |
| DINOv2 + MLP probe | 0.683 | 0.629 | 0.333 | 0.0000 | 0.667 |
| Gemma-4-31B, zero-shot | 0.591 | 0.410 | 0.067 | 0.0013 | 0.933 |
| Gemma-4-31B, image-RAG | 0.604 | 0.511 | 0.187 | 0.0000 | 0.813 |
| Qwen3.5-122B, zero-shot | 0.487 | 0.401 | 0.040 | 0.0025 | **0.960** |
| Nemotron-Nano-VL-8B, zero-shot | 0.634 | 0.425 | 0.600 | 0.0025 | 0.400 |
| Gemma-4-31B, text-RAG | 0.537 | 0.352 | 0.040 | 0.0364 | 0.960 |
| Gemma-4-31B, text+image-RAG | 0.485 | 0.344 | 0.147 | 0.0578 | 0.853 |

(`missed-flood rate` = P(predict ≠ flood \| true = flood); this is the safety
metric — see `eval/metrics.py` docstring.)

The pattern is already a real, nuanced finding, not a coin flip:

- **The supervised probe wins on aggregate accuracy/macro-F1** — it's the best
  overall classifier despite training on ~50-70 flood images.
- **The zero-shot LLMs win on flood recall**, sometimes dramatically (Qwen3.5
  misses only 4% of true floods vs. the probe's 39%), at the cost of overall
  accuracy and (for text-RAG) a much higher false-alarm rate.
- **Image-RAG (retrieval-augmented in-context exemplars from the exact same
  low-data bank the probe trains on) is the one LLM arm that improves over
  zero-shot on macro-F1 for Gemma** — but the same manipulation is
  catastrophic for Nemotron-Nano-VL-8B (flood recall 0.40 → 0.24, not shown
  above but in the raw files). The in-context few-shot boost is **model-
  dependent, not universal** — a real risk for the "LLMs are good at few-shot"
  premise and something a paper must show rather than assume.
- Text-RAG (a written criteria document, not exemplars) consistently hurts —
  it raises false alarms without buying recall, suggesting the LLMs are not
  using free-text grounding well here.

This is the actual paper-worthy result: **not** "LLMs beat classical methods,"
but **"the two approaches fail differently under scarcity — a specialist
optimizes overall correctness, a generalist optimizes not-missing-the-rare-and-
dangerous-class, and which one you want depends on the cost of a missed flood
vs. a false alarm."**

## 4. Pros

- **The scarcity justification is quantifiable and true**, not asserted — 46-73
  training examples per fold for the class of interest is a legitimate few-shot
  regime, and it's the same regime for both arms (the probe and image-RAG draw
  from the identical bank), so the comparison is apples-to-apples on data
  budget.
- **Both arms already run on the same test set, same folds, same labels** —
  no new data pipeline needed, which is usually the most fragile part of this
  kind of comparative study.
- **The result is already interesting and not the "obvious" one** — LLMs don't
  simply win, they trade accuracy for recall on the safety-critical class. That
  asymmetry is a more publishable story than a leaderboard win.
- **Directly extends the user's own published prior work** (segmentation/
  optical-flow CNN baseline, Ranieri et al. 2024, Applied Intelligence) on the
  identical dataset and label taxonomy — built-in novelty framing as "how do
  low-data and foundation-model approaches compare against our own published
  specialist baseline."
- **A safety framing (missed-flood rate vs. false-alarm rate) is more
  compelling to reviewers in a disaster-risk venue than accuracy alone**, and
  the metrics module already centers on this.

## 5. Cons / risks

- **"Few-shot" is doing double duty and needs to be named precisely.** The
  probes are *low-data supervised learning*, not few-shot meta-learning in the
  technical sense (no episodic training, no support/query split, no N-way-K-shot
  protocol). A reviewer who knows the few-shot-learning literature will push
  back if the paper claims "few-shot learning" without qualification. Calling it
  "low-data supervised baselines" is more defensible; if true few-shot framing
  is wanted, a prototypical-network / matching-network baseline with an
  explicit K-shot support set should be added (see §6).
- **Model-dependence of the LLM results is a real threat to the headline
  claim.** Image-RAG helps Gemma-4-31B and hurts Nemotron-Nano-VL-8B. A single-
  model story ("multimodal LLMs do X") won't survive review; the paper has to
  either explain *why* (model size? training data? instruction-tuning for
  visual grounding?) or scope the claim to "results are provider/model-
  dependent," which is a weaker but honest claim.
- **No error bars / significance testing yet.** With only 75 flood examples in
  the whole bank and ~13-40 in the test set per season, single-run point
  estimates for `flood recall` and `missed_flood_rate` will have wide
  confidence intervals. Bootstrap CIs are needed before any "X beats Y"
  statement is trustworthy.
- **Partial/incomplete runs exist** (`baseline_scout_groq.jsonl`: 187/1592,
  `baseline_mistral_small_4.jsonl`: 905/1592, `textrag_qwen3_5_122b.jsonl`:
  145/1592, the `imagerag_fewshot_*_n200` files are 200-example subsamples) —
  these can't go into a comparison table as-is; either finish them or
  explicitly scope the paper to the models with complete n=1,592 runs.
- **`eval/report.py` currently crashes** (`KeyError: 'accuracy'`) on the binary-
  mode result files, because `classification_report` is called with the
  4-class `LABELS` list against binary `flood`/`not_flood` predictions. Fine
  for ad-hoc analysis (worked around above) but blocks an automated
  reproducible report — worth a small fix regardless of which study direction
  is chosen.
- **Cost/latency isn't in the comparison yet.** A probe is ~free at inference
  (a single forward pass through a frozen backbone + linear layer); a
  multimodal LLM call has real $/latency cost, especially with RAG (image-RAG
  sends 3+ extra images per call). A paper arguing for/against LLM adoption in
  an operational flood-monitoring pipeline needs this axis — "X% better flood
  recall at Y× the per-classification cost" is the practically relevant
  framing for a deployability audience.
- **Faithfulness is unverified.** `eval/faithfulness.py` exists (word-overlap
  check on `cited_evidence` vs. retrieved context) but there's no evidence it's
  been run over these result files. If the RAG arms' rationales don't actually
  ground in the retrieved text/exemplars, "RAG helped" claims are weaker than
  they look — worth checking before writing the RAG parts of the paper.
- **CC-BY-NC-SA-4.0 license** on the dataset restricts to non-commercial use —
  fine for an academic paper, but worth stating explicitly in any writeup.

## 6. How to proceed

Roughly in priority order:

1. **Fix `eval/report.py`'s crash on binary-mode files** (or just skip
   binary results in that script) so there's one working, reproducible
   report generator instead of ad-hoc analysis.
2. **Finish or drop the partial runs.** Either complete `baseline_scout_groq`,
   `baseline_mistral_small_4`, and `textrag_qwen3_5_122b` to n=1,592, or
   explicitly exclude those provider/arm combinations from the study and say
   so.
3. **Add bootstrap confidence intervals** on `flood_recall`,
   `missed_flood_rate`, and `macro_f1` (resample the test set with
   replacement, re-score, report 95% CI) — given n≈13-40 flood examples per
   season this is not optional if the paper makes any comparative claim.
4. **Decide on the few-shot framing precisely** — either (a) keep the current
   framing ("low-data supervised probe" vs. "zero/few-shot LLM,
   optionally in-context-augmented") and drop the word "few-shot" for the
   probe side, or (b) add a true K-shot baseline (e.g. a prototypical network
   or a k-NN classifier over DINOv2 embeddings with an explicit, fixed
   K∈{5,10,20} support set per class) so "few-shot" is technically accurate on
   both sides and directly comparable at matched K.
5. **Investigate the Gemma-vs-Nemotron image-RAG divergence** — this is
   currently the paper's biggest liability if left unexplained. At minimum,
   report per-model results (don't average across models) and inspect a sample
   of Nemotron's image-RAG rationales to see whether it's ignoring, misreading,
   or being confused by the injected exemplars.
6. **Run `eval/faithfulness.py` over the RAG result files** and report the
   citation-overlap distribution — needed to support any claim that RAG
   context is actually being used rather than decorating an otherwise
   zero-shot guess.
7. **Add a cost/latency column** (token usage is already recorded per
   example in the result files — `usage.total_tokens` — so this is
   aggregation, not new instrumentation) and frame results as recall-per-
   dollar or recall-per-second where relevant.
8. **Optional but strengthens the paper**: a qualitative error analysis on the
   disagreement set — cases where the probe and the best LLM arm disagree,
   especially missed-flood cases from each side — to characterize *why* each
   approach fails (e.g., does the probe miss floods that look visually similar
   to `high`? does the LLM miss floods at night, or in unusual lighting?).

None of this requires new data collection or a new pipeline — it's tightening
what's already run.

## 7. Publishability

**Realistic, not aspirational assessment:**

- This is **not** a top-tier general-ML venue (NeurIPS/ICML/CVPR) paper as
  scoped — the methodological contribution (probes vs. prompted VLMs under
  imbalance) is not novel at the level ML methods venues want; that comparison
  has been made in other domains already. Framing it as a *methods* paper there
  would draw exactly this objection.
- It **is** a credible fit for:
  - **A natural-hazards / disaster-informatics domain venue** (e.g. *Natural
    Hazards*, *Journal of Hydrology*, or an ISCRAM/disaster-response-informatics
    workshop) as a direct, data-grounded follow-up to the user's own Applied
    Intelligence paper — "how do foundation-model and low-data approaches
    compare against our published specialist baseline, under the same
    scarcity constraint that motivated the original method." This is the
    strongest venue fit: it leverages the existing publication instead of
    competing with generic ML papers.
  - **An applied-ML / AI-for-good workshop** (e.g. "Tackling Climate Change
    with ML" at NeurIPS, or a remote-sensing/Earth-observation workshop) if
    framed around the practical question — "is a zero-shot VLM a viable
    stopgap for flood monitoring at sites with too little labeled data to
    train a specialist model" — with the cost/latency and safety-metric
    framing from §6 front and center.
  - **A short applied/benchmark paper** rather than a full methods paper —
    the honest contribution is a rigorous, safety-metric-centered empirical
    comparison on a real, physically-grounded, published dataset, not a new
    algorithm.
- **What would sink it in review**: (a) claiming "few-shot" imprecisely (see
  §5), (b) generalizing from one model's image-RAG improvement to "LLMs learn
  well from few-shot exemplars" when a same-family model (Nemotron) shows the
  opposite, (c) reporting point estimates without CIs on an evaluation set
  where the flood class has n≈75, (d) omitting cost/latency when the paper's
  practical hook is "should you use an LLM instead of training a specialist."
- **Minimum viable version**: items 1-4 in §6, run on the 3-4 models with
  complete data, with CIs and a precise few-shot framing, is enough for a
  workshop-length paper. A journal-length version additionally wants items
  5-8 (the divergence investigation, faithfulness check, cost framing, and
  qualitative error analysis).

**Bottom line**: worth doing. The scarcity argument is real and already
demonstrated in your own data (46-73 flood training examples per fold), the
infrastructure for both arms already exists and runs on matched folds, and the
current numbers already show a genuine trade-off (accuracy vs. safety-critical
recall) rather than a predictable LLM-wins-or-loses result — that's the
difference between a paper reviewers find obvious and one they find useful.
The best-fit venue is a domain/applied one that leverages the existing
Applied Intelligence publication, not a general ML methods venue.
