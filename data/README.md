# Data pipeline

## Annotation CSV (already in this repo)

`data/raw/flood_images_annot_2.csv` — 68,599 rows, columns `datetime, place, path,
level`. This is the labeled subset actually used in the source paper: SHOP/SHOP2 camera
only, Nov-Feb rainy season across 4 years. Copied from the user's local copy
(originally at `~/Downloads/flood_images_annot_2.csv`).

## Images

The **full** Kaggle dataset is ~73GB (`caetanoranieri/river-images-at-sao-carlos`,
CC-BY-NC-SA-4.0) — every month of the year and both camera location pairs, most of
which isn't needed: only SHOP/SHOP2 (not SESC/SESC2) was carefully labeled and used in
the paper, and only the Nov-Feb rainy-season subset is in the annotation CSV. Fetching
the full dataset is impractical for casual/interactive use — plan to run the real
download on a machine with the disk/bandwidth for it (see the SSH-server discussion in
project notes), selecting only the files referenced in the CSV:

```bash
# Kaggle CLI must be authenticated (~/.kaggle/kaggle.json)
kaggle datasets download -d caetanoranieri/river-images-at-sao-carlos \
  -f "enoe/enoe2/<path from CSV>" -p data/raw/sample_images
# CSV `path` column values need the "enoe/enoe2/" prefix to match the dataset's
# internal file paths. See data/enoe_images.py:KAGGLE_PREFIX.
```

A small **dev sample** (74 images, stratified across all 4 seasons and all 4 classes,
prioritizing the rare medium/high/flood examples) is already downloaded to
`data/raw/sample_images/` for building/testing the pipeline without pulling the full
dataset.

```bash
python -m data.enoe_images
# -> data/processed/examples.jsonl, built from whichever images are present locally
# (74 in the dev sample; re-run after a fuller download to include more)
```

## Label taxonomy

Ordinal, physically defined via a fixed reference marker at the canal:
`low < medium < high < flood`, where `flood` specifically means water has risen above
the top of the concrete canal wall and is touching the grass bank — a specific
threshold, not just "a lot of water." See
`knowledge_base/corpus/mineirinho_creek_criteria.md` for the full criteria (this is also
what the agent retrieves via RAG).

## Class distribution (severe imbalance)

| level | count | % |
|---|---:|---:|
| low | 67,803 | 98.84% |
| medium | 535 | 0.78% |
| high | 186 | 0.27% |
| flood | 75 | 0.11% |

Accuracy is close to meaningless here — report per-class recall (see `eval/metrics.py`).

## Seasons (for leave-one-season-out CV, matching the source paper)

Nov-Feb rainy season, 4 years: `2018-2019` (8,175 images, only 2 flood + 13 high — by
far the weakest fold for evaluating the rare classes), `2019-2020` (7,755), `2020-2021`
(24,647), `2021-2022` (28,022).

## Camera relocation

`place=SHOP` covers only 2018-11-01 to 2019-01-24 (~3 months, the start of the
2018-2019 season, 7,434 images); `place=SHOP2` covers everything after, through
2022-02-28 (61,165 images). One-time early relocation, not a per-season alternation —
the 2018-2019 season is a mix of both camera angles.

## Day/night split

~47% of images overall are nighttime shots, but the rarer/more severe classes skew
disproportionately toward night: flood is 64% nighttime, high is 55%, medium is 54%,
vs. 47% for low. Nighttime frames are often dominated by severe streetlight glare that
can nearly obscure the waterline (confirmed by inspecting real samples). `is_night` is
included as metadata on every example (`data/enoe_images.py`) — plan to report metrics
stratified by day/night, not just pooled, given how much this affects visibility.

## Known limitations

- The 2018-2019 season fold has only 2 flood and 13 high examples total — any
  leave-one-season-out recall computed on that held-out fold will be extremely noisy
  (a single misclassification swings flood recall by 50 percentage points).
- Nighttime visibility is a real, unresolved challenge, not just a labeling nuance —
  worth treating as a first-class evaluation axis (day vs. night performance), not an
  afterthought.
