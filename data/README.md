# Data pipeline

Run in this order (all commands from the repo root, with `.venv` activated):

```bash
python -m data.usgs_nwis
# -> data/raw/usgs_daily_values.csv
# Live-pulls daily discharge/gauge-height from waterservices.usgs.gov for the curated
# gauges in CASE_STUDY_SITES (Sandy/Harvey/Irma), no download/registration needed.

curl -L -o data/raw/crisisnlp_benchmarks.tar.gz \
  "https://crisisnlp.qcri.org/data/crisis_datasets_benchmarks/crisis_datasets_benchmarks_v1.0.tar.gz"
tar xzf data/raw/crisisnlp_benchmarks.tar.gz -C data/raw --exclude='._*'
# -> data/raw/data/event_aware_en/*.tsv (~130MB tarball; the CrisisBench benchmark,
# Alam et al. ICWSM 2021). Only event_aware_en/*_train.tsv is used by data/crisis_text.py.

python -m data.align
# -> data/processed/examples.jsonl
# Joins the two sources above into (event, site, target_date) -> label examples.
# See data/align.py docstring for why labels are anchored to each hurricane's public
# landfall timeline rather than derived from the gauge data itself (avoids circularity
# with the threshold baseline).
```

## Known limitations (read before treating this as paper-ready)

- **CrisisNLP text reports are event-level, not date-specific.** The consolidated
  dataset has no reliable per-tweet timestamp, so every target_date within an event
  gets the same sampled text reports. The text branch currently can't discriminate
  within-event dates on its own -- it contributes general event context and
  informative/off-topic filtering signal, not date-level evidence.
- **Only 3 joint-track events** (Sandy, Harvey, Irma) -- small-sample, not enough for a
  held-out test split beyond leave-one-event-out CV (see baselines/ml_baseline.py).
  Before publishing results, add more events (there are other flood-tagged events in
  the CrisisNLP corpus, e.g. `2013_colorado_floods`, `2014_india_floods` -- the latter
  has no US gauge coverage, but Colorado does).
- **Event-timeline label anchors are the author's recollection of public-record
  landfall/impact dates**, not sourced from an official NOAA record -- cross-check
  `data/align.py::EVENT_LABEL_WINDOWS` against NOAA tropical cyclone reports before
  using results externally.
- **CrisisLexT6's Alberta/Queensland flood subsets are not joinable** with US gauge
  data (no USGS coverage in Canada/Australia) -- they're loaded via the same
  `event_aware_en` file (`data.crisis_text.EVENT_TAGS`) but reserved for a text-only
  out-of-domain generalization check, not the joint task.
