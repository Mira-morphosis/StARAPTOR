# StARAPTOR study results — App 1086940 (Baldur's Gate 3)

Run date: 2026-11-08 · git commit `c813f5b` · database `~/.staraptor/1086940.db` · fixed seed **42**.
Full reproducibility record (commands, seeds, package versions): `run_manifest.json`.
Figures: `figures/*.pdf`. Raw artifacts: `rq1/`, `rq2/`, `rq3/`.

Reproduce with:
```bash
poetry run src/run_rq_studies.py --app-id 1086940 --rq all
```

---

## RQ1 — Is LLM-based NLP consistent enough to be used in analysis?

**Design:** frozen stratified sample of **500 reviews drawn from the RAW live Steam stream**
(`source=steam_live`, pool of 866 reviews, half top by weighted vote score + half random, seed 42),
re-processed by **4 local models** with the production prompt and decoding (temperature 0).
Because the sample is raw (spam included), spam recognition is benchmarked too — the stored DB
cannot do that, since the pipeline already filters spam before storage.

**Spam detection (the key new result):** nominal Krippendorff's alpha over the `isSpam` flag =
**0.715** (bootstrap 95% CI [0.666, 0.757]); **159 of 500** reviews were flagged spam by a
majority of models. That is *below* the reliable threshold (0.800) and only at the tentative
band (≥ 0.667) — models agree on obvious spam but disagree on borderline cases.

**Per-aspect reliability** (ordinal alpha on the full 0–5 scale, CI in brackets; verdict per
Krippendorff: ≥0.800 reliable, ≥0.667 tentative):

| aspect | alpha | verdict | rating-only alpha (jointly detected) |
|---|---|---|---|
| multiplayer | 0.857 [0.74, 0.94] | **reliable** | n/a (13 units) |
| story | 0.821 [0.77, 0.87] | **reliable** | 0.884 (70) |
| audio | 0.769 [0.59, 0.89] | tentative | n/a (6) |
| replayability | 0.771 [0.71, 0.82] | tentative | 0.321 (49) |
| graphics | 0.753 [0.62, 0.85] | tentative | n/a (10) |
| community | 0.728 [0.00, 1.00] | tentative | n/a (2) |
| controls | 0.660 [0.33, 0.86] | below | n/a (3) |
| customization | 0.621 [0.47, 0.76] | below | n/a (6) |
| difficulty | 0.614 [0.46, 0.74] | below | n/a (8) |
| gameplay | 0.513 [0.46, 0.57] | below | 0.832 (72) |
| immersion | 0.475 [0.37, 0.56] | below | n/a (11) |
| monetizationModel | 0.372 [0.16, 0.56] | below | n/a (1) |

Detection (presence vs absence) and overall alphas are nearly identical → the disagreement is
not mainly *whether* an aspect is mentioned, but *how it is scored*. Where enough units are
jointly detected, rating-only alpha is high for story (0.884) and gameplay (0.832) but low for
replayability (0.321).

**Aggregate-level dispersion** (across-model CV of the daily features the forecaster consumes):
low for `positive_ratio` (0.034) and `replayability` (0.035); high for `monetizationModel`
(1.07), `controls` (0.71), `community` (0.66) — these are the features least stable across models.

All 4 models extracted 500/500 reviews (0 request failures). Details: `rq1/report.json`,
`rq1/alpha_per_aspect.csv`, `rq1/aggregate_dispersion.csv`; figures
`figures/rq1_alpha_per_aspect.pdf`, `figures/rq1_aggregate_dispersion.pdf`.

**Answer to RQ1:** aspect extraction is reliable only for a subset of aspects (multiplayer, story);
spam detection and several aspects (immersion, monetization, gameplay scoring) are not yet at
reliable agreement → the NLP stage is usable but model-sensitive; prefer aspects with
alpha ≥ 0.8 downstream, or ensembling.

---

## RQ2 — Are the mined association rules actionable, or noise?

**Persistence:** of **3,789** mined rules, only **141 (3.7%)** persist across windows — most rules
are one-off artifacts of a single 14-day window. In the inspected sample of 45 rules, 0 were
persistent.

**Manual inspection: NOT DONE YET.** `rq2/sample_annotated.csv` still contains the auto-placeholder
categories, so `actionability_rate` / `residual_noise_rate` are `NaN` and `n_inspected = 0` in
`rq2/report.json`. To finish RQ2, fill the `category` and `evidence` columns (categories:
`actionable`, `semantically_trivial`, `taxonomy_artifact`, `unclear`), then re-run:
```bash
poetry run src/run_rq_studies.py --app-id 1086940 --rq rq2
```

---

## RQ3 — Do review-derived features and mined rules improve the forecast?

Forecast of daily concurrent players (70/30 split, 30-day evaluation, 150 Optuna trials per config):

| config | features | 1-step MAPE | multi-step MAPE | 1-step RMSE |
|---|---|---|---|---|
| C1 baseline (KPI + calendar) | 13 | 8.04% | 53.3% | 4,525 |
| C2 + review-derived features | 27 | **5.83%** | **36.5%** | **3,358** |
| C3 + mined rules (C2 + rules) | 3,670 | 7.06% | 56.5% | 4,050 |

`monotonic_improvement_C1_C2_C3 = false`. Review-derived features clearly help (C1→C2: 1-step MAPE
−28% relative); adding the thousands of mined rules **hurts** (C2→C3 worse) — consistent with RQ2's
low persistence (3.7%). Top C3 features are still KPI-driven (`weekend_boost_ratio`,
`kpi_roll_7_max`, `pct_change`); see `rq3/feature_importance_C*.csv`.

**Answer to RQ3:** review-derived daily aggregates improve the forecast; raw mined rules do not
add predictive value beyond them.

---

## Caveats / notes

- RQ1's sample comes from the live Steam review API. Note: Valve disabled the old
  `store.steampowered.com/appreviews/<id>?json=1` endpoint on 2026-10-22 (it now redirects to the
  storefront); the fetcher uses the replacement `api.steampowered.com/IUserReviewsService/GetAppReviews/v1/`
  (no key needed). Review payloads from the new endpoint lack the `hardware` block, so `hw_*`
  fields are mostly `N/A`.
- `rq1/sample.csv` is the frozen raw sample — re-runs reuse it automatically (no new Steam calls);
  use `--rq1-redraw-sample` to draw a fresh one.
- RQ2's actionability metrics are pending the manual annotation described above.
