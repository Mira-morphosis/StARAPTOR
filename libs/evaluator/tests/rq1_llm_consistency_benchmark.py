"""
RQ1 — LLM consistency benchmark.

Answers RQ1 ("Is LLM-based NLP consistent enough to be used in analysis?") by measuring
the inter-model reliability of the structured attributes extracted from a frozen,
stratified sample of reviews (methodology, "LLM-based NLP Performance Benchmark"):

  * a sample of N reviews is drawn once — half top-ranked by weighted vote score, half
    uniform random from the remainder, with a fixed seed — and frozen to disk, so the
    same review set can be re-used to benchmark future models; by default the pool is
    fetched RAW from the Steam review API (--no-fetch-live falls back to the stored
    history), because the stored reviews are already spam-filtered by the pipeline and
    would make the models' spam detection unmeasurable — the raw pool is what allows the
    benchmark to also verify differences in spam recognition;
  * every model exposed by the configured (local) provider re-processes the whole sample
    with the identical prompt and decoding configuration used in production (temperature 0,
    reasoning disabled), so the model is the only variable of the experiment; additional
    frontier/cloud APIs can be added at execution time via repeatable --extra-provider
    arguments (credentials are passed only on the CLI, never read from .env), labeled
    "<provider>/<model>"; previously computed raters can be reused via --merge-extractions,
    so frontier results are added on top of the local ones without re-running them;
  * per aspect, Krippendorff's alpha is computed on the full 0-5 scale (an aspect a
    model does not score is a prediction of absence, encoded as 0), together with the
    nominal detection coefficient (presence vs absence) and the rating-only ordinal
    coefficient (restricted to review-aspect pairs that all models detect), with
    bootstrap confidence intervals over the sampled reviews;
  * at the aggregate level, the daily per-aspect means and the daily positive-review
    ratio — the exact features consumed by the forecasting pipeline — are derived once
    per model and compared across models through their relative dispersion;
  * finally, the base run of all models — the rater that agrees the most with the
    cross-model majority vote over every extraction field (the consensus criterion,
    recorded alongside the isUseful fraction in the rq1_model_runs table) — is stored
    into the shared study database (default <out>/<app_id>/<app_id>.db, the appid folder;
    --db-path overrides it): its extractions over the frozen sample and, in live mode,
    over the whole fetched pool are upserted into the reviews table, so RQ2 and RQ3 work
    from there.

Following Krippendorff's interpretation, alpha >= 0.800 is treated as reliable agreement
and alpha >= 0.667 as the lowest limit for tentative conclusions.

Figures vs tables (thesis vademecum: figures must not be abundant): RQ1 emits exactly ONE
PDF figure — the per-aspect alpha with bootstrap CI and thresholds (rq1_alpha_per_aspect.pdf).
Everything else (spam alpha, detection/rating coefficients, aggregate dispersion) is exported
as CSV (alpha_per_aspect.csv, aggregate_dispersion.csv, ...) for the LaTeX tables.

Outputs are written under out/<app_id>/rq1/.
"""
import argparse
import asyncio
import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless: render straight to PDF files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from evaluator.evaluator_interface import (
    SYSTEM_PROMPT,
    ReviewResponse,
    get_available_models,
    get_instructor_client,
)
from evaluator.instructor_factory import (
    ExtraProvider,
    get_client_for_extra,
    get_request_extras,
    get_decoding_metadata,
    list_models_for_extra,
)
from storage.storage_utils import db_connection, get_existing_review_ids, insert_reviews

load_dotenv()

# Columns of the reviews table (mirrors storage.storage_utils._init_reviews_db), split
# into the raw fetched fields and the LLM-extracted annotation fields.
RAW_REVIEW_COLUMNS: List[str] = [
    "recommendationid", "review", "language", "timestamp_updated",
    "steam_purchase", "received_for_free", "voted_up", "weighted_vote_score",
    "playtime_at_review", "primarily_steam_deck", "has_hardware_info",
    "hw_resolution", "hw_ram", "hw_vram", "hw_gpu", "hw_os", "hw_cpu_vendor",
]
EXTRACTION_COLUMNS: List[str] = [
    "isSpam", "multiplayer", "immersion", "community", "replayability", "story",
    "monetizationModel", "gameplay", "controls", "graphics", "customization",
    "audio", "difficulty", "isUseful", "containsPositiveAspects",
    "containsNegativeAspects", "actuallyRecommendsTitle", "containsBugDescription",
]
REVIEW_COLUMNS: List[str] = RAW_REVIEW_COLUMNS + EXTRACTION_COLUMNS

ASPECTS: List[str] = [
    "multiplayer", "immersion", "community", "replayability", "story",
    "monetizationModel", "gameplay", "controls", "graphics", "customization",
    "audio", "difficulty",
]

RELIABLE_THRESHOLD = 0.800
TENTATIVE_THRESHOLD = 0.667

# A model whose extraction rate falls below this fraction of the sample is almost
# certainly misconfigured (provider errors swallowed by the retry logic), not
# "sincerely spam-flagging"; its all-zero column would silently deflate alpha.
MIN_EXTRACTION_RATE_WARN = 0.90


# ---------------------------------------------------------------------------
# Krippendorff's alpha (complete-data formulation, nominal and ordinal levels)
# ---------------------------------------------------------------------------

def _onehot(ratings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-hot encodes a complete ratings matrix.
    :param ratings: Array of shape (n_units, n_raters) with no missing values.
    :return: A tuple (H, values) where H has shape (n_units, n_raters, n_categories).
    """
    values = np.unique(ratings)
    H = (ratings[:, :, None] == values[None, None, :]).astype(float)
    return H, values


def _alpha_from_onehot(H: np.ndarray, values: np.ndarray, level: str) -> float:
    """
    Computes Krippendorff's alpha from a one-hot ratings matrix via the coincidence matrix.
    :param H: One-hot array of shape (n_units, n_raters, n_categories).
    :param values: The category values corresponding to the last axis of H.
    :param level: "nominal" or "ordinal".
    :return: Alpha, or NaN when undefined (e.g. zero expected disagreement with observed disagreement).
    """
    n_units, m, K = H.shape
    if m < 2 or n_units < 2:
        return float("nan")

    s = H.sum(axis=1)                                # (n_units, K): values per unit per category
    A = s.T @ s                                      # sum_{u,i,j} H[u,i,c] H[u,j,k]
    B = np.einsum("uic,uik->ck", H, H)               # sum_{u,i}   H[u,i,c] H[u,i,k]
    o = (A - B) / (m - 1)                            # coincidence matrix, ordered rater pairs
    n = o.sum()
    nc = o.sum(axis=1)                               # marginal per category

    if level == "nominal":
        delta = 1.0 - np.eye(K)
    elif level == "ordinal":
        # Krippendorff's ordinal distance: squared difference of the midranks of the
        # categories in the pooled marginal distribution (values are already sorted).
        cum = np.cumsum(nc) - nc
        midrank = cum + (nc + 1) / 2.0
        delta = (midrank[:, None] - midrank[None, :]) ** 2
    else:
        raise ValueError(f"Unsupported level of measurement: {level}")

    Do = float((o * delta).sum() / n)
    De = float((np.outer(nc, nc) * delta).sum() / (n * (n - 1)))
    if De == 0.0:
        return 1.0 if Do == 0.0 else float("nan")
    return 1.0 - Do / De


def krippendorff_alpha(ratings: np.ndarray, level: str) -> float:
    """
    Computes Krippendorff's alpha for a complete ratings matrix.
    :param ratings: Array of shape (n_units, n_raters); units are rows, raters are columns.
    :param level: "nominal" or "ordinal".
    :return: The alpha coefficient.
    """
    H, values = _onehot(np.asarray(ratings, dtype=float))
    return _alpha_from_onehot(H, values, level)


def bootstrap_alpha_ci(
    ratings: np.ndarray,
    level: str,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Bootstraps a 95% confidence interval for alpha by resampling units (reviews) with replacement.
    :param ratings: Complete ratings matrix (n_units, n_raters).
    :param level: "nominal" or "ordinal".
    :param n_boot: Number of bootstrap replicates.
    :param rng: A seeded numpy Generator.
    :return: The (2.5, 97.5) percentiles of the bootstrap distribution.
    """
    H, values = _onehot(np.asarray(ratings, dtype=float))
    n_units = H.shape[0]
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        stats[b] = _alpha_from_onehot(H[idx], values, level)
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Frozen stratified sample
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """
    Draws the stratified benchmark sample from a pool of raw reviews: half top-ranked by
    weighted vote score, half uniform random from the remainder, mirroring the daily
    sampling mix consumed by the pipeline.
    :param df: The pool of raw reviews (must contain recommendationid and weighted_vote_score).
    :param sample_size: Total sample size (split in half between the two strata).
    :param seed: The fixed sampling seed identifying the frozen sample.
    :return: A DataFrame with one row per sampled review.
    """
    n_top = sample_size // 2
    n_rand = sample_size - n_top

    top = df.nlargest(n_top, "weighted_vote_score")
    remainder = df.drop(top.index)
    rand = remainder.sample(n=min(n_rand, len(remainder)), random_state=seed) if len(remainder) else remainder

    sample = pd.concat([top, rand]).reset_index(drop=True)
    sample["sample_seed"] = seed
    return sample


def draw_sample(app_id: int, sample_size: int, seed: int) -> pd.DataFrame:
    """
    Draws the frozen stratified benchmark sample from the STORED review history.
    Note: stored reviews have already been through the production pipeline, so this
    pool is already spam-filtered and CANNOT be used to benchmark spam detection;
    draw_live_sample() (the default) is required for that (opt out via --no-fetch-live).
    :param app_id: The Steam unique identifier for the application.
    :param sample_size: Total sample size.
    :param seed: The fixed sampling seed identifying the frozen sample.
    :return: A DataFrame with one row per sampled review.
    """
    with db_connection(app_id) as conn:
        df = conn.execute("""
            SELECT recommendationid, review, language, weighted_vote_score, timestamp_updated
            FROM reviews
            WHERE review IS NOT NULL AND TRIM(review) <> ''
        """).df()

    if df.empty:
        raise RuntimeError(f"No reviews with text found for App ID {app_id}.")

    sample = stratified_sample(df, sample_size, seed)
    sample["source"] = "db"
    return sample


async def draw_live_sample(
    app_id: int,
    sample_size: int,
    seed: int,
    max_days: int = 14,
    num_reviews_per_day: int = 250,
    top_n_per_day: int = 125,
    delay: float = 2.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """
    Draws the frozen stratified benchmark sample from reviews fetched LIVE from the Steam
    review API (via the production fetcher), not from the stored database. The stored
    table is already spam-filtered by the pipeline, so it cannot be used to benchmark the
    models' spam detection; the live pool contains raw reviews, spam included.
    :param app_id: The Steam unique identifier for the application.
    :param sample_size: Total sample size.
    :param seed: The fixed sampling seed identifying the frozen sample.
    :param max_days: How many days back the live pool is fetched (Steam API hard limit: 140).
    :param num_reviews_per_day: Per-day fetch cap used while building the pool (production mix).
    :param top_n_per_day: Per-day top-by-weighted-vote-score count used while building the pool.
    :param delay: Delay between API pages, seconds.
    :param verbose: Regulates logging output.
    :return: A tuple (sample, pool): the DataFrame with one row per sampled review
             (plus a "source" column), and the full raw pool DataFrame it was drawn from
             (used to store the best model's run over the whole window for RQ2/RQ3).
    """
    from fetcher.fetcher_interface import fetch_review_history  # local import: evaluator does not depend on fetcher

    pool = await fetch_review_history(
        app_id=app_id,
        max_days=max_days,
        existing_ids=None,  # fresh pool: we WANT raw duplicates-free reviews, spam included
        num_reviews=num_reviews_per_day,
        top_n=top_n_per_day,
        delay=delay,
        verbose=verbose,
    )
    df = pd.DataFrame(pool)
    if df.empty:
        raise RuntimeError(f"Live fetch returned no reviews for App ID {app_id}.")
    df = df[df["review"].notna() & (df["review"].str.strip() != "")].reset_index(drop=True)

    sample = stratified_sample(df, sample_size, seed)
    sample["source"] = "steam_live"
    return sample, df


# ---------------------------------------------------------------------------
# Model extraction (identical prompt/decoding config as production)
# ---------------------------------------------------------------------------

def parse_extra_provider(arg: str) -> ExtraProvider:
    """
    Parses one --extra-provider CLI value into an ExtraProvider. Key=value pairs are
    separated by ';', e.g.:
      "name=openai;type=openai_compatible;api_key=sk-...;base_url=https://api.openai.com/v1;models=gpt-4o,gpt-4o-mini"
    `models` is comma-separated and optional (if omitted, models are listed live from the
    endpoint — openai_compatible only). `base_url` optional (empty = cloud default).
    Credentials are accepted only here, at execution time — never from .env.
    :param arg: The raw CLI string.
    :return: The parsed ExtraProvider.
    """
    kv: Dict[str, str] = {}
    for pair in arg.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Bad --extra-provider pair '{pair}' (expected key=value, pairs separated by ';').")
        k, v = pair.split("=", 1)
        kv[k.strip()] = v.strip()

    name = kv.get("name")
    api_key = kv.get("api_key")
    if not name or not api_key:
        raise ValueError("--extra-provider requires at least name=... and api_key=...")
    ptype = kv.get("type", "openai_compatible").lower()
    if ptype not in ("openai_compatible", "anthropic"):
        raise ValueError(f"--extra-provider type must be openai_compatible or anthropic, got '{ptype}'.")
    models = [m.strip() for m in kv.get("models", "").split(",") if m.strip()]
    return ExtraProvider(
        name=name,
        type=ptype,
        api_key=api_key,
        base_url=kv.get("base_url") or None,
        models=models,
    )


async def collect_targets(extra_specs: List[ExtraProvider]) -> List[Dict[str, Any]]:
    """
    Builds the list of benchmark targets (raters): every model of the configured local
    provider (rater label = model id), plus every model of each extra (e.g. frontier)
    provider passed on the CLI (rater label = "<provider>/<model>").
    :param extra_specs: Extra providers parsed from --extra-provider arguments.
    :return: A list of dicts with keys rater, client, model, request_extras.
    """
    targets: List[Dict[str, Any]] = []

    local_models = await get_available_models()
    if local_models:
        local_client = get_instructor_client()
        for m in local_models:
            targets.append({
                "rater": m, "client": local_client, "model": m,
                "request_extras": get_request_extras(),
            })

    for spec in extra_specs:
        client = get_client_for_extra(spec)
        models = await list_models_for_extra(spec)
        # Frontier APIs don't understand llama.cpp's chat-template kwargs; disable
        # reasoning only where the API supports it, otherwise send no extras.
        extras = {"thinking": {"type": "disabled"}} if spec.type == "anthropic" else {}
        for m in models:
            targets.append({
                "rater": f"{spec.name}/{m}", "client": client, "model": m,
                "request_extras": extras,
            })

    return targets


async def extract_with_model(
    client,
    model: str,
    rater: Optional[str] = None,
    reviews: List[Dict[str, Any]] = None,
    concurrency: int = 100,
    request_extras: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Re-processes the sample with one model using the same request shape as
    ReviewProcessor._analyze_single (temperature 0, retries, reasoning disabled).
    :param client: The Instructor-wrapped client.
    :param model: The model identifier sent to the API; the only variable of the experiment.
    :param rater: The label under which the extractions are recorded (defaults to `model`;
                  "<provider>/<model>" for extra providers).
    :param reviews: The sampled review records.
    :param concurrency: Max concurrent requests.
    :param request_extras: Provider-specific extra_body (decoding guards); defaults to the
                           production provider's extras.
    :return: A tuple (extractions, n_failures): the DataFrame with one row per
             successfully extracted review, and the number of failed requests
             (empty-text reviews excluded). A non-zero failure count flags a
             misconfigured model instead of silently producing an all-zero column.
    """
    if rater is None:
        rater = model
    if reviews is None:
        reviews = []
    extras = request_extras if request_extras is not None else get_request_extras()
    semaphore = asyncio.Semaphore(concurrency)
    failures = 0

    async def _one(review: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        nonlocal failures
        text = (review.get("review") or "").strip()
        if not text:
            return None
        async with semaphore:
            try:
                result: ReviewResponse = await client.chat.completions.create(
                    model=model,
                    response_model=ReviewResponse,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Review text to analyze:\n{text}"},
                    ],
                    temperature=0.0,
                    max_retries=1,
                    extra_body=extras if extras else None,
                )
                return {"recommendationid": review["recommendationid"], "rater": rater, "model": model, **result.model_dump()}
            except Exception:
                failures += 1
                return None

    raw = await tqdm_asyncio.gather(*[_one(r) for r in reviews], desc=f"RQ1 extracting [{rater}]", unit="review")
    rows = [r for r in raw if r is not None]

    n_extracted = len(rows)
    print(f"RQ1 [{rater}] extracted {n_extracted}/{len(reviews)} reviews ({failures} request failures).")
    if n_extracted < MIN_EXTRACTION_RATE_WARN * len(reviews):
        print(
            f"RQ1 [{rater}] WARNING: extraction rate below "
            f"{MIN_EXTRACTION_RATE_WARN:.0%} — the model is probably misconfigured "
            f"(check the provider logs) and its all-zero column would bias alpha."
        )

    return pd.DataFrame(rows), failures


def build_ratings_cube(
    extractions: Dict[str, pd.DataFrame],
    sample: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Builds, per aspect, the complete ratings matrix (n_reviews x n_models) on the 0-5 scale;
    an aspect a model does not score (or a spam-flagged review) contributes 0.
    :param extractions: model -> extraction DataFrame.
    :param sample: The frozen sample (defines the unit order).
    :return: aspect -> ratings matrix.
    """
    models = list(extractions.keys())
    units = sample["recommendationid"].tolist()
    cube: Dict[str, np.ndarray] = {}
    for aspect in ASPECTS:
        mat = np.zeros((len(units), len(models)), dtype=float)
        for j, model in enumerate(models):
            df = extractions[model]
            if df.empty:
                continue  # model produced no extractions: its column stays all-zero
            vals = df.set_index("recommendationid").reindex(units)[aspect]
            mat[:, j] = vals.fillna(0).astype(float).to_numpy()
        cube[aspect] = mat
    return cube


# ---------------------------------------------------------------------------
# Aggregate-level dispersion (the features the forecaster actually consumes)
# ---------------------------------------------------------------------------

def aggregate_dispersion(
    extractions: Dict[str, pd.DataFrame],
    sample: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derives, once per model, the daily per-aspect means and the daily positive-review
    ratio over that model's non-spam extractions, then compares the models through the
    relative dispersion (coefficient of variation) of each daily feature.
    :param extractions: model -> extraction DataFrame.
    :param sample: The frozen sample, carrying timestamp_updated.
    :return: A tuple (summary, daily_detail): summary has one row per feature with the mean
             across-model CV; daily_detail has the per-model daily aggregates.
    """
    frames = []
    for model, df in extractions.items():
        if df.empty:
            continue
        non_spam = df[df["isSpam"] == False].copy()  # noqa: E712
        non_spam = non_spam.merge(sample[["recommendationid", "timestamp_updated"]], on="recommendationid", how="left")
        non_spam["date"] = pd.to_datetime(non_spam["timestamp_updated"], unit="s").dt.date
        g = non_spam.assign(model=model).groupby(["model", "date"])
        means = g[ASPECTS].mean().fillna(0.0)
        pos = g["actuallyRecommendsTitle"].apply(lambda s: s.fillna(False).astype(float).mean())
        daily = means.copy()
        daily["positive_ratio"] = pos
        frames.append(daily.reset_index())

    daily_all = pd.concat(frames, ignore_index=True)
    records = []
    for feature in ASPECTS + ["positive_ratio"]:
        pivot = daily_all.pivot_table(index="date", columns="model", values=feature)
        vals = pivot.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(np.abs(vals.mean(axis=1)) > 1e-9, vals.std(axis=1) / np.abs(vals.mean(axis=1)), np.nan)
        records.append({
            "feature": feature,
            "mean_cv_across_models": float(np.nanmean(cv)),
            "n_days": int(pivot.shape[0]),
        })
    summary = pd.DataFrame(records)
    summary["mean_cv_overall"] = float(summary["mean_cv_across_models"].mean())
    return summary, daily_all


# ---------------------------------------------------------------------------
# Storing the most "useful" run of all models into the shared study database
# ---------------------------------------------------------------------------

def _py(value: Any) -> Any:
    """
    Converts numpy/pandas scalar values to plain Python types DuckDB can bind; NaN becomes None.
    :param value: The scalar to convert.
    :return: The converted value.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if pd.isna(value) else float(value)
    return value


def run_usefulness(extractions: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Scores every model's run by its usefulness: the fraction of its extracted reviews
    flagged isUseful=True (spam-flagged extractions carry no isUseful and count as not useful).
    :param extractions: model -> extraction DataFrame.
    :return: rater -> metrics dict (n_extracted, n_spam, n_useful, useful_fraction).
    """
    metrics: Dict[str, Dict[str, Any]] = {}
    for rater, df in extractions.items():
        n = len(df)
        n_spam = int((df["isSpam"] == True).sum()) if n and "isSpam" in df else 0  # noqa: E712
        n_useful = int((df["isUseful"] == True).sum()) if n and "isUseful" in df else 0  # noqa: E712
        metrics[rater] = {
            "n_extracted": n,
            "n_spam": n_spam,
            "n_useful": n_useful,
            "useful_fraction": (n_useful / n) if n else 0.0,
        }
    return metrics


AGREEMENT_FIELDS: List[str] = (
    ["isSpam"] + ASPECTS
    + ["isUseful", "containsPositiveAspects", "containsNegativeAspects",
       "actuallyRecommendsTitle", "containsBugDescription"]
)


def majority_agreement_scores(
    extractions: Dict[str, pd.DataFrame],
    sample: pd.DataFrame,
) -> Dict[str, float]:
    """
    Scores every model's run by its agreement with the cross-model majority (consensus):
    for each sampled review and each extraction field, the majority label across raters is
    computed, and a rater's score is the fraction of (review, field) pairs where its label
    equals the majority. Missing labels count as a label (None == None agrees).
    :param extractions: model -> extraction DataFrame.
    :param sample: The frozen sample (defines the unit order).
    :return: rater -> majority-agreement score in [0, 1].
    """
    units = sample["recommendationid"].astype(str).tolist()
    frames: Dict[str, pd.DataFrame] = {}
    for rater, df in extractions.items():
        g = df.copy()
        g["recommendationid"] = g["recommendationid"].astype(str)
        frames[rater] = g.set_index("recommendationid").reindex(units)

    scores: Dict[str, float] = {}
    for field in AGREEMENT_FIELDS:
        # a rater may not have the column at all (empty extraction frame): treat as all-None
        cols = pd.concat(
            {r: (f[field] if field in f.columns else pd.Series(None, index=f.index, dtype=object))
             for r, f in frames.items()},
            axis=1,
        )  # units x raters
        maj = cols.apply(lambda row: (row.mode().iloc[0] if not row.mode().empty else None), axis=1)
        for rater, f in frames.items():
            a = (f[field] if field in f.columns else pd.Series(None, index=f.index, dtype=object)).to_numpy(dtype=object)
            b = maj.to_numpy(dtype=object)
            eq = (a == b) | (pd.isna(a) & pd.isna(b))
            scores[rater] = scores.get(rater, 0.0) + float(eq.mean())
    n_fields = len(AGREEMENT_FIELDS)
    return {r: s / n_fields for r, s in scores.items()}


def select_base_run(
    metrics: Dict[str, Dict[str, Any]],
    agreement: Dict[str, float],
) -> str:
    """
    Picks the base run to store in the DB among all models: the rater that agrees with the
    cross-model majority the most (defensible consensus criterion); ties are broken by
    the isUseful fraction, then the number of extracted reviews, then the rater name.
    :param metrics: The per-rater metrics from run_usefulness().
    :param agreement: The per-rater majority-agreement scores.
    :return: The selected rater label.
    """
    return max(
        metrics,
        key=lambda r: (agreement.get(r, 0.0), metrics[r]["useful_fraction"], metrics[r]["n_extracted"], r),
    )


def store_run_in_db(
    app_id: int,
    sample: pd.DataFrame,
    extractions: Dict[str, pd.DataFrame],
    rater: str,
    metrics: Dict[str, Dict[str, Any]],
    agreement: Optional[Dict[str, float]] = None,
    record_run: bool = True,
) -> Dict[str, Any]:
    """
    Stores one model's run into the shared study database so RQ2/RQ3 work from it:
    the run's extractions are upserted into the reviews table (new sampled reviews are
    inserted with their raw fetched fields, reviews already present get their annotation
    columns updated), and every model's run metrics plus the selection are recorded in
    the rq1_model_runs table.
    :param app_id: The Steam unique identifier for the application.
    :param sample: The frozen sample (carries the raw review fields per recommendationid).
    :param extractions: model -> extraction DataFrame.
    :param rater: The selected rater whose run is stored.
    :param metrics: The per-rater metrics from run_usefulness().
    :param agreement: The per-rater majority-agreement scores from majority_agreement_scores().
    :param record_run: If True, also (re)write the rq1_model_runs table; False for the
                       whole-pool storage pass, which must not overwrite the benchmark
                       selection recorded from the sample.
    :return: A dict describing what was stored (selected rater, inserted/updated counts).
    """
    df = extractions[rater]
    # ids must be compared as strings: sample.csv round-trips them as int64, extraction
    # dicts keep them as str — cast so the raw-field join never silently misses.
    raw = sample.copy()
    raw["recommendationid"] = raw["recommendationid"].astype(str)
    raw = raw.set_index("recommendationid")

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rid = str(row["recommendationid"])
        rec: Dict[str, Any] = {c: None for c in REVIEW_COLUMNS}
        if rid in raw.index:
            base = raw.loc[rid]
            for c in RAW_REVIEW_COLUMNS:
                if c in base.index and pd.notna(base[c]):
                    rec[c] = _py(base[c])
        for c in EXTRACTION_COLUMNS:
            if c in row.index:
                rec[c] = _py(row[c])
        rec["recommendationid"] = rid
        records.append(rec)

    existing = get_existing_review_ids(app_id)
    to_insert = [r for r in records if r["recommendationid"] not in existing]
    to_update = [r for r in records if r["recommendationid"] in existing]

    if to_insert:
        # chunked: insert_reviews() runs one executemany, keep it memory-bounded on pool-sized runs
        chunk = 5000
        for i in range(0, len(to_insert), chunk):
            insert_reviews(app_id, to_insert[i:i + chunk])  # creates the reviews table if missing
    if to_update:
        # Annotation columns are always overwritten; raw fetched fields use COALESCE so a
        # record that lacks them (e.g. a db-sourced sample) never wipes the stored values.
        set_cols = [f"{c} = ?" for c in EXTRACTION_COLUMNS] + [
            f"{c} = COALESCE(?, reviews.{c})" for c in RAW_REVIEW_COLUMNS if c != "recommendationid"
        ]
        raw_order = [c for c in RAW_REVIEW_COLUMNS if c != "recommendationid"]
        sql = f"UPDATE reviews SET {', '.join(set_cols)} WHERE recommendationid = ?"
        with db_connection(app_id) as conn:
            conn.executemany(
                sql,
                [[r[c] for c in EXTRACTION_COLUMNS] + [r[c] for c in raw_order] + [r["recommendationid"]]
                 for r in to_update],
            )

    if record_run:
        now = datetime.datetime.now(datetime.timezone.utc)
        with db_connection(app_id) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rq1_model_runs (
                    rater VARCHAR PRIMARY KEY,
                    n_extracted INTEGER,
                    n_spam INTEGER,
                    n_useful INTEGER,
                    useful_fraction DOUBLE,
                    majority_agreement DOUBLE,
                    selected BOOLEAN,
                    stored_at TIMESTAMP
                )
            """)
            conn.execute("ALTER TABLE rq1_model_runs ADD COLUMN IF NOT EXISTS majority_agreement DOUBLE")
            for r, m in metrics.items():
                conn.execute(
                    "INSERT OR REPLACE INTO rq1_model_runs "
                    "(rater, n_extracted, n_spam, n_useful, useful_fraction, majority_agreement, selected, stored_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [r, int(m["n_extracted"]), int(m["n_spam"]), int(m["n_useful"]),
                     float(m["useful_fraction"]),
                     float(agreement[r]) if agreement and r in agreement else None,
                     r == rater, now],
                )

    return {
        "rater": rater,
        "useful_fraction": float(metrics[rater]["useful_fraction"]),
        "majority_agreement": float(agreement[rater]) if agreement and rater in agreement else None,
        "n_extracted": int(metrics[rater]["n_extracted"]),
        "n_useful": int(metrics[rater]["n_useful"]),
        "n_inserted": len(to_insert),
        "n_updated": len(to_update),
        "db_path": os.getenv("STARAPTOR_DB_PATH"),
    }


# ---------------------------------------------------------------------------
# Figure (vector PDF, ready for inclusion in the thesis)
# ---------------------------------------------------------------------------
# The whole RQ pipeline emits at most three PDF figures (see the thesis vademecum:
# figures must not be abundant — most results are tables). RQ1 contributes exactly
# one: the per-aspect reliability with its bootstrap CI and the interpretation
# thresholds. Everything else (spam alpha, per-aspect detection/rating coefficients,
# aggregate dispersion) is tabular and exported as CSV for the LaTeX tables.

def plot_rq1_figures(
    per_aspect: List[Dict[str, Any]],
    fig_dir: str | Path,
) -> List[str]:
    """
    Renders the single RQ1 figure as a PDF: per-aspect Krippendorff's alpha (ordinal,
    0-5 scale) with bootstrap 95% CI and the reliable/tentative threshold lines.
    :param per_aspect: The per-aspect alpha records.
    :param fig_dir: Directory the PDF figure is written to.
    :return: The list of written figure paths (exactly one).
    """
    fig_path = Path(fig_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(per_aspect).sort_values("alpha_ordinal_0_5", na_position="first")
    y = np.arange(len(df))
    alpha = df["alpha_ordinal_0_5"].to_numpy(dtype=float)
    ci = df["alpha_ordinal_ci95"].tolist()
    lower = np.array([a - c[0] if not np.isnan(a) else 0.0 for a, c in zip(alpha, ci)])
    upper = np.array([c[1] - a if not np.isnan(a) else 0.0 for a, c in zip(alpha, ci)])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(y, np.nan_to_num(alpha), xerr=[lower, upper], color="steelblue",
            error_kw=dict(ecolor="black", capsize=3), height=0.65)
    ax.axvline(RELIABLE_THRESHOLD, color="green", linestyle="--", linewidth=1,
               label=f"reliable ≥ {RELIABLE_THRESHOLD}")
    ax.axvline(TENTATIVE_THRESHOLD, color="orange", linestyle="--", linewidth=1,
               label=f"tentative ≥ {TENTATIVE_THRESHOLD}")
    ax.set_yticks(y, df["aspect"])
    ax.set_xlabel("Krippendorff's $\\alpha$ (ordinal, 0–5 scale)")
    ax.set_title("RQ1 — inter-model reliability per aspect (bootstrap 95% CI)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = fig_path / "rq1_alpha_per_aspect.pdf"
    fig.savefig(out)
    plt.close(fig)

    return [str(out)]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def run_rq1_benchmark(
    app_id: int,
    sample_size: int = 500,
    seed: int = 42,
    n_boot: int = 1000,
    min_rating_units: int = 30,
    sample_csv: Optional[str] = None,
    out_dir: str = "./out",
    fig_dir: Optional[str] = None,
    extra_providers: Optional[List[ExtraProvider]] = None,
    merge_extractions: Optional[str] = None,
    rerun: bool = False,
    fetch_live: bool = True,
    fetch_max_days: int = 14,
    fetch_delay: float = 2.0,
    redraw_sample: bool = False,
    db_path: Optional[str] = None,
    store_pool: bool = True,
    store_pool_max: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs the full RQ1 benchmark for one game.
    :param app_id: The Steam unique identifier for the application.
    :param sample_size: Size of the frozen stratified sample.
    :param seed: Fixed sampling seed (also used for the bootstrap).
    :param n_boot: Number of bootstrap replicates for the alpha confidence intervals.
    :param min_rating_units: Minimum jointly-detected units for a stable rating-only alpha.
    :param sample_csv: Optional path to a previously frozen sample CSV; skips re-drawing.
    :param out_dir: Root output directory; artifacts land in <out_dir>/<app_id>/rq1/.
    :param fig_dir: Directory for the PDF figures; defaults to <out_dir>/<app_id>/rq1/figures/.
    :param extra_providers: Extra (e.g. frontier) providers passed on the CLI; their models
                            are added as raters labeled "<provider>/<model>".
    :param merge_extractions: Optional CSV of extractions from a previous run (e.g.
                              extractions_all_models.csv): raters already present are reused
                              instead of recomputed, so frontier results can be added on top
                              of the local ones without re-paying the local inference cost.
    :param rerun: If True, ignore merge_extractions and recompute every rater.
    :param fetch_live: If True (default), draw the sample from reviews fetched LIVE from
                       the Steam review API (raw, spam included) instead of the stored DB,
                       so the models' spam detection is benchmarked too; the stored DB is
                       already spam-filtered and would make spam recognition unmeasurable.
    :param fetch_max_days: Days of history to fetch when fetch_live=True (API hard limit 140).
    :param fetch_delay: Delay between Steam API pages, seconds.
    :param redraw_sample: If True, ignore the frozen <out>/<app_id>/rq1/sample.csv and draw
                          a fresh sample; otherwise the frozen sample is reused so every
                          run (and every model) sees the identical review set.
    :param db_path: The shared study database to store the most useful run into
                    (default: <out>/<app_id>/<app_id>.db, the appid folder; resolved via
                    the STARAPTOR_DB_PATH env var by storage.storage_utils).
    :param store_pool: In live mode, after benchmarking the sample, run the most useful
                       model over the WHOLE fetched pool and store those extractions in the
                       study db, so RQ2/RQ3 get a dense review table instead of just the
                       sample. The benchmark itself still runs on the frozen sample only.
    :param store_pool_max: Optional cap on the pool run (seeded draw from the pool);
                           None stores the whole pool.
    :return: The report dict (also written to report.json).
    """
    if db_path:
        os.environ["STARAPTOR_DB_PATH"] = str(Path(db_path).resolve())
    out_path = Path(out_dir) / str(app_id) / "rq1"
    out_path.mkdir(parents=True, exist_ok=True)

    sample_source = "unknown"
    pool_size: Optional[int] = None
    pool: Optional[pd.DataFrame] = None  # full live pool (live mode only), used for the stored pool run
    frozen = out_path / "sample.csv"
    pool_csv = out_path / "pool.csv"
    if sample_csv:
        sample = pd.read_csv(sample_csv)
        sample_source = str(sample["source"].iloc[0]) if "source" in sample.columns else "csv"
        print(f"[App {app_id}] Loaded frozen sample from {sample_csv} ({len(sample)} reviews).")
    elif fetch_live:
        # Always make sure the raw pool is available (persisted as pool.csv): the sample is
        # drawn/frozen only when missing or --redraw-sample, but the pool is (re)fetched
        # whenever pool.csv is absent, so the whole-pool storage pass can always run.
        if redraw_sample or not frozen.exists() or not pool_csv.exists():
            new_sample, pool = await draw_live_sample(
                app_id, sample_size, seed,
                max_days=fetch_max_days, delay=fetch_delay, verbose=True,
            )
            pool_size = len(pool)
            pool.to_csv(pool_csv, index=False)
            if redraw_sample or not frozen.exists():
                sample = new_sample
                sample.to_csv(frozen, index=False)
                print(f"[App {app_id}] Frozen LIVE sample drawn with seed {seed} from a pool of {pool_size} "
                      f"raw Steam reviews ({len(sample)} sampled: {sample_size // 2} top by weighted_vote_score "
                      f"+ {len(sample) - sample_size // 2} random).")
            else:
                sample = pd.read_csv(frozen)
                print(f"[App {app_id}] Reusing frozen sample {frozen} ({len(sample)} reviews, seed {seed}); "
                      f"refreshed the live pool ({pool_size} raw reviews) -> {pool_csv}.")
        else:
            sample = pd.read_csv(frozen)
            pool = pd.read_csv(pool_csv)
            pool_size = len(pool)
            print(f"[App {app_id}] Reusing frozen sample {frozen} and frozen pool {pool_csv} "
                  f"({len(sample)} sampled / {pool_size} raw reviews).")
        sample_source = str(sample["source"].iloc[0]) if "source" in sample.columns else "steam_live"
        if sample_source != "steam_live":
            print(f"[App {app_id}] WARNING: the frozen sample comes from the stored, already "
                  f"spam-filtered history, so spam-detection agreement cannot be measured on it; "
                  f"pass --redraw-sample to draw a fresh raw (live) sample.")
    elif not redraw_sample and frozen.exists():
        sample = pd.read_csv(frozen)
        sample_source = str(sample["source"].iloc[0]) if "source" in sample.columns else "csv"
        print(f"[App {app_id}] Reusing frozen sample {frozen} (source={sample_source}, "
              f"{len(sample)} reviews, seed {seed}).")
    else:
        sample = draw_sample(app_id, sample_size, seed)
        sample_source = "db"
        sample.to_csv(out_path / "sample.csv", index=False)
        print(f"[App {app_id}] Frozen sample drawn from stored history with seed {seed}: {len(sample)} reviews "
              f"({sample_size // 2} top by weighted_vote_score + {len(sample) - sample_size // 2} random).")

    concurrency = int(os.getenv("STARAPTOR_CONCURRENCY_LIMIT", "100"))
    reviews = sample.to_dict("records")

    targets = await collect_targets(extra_providers or [])
    if not targets:
        raise RuntimeError(
            "No models available: neither from the configured local provider nor from "
            "--extra-provider arguments."
        )
    targets_by_rater = {t["rater"]: t for t in targets}

    # Reuse extractions from a previous run (e.g. the local-only pass): raters already
    # present in the merge file are kept, only the missing ones (e.g. frontier models)
    # are computed now.
    extractions: Dict[str, pd.DataFrame] = {}
    if merge_extractions and Path(merge_extractions).exists() and not rerun:
        old = pd.read_csv(merge_extractions)
        if "rater" not in old.columns and "model" in old.columns:
            old = old.rename(columns={"model": "rater"})
        if "rater" in old.columns:
            for rater, g in old.groupby("rater"):
                extractions[rater] = g.reset_index(drop=True)
            print(f"[App {app_id}] Reusing {len(extractions)} rater(s) from {merge_extractions}: {sorted(extractions)}")

    extraction_rates: Dict[str, Dict[str, int]] = {}
    for t in targets:
        rater = t["rater"]
        if rater in extractions and not rerun:
            print(f"[App {app_id}] Skipping {rater} (already present in the merge file).")
            continue
        df_model, n_failures = await extract_with_model(
            t["client"], t["model"], rater, reviews, concurrency, t["request_extras"]
        )
        extractions[rater] = df_model
        extraction_rates[rater] = {
            "extracted": int(len(df_model)),
            "failures": int(n_failures),
            "sample": int(len(reviews)),
        }
        path = out_path / f"extractions_{rater.replace('/', '_')}.csv"
        extractions[rater].to_csv(path, index=False)

    if not extractions:
        raise RuntimeError("No extractions available: no targets to run and nothing to merge.")

    raters = list(extractions.keys())
    print(f"[App {app_id}] Benchmarking {len(raters)} raters: {raters}")

    all_extractions = pd.concat(extractions.values(), ignore_index=True)
    all_extractions.to_csv(out_path / "extractions_all_models.csv", index=False)

    cube = build_ratings_cube(extractions, sample)
    rng = np.random.default_rng(seed)

    # Spam detection is part of the extraction schema and only measurable on raw
    # (non-spam-filtered) reviews: inter-rater nominal alpha over the isSpam flag.
    units = sample["recommendationid"].tolist()
    raters_list = list(extractions.keys())
    spam_mat = np.zeros((len(units), len(raters_list)), dtype=float)
    for j, rater in enumerate(raters_list):
        df = extractions[rater]
        if df.empty:
            continue
        vals = df.set_index("recommendationid").reindex(units)["isSpam"]
        spam_mat[:, j] = vals.fillna(False).astype(float).to_numpy()
    alpha_spam = krippendorff_alpha(spam_mat, "nominal")
    spam_lo, spam_hi = bootstrap_alpha_ci(spam_mat, "nominal", n_boot, rng)
    spam_report = {
        "alpha_nominal": alpha_spam,
        "ci95": [spam_lo, spam_hi],
        "n_reviews": int(spam_mat.shape[0]),
        "n_spam_flagged_by_majority": int((spam_mat.mean(axis=1) >= 0.5).sum()),
        "measurable": sample_source == "steam_live",
    }
    if sample_source != "steam_live":
        spam_report["note"] = (
            f"Sample source '{sample_source}' is already spam-filtered by the pipeline, so "
            "this coefficient does NOT measure spam recognition; a raw (live) sample is "
            "required for that (the default)."
        )
        print(f"[App {app_id}] WARNING: sample source '{sample_source}' is already spam-filtered; "
              f"spam alpha is not meaningful here. Use live sampling (the default) to benchmark "
              f"spam recognition.")
    print(f"[App {app_id}] spam detection: alpha(nominal)={alpha_spam:.3f} "
          f"CI95=[{spam_lo:.3f}, {spam_hi:.3f}] over {spam_mat.shape[0]} reviews (source={sample_source})")

    per_aspect = []
    for aspect in ASPECTS:
        ratings = cube[aspect]
        alpha_ord = krippendorff_alpha(ratings, "ordinal")
        lo, hi = bootstrap_alpha_ci(ratings, "ordinal", n_boot, rng)

        detection = (ratings > 0).astype(float)
        alpha_det = krippendorff_alpha(detection, "nominal")

        jointly = ratings[(ratings > 0).all(axis=1)]
        n_joint = int(jointly.shape[0])
        if n_joint >= min_rating_units:
            alpha_rating = krippendorff_alpha(jointly, "ordinal")
            rating_status = "ok"
        else:
            alpha_rating = float("nan")
            rating_status = "insufficient_data"

        if np.isnan(alpha_ord):
            verdict = "insufficient_data"
        elif alpha_ord >= RELIABLE_THRESHOLD:
            verdict = "reliable"
        elif alpha_ord >= TENTATIVE_THRESHOLD:
            verdict = "tentative"
        else:
            verdict = "below_tentative"

        per_aspect.append({
            "aspect": aspect,
            "n_units": int(ratings.shape[0]),
            "n_models": int(ratings.shape[1]),
            "alpha_ordinal_0_5": alpha_ord,
            "alpha_ordinal_ci95": [lo, hi],
            "alpha_detection_nominal": alpha_det,
            "alpha_rating_ordinal": alpha_rating,
            "rating_units_jointly_detected": n_joint,
            "rating_status": rating_status,
            "verdict": verdict,
        })
        print(f"[App {app_id}] {aspect:<18} alpha(ordinal, 0-5)={alpha_ord:.3f} "
              f"CI95=[{lo:.3f}, {hi:.3f}] detection={alpha_det:.3f} rating={alpha_rating:.3f} "
              f"(n={n_joint}) -> {verdict}")

    dispersion_summary, daily_all = aggregate_dispersion(extractions, sample)
    dispersion_summary.drop(columns=["mean_cv_overall"]).to_csv(out_path / "aggregate_dispersion.csv", index=False)
    daily_all.to_csv(out_path / "daily_aggregates_per_model.csv", index=False)

    # Store the most "useful" run of all models into the shared study database, so that
    # RQ2 and RQ3 work from it: the winning rater's extractions are upserted into the
    # reviews table and the selection is recorded in the rq1_model_runs table.
    run_metrics = run_usefulness(extractions)
    agreement = majority_agreement_scores(extractions, sample)
    best_rater = select_base_run(run_metrics, agreement)
    stored = store_run_in_db(app_id, sample, extractions, best_rater, run_metrics, agreement)
    print(f"[App {app_id}] stored the base run in the DB: rater={best_rater} "
          f"(majority_agreement={stored['majority_agreement']:.4f}, "
          f"useful_fraction={stored['useful_fraction']:.3f}, "
          f"inserted {stored['n_inserted']} / updated {stored['n_updated']} reviews) "
          f"-> {stored['db_path']}")

    # Whole-pool storage pass (live mode only): the benchmark stays on the frozen sample,
    # but the winning model re-processes the ENTIRE fetched pool so that RQ2 (mining) and
    # RQ3 (daily review aggregates) work from a dense review table in the shared db.
    stored_pool: Optional[Dict[str, Any]] = None
    if fetch_live and store_pool and pool is not None and not pool.empty:
        best_target = targets_by_rater[best_rater]
        pool_df = pool
        if store_pool_max is not None and len(pool_df) > store_pool_max:
            pool_df = pool_df.sample(n=store_pool_max, random_state=seed).reset_index(drop=True)
        print(f"[App {app_id}] storing the best run over the WHOLE live pool: rater={best_rater}, "
              f"{len(pool_df)} reviews (of {len(pool)} fetched)...")
        pool_extractions, pool_failures = await extract_with_model(
            best_target["client"], best_target["model"], best_rater,
            pool_df.to_dict("records"), concurrency, best_target["request_extras"],
        )
        stored_pool = store_run_in_db(
            app_id, pool_df, {best_rater: pool_extractions}, best_rater, run_metrics,
            agreement, record_run=False,
        )
        stored_pool["failures"] = int(pool_failures)
        print(f"[App {app_id}] pool run stored: rater={best_rater}, "
              f"inserted {stored_pool['n_inserted']} / updated {stored_pool['n_updated']} reviews, "
              f"{pool_failures} failures -> {stored_pool['db_path']}")

    figures = plot_rq1_figures(per_aspect, fig_dir or out_path / "figures")
    for f in figures:
        print(f"[App {app_id}] figure written: {f}")

    report = {
        "rq": "RQ1",
        "app_id": app_id,
        "seed": seed,
        "sample_size": int(len(sample)),
        "raters": raters,
        "sample_source": sample_source,
        "sample_pool_size": pool_size,
        "spam_detection": spam_report,
        "providers": {
            "local": [t["rater"] for t in targets if "/" not in t["rater"]],
            "extra": {
                spec.name: [t["rater"] for t in targets if t["rater"].startswith(spec.name + "/")]
                for spec in (extra_providers or [])
            },
        },
        "merged_from": merge_extractions if (merge_extractions and not rerun) else None,
        "decoding": get_decoding_metadata(),
        "extraction_rates": extraction_rates,
        "run_usefulness": run_metrics,
        "majority_agreement": agreement,
        "selection_rule": "highest cross-model majority agreement (tie-break: isUseful fraction, extraction count)",
        "stored_run": stored,
        "stored_pool_run": stored_pool,
        "thresholds": {"reliable": RELIABLE_THRESHOLD, "tentative": TENTATIVE_THRESHOLD},
        "n_bootstraps": n_boot,
        "per_aspect": per_aspect,
        "aggregate_dispersion_mean_cv": dispersion_summary.set_index("feature")["mean_cv_across_models"].to_dict(),
        "outputs": {
            "sample": str(out_path / "sample.csv"),
            "extractions": str(out_path / "extractions_all_models.csv"),
            "alpha_per_aspect": str(out_path / "alpha_per_aspect.csv"),
            "aggregate_dispersion": str(out_path / "aggregate_dispersion.csv"),
            "daily_aggregates": str(out_path / "daily_aggregates_per_model.csv"),
            "figures": figures,
        },
    }
    pd.DataFrame(per_aspect).to_csv(out_path / "alpha_per_aspect.csv", index=False)
    with open(out_path / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n[App {app_id}] RQ1 report written to {out_path / 'report.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="RQ1 — LLM consistency benchmark")
    parser.add_argument("--app-id", type=int, default=int(os.getenv("STARAPTOR_APP_ID", 1086940)))
    parser.add_argument("--db-path", type=str, default=None,
                        help="Shared study database the most useful run is stored into "
                             "(default: <out>/<app_id>/<app_id>.db, the appid folder).")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--min-rating-units", type=int, default=30)
    parser.add_argument("--sample-csv", type=str, default=None, help="Reuse a previously frozen sample CSV.")
    parser.add_argument("--out", type=str, default="./out")
    parser.add_argument("--fig-dir", type=str, default=None, help="Directory for the PDF figures.")
    parser.add_argument(
        "--extra-provider", action="append", default=[], metavar="KV",
        help="Extra (e.g. frontier) API provider as ';'-separated key=value pairs: "
             "name=...;api_key=...;type=openai_compatible|anthropic;base_url=...;models=m1,m2 "
             "(models optional; listed live from the endpoint if omitted, openai_compatible "
             "only). Repeatable for multiple providers. Credentials are accepted only here "
             "at execution time, never from .env. Example: "
             "\"name=openai;type=openai_compatible;api_key=$OPENAI_KEY;models=gpt-4o,gpt-4o-mini\"")
    parser.add_argument("--merge-extractions", type=str, default=None,
                        help="CSV of extractions from a previous run (e.g. extractions_all_models.csv): "
                             "raters already present are reused, only missing ones are computed.")
    parser.add_argument("--rerun", action="store_true",
                        help="Ignore --merge-extractions and recompute every rater.")
    parser.add_argument("--fetch-live", action=argparse.BooleanOptionalAction, default=True,
                        help="Draw the sample from reviews fetched LIVE from the Steam review API "
                             "(raw, spam included) instead of the stored DB, so spam detection is "
                             "benchmarked too. This is the DEFAULT because the stored DB is already "
                             "spam-filtered; pass --no-fetch-live to sample the stored history "
                             "instead. The sample is frozen by seed and saved to sample.csv.")
    parser.add_argument("--fetch-max-days", type=int, default=14,
                        help="Days of history to fetch when --fetch-live (Steam API hard limit: 140).")
    parser.add_argument("--fetch-delay", type=float, default=2.0,
                        help="Delay between Steam API pages, seconds (same default as the "
                             "production fetcher; raise it if Steam rate-limits).")
    parser.add_argument("--redraw-sample", action="store_true",
                        help="Ignore the frozen <out>/<app_id>/rq1/sample.csv and draw a fresh sample.")
    parser.add_argument("--store-pool", action=argparse.BooleanOptionalAction, default=True,
                        help="In live mode, after benchmarking the frozen sample, run the most useful "
                             "model over the WHOLE fetched pool and store those extractions in the "
                             "study db, so RQ2/RQ3 get a dense review table. The benchmark itself "
                             "still runs on the sample only. Pass --no-store-pool to store the "
                             "sample run only.")
    parser.add_argument("--store-pool-max", type=int, default=None,
                        help="Cap the whole-pool storage run to N reviews (seeded draw from the "
                             "pool); default: the whole pool.")
    args = parser.parse_args()
    extra_specs = [parse_extra_provider(a) for a in args.extra_provider]
    db_path = args.db_path or str((Path(args.out).resolve() / str(args.app_id) / f"{args.app_id}.db"))
    asyncio.run(run_rq1_benchmark(
        app_id=args.app_id,
        db_path=db_path,
        sample_size=args.sample_size,
        seed=args.seed,
        n_boot=args.bootstraps,
        min_rating_units=args.min_rating_units,
        sample_csv=args.sample_csv,
        out_dir=args.out,
        fig_dir=args.fig_dir,
        extra_providers=extra_specs,
        merge_extractions=args.merge_extractions,
        rerun=args.rerun,
        fetch_live=args.fetch_live,
        fetch_max_days=args.fetch_max_days,
        fetch_delay=args.fetch_delay,
        redraw_sample=args.redraw_sample,
        store_pool=args.store_pool,
        store_pool_max=args.store_pool_max,
    ))


if __name__ == "__main__":
    main()
