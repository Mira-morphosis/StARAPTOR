import copy
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from xgboost import XGBRegressor

from analyzer.regressor import Regressor, XGB_BEST_PARAMS, DEFAULT_DECAY_PARAMS, make_asymmetric_capacity_objective
from storage.storage_utils import (
    get_params_for_size,
    get_latest_tuning_record,
    list_tuning_history,
    save_tuned_params,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =====================================================================================
# PROGRESS / ETA
# =====================================================================================
# Tracks weighted progress across every tier of a tuning cycle, so the ETA reflects
# real cost (bigger tiers = slower trials) instead of assuming every trial is equal.

def _format_duration(seconds: float) -> str:
    """
    Formats a duration in seconds as a compact human-readable string.
    :param seconds: The duration in seconds.
    :return: A string like "1h02m03s", "2m03s", "3s", or "—" if not finite.
    """
    if not np.isfinite(seconds):
        return "—"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class TuningProgress:
    """
    Fine-grained, thread-safe progress tracker for a tuning cycle.

    Progress is recorded per completed fold (not per trial), weighted by
    (training rows in that fold) x (n_estimators used by that specific trial). This is more
    accurate than a flat "tier_size x n_folds" weight because: n_estimators varies trial to
    trial, walk-forward folds grow in size (expanding window), and pruned trials only count
    the work actually completed before being cut off.

    Throughput is estimated over a moving time window (instead of a cumulative average) so it
    adapts quickly when pruning behavior or tier size changes; falls back to the cumulative
    average when the window doesn't have enough samples yet.
    """

    def __init__(
        self,
        tier_plan: List[Tuple[int, int, int]],
        app_id: int,
        window_seconds: float = 45.0,
        min_window_span: float = 3.0,
        print_interval: float = 1.0,
    ):
        """
        :param tier_plan: A list of (tier_size, n_trials) pairs; expected work per tier is computed from real fold boundaries plus expected n_estimators (see _expected_tier_work), not a flat estimate.
        :param app_id: The Steam unique identifier for the application being tuned.
        :param window_seconds: The size of the moving time window used to estimate throughput.
        :param min_window_span: The minimum time span the window must cover before it's trusted over the cumulative average.
        :param print_interval: The minimum number of seconds between progress print lines.
        """
        self.app_id = app_id
        self.total_work = sum(_expected_tier_work(size, trials) for size, trials in tier_plan)
        self.completed_work = 0.0
        self.start_time = time.time()
        self.window_seconds = window_seconds
        self.min_window_span = min_window_span
        self.print_interval = print_interval
        self._samples: "deque[Tuple[float, float]]" = deque([(self.start_time, 0.0)])
        self._last_print = 0.0
        self._lock = threading.Lock()

    def _rate_locked(self, now: float) -> float:
        while len(self._samples) > 1 and now - self._samples[0][0] > self.window_seconds:
            self._samples.popleft()
        oldest_t, oldest_w = self._samples[0]
        span_t = now - oldest_t
        span_w = self.completed_work - oldest_w
        if span_t >= self.min_window_span and span_w > 0:
            return span_w / span_t
        elapsed = now - self.start_time
        return self.completed_work / elapsed if elapsed > 0 else 0.0

    def _snapshot_locked(self, now: float) -> Tuple[float, float, float, float]:
        rate = self._rate_locked(now)
        elapsed = now - self.start_time
        remaining = max(0.0, self.total_work - self.completed_work)
        eta = remaining / rate if rate > 0 else float("inf")
        pct = (self.completed_work / self.total_work * 100.0) if self.total_work > 0 else 100.0
        return elapsed, eta, pct, rate

    def tick(self, weight: float) -> Tuple[float, float, float, float, bool]:
        """
        Records a completed fold's work.
        :param weight: The work weight to add, typically (training rows) x (n_estimators) for the fold just completed.
        :return: A tuple (elapsed, eta, pct, rate, should_print).
        """
        with self._lock:
            self.completed_work += weight
            now = time.time()
            self._samples.append((now, self.completed_work))
            elapsed, eta, pct, rate = self._snapshot_locked(now)
            should_print = (now - self._last_print) >= self.print_interval
            if should_print:
                self._last_print = now
            return elapsed, eta, pct, rate, should_print

    def snapshot(self) -> Tuple[float, float, float]:
        """
        Reads the current progress state without recording new work (for end-of-trial checkpoints).
        :return: A tuple (elapsed, eta, pct).
        """
        with self._lock:
            now = time.time()
            elapsed, eta, pct, _rate = self._snapshot_locked(now)
            return elapsed, eta, pct


def _expected_tier_work(tier_size: int, n_trials: int) -> float:
    """
    Estimates the total work for a tier from its real walk-forward fold boundaries.
    :param tier_size: The number of rows in the tier.
    :param n_trials: The number of Optuna trials planned for the tier.
    :return: The expected work units: sum of training rows per fold, times expected n_estimators, times n_trials.
    """
    test_size, n_folds = adaptive_fold_params(tier_size)
    folds = _generate_folds(tier_size, n_folds=n_folds, test_size=test_size)
    if not folds:
        return float(tier_size * n_trials)
    max_n_estimators = _max_n_estimators_for_tier(tier_size)
    expected_n_estimators = (50 + max_n_estimators) / 2.0
    rows_per_trial = sum(train_end for _, train_end, _ in folds)  # train_start is always 0
    return rows_per_trial * expected_n_estimators * n_trials


# =====================================================================================
# CONTROL INTERFACE
# =====================================================================================
# Determines whether a new tuning cycle is needed. Fetch/store operations are entirely
# delegated to storage.storage_utils.

def needs_retuning(
    app_id: int,
    current_size: int,
    overshoot_ratio: float = 1.5,
    staleness_days: Optional[int] = 90,
) -> Tuple[bool, str]:
    """
    Decides whether the current dataset requires a new tuning cycle.

    Checks two independent conditions: (1) the current dataset exceeds the largest
    previously tuned tier by more than overshoot_ratio, meaning existing params were
    optimized on too little data relative to what's available now; (2) the last tuning run
    is older than staleness_days (skipped if staleness_days is None).
    :param app_id: The Steam unique identifier for the application.
    :param current_size: The number of rows in the current dataset.
    :param overshoot_ratio: How many times larger than the max tuned tier the dataset must be to trigger retuning.
    :param staleness_days: The maximum age in days of the last tuning run before it's considered stale, or None to disable this check.
    :return: A tuple (should_retune, reason) so the decision can be logged or inspected.
    """
    history = list_tuning_history(app_id)

    if history.empty:
        return True, "No previous tuning found for this App ID."

    max_tier = int(history["tier_size"].max())
    if current_size > max_tier * overshoot_ratio:
        return True, (
            f"Current dataset ({current_size} rows) exceeds the max tuned tier "
            f"({max_tier} rows) by more than {overshoot_ratio}x."
        )

    if staleness_days is not None:
        latest = get_latest_tuning_record(app_id)
        latest_ts = pd.to_datetime(latest["timestamp"])
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.tz_localize("UTC")
        now = pd.Timestamp.now(tz=latest_ts.tzinfo)
        age_days = (now - latest_ts).days
        if age_days > staleness_days:
            return True, f"Last tuning was {age_days} days ago (threshold: {staleness_days})."

    return False, "Current parameters are still valid, no tuning needed."


# =====================================================================================
# TIER GENERATION AND WALK-FORWARD CROSS-VALIDATION
# =====================================================================================

def generate_size_tiers(
    max_observed_size: int,
    base: int = 140,
    growth: float = 1.6,
    cap: int = 5000,
) -> List[int]:
    """
    Generates dataset-size tiers with geometric growth: dense near cold-start (where
    hyperparameter sensitivity is highest), sparse at large sizes (where the marginal effect
    of one more day flattens out). Always includes max_observed_size as the upper bound, so
    tuning never happens on data that doesn't exist yet.
    :param max_observed_size: The largest dataset size currently available.
    :param base: The size of the smallest tier.
    :param growth: The geometric growth factor between tiers.
    :param cap: The largest tier size to generate via geometric growth (max_observed_size can still exceed this as the final tier).
    :return: A list of tier sizes in increasing order.
    """
    if max_observed_size < base:
        return [max_observed_size]

    tiers = [base]
    while tiers[-1] * growth < min(max_observed_size, cap):
        tiers.append(int(tiers[-1] * growth))
    if tiers[-1] < max_observed_size:
        tiers.append(max_observed_size)
    return tiers


def adaptive_fold_params(n_samples: int) -> Tuple[int, int]:
    """
    Scales cross-validation fold size and count to the dataset size.
    :param n_samples: The number of rows available.
    :return: A tuple (test_size, n_folds).
    """
    test_size = max(7, n_samples // 14)
    min_train = max(40, n_samples // 3)
    max_folds = max(1, (n_samples - min_train) // test_size)
    n_folds = max(3, min(6, max_folds))
    return test_size, n_folds


def _generate_folds(
    n_samples: int,
    n_folds: int,
    test_size: int,
    min_train_size: int = 40,
) -> List[Tuple[int, int, int]]:
    """
    Generates walk-forward folds as (train_start, train_end, test_end), always with an
    expanding window: the training set always starts at the beginning of the series and
    grows, rather than sliding forward and discarding old data.

    This is cross-validation adapted for time series. Regular k-fold CV
    shuffles data randomly into folds, which would let the model "see the future" when
    predicting the past. Walk-forward CV instead always trains on an earlier chunk of time
    and tests on the chunk right after it, moving that boundary forward fold by fold.
    :param n_samples: The total number of rows available.
    :param n_folds: The number of folds to attempt to generate.
    :param test_size: The number of rows in each fold's test window.
    :param min_train_size: The minimum number of training rows required to keep generating folds.
    :return: A list of (train_start, train_end, test_end) tuples, in chronological order.
    """
    folds = []
    max_test_end = n_samples
    for i in range(n_folds):
        test_end = max_test_end - i * test_size
        train_end = test_end - test_size
        if train_end < min_train_size:
            break
        folds.append((0, train_end, test_end))
    return list(reversed(folds))


def _run_cv_benchmark(
    regressor: Regressor,
    model: XGBRegressor,
    X: pd.DataFrame,
    Y: pd.Series,
    df_full: pd.DataFrame,
    n_folds: int,
    test_size: int,
    decay_rate: float = 0.95,
    rule_decay_rate: float = 0.995,
    peak_weight_alpha: float = 1.0,
    trial: Optional[optuna.Trial] = None,
) -> Dict[str, float]:
    """
    Runs walk-forward cross-validation for a given model and aggregates per-fold metrics.
    :param regressor: The Regressor instance providing app_id context.
    :param model: The XGBRegressor to evaluate.
    :param X: The full feature matrix.
    :param Y: The full target series.
    :param df_full: The full merged frame, including target_y_abs.
    :param n_folds: The number of walk-forward folds to generate.
    :param test_size: The number of rows in each fold's test window.
    :param decay_rate: Passed through to each fold's benchmark() call.
    :param rule_decay_rate: Passed through to each fold's benchmark() call.
    :param peak_weight_alpha: Passed through to each fold's benchmark() call.
    :param trial: If provided, an Optuna trial used to report intermediate scores and allow pruning.
    :return: A dict of aggregated (mean/std) metrics across folds, or {} if no folds were generated.
    """
    folds = _generate_folds(len(X), n_folds=n_folds, test_size=test_size)
    if not folds:
        return {}

    fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_start, train_end, test_end) in enumerate(folds):
        X_fold = X.iloc[train_start:test_end].reset_index(drop=True)
        Y_fold = Y.iloc[train_start:test_end].reset_index(drop=True)
        df_fold = df_full.iloc[train_start:test_end].reset_index(drop=True)

        split_ratio = train_end / test_end

        fold_regressor = Regressor(app_id=regressor.app_id, model=copy.deepcopy(model), capacity_mode=True)

        metrics, _ = fold_regressor.benchmark(
            model=model,
            split_ratio=split_ratio,
            decay_rate=decay_rate,
            rule_decay_rate=rule_decay_rate,
            peak_weight_alpha=peak_weight_alpha,
            eval_days=14,  # Truncated evaluation horizon for fast tuning
            data_tuple=(X_fold, Y_fold, df_fold),
        )
        if metrics:
            fold_metrics.append(metrics)

        # Early pruning if this fold's performance is already poor
        if trial is not None and metrics:
            intermediate_score = metrics.get("mape_multistep_pct", 100.0)
            trial.report(intermediate_score, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    if not fold_metrics:
        return {}

    agg: Dict[str, float] = {"n_folds": len(fold_metrics)}
    for key in fold_metrics[0]:
        if isinstance(fold_metrics[0][key], (int, float)):
            vals = [m[key] for m in fold_metrics]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

    return agg


def _max_n_estimators_for_tier(
    tier_size: int, hard_cap: int = 250, base: int = 50, step: int = 25
) -> int:
    """
    Caps n_estimators based on tier size, since small tiers gain nothing from very deep
    ensembles and it only adds overhead from the custom objective's per-round Python call.
    The floor at 100 still leaves search room even on the smallest tier (base=140).

    The result is always aligned to the grid used by trial.suggest_int(base, upper,
    step=step) — otherwise Optuna silently narrows the range (e.g. [50, 80] with step=25
    becomes [50, 75]) and emits a UserWarning on every trial.
    :param tier_size: The number of rows in the tier.
    :param hard_cap: The absolute maximum n_estimators to allow.
    :param base: The minimum n_estimators to allow.
    :param step: The step size used by the Optuna search grid.
    :return: The maximum n_estimators for this tier, aligned to the (base, step) grid.
    """
    raw = int(np.clip(tier_size // 2, 100, hard_cap))
    n_steps = max(0, (raw - base) // step)
    return base + n_steps * step


def _tune_tier(
        app_id: int,
        tier_size: int,
        data_tuple: Tuple[pd.DataFrame, pd.Series, pd.DataFrame],
        n_trials: int = 35,
        n_jobs: int = -1,
        progress: Optional[TuningProgress] = None,
        tier_idx: int = 1,
        n_tiers: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    """
    Runs an Optuna hyperparameter search for a single dataset-size tier.

    Instead of trying every combination (grid search) or purely random guesses, Optuna uses a Bayesian method called TPE
    (Tree-structured Parzen Estimator) that models which regions of the search space tend to
    score well, and samples new trials more often from those regions as it learns. It's also
    combined with a "pruner" here, which watches a trial's score partway through (after each
    walk-forward fold) and kills it early if it's clearly worse than what's already been
    found, saving time by not fully training doomed candidates.
    :param app_id: The Steam unique identifier for the application.
    :param tier_size: The number of rows in this tier.
    :param data_tuple: The (X, Y, df_full) slice for this tier, held in memory to avoid DB lock contention across trials.
    :param n_trials: The number of Optuna trials to run.
    :param n_jobs: The number of trials to run in parallel (-1 for all cores).
    :param progress: If provided, a TuningProgress tracker to report fold-level progress to.
    :param tier_idx: The 1-based index of this tier within the overall tuning cycle, for progress printing.
    :param n_tiers: The total number of tiers in the overall tuning cycle, for progress printing.
    :return: A tuple (best_params, metrics, n_folds).
    """
    X_tier, Y_tier, df_tier = data_tuple
    test_size, n_folds = adaptive_fold_params(tier_size)
    folds = _generate_folds(len(X_tier), n_folds=n_folds, test_size=test_size)
    max_n_estimators = _max_n_estimators_for_tier(tier_size)

    def objective(trial: optuna.Trial) -> float:
        asymmetric_alpha = trial.suggest_float("asymmetric_alpha", 1.5, 15.0, log=True)
        custom_obj = make_asymmetric_capacity_objective(alpha=asymmetric_alpha)

        xgb_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, max_n_estimators, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            "objective": custom_obj,
            "random_state": 42,
            # Single-threaded per fit: parallelism happens across trials (n_jobs at the
            # Optuna level below), not inside each individual XGBoost fit.
            "n_jobs": 1,
        }

        peak_weight_alpha = trial.suggest_float("peak_weight_alpha", 0.0, 5.0)

        fold_scores = []

        for fold_idx, (train_start, train_end, test_end) in enumerate(folds):
            X_tr, Y_tr = X_tier.iloc[train_start:train_end], Y_tier.iloc[train_start:train_end]
            X_te, Y_te = X_tier.iloc[train_end:test_end], Y_tier.iloc[train_end:test_end]

            weights = None
            if 'is_weekend' in X_tr.columns and 'mean_concurrent' in X_tr.columns:
                wk_mask = X_tr['is_weekend'] == 1
                wk_mean = X_tr.loc[wk_mask, 'mean_concurrent'].mean()
                weekday_mean = X_tr.loc[~wk_mask, 'mean_concurrent'].mean()
                natural_boost = max(0.0, (wk_mean / (weekday_mean + 1e-5)) - 1.0)
                wk_weight = 1.0 + (peak_weight_alpha * natural_boost)
                weights = np.where(wk_mask, wk_weight, 1.0)

            model = XGBRegressor(**xgb_params)
            model.fit(X_tr, Y_tr, sample_weight=weights)

            preds_log = model.predict(X_te)
            anchor_levels = X_te['mean_concurrent'].to_numpy()
            preds_abs = np.maximum(0.0, anchor_levels * np.exp(np.clip(preds_log, -2.0, 2.0)))
            actuals = df_tier.iloc[train_end:test_end]["target_y_abs"].to_numpy()

            mape_fold = np.mean(np.abs(actuals - preds_abs) / np.maximum(1.0, actuals)) * 100.0
            fold_scores.append(mape_fold)

            # Weight = actual training rows in this fold x actual n_estimators sampled for
            # this trial: it is exact, not an a-priori estimate, so pruned trials only get credit
            # for the folds they actually completed.
            if progress is not None:
                fold_work = (train_end - train_start) * xgb_params["n_estimators"]
                elapsed, eta, pct, rate, should_print = progress.tick(fold_work)
                if should_print:
                    print(
                        f"[App {app_id}] tier {tier_idx}/{n_tiers} (n={tier_size}) "
                        f"| trial {trial.number + 1}/{n_trials} fold {fold_idx + 1}/{len(folds)} "
                        f"| MAPE so far {np.mean(fold_scores):.2f}% "
                        f"| overall {min(pct, 100.0):5.1f}% | elapsed {_format_duration(elapsed)} "
                        f"| ETA {_format_duration(eta)} | {rate:,.0f} work/s",
                        flush=True,
                    )

            # Report the running mean after each fold and let the pruner decide whether
            # this trial is worth finishing.
            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_mape = float(np.mean(fold_scores))
        trial.set_user_attr("mape_multistep_pct", mean_mape)
        trial.set_user_attr("mape_1step_pct", mean_mape)
        trial.set_user_attr("mape_multistep_std", float(np.std(fold_scores)))
        trial.set_user_attr("mape_weekday_pct", mean_mape)
        trial.set_user_attr("mape_weekend_pct", mean_mape)
        trial.set_user_attr("underestim_weekend_pct", 0.0)

        return mean_mape

    sampler = TPESampler(multivariate=True, seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2, n_startup_trials=8)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.set_user_attr("n_trials_target", n_trials)

    callbacks = []
    if progress is not None:
        def _trial_end_checkpoint(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            # Read-only: the work was already ticked fold-by-fold inside the objective
            # above, so this must not add weight again: it just prints a clean per-trial
            # checkpoint line marking completed/pruned.
            elapsed, eta, pct = progress.snapshot()
            status = "pruned" if trial.state == optuna.trial.TrialState.PRUNED else "done"
            best = f"{study.best_value:.2f}%" if study.best_trial is not None else "n/a"
            print(
                f"[App {app_id}] tier {tier_idx}/{n_tiers} (n={tier_size}) "
                f"| trial {trial.number + 1}/{n_trials} {status} "
                f"| best MAPE {best} "
                f"| overall {min(pct, 100.0):5.1f}% | elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}",
                flush=True,
            )

        callbacks.append(_trial_end_checkpoint)

    # Trials are independent, in-memory-only units of work (no shared DB handle inside the
    # objective), so they can run in parallel across cores via n_jobs.
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False, callbacks=callbacks)

    # Optuna may run a few extra trials past n_trials when n_jobs>1 (in-flight trials from
    # other workers finish after the target is reached) — only count trials that actually
    # completed when picking "best".
    best = study.best_trial

    xgb_param_keys = {
        "n_estimators", "max_depth", "learning_rate", "min_child_weight",
        "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda",
    }
    best_xgb_params = {k: v for k, v in best.params.items() if k in xgb_param_keys}
    best_xgb_params["asymmetric_alpha"] = best.params.get("asymmetric_alpha", 5.0)
    best_xgb_params["random_state"] = 42
    best_xgb_params["n_jobs"] = 2

    best_params = {
        "xgb_params": best_xgb_params,
        "decay_params": {
            "decay_rate": 0.95,
            "rule_decay_rate": best.params.get("rule_decay_rate", 0.995),
        },
        # Not part of xgb_params: peak_weight_alpha controls sample_weight during .fit()
        # (see Regressor.fit()), it isn't an XGBoost constructor argument. Kept top-level so
        # it survives being read back without leaking into a direct XGBRegressor(**xgb_params)
        # call downstream.
        "peak_weight_alpha": best.params.get("peak_weight_alpha", 1.0),
    }

    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)

    metrics = {
        "score": best.value,
        "mape_multistep_pct": best.user_attrs["mape_multistep_pct"],
        "mape_1step_pct": best.user_attrs["mape_1step_pct"],
        "mape_multistep_std": best.user_attrs["mape_multistep_std"],
        "mape_weekday_pct": best.user_attrs["mape_weekday_pct"],
        "mape_weekend_pct": best.user_attrs["mape_weekend_pct"],
        "underestim_weekend_pct": best.user_attrs["underestim_weekend_pct"],
        "n_trials_complete": n_complete,
        "n_trials_pruned": n_pruned,
    }

    return best_params, metrics, n_folds


def run_tuning_cycle(
    app_id: int,
    n_trials: int = 150,
    force: bool = False,
    verbose: bool = True,
    n_jobs: int = -1,
) -> Dict[int, Dict[str, Any]]:
    """
    Tunes every dataset-size tier not yet present in the tuning history, saving each result to the DB.
    :param app_id: The Steam unique identifier for the application.
    :param n_trials: The number of Optuna trials to run per tier.
    :param force: If True, ignores tuning history and re-tunes every tier from scratch (used by tuning_tester.py for a clean end-to-end run, where a baseline/tuned comparison needs a full cycle rather than partially recycled parameters).
    :param verbose: Regulates logging output.
    :param n_jobs: The number of trials to run in parallel per tier (-1 for all cores).
    :return: A dict mapping tier_size -> {"params": ..., "metrics": ...} for every tier tuned in this call.
    """
    regressor = Regressor(app_id=app_id)
    X, Y, df_full = regressor.generator.build_daily_matrix()
    current_size = len(X)

    history = list_tuning_history(app_id)
    already_tuned = set(history["tier_size"].tolist()) if (not history.empty and not force) else set()

    tiers = generate_size_tiers(max_observed_size=current_size)
    tiers_to_tune = [t for t in tiers if t not in already_tuned and t <= current_size]

    if not tiers_to_tune:
        if verbose:
            print(f"[App {app_id}] No new tiers to tune (dataset: {current_size} rows).")
        return {}

    # Precompute (tier_size, n_trials) for every tier up-front so the progress tracker knows
    # the exact weighted workload before starting, and the ETA is meaningful from the very
    # first fold of the very first tier.
    tier_plan: List[Tuple[int, int]] = [(tier_size, n_trials) for tier_size in tiers_to_tune]

    progress = TuningProgress(tier_plan, app_id=app_id) if verbose else None

    if verbose:
        print(f"[App {app_id}] Tuning plan: {len(tiers_to_tune)} tiers "
              f"({', '.join(str(t) for t in tiers_to_tune)}), {n_trials} trials/tier, force={force}.")

    results: Dict[int, Dict[str, Any]] = {}
    cycle_start = time.time()

    for i, tier_size in enumerate(tiers_to_tune, start=1):
        data_tuple = (
            X.iloc[-tier_size:].reset_index(drop=True),
            Y.iloc[-tier_size:].reset_index(drop=True),
            df_full.iloc[-tier_size:].reset_index(drop=True),
        )

        if verbose:
            print(f"\n[App {app_id}] === Tier {i}/{len(tiers_to_tune)}: {tier_size} rows "
                  f"({n_trials} trials, n_jobs={n_jobs}) ===")

        params, metrics, n_folds = _tune_tier(
            app_id, tier_size, data_tuple, n_trials,
            n_jobs=n_jobs, progress=progress, tier_idx=i, n_tiers=len(tiers_to_tune),
        )

        save_tuned_params(
            app_id=app_id,
            tier_size=tier_size,
            dataset_size_at_tuning=current_size,
            params=params,
            metrics=metrics,
            n_trials=n_trials,
            n_folds=n_folds,
        )

        results[tier_size] = {"params": params, "metrics": metrics}

        if verbose:
            print(f"[App {app_id}] Tier {tier_size} complete — "
                  f"multi-step MAPE: {metrics['mape_multistep_pct']:.2f}%, "
                  f"1-step MAPE: {metrics['mape_1step_pct']:.2f}%, "
                  f"pruned {metrics.get('n_trials_pruned', 0)}/{n_trials} trials")

    if verbose:
        print(f"\n[App {app_id}] Tuning cycle complete in {_format_duration(time.time() - cycle_start)}.")

    return results


def ensure_up_to_date_params(
    app_id: int,
    n_trials: int = 150,
    overshoot_ratio: float = 1.5,
    staleness_days: Optional[int] = 90,
    verbose: bool = True,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Checks whether new hyperparameters are needed for the given App ID, tunes if so, and
    returns the current parameters best suited to the dataset's current size. If no tuning
    has ever run for this app, tuning is forced.
    :param app_id: The Steam unique identifier for the application.
    :param n_trials: The number of Optuna trials to run per tier, if tuning is triggered.
    :param overshoot_ratio: Passed through to needs_retuning().
    :param staleness_days: Passed through to needs_retuning().
    :param verbose: Regulates logging output.
    :param n_jobs: The number of trials to run in parallel per tier, if tuning is triggered.
    :return: The resolved hyperparameter dict for the current dataset size.
    """
    regressor = Regressor(app_id=app_id)
    X, _, _ = regressor.generator.build_daily_matrix()
    current_size = len(X)

    should_tune, reason = needs_retuning(
        app_id, current_size,
        overshoot_ratio=overshoot_ratio,
        staleness_days=staleness_days,
    )

    if verbose:
        print(f"[App {app_id}] needs_retuning={should_tune} — {reason}")

    if should_tune:
        run_tuning_cycle(app_id, n_trials=n_trials, verbose=verbose, n_jobs=n_jobs)

    params = get_params_for_size(app_id, current_size)
    if params is None:
        raise RuntimeError(
            f"Tuning ran but no parameters were found for App ID {app_id}. "
            "Check that build_daily_matrix() produces valid data."
        )

    return params


def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes a hyperparameter dict that may have been saved before the nested
    xgb_params/decay_params structure existed (older tuning records already in the DB for
    some apps). Dicts already in the current format are returned unchanged.
    :param params: The raw hyperparameter dict as read from storage.
    :return: A dict guaranteed to have "xgb_params", "decay_params", and "peak_weight_alpha" keys.
    """
    if "xgb_params" in params and "decay_params" in params:
        normalized = dict(params)
        normalized.setdefault("peak_weight_alpha", 1.0)
        return normalized
    return {
        "xgb_params": dict(params),
        "decay_params": dict(DEFAULT_DECAY_PARAMS),
        "peak_weight_alpha": 1.0,
    }


def resolve_params(
    app_id: int,
    n_trials: int = 150,
    auto_tune: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Resolves hyperparameters from storage (or a static fallback) and attaches the compiled
    custom objective function before returning.
    :param app_id: The Steam unique identifier for the application.
    :param n_trials: The number of Optuna trials to run per tier, if auto_tune triggers a tuning cycle.
    :param auto_tune: If True, runs ensure_up_to_date_params() to tune if needed before resolving.
    :param verbose: Regulates logging output.
    :return: A dict with "xgb_params" (including a compiled "objective" callable), "decay_params", and "peak_weight_alpha".
    """
    if auto_tune:
        params = ensure_up_to_date_params(app_id, n_trials=n_trials, verbose=verbose)
    else:
        regressor = Regressor(app_id=app_id)
        X, _, _ = regressor.generator.build_daily_matrix()
        raw_params = get_params_for_size(app_id, len(X))

        if raw_params is None:
            params = {
                "xgb_params": dict(XGB_BEST_PARAMS),
                "decay_params": dict(DEFAULT_DECAY_PARAMS),
            }
        else:
            params = _normalize_params(raw_params)

    # Attach the compiled custom loss closure. The code works on a copy: raw_params/DB-cached dicts
    # shouldn't be mutated in place, and callers (regressor_test.py, tuning_tester.py)
    # instantiate XGBRegressor(**params["xgb_params"]) directly, as asymmetric_alpha isn't a
    # real XGBoost parameter. It only exists to build `objective` below, so it must be popped
    # out here or every direct XGBRegressor(**xgb_params) call emits an "unused parameter"
    # warning.
    xgb_params = dict(params["xgb_params"])
    alpha = xgb_params.pop("asymmetric_alpha", 5.0)
    xgb_params["objective"] = make_asymmetric_capacity_objective(alpha=alpha)
    params = dict(params)
    params["xgb_params"] = xgb_params
    # Guaranteed regardless of which branch built `params` above: the auto_tune path returns
    # storage.get_params_for_size()'s raw dict directly (bypassing _normalize_params), and the
    # no-stored-params fallback dict predates this field.
    params.setdefault("peak_weight_alpha", 1.0)

    return params


if __name__ == "__main__":
    ensure_up_to_date_params(app_id=1086940, n_trials=150)