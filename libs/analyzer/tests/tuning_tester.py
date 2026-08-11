import sys
import time
from typing import Any, Dict, Optional
from xgboost import XGBRegressor

from analyzer.regressor import Regressor
from analyzer.regressor_tuner import resolve_params, run_tuning_cycle, _normalize_params, _format_duration
from storage.storage_utils import get_params_for_size


def run_tuning_tester(
        app_id: int,
        n_trials: int = 150,
        split_ratio: float = 0.70,
        decay_rate: Optional[float] = None,
        rule_decay_rate: Optional[float] = None,
        peak_weight_alpha: Optional[float] = None,
        force: bool = True,
        n_jobs: int = -1,
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs baseline benchmark, executes tuning cycle, and returns delta improvements,
    including weekday vs. weekend split performance and underestimation ratios.

    force=True by design: this is the end-to-end regression test for the tuning
    pipeline, so every tier is retrained from scratch on every run rather than
    reusing whatever happens to already be in storage_utils/history. That's what
    makes the baseline-vs-tuned delta meaningful here — do not flip this default.
    """
    def _resolve_decay(decay_params: Dict[str, float]) -> Dict[str, float]:
        resolved = dict(decay_params)
        if decay_rate is not None:
            resolved["decay_rate"] = decay_rate
        if rule_decay_rate is not None:
            resolved["rule_decay_rate"] = rule_decay_rate
        return resolved

    run_start = time.time()

    # 1. Baseline Benchmark using existing stored DB params (or default fallback)
    if verbose:
        print(f"\n--- 1. Evaluating Baseline Metrics for App ID {app_id} ---")
    baseline_params = resolve_params(app_id=app_id, auto_tune=False, verbose=verbose)
    baseline_model = XGBRegressor(**baseline_params["xgb_params"])
    baseline_regressor = Regressor(app_id=app_id, model=baseline_model)
    baseline_decay = _resolve_decay(baseline_params["decay_params"])
    baseline_alpha = peak_weight_alpha if peak_weight_alpha is not None else baseline_params.get("peak_weight_alpha", 1.0)

    baseline_metrics, _ = baseline_regressor.benchmark(
        split_ratio=split_ratio,
        peak_weight_alpha=baseline_alpha,
        **baseline_decay
    )
    if verbose:
        print(f"  baseline decay_params: {baseline_decay}")
        print(f"  baseline peak_weight_alpha: {baseline_alpha}")

    # 2. Run Optimization Cycle (Force Retraining — every tier, from scratch)
    if verbose:
        print(f"\n--- 2. Running Tuning Cycle ({n_trials} trials/tier, force={force}, n_jobs={n_jobs}) ---")
    tuning_start = time.time()
    run_tuning_cycle(app_id=app_id, n_trials=n_trials, force=force, verbose=verbose, n_jobs=n_jobs)
    if verbose:
        print(f"--- Tuning cycle wall time: {_format_duration(time.time() - tuning_start)} ---")

    # 3. Post-Tuning Benchmark
    if verbose:
        print(f"\n--- 3. Evaluating Post-Tuning Metrics ---")
    regressor = Regressor(app_id=app_id)
    X, _, _ = regressor.generator.build_daily_matrix()
    raw_tuned_params = get_params_for_size(app_id, len(X)) or baseline_params
    tuned_params = _normalize_params(raw_tuned_params)

    tuned_model = XGBRegressor(**tuned_params["xgb_params"])
    tuned_regressor = Regressor(app_id=app_id, model=tuned_model)
    tuned_decay = _resolve_decay(tuned_params["decay_params"])
    tuned_alpha = peak_weight_alpha if peak_weight_alpha is not None else tuned_params.get("peak_weight_alpha", 1.0)

    tuned_metrics, _ = tuned_regressor.benchmark(
        split_ratio=split_ratio,
        peak_weight_alpha=tuned_alpha,
        **tuned_decay
    )
    if verbose:
        print(f"  tuned decay_params: {tuned_decay}")
        print(f"  tuned peak_weight_alpha: {tuned_alpha}")

    # 4. Compute Metric Deltas
    deltas: Dict[str, float] = {}
    pct_improvements: Dict[str, float] = {}
    evaluated_keys = [
        "mae_1step", "rmse_1step", "mape_1step_pct",
        "mae_multistep", "rmse_multistep", "mape_multistep_pct",
        "mape_multistep_weekday_pct", "mape_multistep_weekend_pct",
        "underestimation_pct", "underestimation_weekend_pct"
    ]

    for key in evaluated_keys:
        b_val = baseline_metrics.get(key, 0.0)
        t_val = tuned_metrics.get(key, 0.0)

        # Delta = Baseline - Tuned (Positive delta indicates error reduction / improvement)
        delta = b_val - t_val
        pct_imp = (delta / b_val * 100.0) if b_val != 0 else 0.0

        deltas[key] = delta
        pct_improvements[f"{key}_pct_imp"] = pct_imp

    if verbose:
        print(f"\n=== Tuning Delta Improvement Summary (App ID: {app_id}) ===")
        print(f"{'Metric':<30} | {'Baseline':<10} | {'Tuned':<10} | {'Delta':<10} | {'Improvement %':<12}")
        print("-" * 82)
        for key in evaluated_keys:
            b_val = baseline_metrics.get(key, 0.0)
            t_val = tuned_metrics.get(key, 0.0)
            d_val = deltas[key]
            p_val = pct_improvements[f"{key}_pct_imp"]
            print(f"{key:<30} | {b_val:<10.4f} | {t_val:<10.4f} | {d_val:<+10.4f} | {p_val:<+11.2f}%")
        print(f"\nrule_decay_rate: {baseline_decay.get('rule_decay_rate')} -> {tuned_decay.get('rule_decay_rate')}")
        print(f"\nTotal run time: {_format_duration(time.time() - run_start)}")

    return {
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "baseline_decay_params": baseline_decay,
        "tuned_decay_params": tuned_decay,
        "deltas": deltas,
        "pct_improvements": pct_improvements,
        "total_run_seconds": time.time() - run_start,
    }


if __name__ == "__main__":
    app_id_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1086940
    run_tuning_tester(app_id=app_id_arg, n_trials=150, force=True)