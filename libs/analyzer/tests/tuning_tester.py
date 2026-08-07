import sys
from typing import Any, Dict
from xgboost import XGBRegressor

from analyzer.regressor import Regressor
from analyzer.regressor_tuner import resolve_params, run_tuning_cycle
from storage.storage_utils import get_params_for_size


def run_tuning_tester(
        app_id: int,
        n_trials: int = 150,
        split_ratio: float = 0.70,
        decay_rate: float = 0.95,
        force: bool = True,  # <-- Added parameter with default True
        verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs baseline benchmark, executes tuning cycle, and returns delta improvements.
    """
    # 1. Baseline Benchmark using existing stored DB params (or default fallback)
    if verbose:
        print(f"\n--- 1. Evaluating Baseline Metrics for App ID {app_id} ---")
    baseline_params = resolve_params(app_id=app_id, auto_tune=False, verbose=verbose)
    baseline_model = XGBRegressor(**baseline_params)
    baseline_regressor = Regressor(app_id=app_id, model=baseline_model)
    baseline_metrics, _ = baseline_regressor.benchmark(
        split_ratio=split_ratio, decay_rate=decay_rate
    )

    # 2. Run Optimization Cycle (Force Retraining)
    if verbose:
        print(f"\n--- 2. Running Tuning Cycle ({n_trials} trials, force={force}) ---")
    run_tuning_cycle(app_id=app_id, n_trials=n_trials, force=force, verbose=verbose)  # <-- Passed force=force

    # 3. Post-Tuning Benchmark
    if verbose:
        print(f"\n--- 3. Evaluating Post-Tuning Metrics ---")
    regressor = Regressor(app_id=app_id)
    X, _, _ = regressor.generator.build_daily_matrix()
    tuned_params = get_params_for_size(app_id, len(X)) or baseline_params

    tuned_model = XGBRegressor(**tuned_params)
    tuned_regressor = Regressor(app_id=app_id, model=tuned_model)
    tuned_metrics, _ = tuned_regressor.benchmark(
        split_ratio=split_ratio, decay_rate=decay_rate
    )

    # 4. Compute Metric Deltas
    deltas: Dict[str, float] = {}
    pct_improvements: Dict[str, float] = {}
    evaluated_keys = [
        "mae_1step", "rmse_1step", "mape_1step_pct",
        "mae_multistep", "rmse_multistep", "mape_multistep_pct"
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
        print(f"{'Metric':<20} | {'Baseline':<10} | {'Tuned':<10} | {'Delta':<10} | {'Improvement %':<12}")
        print("-" * 72)
        for key in evaluated_keys:
            b_val = baseline_metrics.get(key, 0.0)
            t_val = tuned_metrics.get(key, 0.0)
            d_val = deltas[key]
            p_val = pct_improvements[f"{key}_pct_imp"]
            print(f"{key:<20} | {b_val:<10.4f} | {t_val:<10.4f} | {d_val:<+10.4f} | {p_val:<+11.2f}%")

    return {
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "deltas": deltas,
        "pct_improvements": pct_improvements,
    }


if __name__ == "__main__":
    app_id_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1086940
    run_tuning_tester(app_id=app_id_arg, n_trials=150, force=True)