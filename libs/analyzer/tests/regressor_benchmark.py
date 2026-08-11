import sys
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from xgboost import XGBRegressor

from analyzer.regressor import Regressor
from analyzer.regressor_tuner import resolve_params

def run_regressor_benchmark(
    app_id: int,
    split_ratio: float = 0.70,
    decay_rate: Optional[float] = None,
    rule_decay_rate: Optional[float] = None,
    peak_weight_alpha: Optional[float] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluates benchmark metrics for a regressor using stored hyperparameters,
    including weekday vs. weekend split performance and underestimation ratios.
    """
    params = resolve_params(app_id=app_id, auto_tune=False, verbose=verbose)
    model = XGBRegressor(**params["xgb_params"])
    regressor = Regressor(app_id=app_id, model=model)

    decay_params = dict(params["decay_params"])
    if decay_rate is not None:
        decay_params["decay_rate"] = decay_rate
    if rule_decay_rate is not None:
        decay_params["rule_decay_rate"] = rule_decay_rate

    # Extract peak_weight_alpha from params or fallback to default 1.0
    alpha = peak_weight_alpha if peak_weight_alpha is not None else params["xgb_params"].get("peak_weight_alpha", 1.0)

    metrics, benchmark_df = regressor.benchmark(
        split_ratio=split_ratio,
        peak_weight_alpha=alpha,
        **decay_params,
    )

    if verbose:
        print(f"\n=== Benchmark Metrics for App ID: {app_id} ===")
        print(f"  decay_params used     : {decay_params}")
        print(f"  peak_weight_alpha used: {alpha}")
        print("-" * 50)
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key:<30}: {val:.4f}")
            else:
                print(f"  {key:<30}: {val}")

    return metrics, benchmark_df

if __name__ == "__main__":
    app_id_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1086940
    run_regressor_benchmark(app_id=app_id_arg)