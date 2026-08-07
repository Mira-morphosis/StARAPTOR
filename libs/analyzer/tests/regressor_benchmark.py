import sys
from typing import Any, Dict, Tuple
import pandas as pd
from xgboost import XGBRegressor

from analyzer.regressor import Regressor
from analyzer.regressor_tuner import resolve_params

def run_regressor_benchmark(
    app_id: int,
    split_ratio: float = 0.70,
    decay_rate: float = 0.95,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluates benchmark metrics for a regressor using stored hyperparameters.
    """
    params = resolve_params(app_id=app_id, auto_tune=False, verbose=verbose)
    model = XGBRegressor(**params)
    regressor = Regressor(app_id=app_id, model=model)

    metrics, benchmark_df = regressor.benchmark(
        split_ratio=split_ratio,
        decay_rate=decay_rate,
    )

    if verbose:
        print(f"\n=== Benchmark Metrics for App ID: {app_id} ===")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key:<22}: {val:.4f}")
            else:
                print(f"  {key:<22}: {val}")

    return metrics, benchmark_df

if __name__ == "__main__":
    app_id_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1086940
    run_regressor_benchmark(app_id=app_id_arg)