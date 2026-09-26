import pandas as pd
import sys
from analyzer.regressor import Regressor
from analyzer.regressor_tuner import resolve_params
from pathlib import Path
from typing import Optional
from xgboost import XGBRegressor


def run_regressor_test(
    app_id: int,
    days: int = 30,
    decay_rate: Optional[float] = None,
    rule_decay_rate: Optional[float] = None,
    output_dir: str = "./out",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Executes a forecast and writes result chart data to <output_dir>/<app_id>.csv.

    decay_rate / rule_decay_rate: se omessi (None), vengono usati i decay_params
    associati ai parametri risolti (tunati, se presenti, altrimenti i default).
    """
    params = resolve_params(app_id=app_id, auto_tune=False, verbose=verbose)
    model = XGBRegressor(**params["xgb_params"])
    regressor = Regressor(app_id=app_id, model=model)

    decay_params = dict(params["decay_params"])
    if decay_rate is not None:
        decay_params["decay_rate"] = decay_rate
    if rule_decay_rate is not None:
        decay_params["rule_decay_rate"] = rule_decay_rate

    peak_weight_alpha = params.get("peak_weight_alpha", 1.0)

    forecast_df = regressor.forecast_days(
        days=days,
        export_chart=True,
        output_dir=output_dir,
        peak_weight_alpha=peak_weight_alpha,
        **decay_params,
    )

    out_file = Path(output_dir) / f"{app_id}.csv"
    if verbose:
        print(f"\n[App {app_id}] {days}-day forecast completed.")
        print(f"decay_params used: {decay_params}")
        print(f"peak_weight_alpha used: {peak_weight_alpha}")
        print(f"Chart output generated: {out_file.resolve()}")

    return forecast_df

if __name__ == "__main__":
    app_id_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1086940
    run_regressor_test(app_id=app_id_arg, days=30, output_dir="./out")