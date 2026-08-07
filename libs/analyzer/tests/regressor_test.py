import sys
from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor

from analyzer.regressor import Regressor
from analyzer.regressor_tuner import resolve_params

def run_regressor_test(
    app_id: int,
    days: int = 30,
    decay_rate: float = 0.95,
    output_dir: str = "./out",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Executes a forecast and writes result chart data to <output_dir>/<app_id>.csv.
    """
    params = resolve_params(app_id=app_id, auto_tune=False, verbose=verbose)
    model = XGBRegressor(**params)
    regressor = Regressor(app_id=app_id, model=model)

    forecast_df = regressor.forecast_days(
        days=days,
        decay_rate=decay_rate,
        export_chart=True,
        output_dir=output_dir,
    )

    out_file = Path(output_dir) / f"{app_id}.csv"
    if verbose:
        print(f"\n[App {app_id}] {days}-day forecast completed.")
        print(f"Chart output generated: {out_file.resolve()}")

    return forecast_df

if __name__ == "__main__":
    app_id_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1086940
    run_regressor_test(app_id=app_id_arg, days=30, output_dir="./out")