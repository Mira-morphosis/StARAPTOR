import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from analyzer.matrix_generator import MatrixGenerator
from storage.storage_utils import DB_ROOT


def make_asymmetric_capacity_objective(alpha: float = 5.0):
    """
    Builds a custom XGBoost objective that penalizes underestimation more heavily than overestimation.

    A normal regression loss treats "predicted too high" and "predicted too
    low" the same. For server-capacity planning, predicting too few players is worse than
    predicting too many — under-provisioning causes outages, over-provisioning just costs a
    bit extra. This objective multiplies the gradient/hessian by `alpha` whenever the model
    underestimates, so XGBoost is pushed to err on the side of predicting higher.
    :param alpha: The penalty multiplier applied to underestimation errors (>1.0 biases toward higher predictions).
    :return: A callable (y_true, y_pred, sample_weight=None) -> (gradient, hessian), matching XGBRegressor's expected objective signature.
    """

    def custom_objective(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        errors = y_pred - y_true  # positive error = overestimation, negative error = underestimation

        is_over = errors >= 0
        weights = np.where(is_over, 1.0, alpha)

        grad = (errors * weights).astype(np.float32)
        hess = np.float32(weights)

        if sample_weight is not None and len(sample_weight) > 0:
            grad *= sample_weight
            hess *= sample_weight

        return grad, hess

    return custom_objective

XGB_BEST_PARAMS: Dict[str, Any] = {
    "n_estimators": 150,
    "max_depth": 6,
    "learning_rate": 0.04406053468266097,
    "subsample": 0.8458975541792212,
    "colsample_bytree": 0.9864080480804699,
    "reg_alpha": 3.3707760847157993e-07,
    "reg_lambda": 1.9781349361493368,
    "random_state": 42,
    "n_jobs": 1
}

DEFAULT_DECAY_PARAMS: Dict[str, float] = {
    "decay_rate": 0.95,
    "rule_decay_rate": 0.995,
}

class Regressor:
    """Trains an XGBoost model on the daily feature matrix and produces multi-day CCU forecasts."""

    def __init__(self, app_id: int, model: Optional[XGBRegressor] = None, capacity_mode: bool = True):
        self.app_id = app_id
        self.generator = MatrixGenerator(app_id)
        self.capacity_mode = capacity_mode

        if model is not None:
            self.model = model
        else:
            params = dict(XGB_BEST_PARAMS)
            if self.capacity_mode:
                alpha = params.get("asymmetric_alpha", 5.0)
                params["objective"] = make_asymmetric_capacity_objective(alpha=alpha)
            self.model = XGBRegressor(**params)

        self.is_trained: bool = False
        self.feature_names: List[str] = []
        self.residual_var = None

    def fit(
            self,
            run_id: Optional[str] = None,
            data_tuple: Optional[Tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = None,
            peak_weight_alpha: float = 1.0
    ) -> Dict[str, Any]:
        """
        Trains the model on the daily feature matrix, weighting weekend samples by their natural boost.
        :param run_id: If provided, restricts association-rule features to this mining run.
        :param data_tuple: Pre-built (X, Y, df_full) to train on, bypassing a fresh matrix build.
        :param peak_weight_alpha: Scales how much extra sample weight weekend rows get, based on their observed CCU boost over weekdays.
        :return: A dict with "samples", "features_count", and "feature_names", or {} if there was no data to train on.
        """
        if data_tuple is not None:
            X, Y, df_full = data_tuple
        else:
            X, Y, df_full = self.generator.build_daily_matrix(run_id=run_id)

        if X.empty:
            return {}

        self.feature_names = [str(col) for col in X.columns]

        sample_weights = None
        if 'is_weekend' in X.columns and 'mean_concurrent' in X.columns:
            wk_mask = X['is_weekend'] == 1
            wk_mean = X.loc[wk_mask, 'mean_concurrent'].mean()
            weekday_mean = X.loc[~wk_mask, 'mean_concurrent'].mean()

            natural_boost = max(0.0, (wk_mean / (weekday_mean + 1e-5)) - 1.0)
            wk_weight = 1.0 + (peak_weight_alpha * natural_boost)
            sample_weights = np.where(wk_mask, wk_weight, 1.0)

        alpha_param = self.model.get_params().get("asymmetric_alpha", 5.0)
        if self.capacity_mode:
            self.model.set_params(objective=make_asymmetric_capacity_objective(alpha=alpha_param))

        self.model.fit(X, Y, sample_weight=sample_weights)
        self.is_trained = True

        train_preds = self.model.predict(X)
        residuals = Y - train_preds
        self.residual_var = float(np.var(residuals))

        return {
            "samples": len(X),
            "features_count": len(self.feature_names),
            "feature_names": self.feature_names
        }

    def forecast_days(
            self,
            run_id=None,
            decay_rate=0.95,
            rule_decay_rate=0.995,
            days=30,
            data_tuple=None,
            export_chart: bool = True,
            output_dir: None | str | Path = None,
            safety_buffer: float = 1.00,  # e.g. 1.05 for +5% operational headroom
            peak_weight_alpha: float = 1.0
    ) -> pd.DataFrame:
        """
        Recursively forecasts daily mean concurrent players for a number of days ahead.

        The model predicts one day ahead at a time (a log-ratio), and each
        prediction is fed back in as the new "yesterday" to predict the day after that. Since
        errors compound step by step, a naive multi-step forecast tends to run systematically
        low (a consequence of Jensen's inequality: averaging in log-space and then
        exponentiating underestimates the true expected level). The single-step training
        residual variance is used to apply a small upward correction that grows with the
        forecast horizon, counteracting that bias without ever feeding the corrected value
        back into the recursion (which would make the bug compound instead of just appearing
        once).
        :param run_id: If provided, restricts association-rule features to this mining run.
        :param decay_rate: Per-day decay applied to sentiment-derived features as they drift toward their historical baseline.
        :param rule_decay_rate: Per-day decay applied to association-rule features as they drift toward their historical baseline.
        :param days: The number of days ahead to forecast.
        :param data_tuple: Pre-built (X, Y, df_full) to forecast from, bypassing a fresh matrix build.
        :param export_chart: Whether to write the forecast to a CSV chart file.
        :param output_dir: Directory to write the chart CSV to; defaults to DB_ROOT/charts.
        :param safety_buffer: A flat multiplier applied to the reported forecast (e.g. 1.05 for +5% headroom).
        :param peak_weight_alpha: Passed through to fit() if the model hasn't been trained yet.
        :return: A DataFrame with "date" and "predicted_mean_concurrent" columns, one row per forecast day.
        """
        if data_tuple is not None:
            X, Y, df_full = data_tuple
        else:
            X, Y, df_full = self.generator.build_daily_matrix(run_id=run_id)

        if not self.is_trained:
            self.fit(run_id=run_id, data_tuple=(X, Y, df_full), peak_weight_alpha=peak_weight_alpha)

        if df_full.empty:
            return pd.DataFrame()

        last_date = pd.to_datetime(str(df_full.iloc[-1]['date']))
        columns = list(X.columns)
        col_idx = {c: i for i, c in enumerate(columns)}
        current_row = X.iloc[-1].to_numpy(dtype=float).copy()

        sentiment_decay_idx = [col_idx[c] for c in columns if c.startswith('avg_') or c == 'positive_ratio']
        rule_decay_idx = [col_idx[c] for c in columns if c.startswith('rule_lift_')]

        sentiment_baseline = X.iloc[:, sentiment_decay_idx].mean().to_numpy() if sentiment_decay_idx else None
        rule_baseline = X.iloc[:, rule_decay_idx].mean().to_numpy() if rule_decay_idx else None

        kpi_history: List[float] = [float(v) for v in df_full['mean_concurrent'].values]
        predictions: List[float] = []
        future_dates: List[pd.Timestamp] = []

        lag1_i = col_idx.get('kpi_lag_1')
        lag7_i = col_idx.get('kpi_lag_7')
        roll_mean_i = col_idx.get('kpi_roll_7_mean')
        roll_max_i = col_idx.get('kpi_roll_7_max')
        inter_i = col_idx.get('lag7_weekend_interaction')
        pct_i = col_idx.get('pct_change')
        dow_i = col_idx.get('dayofweek')
        weekend_i = col_idx.get('is_weekend')
        sin_i = col_idx.get('dow_sin')
        cos_i = col_idx.get('dow_cos')
        level_i = col_idx.get('mean_concurrent')

        current_level = kpi_history[-1]

        # Bias correction: a t-day-ahead forecast is t chained single-step log-ratio
        # predictions. Under the standard independent-steps approximation (the same
        # assumption behind a geometric random walk), the variance of the t-step-ahead
        # log-level grows linearly with t: Var(log level_t) ~ t * residual_var. The
        # correct Jensen's-inequality correction at horizon t is therefore
        # exp(t * residual_var / 2), applied per absolute step rather than depending
        # on the total number of days requested.
        #
        # This is still an approximation: this is a conservative (upper-bound-ish) choice, not a
        # guarantee of zero long-horizon drift.
        residual_var = self.residual_var if self.residual_var is not None else 0.0

        for day in range(1, days + 1):
            next_date = last_date + pd.Timedelta(days=day)
            future_dates.append(next_date)

            pred_log_ratio = float(self.model.predict(current_row.reshape(1, -1))[0])
            pred_log_ratio = float(np.clip(pred_log_ratio, -2.0, 2.0))

            # Raw/median next level: no bias correction, no safety buffer. This feeds
            # the recursion (lags, rolling stats, level feature), keeping it consistent
            # with what the model actually saw during training.
            raw_level = max(0.0, current_level * float(np.exp(pred_log_ratio)))

            # Reported forecast: horizon-aware correction applied once, never fed back
            # into the loop, so it cannot compound.
            step_correction = float(np.exp(day * residual_var / 2.0))
            reported_kpi = max(0.0, raw_level * step_correction * safety_buffer)

            predictions.append(reported_kpi)
            kpi_history.append(raw_level)
            current_level = raw_level

            if level_i is not None:
                current_row[level_i] = current_level
            if lag1_i is not None:
                current_row[lag1_i] = kpi_history[-1]
            if lag7_i is not None:
                current_row[lag7_i] = kpi_history[-7] if len(kpi_history) >= 7 else kpi_history[0]
            if roll_mean_i is not None:
                current_row[roll_mean_i] = float(np.mean(kpi_history[-7:]))
            if roll_max_i is not None:
                current_row[roll_max_i] = float(np.max(kpi_history[-7:]))
            if pct_i is not None and len(kpi_history) >= 2:
                prev_val = kpi_history[-2]
                current_row[pct_i] = (raw_level - prev_val) / prev_val if prev_val != 0 else 0.0

            if dow_i is not None:
                dow = next_date.dayofweek
                is_wk = 1 if dow in (5, 6) else 0
                current_row[dow_i] = dow
                if weekend_i is not None:
                    current_row[weekend_i] = is_wk
                if sin_i is not None:
                    current_row[sin_i] = np.sin(2 * np.pi * dow / 7.0)
                if cos_i is not None:
                    current_row[cos_i] = np.cos(2 * np.pi * dow / 7.0)
                if inter_i is not None and lag7_i is not None:
                    current_row[inter_i] = current_row[lag7_i] * is_wk

            if sentiment_decay_idx and sentiment_baseline is not None:
                current_row[sentiment_decay_idx] = (
                    current_row[sentiment_decay_idx] * decay_rate + sentiment_baseline * (1.0 - decay_rate)
                )
            if rule_decay_idx and rule_baseline is not None:
                current_row[rule_decay_idx] = (
                    current_row[rule_decay_idx] * rule_decay_rate + rule_baseline * (1.0 - rule_decay_rate)
                )

        forecast_df = pd.DataFrame({"date": future_dates, "predicted_mean_concurrent": predictions})

        if export_chart and not forecast_df.empty:
            charts_dir = Path(output_dir) if output_dir is not None else Path(DB_ROOT).expanduser() / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)

            chart_df = forecast_df.rename(columns={"date": "timestamp", "predicted_mean_concurrent": "kpi"})[["timestamp", "kpi"]]
            chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            chart_df.to_csv(charts_dir / f"{self.app_id}.csv", index=False)

        return forecast_df

    def benchmark(
            self,
            model: Optional[XGBRegressor] = None,
            split_ratio: float = 0.70,
            decay_rate: float = 0.95,
            rule_decay_rate: float = 0.995,
            peak_weight_alpha: float = 1.0,
            eval_days: Optional[int] = None,
            run_id: Optional[str] = None,
            data_tuple: Optional[Tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Trains on a leading split of the data and evaluates both 1-step and multi-step forecast accuracy on the rest.
        :param model: The XGBRegressor to benchmark; defaults to self.model.
        :param split_ratio: The fraction of rows used for training, in chronological order.
        :param decay_rate: Passed through to forecast_days().
        :param rule_decay_rate: Passed through to forecast_days().
        :param peak_weight_alpha: Passed through to fit() and forecast_days().
        :param eval_days: Truncates the evaluation horizon to this many days, for faster benchmarking during tuning.
        :param run_id: If provided, restricts association-rule features to this mining run.
        :param data_tuple: Pre-built (X, Y, df_full) to benchmark on, bypassing a fresh matrix build.
        :return: A tuple (metrics dict, benchmark_df) with per-day actual vs. predicted values and error columns.
        """
        if data_tuple is not None:
            X, Y, df_full = data_tuple
        else:
            X, Y, df_full = self.generator.build_daily_matrix(run_id=run_id)

        if X.empty or len(X) < 10:
            return {}, pd.DataFrame()

        train_size = int(len(X) * split_ratio)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        Y_train, Y_test = Y.iloc[:train_size], Y.iloc[train_size:]
        df_train = df_full.iloc[:train_size].copy()
        df_test = df_full.iloc[train_size:].copy()

        target_model = model if model is not None else self.model
        bench_model = copy.deepcopy(target_model)
        bench_regressor = Regressor(app_id=self.app_id, model=bench_model, capacity_mode=self.capacity_mode)

        bench_regressor.feature_names = [str(col) for col in X_train.columns]
        bench_regressor.fit(
            data_tuple=(X_train, Y_train, df_train),
            peak_weight_alpha=peak_weight_alpha
        )
        bench_regressor.is_trained = True

        preds_1s_log = bench_model.predict(X_test)
        anchor_levels_1s = X_test['mean_concurrent'].to_numpy()
        preds_1s_arr = np.maximum(0.0, anchor_levels_1s * np.exp(np.clip(preds_1s_log, -2.0, 2.0)))

        horizon = min(len(X_test), eval_days) if eval_days is not None else len(X_test)

        forecast_df = bench_regressor.forecast_days(
            run_id=run_id,
            decay_rate=decay_rate,
            rule_decay_rate=rule_decay_rate,
            days=horizon,
            data_tuple=(X_train, Y_train, df_train),
            export_chart=False
        )

        preds_ms_arr = forecast_df["predicted_mean_concurrent"].to_numpy()

        df_test_eval = df_test.iloc[:horizon]
        preds_1s_arr = preds_1s_arr[:horizon]
        actual_kpi_arr = df_test_eval["target_y_abs"].to_numpy()

        benchmark_df = pd.DataFrame({
            "date": df_test_eval["date"].values,
            "actual_kpi": actual_kpi_arr,
            "pred_1step": preds_1s_arr,
            "abs_err_1step": np.abs(actual_kpi_arr - preds_1s_arr),
            "pred_multistep": preds_ms_arr,
            "abs_err_multistep": np.abs(actual_kpi_arr - preds_ms_arr),
            "is_weekend": df_test_eval["is_weekend"].values if "is_weekend" in df_test_eval.columns else 0
        })

        benchmark_df["pct_err_1step"] = np.where(
            benchmark_df["actual_kpi"] != 0,
            (benchmark_df["abs_err_1step"] / benchmark_df["actual_kpi"]) * 100,
            0.0
        )
        benchmark_df["pct_err_multistep"] = np.where(
            benchmark_df["actual_kpi"] != 0,
            (benchmark_df["abs_err_multistep"] / benchmark_df["actual_kpi"]) * 100,
            0.0
        )

        mae_1s_val = cast(float, mean_absolute_error(actual_kpi_arr, preds_1s_arr))
        mse_1s_val = cast(float, mean_squared_error(actual_kpi_arr, preds_1s_arr))
        rmse_1s_val = float(np.sqrt(mse_1s_val))

        mae_ms_val = cast(float, mean_absolute_error(actual_kpi_arr, preds_ms_arr))
        mse_ms_val = cast(float, mean_squared_error(actual_kpi_arr, preds_ms_arr))
        rmse_ms_val = float(np.sqrt(mse_ms_val))

        mape_1s_val = cast(float, np.mean(benchmark_df["pct_err_1step"].to_numpy()))
        mape_ms_val = cast(float, np.mean(benchmark_df["pct_err_multistep"].to_numpy()))

        wk_mask = benchmark_df["is_weekend"] == 1
        wd_mask = ~wk_mask

        raw_err_ms_arr = actual_kpi_arr - preds_ms_arr

        actual_wk = actual_kpi_arr[wk_mask]
        preds_ms_wk = preds_ms_arr[wk_mask]
        raw_err_wk = raw_err_ms_arr[wk_mask]

        mape_ms_weekend = float(np.mean(benchmark_df.loc[wk_mask, "pct_err_multistep"].to_numpy())) if np.any(
            wk_mask) else mape_ms_val
        underestimation_weekend_pct = float(
            np.mean(np.maximum(0.0, raw_err_wk / (actual_wk + 1e-5))) * 100.0
        ) if np.any(wk_mask) else 0.0

        mape_ms_weekday = float(np.mean(benchmark_df.loc[wd_mask, "pct_err_multistep"].to_numpy())) if np.any(
            wd_mask) else mape_ms_val

        metrics: Dict[str, Any] = {
            "train_samples": train_size,
            "test_samples": len(X_test),
            "mae_1step": mae_1s_val,
            "rmse_1step": rmse_1s_val,
            "mape_1step_pct": mape_1s_val,
            "mae_multistep": mae_ms_val,
            "rmse_multistep": rmse_ms_val,
            "mape_multistep_pct": mape_ms_val,
            "mape_multistep_weekday_pct": mape_ms_weekday,
            "mape_multistep_weekend_pct": mape_ms_weekend,
            "underestimation_pct": float(np.mean(np.maximum(0.0, raw_err_ms_arr / (actual_kpi_arr + 1e-5))) * 100.0),
            "underestimation_weekend_pct": underestimation_weekend_pct
        }
        return metrics, benchmark_df