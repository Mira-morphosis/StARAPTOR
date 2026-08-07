import copy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from analyzer.matrix_generator import MatrixGenerator
from storage.storage_utils import DB_ROOT

# Costante globale con gli iperparametri ottimizzati tramite Optuna
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

class Regressor:
    """
    Modulo di Regressione XGBoost per la previsione a breve e lungo termine delle KPI.
    Gestisce il fitting del modello e la proiezione iterativa fino a 30 giorni tramite decay.
    """

    def __init__(self, app_id: int, model: Optional[XGBRegressor] = None):
        self.app_id = app_id
        self.generator = MatrixGenerator(app_id)
        self.model: XGBRegressor = model if model is not None else XGBRegressor(**XGB_BEST_PARAMS)
        self.is_trained: bool = False
        self.feature_names: List[str] = []

    def fit(
        self,
        run_id: Optional[str] = None,
        data_tuple: Optional[Tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Costruisce la matrice di dati (o usa quella fornita) e addestra il regressore.
        """
        if data_tuple is not None:
            X, Y, _ = data_tuple
        else:
            X, Y, _ = self.generator.build_daily_matrix(run_id=run_id)

        if X.empty:
            print("W: Feature matrix vuota. Impossibile completare il fit.")
            return {}

        self.feature_names = [str(col) for col in X.columns]
        self.model.fit(X, Y)
        self.is_trained = True

        return {
            "samples": len(X),
            "features_count": len(self.feature_names),
            "feature_names": self.feature_names
        }

    def predict_next_day(self, latest_x_row: pd.DataFrame) -> float:
        """
        Esegue la stima puntuale per il giorno t+1.
        """
        if not self.is_trained:
            raise RuntimeError("Il modello non è ancora stato addestrato. Chiamare .fit() prima.")

        preds = self.model.predict(latest_x_row)
        return float(preds[0])

    def forecast_days(
            self,
            run_id=None,
            decay_rate=0.95,
            days=30,
            data_tuple=None,
            export_chart:bool = True,
            output_dir: None|str|Path = None,) -> pd.DataFrame:
        if data_tuple is not None:
            X, Y, df_full = data_tuple
        else:
            X, Y, df_full = self.generator.build_daily_matrix(run_id=run_id)

        if not self.is_trained:
            self.fit(run_id=run_id, data_tuple=(X, Y, df_full))

        if df_full.empty:
            return pd.DataFrame()

        last_date = pd.to_datetime(str(df_full.iloc[-1]['date']))
        columns = list(X.columns)
        col_idx = {c: i for i, c in enumerate(columns)}
        current_row = X.iloc[-1].to_numpy(dtype=float).copy()  # 1D array, work in numpy from here

        decay_idx = [col_idx[c] for c in columns
                     if c.startswith('avg_') or c.startswith('rule_lift_') or c == 'positive_ratio']

        # Media storica per il decadimento verso la baseline
        baseline_means = X.iloc[:, decay_idx].mean().to_numpy() if decay_idx else None

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

        for day in range(1, days + 1):
            next_date = last_date + pd.Timedelta(days=day)
            future_dates.append(next_date)

            pred_kpi = max(0.0, float(self.model.predict(current_row.reshape(1, -1))[0]))
            predictions.append(pred_kpi)
            kpi_history.append(pred_kpi)

            # 1. Aggiornamento Lag Base
            if lag1_i is not None:
                current_row[lag1_i] = kpi_history[-1]
            if lag7_i is not None:
                current_row[lag7_i] = kpi_history[-7] if len(kpi_history) >= 7 else kpi_history[0]

            # 2. Aggiornamento dinamico delle statistiche su finestra mobile (7d)
            if roll_mean_i is not None:
                current_row[roll_mean_i] = float(np.mean(kpi_history[-7:]))
            if roll_max_i is not None:
                current_row[roll_max_i] = float(np.max(kpi_history[-7:]))

            if pct_i is not None and len(kpi_history) >= 2:
                prev_val = kpi_history[-2]
                current_row[pct_i] = (pred_kpi - prev_val) / prev_val if prev_val != 0 else 0.0

            # 3. Aggiornamento calendari e interazioni
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

            # 4. Decadimento verso la media storica di baseline (non verso zero)
            if decay_idx and baseline_means is not None:
                current_row[decay_idx] = (
                    current_row[decay_idx] * decay_rate + baseline_means * (1.0 - decay_rate)
                )

        forecast_df = pd.DataFrame({"date": future_dates, "predicted_mean_concurrent": predictions})
        if export_chart and not forecast_df.empty:
            if output_dir is not None:
                charts_dir = Path(output_dir)
            else:
                charts_dir = Path(DB_ROOT).expanduser() / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)

            chart_df = forecast_df.rename(
                columns={
                    "date": "timestamp",
                    "predicted_mean_concurrent": "kpi"
                }
            )[["timestamp", "kpi"]]

            chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            output_file = charts_dir / f"{self.app_id}.csv"
            chart_df.to_csv(output_file, index=False)

        return forecast_df


    def benchmark(
        self,
        model: Optional[XGBRegressor] = None,
        split_ratio: float = 0.70,
        decay_rate: float = 0.95,
        run_id: Optional[str] = None,
        data_tuple: Optional[Tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Esegue il benchmarking dividendo i dati in Train/Test.
        Vettorizzato per massime prestazioni durante l'ottimizzazione.
        """
        if data_tuple is not None:
            X, Y, df_full = data_tuple
        else:
            X, Y, df_full = self.generator.build_daily_matrix(run_id=run_id)

        if X.empty or len(X) < 10:
            print("W: Dati insufficienti per eseguire il benchmark.")
            return {}, pd.DataFrame()

        # 1. Split Temporale Train / Test
        train_size = int(len(X) * split_ratio)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        Y_train, Y_test = Y.iloc[:train_size], Y.iloc[train_size:]
        df_train = df_full.iloc[:train_size].copy()
        df_test = df_full.iloc[train_size:].copy()

        target_model = model if model is not None else self.model

        # 2. Clona il modello per non alterare lo stato dell'istanza principale
        bench_model = copy.deepcopy(target_model)
        bench_regressor = Regressor(app_id=self.app_id, model=bench_model)

        # Fit usando i dati in memoria
        bench_regressor.feature_names = [str(col) for col in X_train.columns]
        bench_regressor.model.fit(X_train, Y_train)
        bench_regressor.is_trained = True

        # 3. Predizioni 1-Step (VETTORIZZATE: ~100x più veloce rispetto al loop iloc)
        preds_1s_arr = bench_model.predict(X_test)

        # 4. Predizioni Multi-Step con decay
        forecast_df = bench_regressor.forecast_days(
            run_id=run_id,
            decay_rate=decay_rate,
            days=len(X_test),
            data_tuple=(X_train, Y_train, df_train),  # Passa i dati train per evitare re-query DB
            export_chart=False
        )

        preds_ms_arr = forecast_df["predicted_mean_concurrent"].to_numpy()

        # 5. Costruzione DataFrame degli scarti
        benchmark_df = pd.DataFrame({
            "date": df_test["date"].values,
            "actual_kpi": Y_test.values,
            "pred_1step": preds_1s_arr,
            "abs_err_1step": np.abs(Y_test.values - preds_1s_arr),
            "pred_multistep": preds_ms_arr,
            "abs_err_multistep": np.abs(Y_test.values - preds_ms_arr)
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

        # 6. Metriche di sintesi
        mae_1s_val = cast(float, mean_absolute_error(Y_test, preds_1s_arr))
        mse_1s_val = cast(float, mean_squared_error(Y_test, preds_1s_arr))
        rmse_1s_val = float(np.sqrt(mse_1s_val))

        mae_ms_val = cast(float, mean_absolute_error(Y_test, preds_ms_arr))
        mse_ms_val = cast(float, mean_squared_error(Y_test, preds_ms_arr))
        rmse_ms_val = float(np.sqrt(mse_ms_val))

        mape_1s_val = cast(float, np.mean(benchmark_df["pct_err_1step"].to_numpy()))
        mape_ms_val = cast(float, np.mean(benchmark_df["pct_err_multistep"].to_numpy()))

        metrics: Dict[str, Any] = {
            "train_samples": train_size,
            "test_samples": len(X_test),
            "mae_1step": mae_1s_val,
            "rmse_1step": rmse_1s_val,
            "mape_1step_pct": mape_1s_val,
            "mae_multistep": mae_ms_val,
            "rmse_multistep": rmse_ms_val,
            "mape_multistep_pct": mape_ms_val,
        }

        return metrics, benchmark_df