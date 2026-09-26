import numpy as np
import pandas as pd
from storage.storage_utils import db_connection
from typing import Tuple, List, Optional


class MatrixGenerator:
    """Builds the daily feature matrix (X) and forecasting target (Y) used to train the regressor."""

    def __init__(self, app_id: int):
        self.app_id = app_id

    def build_daily_matrix(self, run_id: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Assembles the daily feature matrix and target series from KPI, review, and association-rule data.
        :param run_id: If provided, only association rules from this mining run are used as features; otherwise all runs are used.
        :return: A tuple (X, Y, df_clean) where X is the feature matrix, Y is the log-ratio target, and df_clean is the full merged frame (including target_y_abs, kept for evaluation only).
        """
        with db_connection(self.app_id) as conn:
            # 1. Daily KPI time series
            kpi_df: pd.DataFrame = conn.execute("""
                                                SELECT date, mean_concurrent, pct_change
                                                FROM daily_kpis
                                                ORDER BY date ASC
                                                """).df()

            if kpi_df.empty:
                raise ValueError(f"No KPI data found for App ID {self.app_id}.")

            # 2. Daily review aggregates (12 pillars + sentiment ratio)
            reviews_df: pd.DataFrame = conn.execute("""
                                                    SELECT CAST(to_timestamp(timestamp_updated) AS DATE) AS date,
                    COUNT(*) AS daily_review_count,
                    AVG(multiplayer) AS avg_multiplayer,
                    AVG(immersion) AS avg_immersion,
                    AVG(community) AS avg_community,
                    AVG(replayability) AS avg_replayability,
                    AVG(story) AS avg_story,
                    AVG(monetizationModel) AS avg_monetization,
                    AVG(gameplay) AS avg_gameplay,
                    AVG(controls) AS avg_controls,
                    AVG(graphics) AS avg_graphics,
                    AVG(customization) AS avg_customization,
                    AVG(audio) AS avg_audio,
                    AVG(difficulty) AS avg_difficulty,
                    AVG(CASE WHEN voted_up THEN 1.0 ELSE 0.0 END) AS positive_ratio
                                                    FROM reviews
                                                    WHERE isSpam = FALSE
                                                    GROUP BY date
                                                    """).df()

            # 3. FP-Growth association rules
            if run_id:
                rules_query = """
                              SELECT r.run_id, m.timestamp::DATE as date, r.antecedents, r.consequents, r.lift
                              FROM association_rules r
                                  JOIN mining_runs m \
                              ON r.run_id = m.run_id
                              WHERE r.run_id = ? \
                              """
                rules_df: pd.DataFrame = conn.execute(rules_query, [run_id]).df()
            else:
                rules_query = """
                              SELECT r.run_id, m.timestamp::DATE as date, r.antecedents, r.consequents, r.lift
                              FROM association_rules r
                                  JOIN mining_runs m \
                              ON r.run_id = m.run_id \
                              """
                rules_df: pd.DataFrame = conn.execute(rules_query).df()

        # Merge KPI and review data
        df: pd.DataFrame = pd.merge(kpi_df, reviews_df, on="date", how="left").fillna(0)

        kpi_series: pd.Series = df['mean_concurrent']
        df['kpi_lag_1'] = kpi_series.shift(1)
        df['kpi_lag_7'] = kpi_series.shift(7)
        df['kpi_lag_14'] = kpi_series.shift(14)

        # Rolling window statistics
        df['kpi_roll_7_mean'] = kpi_series.shift(1).rolling(window=7, min_periods=1).mean()
        df['kpi_roll_7_max'] = kpi_series.shift(1).rolling(window=7, min_periods=1).max()

        # Calendar features
        df['dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

        # Peak interaction features
        df['lag7_weekend_interaction'] = df['kpi_lag_7'] * df['is_weekend']

        df['weekend_boost_ratio'] = (df['kpi_lag_7'] / (df['kpi_roll_7_mean'] + 1e-5)) * df['is_weekend']

        # 5. Pivot association rules into dynamic per-rule columns
        if not rules_df.empty:
            rules_df['rule_feature'] = "rule_lift_" + rules_df['antecedents'].astype(str) + "_->" + rules_df[
                'consequents'].astype(str)
            rules_pivot = rules_df.pivot_table(
                index='date', columns='rule_feature', values='lift', aggfunc='max'
            ).fillna(0)

            df = pd.merge(df, rules_pivot, on="date", how="left").fillna(0)

        df['date'] = pd.to_datetime(df['date'])


        # 6. Target Y is the log-ratio between tomorrow's and today's KPI level (not the raw level).
        #
        # XGBoost is a tree model, so it predicts by averaging training
        # examples that "look similar": it cannot extrapolate past the largest value it has
        # ever seen. If we trained directly on player-count levels, the model would be blind
        # to any peak higher than its training data. A log-ratio ("player count is up 30%")
        # is a much more repeatable pattern across different absolute scales than "63,552
        # players exactly", so training on it generalizes better to new highs and lows.
        # 'target_y_abs' is the actual player-count level, kept only for scoring predictions
        # afterward, and it must never be used as a training feature.
        target_shift: pd.Series = kpi_series.shift(-1)
        df['target_y_abs'] = target_shift
        with np.errstate(divide='ignore', invalid='ignore'):
            df['target_y'] = np.log(target_shift / kpi_series.replace(0, np.nan))

        # Drop rows made incomplete by the lag/shift windows above (also drops target_y_abs==0
        # rows, since a zero level makes the log-ratio NaN/inf).
        df_clean: pd.DataFrame = df.dropna(subset=['target_y']).reset_index(drop=True)

        # target_y_abs is explicitly excluded from the feature set; it's benchmark()-only.
        ignore_cols = {'date', 'target_y', 'target_y_abs'}
        feature_cols: List[str] = [str(c) for c in df_clean.columns if c not in ignore_cols]

        X: pd.DataFrame = df_clean[feature_cols].copy()
        Y: pd.Series = df_clean['target_y'].copy()

        return X, Y, df_clean