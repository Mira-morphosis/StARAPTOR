import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from storage.storage_utils import db_connection


class MatrixGenerator:
    """
    Classe responsabile della generazione e strutturazione della Feature Matrix X
    e del Vettore Target Y aggregate su base giornaliera.
    """

    def __init__(self, app_id: int):
        self.app_id = app_id

    def build_daily_matrix(self, run_id: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Estrae i dati da DuckDB e costruisce la matrice di feature giornaliera (X)
        e il target (Y = mean_concurrent al giorno t+1).
        """
        with db_connection(self.app_id) as conn:
            # 1. Carica la serie temporale delle KPI giornaliere
            kpi_df: pd.DataFrame = conn.execute("""
                                                SELECT date, mean_concurrent, pct_change
                                                FROM daily_kpis
                                                ORDER BY date ASC
                                                """).df()

            if kpi_df.empty:
                raise ValueError(f"Nessun dato KPI trovato per l'App ID {self.app_id}.")

            # 2. Aggrega le recensioni su base giornaliera (12 Pillars + Sentiment Ratio)
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

            # 3. Carica le regole dall'FP-Growth
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

        # Merge KPI e Recensioni
        df: pd.DataFrame = pd.merge(kpi_df, reviews_df, on="date", how="left").fillna(0)

        # 4. Aggiunta dei Lag Autoregressivi sulla KPI (Tipizzazione esplicita delle colonne)
        kpi_series: pd.Series = df['mean_concurrent']
        df['kpi_lag_1'] = kpi_series.shift(1)
        df['kpi_lag_7'] = kpi_series.shift(7)

        # 1. Rolling Window Statistics (shifted by 1 to prevent data leakage)
        df['kpi_roll_7_mean'] = kpi_series.shift(1).rolling(window=7, min_periods=1).mean()
        df['kpi_roll_7_max'] = kpi_series.shift(1).rolling(window=7, min_periods=1).max()

        # 2. Calendar features
        df['dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

        # 3. Peak Interaction Feature: Forces XGBoost to weigh last week's same-day peak when predicting weekends
        df['lag7_weekend_interaction'] = df['kpi_lag_7'] * df['is_weekend']

        # 5. Mappatura delle regole come colonne dinamiche
        if not rules_df.empty:
            rules_df['rule_feature'] = "rule_lift_" + rules_df['antecedents'].astype(str) + "_->" + rules_df[
                'consequents'].astype(str)
            rules_pivot = rules_df.pivot_table(
                index='date', columns='rule_feature', values='lift', aggfunc='max'
            ).fillna(0)

            df = pd.merge(df, rules_pivot, on="date", how="left").fillna(0)

        df['date'] = pd.to_datetime(df['date'])


        # 6. Definizione del Target Y (mean_concurrent del giorno t+1)
        target_shift: pd.Series = kpi_series.shift(-1)
        df['target_y'] = target_shift

        # Pulizia righe incomplete a causa dei lags/lead
        df_clean: pd.DataFrame = df.dropna(subset=['target_y']).reset_index(drop=True)

        # Estrazione X e Y
        ignore_cols = {'date', 'target_y'}
        feature_cols: List[str] = [str(c) for c in df_clean.columns if c not in ignore_cols]

        X: pd.DataFrame = df_clean[feature_cols].copy()
        Y: pd.Series = df_clean['target_y'].copy()

        return X, Y, df_clean