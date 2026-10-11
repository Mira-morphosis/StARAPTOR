import asyncio
import datetime
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from analyzer.analyzer import Analyzer
from analyzer.matrix_generator import MatrixGenerator
from analyzer.regressor import Regressor
from analyzer.regressor_tuner import ensure_up_to_date_params, run_tuning_cycle
from storage.storage_interface import kpi_updater, review_updater
from storage.storage_utils import (
    DB_ROOT,
    export_all_rules,
    clear_mining_rules,
    get_params_for_size,
)

logger = logging.getLogger("staraptor.orchestrator")

FORECAST_DAYS = int(os.getenv("STARAPTOR_FORECAST_DAYS", "45"))


class OperationLockedError(Exception):
    """Raised when an operation is requested while another one is already running."""
    pass


class OrchestratorService:
    def __init__(self):
        # Lock globale async per garantire l'esecuzione esclusiva
        self._lock = asyncio.Lock()
        self._progress: int = 0
        self._current_task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        """Restituisce True se un'operazione è attualmente in corso."""
        return self._lock.locked()

    @property
    def progress(self) -> int:
        """Restituisce la percentuale di avanzamento dinamica dell'operazione corrente (0-100)."""
        return self._progress

    def ensure_export_directory(self) -> str:
        """Assicura l'esistenza della cartella di export e ne restituisce il percorso."""
        export_path = Path(DB_ROOT).expanduser() / "exports"
        export_path.mkdir(parents=True, exist_ok=True)
        return str(export_path)

    def cancel(self):
        """Interrompe il task attualmente in esecuzione (se presente)."""
        if self._current_task and not self._current_task.done():
            logger.info("Richiesta di cancellazione ricevuta per il task orchestrator attivo.")
            self._current_task.cancel()

    def stop(self):
        """Alias per cancel() per compatibilità con l'API del banner."""
        self.cancel()

    async def run_tuning_pipeline(self, app_id: int, n_trials: int = 150, force: bool = False, verbose: bool = True):
        """Esegue il tuning degli iperparametri."""
        if self._lock.locked():
            raise OperationLockedError("Another operation is currently in progress.")

        async with self._lock:
            self._current_task = asyncio.current_task()
            try:
                self._progress = 10
                await asyncio.to_thread(self._sync_tuning_pipeline, app_id, n_trials, force, verbose)
                self._progress = 100
            except asyncio.CancelledError:
                logger.warning(f"Tuning pipeline annullata per App ID {app_id}.")
                raise
            finally:
                self._progress = 0
                self._current_task = None

    def _sync_tuning_pipeline(self, app_id: int, n_trials: int, force: bool, verbose: bool):
        if force:
            if verbose:
                print(f"--- Forced tuning for App ID {app_id} (all tiers) ---")
            run_tuning_cycle(app_id=app_id, n_trials=n_trials, verbose=verbose)
        else:
            ensure_up_to_date_params(app_id=app_id, n_trials=n_trials, verbose=verbose)

    async def run_initial_pipeline(
        self,
        app_id: int,
        max_days: int | None = None,
        verbose: bool = True,
        skip_tuning: bool = False,
    ):
        """Esegue la pipeline di inizializzazione completa (cold-start)."""
        if self._lock.locked():
            raise OperationLockedError("Another operation is currently in progress.")

        async with self._lock:
            self._current_task = asyncio.current_task()
            try:
                self._progress = 5

                if verbose:
                    print("\n==========================================")
                    print(f"  STARTING INITIALIZATION FOR APP ID: {app_id}")
                    print("==========================================\n")

                # Step 1: Download historical KPIs
                self._progress = 10
                if verbose:
                    print("--- Step 1/5: Downloading Historical KPIs ---")
                await kpi_updater(app_id=app_id, max_days=max_days, verbose=verbose)
                self._progress = 30

                # Step 2: Download reviews & execute LLM processing
                if verbose:
                    print("\n--- Step 2/5: Processing Historical Reviews ---")
                await review_updater(app_id=app_id, max_days=max_days, verbose=verbose)
                self._progress = 55

                # Step 3: Mine association rules
                if verbose:
                    print("\n--- Step 3/5: Mining Association Rules ---")
                analyzer = Analyzer(app_id)
                rules_df = analyzer.run(
                    min_support=0.02,
                    min_confidence=0.5,
                    min_lift=1.2,
                    deduplicate_mode="both",
                    save_to_db=True
                )
                self._progress = 75

                if rules_df.empty and verbose:
                    print("W: Mining completed, but no rules met the threshold criteria.")

                # Step 4: Hyperparameter tuning
                if not skip_tuning:
                    if verbose:
                        print("\n--- Step 4/5: Tuning XGBoost Hyperparameters ---")
                    try:
                        await asyncio.to_thread(self._sync_tuning_pipeline, app_id, 150, False, verbose)
                    except Exception as e:
                        if verbose:
                            print(f"W: Initial tuning failed, can be re-run manually: {e}")
                elif verbose:
                    print("\n--- Step 4/5: Tuning XGBoost Hyperparameters (skipped) ---")

                # Step 5: Export forecast CSV
                self._progress = 95
                if verbose:
                    print("\n--- Step 5/5: Exporting Forecast CSV ---")

                try:
                    await asyncio.to_thread(
                        self._sync_export_forecast,
                        app_id,
                        FORECAST_DAYS,
                        verbose
                    )
                except Exception as e:
                    if verbose:
                        print(f"W: Forecast CSV export failed: {e}")

                self._progress = 100

                if verbose:
                    print(f"\n[OK] Initialization complete for App ID {app_id}.\n")

            except asyncio.CancelledError:
                logger.warning(f"Pipeline di inizializzazione annullata per App ID {app_id}.")
                raise
            finally:
                self._progress = 0
                self._current_task = None

    async def run_daily_pipeline(self, app_id: int, verbose: bool = True):
        """Esegue il ciclo di aggiornamento giornaliero aggiornando la percentuale di progresso."""
        if self._lock.locked():
            raise OperationLockedError("Another operation is currently in progress.")

        async with self._lock:
            self._current_task = asyncio.current_task()
            try:
                self._progress = 5
                now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

                if verbose:
                    print(f"\n[{now}] Executing daily update cycle for App ID: {app_id}...")

                # Step 1: Update recent KPIs
                self._progress = 15
                await kpi_updater(app_id=app_id, max_days=7, verbose=verbose)
                self._progress = 30

                # Step 2: Update recent reviews
                if verbose:
                    print("\n--- Updating recent reviews ---")
                await review_updater(app_id=app_id, max_days=1, verbose=verbose)
                self._progress = 60

                # Step 3: Mine updated dataset
                if verbose:
                    print("\n--- Running Association Mining ---")
                analyzer = Analyzer(app_id)
                analyzer.run(
                    min_support=0.02,
                    min_confidence=0.5,
                    min_lift=1.2,
                    deduplicate_mode="both",
                    save_to_db=True
                )
                self._progress = 85

                # Step 4: Check or retune hyperparameters
                if verbose:
                    print("\n--- Checking Hyperparameter Tuning ---")
                try:
                    await asyncio.to_thread(self._sync_tuning_pipeline, app_id, 150, False, verbose)
                except Exception as e:
                    if verbose:
                        print(f"W: Hyperparameter check/tuning failed, will retry next cycle: {e}")

                # Step 5: Export forecast CSV
                self._progress = 95
                if verbose:
                    print("\n--- Exporting Forecast CSV ---")

                try:
                    await asyncio.to_thread(
                        self._sync_export_forecast,
                        app_id,
                        FORECAST_DAYS,
                        verbose
                    )
                except Exception as e:
                    if verbose:
                        print(f"W: Forecast CSV export failed: {e}")

                self._progress = 100

                if verbose:
                    print(f"[{now}] Daily update complete.\n")

            except asyncio.CancelledError:
                logger.warning(f"Aggiornamento giornaliero annullato per App ID {app_id}.")
                raise
            finally:
                self._progress = 0
                self._current_task = None

    async def start_daily_scheduler(self, app_id: int, interval_hours: int = 24, verbose: bool = True):
        """Daemon di schedulazione in background."""
        if verbose:
            print(f"Starting daily orchestrator daemon for App ID {app_id} (Interval: {interval_hours}h)...")

        while True:
            try:
                await self.run_daily_pipeline(app_id=app_id, verbose=verbose)
            except OperationLockedError:
                if verbose:
                    print("W: Scheduled daily update skipped because another operation is running.")
            except asyncio.CancelledError:
                if verbose:
                    print("I: Scheduled daemon task was cancelled.")
                break
            except Exception as e:
                if verbose:
                    print(f"E: An error occurred during the daily update cycle: {e}")

            sleep_seconds = interval_hours * 3600
            if verbose:
                print(f"Sleeping for {interval_hours} hours until next update...")
            await asyncio.sleep(sleep_seconds)

    async def run_force_mining_pipeline(self, app_id: int):
        """Pulisce le regole esistenti ed esegue un re-mining completo."""
        if self._lock.locked():
            raise OperationLockedError("Another operation is currently in progress.")

        async with self._lock:
            self._current_task = asyncio.current_task()
            try:
                self._progress = 10
                await asyncio.to_thread(self._sync_force_mining, app_id)
                self._progress = 100
            except asyncio.CancelledError:
                logger.warning(f"Force mining annullato per App ID {app_id}.")
                raise
            finally:
                self._progress = 0
                self._current_task = None

    def _sync_force_mining(self, app_id: int):
        clear_mining_rules(app_id)
        self._progress = 30
        analyzer = Analyzer(app_id)
        analyzer.run_historical(
            window_days=14,
            step_days=7,
            min_support=0.02,
            min_confidence=0.5,
            min_lift=1.2,
            deduplicate_mode="both"
        )

    def _sync_export_forecast(
        self,
        app_id: int,
        days: int = FORECAST_DAYS,
        verbose: bool = False
    ) -> Optional[Path]:
        """
        Generates the forecast using the regressor and exports it to CSV.

        Produces:
        - exports/{app_id}_forecast_YYYYMMDD_HHMMSS.csv
        - exports/{app_id}_forecast_latest.csv
        """
        days = min(max(1, int(days)), 365)
        export_path = Path(self.ensure_export_directory())

        try:
            generator = MatrixGenerator(app_id)
            X, Y, df_full = generator.build_daily_matrix()
        except Exception as e:
            if verbose:
                print(f"W: Unable to build forecast matrix for App ID {app_id}: {e}")
            return None

        if X.empty or df_full.empty:
            if verbose:
                print(f"W: No data available to forecast App ID {app_id}.")
            return None

        reg = Regressor(app_id=app_id, capacity_mode=True)

        decay_rate = 0.95
        rule_decay_rate = 0.995

        tuned = get_params_for_size(app_id, len(X))
        if tuned:
            xgb_params = {
                str(k).strip(): v
                for k, v in dict(tuned.get("xgb_params", {})).items()
            }
            tuned_decay = dict(tuned.get("decay_params", {}))

            # The dashboard forecast always uses the custom capacity-mode objective.
            xgb_params.pop("objective", None)
            xgb_params.pop("asymmetric_alpha", None)

            if xgb_params:
                try:
                    reg.model.set_params(**xgb_params)
                    if verbose:
                        print(f"Using tuned XGBoost params for dataset size {len(X)}.")
                except Exception as e:
                    if verbose:
                        print(f"W: Unable to apply tuned params: {e}")

            try:
                decay_rate = float(tuned_decay.get("decay_rate", decay_rate))
                rule_decay_rate = float(tuned_decay.get("rule_decay_rate", rule_decay_rate))
            except Exception:
                pass

        # The matrix drops the newest KPI row because its target is unknown.
        # Forecast extra days if needed so the exported window starts from today.
        last_matrix_date = pd.to_datetime(df_full.iloc[-1]["date"]).normalize()
        today = pd.Timestamp.today().normalize()
        extra_days = max(0, int((today - last_matrix_date).days))
        horizon = days + extra_days

        forecast_df = reg.forecast_days(
            days=horizon,
            decay_rate=decay_rate,
            rule_decay_rate=rule_decay_rate,
            data_tuple=(X, Y, df_full),
            export_chart=False,
            safety_buffer=1.00,
        )

        if forecast_df.empty:
            if verbose:
                print(f"W: Empty forecast produced for App ID {app_id}.")
            return None

        forecast_df = forecast_df.copy()
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])

        # Keep only today onward and trim to the requested visible window.
        forecast_df = forecast_df[forecast_df["date"] >= today].head(days)

        if forecast_df.empty:
            if verbose:
                print(f"W: Forecast window is empty after date filtering for App ID {app_id}.")
            return None

        out_df = pd.DataFrame({
            "day": forecast_df["date"].dt.strftime("%Y-%m-%d"),
            "predicted_mean": forecast_df["predicted_mean_concurrent"],
        })

        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        timestamped_path = export_path / f"{app_id}_forecast_{stamp}.csv"
        latest_path = export_path / f"{app_id}_forecast_latest.csv"
        temp_latest_path = export_path / f"{app_id}_forecast_latest.tmp.csv"

        out_df.to_csv(timestamped_path, index=False)
        out_df.to_csv(temp_latest_path, index=False)
        temp_latest_path.replace(latest_path)

        if verbose:
            print(f"[OK] Forecast exported to: {latest_path}")

        return latest_path

    async def export_forecast(self, app_id: int, days: int = FORECAST_DAYS) -> Optional[Path]:
        """
        Public async export operation, similar to export_rules.
        """
        if self._lock.locked():
            raise OperationLockedError("Another operation is currently in progress.")

        async with self._lock:
            self._current_task = asyncio.current_task()
            try:
                self._progress = 20
                path = await asyncio.to_thread(
                    self._sync_export_forecast,
                    app_id,
                    days,
                    True
                )
                self._progress = 100
                return path
            except asyncio.CancelledError:
                logger.warning(f"Forecast export cancelled for App ID {app_id}.")
                raise
            finally:
                self._progress = 0
                self._current_task = None

    async def export_rules(self, app_id: int):
        """Esporta le regole in un file CSV."""
        if self._lock.locked():
            raise OperationLockedError("Another operation is currently in progress.")

        async with self._lock:
            self._current_task = asyncio.current_task()
            try:
                self._progress = 20
                self.ensure_export_directory()
                await asyncio.to_thread(export_all_rules, app_id)
                self._progress = 100
            except asyncio.CancelledError:
                logger.warning(f"Export regole annullato per App ID {app_id}.")
                raise
            finally:
                self._progress = 0
                self._current_task = None


# Istanza Singleton esportata
orchestrator_service = OrchestratorService()