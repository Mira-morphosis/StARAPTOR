import asyncio
import datetime
import logging
from pathlib import Path

from analyzer.analyzer import Analyzer
from analyzer.regressor_tuner import ensure_up_to_date_params, run_tuning_cycle
from storage.storage_interface import kpi_updater, review_updater
from storage.storage_utils import DB_ROOT, export_all_rules, clear_mining_rules

logger = logging.getLogger("staraptor.orchestrator")


class OperationLockedError(Exception):
    """Raised when an operation is requested while another one is already running."""
    pass


class OrchestratorService:
    def __init__(self):
        # Global async lock to prevent concurrent execution of operations
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        """Returns True if an operation is currently in progress."""
        return self._lock.locked()

    def ensure_export_directory(self) -> str:
        """Ensures the export directory exists and returns its string path."""
        export_path = Path(DB_ROOT).expanduser() / "exports"
        export_path.mkdir(parents=True, exist_ok=True)
        return str(export_path)

    async def run_tuning_pipeline(self, app_id: int, n_trials: int = 150, force: bool = False, verbose: bool = True):
        """Runs the hyperparameter tuning pipeline ensuring mutual exclusion."""
        if self._lock.locked():
            raise OperationLockedError("Another operation is currently in progress.")

        async with self._lock:
            await asyncio.to_thread(self._sync_tuning_pipeline, app_id, n_trials, force, verbose)

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
        """Runs the full cold-start initialization pipeline for an app."""
        if self._lock.locked():
            raise OperationLockedError("Another operation is currently in progress.")

        async with self._lock:
            if verbose:
                print(f"\n==========================================")
                print(f"  STARTING INITIALIZATION FOR APP ID: {app_id}")
                print(f"==========================================\n")

            # Step 1: Download historical KPIs
            if verbose:
                print("--- Step 1/4: Downloading Historical KPIs ---")
            await kpi_updater(app_id=app_id, max_days=max_days, verbose=verbose)

            # Step 2: Download reviews & execute LLM processing
            if verbose:
                print("\n--- Step 2/4: Processing Historical Reviews ---")
            await review_updater(app_id=app_id, max_days=max_days, verbose=verbose)

            # Step 3: Mine association rules
            if verbose:
                print("\n--- Step 3/4: Mining Association Rules ---")
            analyzer = Analyzer(app_id)
            rules_df = analyzer.run(
                min_support=0.02,
                min_confidence=0.5,
                min_lift=1.2,
                deduplicate_mode="both",
                save_to_db=True
            )

            if rules_df.empty and verbose:
                print("W: Mining completed, but no rules met the threshold criteria.")

            # Step 4: Hyperparameter tuning
            if not skip_tuning:
                if verbose:
                    print("\n--- Step 4/4: Tuning XGBoost Hyperparameters ---")
                try:
                    await asyncio.to_thread(self._sync_tuning_pipeline, app_id, 150, False, verbose)
                except Exception as e:
                    if verbose:
                        print(f"W: Initial tuning failed, can be re-run manually: {e}")
            elif verbose:
                print("\n--- Step 4/4: Tuning XGBoost Hyperparameters (skipped) ---")

            if verbose:
                print(f"\n[OK] Initialization complete for App ID {app_id}.\n")

    async def run_daily_pipeline(self, app_id: int, verbose: bool = True):
        """Runs a single daily update cycle."""
        if self._lock.locked():
            raise OperationLockedError("Another operation is currently in progress.")

        async with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            if verbose:
                print(f"\n[{now}] Executing daily update cycle for App ID: {app_id}...")

            # Step 1: Update recent KPIs
            await kpi_updater(app_id=app_id, max_days=7, verbose=verbose)

            # Step 2: Update recent reviews
            await review_updater(app_id=app_id, max_days=1, verbose=verbose)

            # Step 3: Mine updated dataset
            analyzer = Analyzer(app_id)
            analyzer.run(
                min_support=0.02,
                min_confidence=0.5,
                min_lift=1.2,
                deduplicate_mode="both",
                save_to_db=True
            )

            # Step 4: Check or retune hyperparameters
            try:
                await asyncio.to_thread(self._sync_tuning_pipeline, app_id, 150, False, verbose)
            except Exception as e:
                if verbose:
                    print(f"W: Hyperparameter check/tuning failed, will retry next cycle: {e}")

            if verbose:
                print(f"[{now}] Daily update complete.\n")

    async def start_daily_scheduler(self, app_id: int, interval_hours: int = 24, verbose: bool = True):
        """Background scheduler loop. Acquires lock during each cycle execution."""
        if verbose:
            print(f"Starting daily orchestrator daemon for App ID {app_id} (Interval: {interval_hours}h)...")

        while True:
            try:
                await self.run_daily_pipeline(app_id=app_id, verbose=verbose)
            except OperationLockedError:
                if verbose:
                    print("W: Scheduled daily update skipped because another operation is running.")
            except Exception as e:
                if verbose:
                    print(f"E: An error occurred during the daily update cycle: {e}")

            sleep_seconds = interval_hours * 3600
            if verbose:
                print(f"Sleeping for {interval_hours} hours until next update...")
            await asyncio.sleep(sleep_seconds)

    async def run_force_mining_pipeline(self, app_id: int):
        """Clears existing rules and executes full re-mining with sliding windows."""
        if self._lock.locked():
            raise OperationLockedError("Another operation is currently in progress.")

        async with self._lock:
            await asyncio.to_thread(self._sync_force_mining, app_id)

    def _sync_force_mining(self, app_id: int):
        clear_mining_rules(app_id)
        analyzer = Analyzer(app_id)
        analyzer.run_historical(
            window_days=14,
            step_days=7,
            min_support=0.02,
            min_confidence=0.5,
            min_lift=1.2,
            deduplicate_mode="both"
        )

    async def export_rules(self, app_id: int):
        """Exports rules to a CSV file."""
        if self._lock.locked():
            raise OperationLockedError("Another operation is currently in progress.")

        async with self._lock:
            self.ensure_export_directory()
            await asyncio.to_thread(export_all_rules, app_id)


# Singleton instance exported for API / CLI usage
orchestrator_service = OrchestratorService()