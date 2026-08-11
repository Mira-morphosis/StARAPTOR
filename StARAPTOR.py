import argparse
import asyncio
import datetime
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from analyzer.analyzer import Analyzer
from analyzer.regressor_tuner import ensure_up_to_date_params, run_tuning_cycle
from storage.storage_interface import kpi_updater, review_updater
from storage.storage_utils import DB_ROOT, export_all_rules, clear_mining_rules

load_dotenv()


def ensure_export_directory() -> str:
    """
    Ensures the default export directory exists before running exports.
    :return: The resolved export directory path as a string.
    """
    export_path = Path(DB_ROOT).expanduser() / "exports"
    export_path.mkdir(parents=True, exist_ok=True)
    return str(export_path)


def run_tuning_pipeline(app_id: int, n_trials: int = 150, force: bool = False, verbose: bool = True):
    """
    Checks whether XGBoost hyperparameters are up to date for the app and, if needed (or if
    force=True), runs a new walk-forward tuning cycle, saving per-tier results to DuckDB.
    :param app_id: The Steam unique identifier for the application.
    :param n_trials: The number of Optuna trials to run per tier.
    :param force: If True, skips the staleness check and re-tunes every tier regardless.
    :param verbose: Regulates logging output.
    :return: None.
    """
    if force:
        print(f"--- Forced tuning for App ID {app_id} (all tiers) ---")
        run_tuning_cycle(app_id=app_id, n_trials=n_trials, verbose=verbose)
    else:
        ensure_up_to_date_params(app_id=app_id, n_trials=n_trials, verbose=verbose)


async def run_initial_pipeline(
    app_id: int,
    max_days: int | None = None,
    verbose: bool = True,
    skip_tuning: bool = False,
):
    """
    Runs the full cold-start pipeline for a new app: downloads historical KPIs and reviews,
    mines association rules, and runs an initial hyperparameter tuning cycle.
    :param app_id: The Steam unique identifier for the application.
    :param max_days: The maximum number of days to look back for KPIs/reviews; None uses each source's own default limit.
    :param verbose: Regulates logging output.
    :param skip_tuning: If True, skips the initial hyperparameter tuning step.
    :return: None.
    """
    print(f"\n==========================================")
    print(f"  STARTING INITIALIZATION FOR APP ID: {app_id}")
    print(f"==========================================\n")

    # Step 1: Download full historical KPIs
    print("--- Step 1/4: Downloading Historical KPIs ---")
    await kpi_updater(app_id=app_id, max_days=max_days, verbose=verbose)

    # Step 2: Download full review history & run LLM evaluator
    print("\n--- Step 2/4: Processing Historical Reviews ---")
    await review_updater(app_id=app_id, max_days=max_days, verbose=verbose)

    # Step 3: Run association rule mining
    print("\n--- Step 3/4: Mined Association Rules ---")
    analyzer = Analyzer(app_id)
    rules_df = analyzer.run(
        min_support=0.02,
        min_confidence=0.5,
        min_lift=1.2,
        deduplicate_mode="both",
        save_to_db=True
    )

    if rules_df.empty:
        print("W: Mining completed, but no rules met the threshold criteria.")

    # Step 4: Initial hyperparameter tuning for the cold-start dataset tier
    if not skip_tuning:
        print("\n--- Step 4/4: Tuning XGBoost Hyperparameters ---")
        try:
            run_tuning_pipeline(app_id=app_id, n_trials=150, force=False, verbose=verbose)
        except Exception as e:
            print(f"W: Initial tuning failed, it can be re-run manually with --tune: {e}")
    else:
        print("\n--- Step 4/4: Tuning XGBoost Hyperparameters (skipped: --skip-tuning) ---")

    print(f"\n[OK] Initialization complete for App ID {app_id}.\n")


async def run_daily_pipeline(app_id: int, verbose: bool = True):
    """
    Runs a single update cycle: refreshes recent KPIs and today's reviews, re-mines
    association rules, and retunes hyperparameters if they've gone stale.
    :param app_id: The Steam unique identifier for the application.
    :param verbose: Regulates logging output.
    :return: None.
    """
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n[{now}] Executing daily update cycle for App ID: {app_id}...")

    # Step 1: Update KPIs (check last 7 days for any adjustments)
    await kpi_updater(app_id=app_id, max_days=7, verbose=verbose)

    # Step 2: Update reviews for today
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

    # Step 4: Lightweight staleness check — only triggers an actual Optuna run when the
    # dataset has outgrown the largest tuned tier or params are stale. Wrapped defensively so
    # a tuning failure never aborts the daily cycle.
    try:
        run_tuning_pipeline(app_id=app_id, n_trials=150, force=False, verbose=verbose)
    except Exception as e:
        print(f"W: Hyperparameter check/tuning failed, will retry next cycle: {e}")

    print(f"[{now}] Daily update complete.\n")


async def start_daily_scheduler(app_id: int, interval_hours: int = 24, verbose: bool = True):
    """
    Runs the daily pipeline immediately, then repeats it every interval_hours indefinitely.
    :param app_id: The Steam unique identifier for the application.
    :param interval_hours: The number of hours to sleep between pipeline runs.
    :param verbose: Regulates logging output.
    :return: None. Runs until interrupted.
    """
    print(f"Starting daily orchestrator daemon for App ID {app_id} (Interval: {interval_hours}h)...")

    while True:
        try:
            await run_daily_pipeline(app_id=app_id, verbose=verbose)
        except Exception as e:
            print(f"E: An error occurred during the daily update cycle: {e}")

        sleep_seconds = interval_hours * 3600
        print(f"Sleeping for {interval_hours} hours until the next scheduled update...")
        await asyncio.sleep(sleep_seconds)

def run_force_mining_pipeline(app_id: int):
    """
    Clears all existing rules/runs for the app and re-mines the full history using sliding windows.
    :param app_id: The Steam unique identifier for the application.
    :return: None.
    """
    # Step 1: Clear existing rules
    clear_mining_rules(app_id)

    # Step 2: Recompute historical sliding-window mining
    analyzer = Analyzer(app_id)
    analyzer.run_historical(
        window_days=14,
        step_days=7,
        min_support=0.02,
        min_confidence=0.5,
        min_lift=1.2,
        deduplicate_mode="both"
    )

def main():
    """
    Parses CLI arguments and dispatches to the selected StARAPTOR pipeline.
    :return: None. Exits the process on completion or on KeyboardInterrupt.
    """
    parser = argparse.ArgumentParser(description="StARAPTOR")
    parser.add_argument(
        "--app-id",
        type=int,
        default=os.getenv("STARAPTOR_APP_ID", 1086940),
        help="Steam App ID to process (default: STARAPTOR_APP_ID env var or 1086940)"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--init",
        action="store_true",
        help="Run initial DB setup: download all reviews/KPIs, mine rules, and tune hyperparameters."
    )
    group.add_argument(
        "--daily",
        action="store_true",
        help="Start continuous daily updates (runs daily pipeline every 24 hours)."
    )
    group.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single daily update cycle immediately and exit."
    )
    group.add_argument(
        "--tune",
        action="store_true",
        help="Check whether XGBoost hyperparameters need updating and retune them if so."
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Limit lookback period in days for review/KPI fetching (e.g., --max-days 14)."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=24,
        help="Interval in hours for --daily mode (default: 24)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=150,
        help="Number of Optuna trials per tier, used with --init and --tune (default: 150)."
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip automatic hyperparameter tuning after initialization (only with --init)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Used with --tune: skip the staleness check and retune all tiers regardless."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose output logging."
    )
    group.add_argument(
        "--export",
        action="store_true",
        help="Exports rules found to CSV."
    )
    group.add_argument(
        "--force-mining",
        action="store_true",
        help="Clear existing rules in DB and re-mine full history using sliding windows."
    )
    args = parser.parse_args()
    verbose = not args.quiet

    try:
        if args.init:
            asyncio.run(run_initial_pipeline(
                app_id=args.app_id,
                max_days=args.max_days,
                verbose=verbose,
                skip_tuning=args.skip_tuning
            ))
        elif args.daily:
            asyncio.run(start_daily_scheduler(
                app_id=args.app_id,
                interval_hours=args.interval,
                verbose=verbose
            ))
        elif args.run_once:
            asyncio.run(run_daily_pipeline(
                app_id=args.app_id,
                verbose=verbose
            ))
        elif args.tune:
            run_tuning_pipeline(
                app_id=args.app_id,
                n_trials=args.n_trials,
                force=args.force,
                verbose=verbose
            )
        elif args.export:
            export_all_rules(app_id=args.app_id)
        elif args.force_mining if hasattr(args, "force_mining") and args.force_mining else False:
            run_force_mining_pipeline(app_id=args.app_id)
    except KeyboardInterrupt:
        print("\nOrchestrator stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()