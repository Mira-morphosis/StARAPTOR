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

# Load environment variables
load_dotenv()


def ensure_export_directory() -> str:
    """Ensures the default export directory exists before running exports."""
    export_path = Path(DB_ROOT).expanduser() / "exports"
    export_path.mkdir(parents=True, exist_ok=True)
    return str(export_path)


def run_tuning_pipeline(app_id: int, n_trials: int = 150, force: bool = False, verbose: bool = True):
    """
    Verifica se gli iperparametri XGBoost sono aggiornati per l'App ID e, se
    necessario (o se force=True), esegue un nuovo ciclo di tuning walk-forward,
    salvando i risultati per tier in DuckDB tramite analyzer/tuner.py.
    """
    if force:
        print(f"--- Tuning forzato per App ID {app_id} (tutti i tier mancanti) ---")
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
    1. Downloads all historical KPIs for the app.
    2. Downloads and evaluates all historical reviews.
    3. Runs the association rule mining algorithm.
    4. Runs an initial XGBoost hyperparameter tuning cycle for the cold-start
       dataset tier (skipped if skip_tuning=True).
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
        print("\n--- Step 4/4: Tuning Iperparametri XGBoost ---")
        try:
            run_tuning_pipeline(app_id=app_id, n_trials=150, force=False, verbose=verbose)
        except Exception as e:
            print(f"W: Tuning iniziale fallito, si può rilanciare manualmente con --tune: {e}")
    else:
        print("\n--- Step 4/4: Tuning Iperparametri XGBoost (saltato: --skip-tuning) ---")

    print(f"\n[✓] Initialization complete for App ID {app_id}.\n")


async def run_daily_pipeline(app_id: int, verbose: bool = True):
    """
    Executes a single update cycle:
    1. Fetches recent KPI changes.
    2. Fetches newly added reviews for today.
    3. Re-runs association rule mining.
    4. Checks whether hyperparameters need retuning and retunes if so.
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

    # Step 4: Lightweight staleness check — only triggers an actual Optuna run
    # when the dataset has outgrown the largest tuned tier or params are stale.
    # Wrapped defensively so a tuning failure never aborts the daily cycle.
    try:
        run_tuning_pipeline(app_id=app_id, n_trials=150, force=False, verbose=verbose)
    except Exception as e:
        print(f"W: Controllo/tuning iperparametri fallito, si riproverà al prossimo ciclo: {e}")

    print(f"[{now}] Daily update complete.\n")


async def start_daily_scheduler(app_id: int, interval_hours: int = 24, verbose: bool = True):
    """
    Runs the daily pipeline immediately and then schedules it to repeat every interval_hours.
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
    Cancella tutte le regole/run presenti nel DB per l'App ID e ricalcola
    il mining su tutto lo storico suddiviso in sliding window.
    """
    # Step 1: Ripulitura regole esistenti
    clear_mining_rules(app_id)

    # Step 2: Calcolo storico a sliding window
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
        help="Verifica se gli iperparametri XGBoost necessitano un aggiornamento e li ritunizza se necessario."
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
        help="Numero di trial Optuna per tier, usato con --init e --tune (default: 150)."
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Salta il tuning automatico degli iperparametri dopo l'inizializzazione (solo con --init)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Usato con --tune: ignora il controllo di staleness e ritunizza tutti i tier mancanti."
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