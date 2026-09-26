#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys
from dotenv import load_dotenv

from orchestrator.orchestrator_service import orchestrator_service, OperationLockedError

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="StARAPTOR CLI")

    parser.add_argument(
        "--app-id",
        type=int,
        default=os.getenv("STARAPTOR_APP_ID", 1086940),
        help="Steam App ID to process (default: STARAPTOR_APP_ID env var or 1086940)"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init", action="store_true", help="Run initial DB setup.")
    group.add_argument("--daily", action="store_true", help="Start continuous daily updates daemon.")
    group.add_argument("--run-once", action="store_true", help="Run a single daily update cycle and exit.")
    group.add_argument("--tune", action="store_true", help="Check and retune XGBoost hyperparameters.")
    group.add_argument("--export", action="store_true", help="Exports rules found to CSV.")
    group.add_argument("--force-mining", action="store_true", help="Clear rules and re-mine full history.")

    parser.add_argument("--max-days", type=int, default=None, help="Limit lookback period in days.")
    parser.add_argument("--interval", type=int, default=24, help="Interval in hours for --daily mode.")
    parser.add_argument("--n-trials", type=int, default=150, help="Number of Optuna trials per tier.")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter tuning during --init.")
    parser.add_argument("--force", action="store_true", help="Used with --tune to force retuning all tiers.")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output logging.")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    verbose = not args.quiet

    try:
        if args.init:
            asyncio.run(orchestrator_service.run_initial_pipeline(
                app_id=args.app_id,
                max_days=args.max_days,
                verbose=verbose,
                skip_tuning=args.skip_tuning
            ))
        elif args.daily:
            asyncio.run(orchestrator_service.start_daily_scheduler(
                app_id=args.app_id,
                interval_hours=args.interval,
                verbose=verbose
            ))
        elif args.run_once:
            asyncio.run(orchestrator_service.run_daily_pipeline(
                app_id=args.app_id,
                verbose=verbose
            ))
        elif args.tune:
            asyncio.run(orchestrator_service.run_tuning_pipeline(
                app_id=args.app_id,
                n_trials=args.n_trials,
                force=args.force,
                verbose=verbose
            ))
        elif args.export:
            asyncio.run(orchestrator_service.export_rules(app_id=args.app_id))
        elif args.force_mining:
            asyncio.run(orchestrator_service.run_force_mining_pipeline(app_id=args.app_id))

    except OperationLockedError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOrchestrator stopped by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()