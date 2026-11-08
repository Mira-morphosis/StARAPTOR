#!/usr/bin/env python3
"""
Controlled runner for the Research Question scripts.

Runs the RQ1/RQ2/RQ3 study scripts with fixed, seed-controlled parameters so that
anyone with the same per-game database gets the same outputs:

  RQ1  libs/evaluator/tests/rq1_llm_consistency_benchmark.py  (needs a configured LLM provider)
  RQ2  libs/analyzer/tests/rq2_rules_inspection.py             (sampling is automated; the
       classification of the sampled rules is manual — see below)
  RQ3  libs/analyzer/tests/rq3_forecast_ablation.py            (fully automated)

Prerequisite: the game database must exist, e.g. run first
    ./staraptor --app-id <id> --init
then run this script with the project environment, e.g.
    poetry run python src/run_rq_studies.py --app-id 1086940 --rq all
or  .venv/bin/python src/run_rq_studies.py --app-id 1086940 --rq all

RQ1 draws its frozen sample from RAW reviews fetched live from the Steam API by default
(spam included), because the stored database is already spam-filtered by the pipeline and
would make the models' spam recognition unmeasurable; pass --no-rq1-fetch-live to sample
the stored history instead, or --rq1-redraw-sample to discard a previously frozen sample.

RQ2 has one manual step: after the first run writes out/<app_id>/rq2/sample.csv, fill its
'category' and 'evidence' columns (categories: actionable, semantically_trivial,
taxonomy_artifact, unclear), then re-run with
    --rq rq2 --rq2-annotations out/<app_id>/rq2/sample.csv
to compute the metrics, trace-back evidence and figures.

All artifacts (CSV/JSON reports and PDF figures) land under --out (default ./out), the
figures under --fig-dir (default <out>/<app_id>/figures/). A run_manifest.json recording
the exact commands, seeds, git commit and package versions is written next to them, so a
run is fully reproducible and auditable.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

RQ_SCRIPTS = {
    "rq1": "libs/evaluator/tests/rq1_llm_consistency_benchmark.py",
    "rq2": "libs/analyzer/tests/rq2_rules_inspection.py",
    "rq3": "libs/analyzer/tests/rq3_forecast_ablation.py",
}

# Controlled defaults: identical for every user, overridable only via CLI flags.
CONTROLLED_DEFAULTS = {
    "seed": 42,
    "rq1": {"sample_size": 500, "bootstraps": 1000, "min_rating_units": 30},
    "rq2": {"top_k": 10, "random_k": 5, "traceback_max_reviews": 10},
    "rq3": {"split": 0.70, "n_trials": 150, "eval_days": 30, "n_jobs": 1},
}

TRACKED_PACKAGES = [
    "analyzer", "evaluator", "storage", "fetcher", "orchestrator",
    "pandas", "numpy", "matplotlib", "duckdb", "xgboost", "optuna", "mlxtend",
    "instructor", "openai", "anthropic", "scikit-learn", "statsmodels",
]


def resolve_db_path(app_id: int) -> Path:
    """
    Resolves the per-game DuckDB file the same way storage.storage_utils does.
    :param app_id: The Steam unique identifier for the application.
    :return: The expected path of <app_id>.db under STARAPTOR_DB_ROOT.
    """
    db_root = os.getenv("STARAPTOR_DB_ROOT", "~/.staraptor/")
    return Path(db_root).expanduser() / f"{app_id}.db"


def git_commit() -> Optional[str]:
    """
    :return: The current git commit of the repo, or None if unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def package_versions() -> Dict[str, str]:
    """
    :return: Installed versions of the key packages, for the reproducibility manifest.
    """
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def run_step(label: str, argv: List[str], manifest_runs: Dict[str, Any]) -> bool:
    """
    Runs one RQ script as a subprocess with the current interpreter, streaming its output.
    :param label: The RQ label (rq1/rq2/rq3, optionally with a phase suffix).
    :param argv: The full command line (argv[0] is the python executable).
    :param manifest_runs: The manifest dict to record the command and outcome into.
    :return: True if the subprocess exited with code 0.
    """
    print(f"\n{'=' * 70}\n>>> {label}: {' '.join(argv[1:])}\n{'=' * 70}", flush=True)
    proc = subprocess.run(argv, cwd=ROOT)
    manifest_runs[label] = {"command": argv[1:], "returncode": proc.returncode}
    ok = proc.returncode == 0
    print(f">>> {label} {'succeeded' if ok else 'FAILED'} (exit {proc.returncode})\n", flush=True)
    return ok


def rq2_annotations_path(out_root: str, app_id: int, explicit: Optional[str]) -> Optional[str]:
    """
    Resolves which annotated inspection form to use for the RQ2 metrics phase:
    an explicit --rq2-annotations path, else out/<app>/rq2/sample_annotated.csv if present,
    else the sample.csv itself if its category column was filled in place.
    :return: The path to use, or None if no annotations are available yet.
    """
    if explicit:
        explicit_abs = str(Path(explicit).resolve())  # children run with cwd=ROOT
        if not Path(explicit_abs).exists():
            raise FileNotFoundError(f"--rq2-annotations file not found: {explicit}")
        return explicit_abs

    rq2_dir = Path(out_root) / str(app_id) / "rq2"
    annotated = rq2_dir / "sample_annotated.csv"
    if annotated.exists():
        return str(annotated)

    sample = rq2_dir / "sample.csv"
    if sample.exists():
        try:
            import pandas as pd
            df = pd.read_csv(sample)
            if "category" in df.columns and df["category"].notna().any() and (df["category"].astype(str).str.strip() != "").any():
                return str(sample)
        except Exception:
            pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Controlled runner for the RQ1/RQ2/RQ3 study scripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--app-id", type=int, default=int(os.getenv("STARAPTOR_APP_ID", 1086940)),
                        help="Steam App ID whose database the studies run on.")
    parser.add_argument("--rq", nargs="+", default=["all"], choices=["all", "rq1", "rq2", "rq3"],
                        help="Which research question scripts to run.")
    parser.add_argument("--out", type=str, default="./out", help="Root output directory.")
    parser.add_argument("--fig-dir", type=str, default=None,
                        help="Directory for the PDF figures (default: <out>/<app_id>/figures/).")
    parser.add_argument("--seed", type=int, default=CONTROLLED_DEFAULTS["seed"], help="Fixed seed used by every script.")
    parser.add_argument("--rq2-remine", action="store_true",
                        help="RQ2: clear the stored rules and re-mine the full history first "
                             "(mutates the game database).")
    parser.add_argument("--rq2-annotations", type=str, default=None,
                        help="RQ2: path to the manually annotated sample CSV for the metrics phase.")
    parser.add_argument("--rq1-sample-size", type=int, default=CONTROLLED_DEFAULTS["rq1"]["sample_size"])
    parser.add_argument("--rq1-bootstraps", type=int, default=CONTROLLED_DEFAULTS["rq1"]["bootstraps"])
    parser.add_argument("--rq1-extra-provider", action="append", default=[], metavar="KV",
                        help="Passthrough to the RQ1 script's --extra-provider: a frontier/cloud API as "
                             "';'-separated key=value pairs (name, api_key, type, base_url, models). "
                             "Repeatable. Credentials are never read from .env.")
    parser.add_argument("--rq1-merge-extractions", type=str, default=None,
                        help="RQ1: reuse extractions from this CSV and only compute missing raters. "
                             "Default: <out>/<app_id>/rq1/extractions_all_models.csv if it exists.")
    parser.add_argument("--rq1-rerun", action="store_true",
                        help="RQ1: ignore the merge file and recompute every rater.")
    parser.add_argument("--rq1-fetch-live", action=argparse.BooleanOptionalAction, default=True,
                        help="RQ1: draw the sample from reviews fetched live from the Steam API "
                             "(raw, spam included) instead of the stored DB, so that inter-model "
                             "spam recognition is benchmarked too. This is the DEFAULT because the "
                             "stored DB is already spam-filtered by the pipeline; pass "
                             "--no-rq1-fetch-live to sample the stored history instead.")
    parser.add_argument("--rq1-fetch-days", type=int, default=14,
                        help="RQ1 --fetch-live: days of history to fetch (Steam API limit: 140).")
    parser.add_argument("--rq1-fetch-delay", type=float, default=2.0,
                        help="RQ1 --fetch-live: delay between Steam API pages, seconds (same "
                             "default as the production fetcher). Raise this if Steam "
                             "rate-limits the live backfill (empty/HTML responses).")
    parser.add_argument("--rq1-redraw-sample", action="store_true",
                        help="RQ1: ignore the frozen <out>/<app_id>/rq1/sample.csv and draw a fresh sample.")
    parser.add_argument("--rq3-n-trials", type=int, default=CONTROLLED_DEFAULTS["rq3"]["n_trials"])
    parser.add_argument("--rq3-eval-days", type=int, default=CONTROLLED_DEFAULTS["rq3"]["eval_days"])
    parser.add_argument("--rq3-n-jobs", type=int, default=CONTROLLED_DEFAULTS["rq3"]["n_jobs"],
                        help="Parallel Optuna trials for RQ3; 1 is fully deterministic (same outputs "
                             "for everyone), higher values are faster but trial scheduling may vary.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    selected = ["rq1", "rq2", "rq3"] if "all" in args.rq else args.rq
    # Anchor relative output paths to the repo root (children run with cwd=ROOT),
    # so the artifacts land in the same place no matter where the runner is invoked from.
    out_root = args.out if os.path.isabs(args.out) else str((ROOT / args.out).resolve())
    app_out = Path(out_root) / str(args.app_id)
    fig_dir = args.fig_dir if (args.fig_dir and os.path.isabs(args.fig_dir)) else (
        args.fig_dir if args.fig_dir else str(app_out / "figures"))
    if args.fig_dir and not os.path.isabs(args.fig_dir):
        fig_dir = str((ROOT / args.fig_dir).resolve())
    app_out.mkdir(parents=True, exist_ok=True)

    # ---- preflight -------------------------------------------------------
    for rq in selected:
        script = ROOT / RQ_SCRIPTS[rq]
        if not script.exists():
            print(f"E: missing script {script}")
            return 2

    db_path = resolve_db_path(args.app_id)
    if not db_path.exists():
        print(f"E: no database for App ID {args.app_id} at {db_path}.\n"
              f"   Run first:  ./staraptor --app-id {args.app_id} --init")
        return 2

    if "rq1" in selected:
        provider = os.getenv("STARAPTOR_PROVIDER_TYPE", "openai_compatible").lower()
        local_ok = provider == "anthropic" or bool(os.getenv("STARAPTOR_BASE_URL"))
        if not local_ok and not args.rq1_extra_provider:
            print("E: RQ1 needs a configured LLM provider (STARAPTOR_BASE_URL, or "
                  "STARAPTOR_PROVIDER_TYPE=anthropic with STARAPTOR_API_KEY), or at least one "
                  "--rq1-extra-provider argument.\n"
                  "   Set the local provider in .env, or run with --rq rq2 rq3 to skip RQ1.")
            return 2

    # ---- run the selected studies ----------------------------------------
    manifest: Dict[str, Any] = {
        "runner": "src/run_rq_studies.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "app_id": args.app_id,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "controlled_defaults": {**CONTROLLED_DEFAULTS, "seed": args.seed},
        "db_path": str(db_path),
        "out_root": out_root,
        "fig_dir": fig_dir,
        "runs": {},
    }
    runs: Dict[str, Any] = manifest["runs"]
    all_ok = True

    py = sys.executable

    if "rq1" in selected:
        argv = [py, RQ_SCRIPTS["rq1"],
                "--app-id", str(args.app_id),
                "--sample-size", str(args.rq1_sample_size),
                "--seed", str(args.seed),
                "--bootstraps", str(args.rq1_bootstraps),
                "--min-rating-units", str(CONTROLLED_DEFAULTS["rq1"]["min_rating_units"]),
                "--out", out_root, "--fig-dir", fig_dir]
        for kv in args.rq1_extra_provider:
            argv += ["--extra-provider", kv]
        merge = args.rq1_merge_extractions
        if merge is None and not args.rq1_rerun:
            default_merge = app_out / "rq1" / "extractions_all_models.csv"
            if default_merge.exists():
                merge = str(default_merge)
        if merge:
            argv += ["--merge-extractions", merge]
        if args.rq1_rerun:
            argv.append("--rerun")
        if args.rq1_fetch_live:
            argv += ["--fetch-live", "--fetch-max-days", str(args.rq1_fetch_days),
                     "--fetch-delay", str(args.rq1_fetch_delay)]
        else:
            argv.append("--no-fetch-live")
        if args.rq1_redraw_sample:
            argv.append("--redraw-sample")
        all_ok &= run_step("rq1", argv, runs)

    if "rq2" in selected:
        argv = [py, RQ_SCRIPTS["rq2"],
                "--app-id", str(args.app_id),
                "--seed", str(args.seed),
                "--top-k", str(CONTROLLED_DEFAULTS["rq2"]["top_k"]),
                "--random-k", str(CONTROLLED_DEFAULTS["rq2"]["random_k"]),
                "--out", out_root]
        if args.rq2_remine:
            argv.append("--remine")
        all_ok &= run_step("rq2 (sampling)", argv, runs)

        annotations = rq2_annotations_path(out_root, args.app_id, args.rq2_annotations)
        if annotations:
            argv = [py, RQ_SCRIPTS["rq2"],
                    "--app-id", str(args.app_id),
                    "--seed", str(args.seed),
                    "--top-k", str(CONTROLLED_DEFAULTS["rq2"]["top_k"]),
                    "--random-k", str(CONTROLLED_DEFAULTS["rq2"]["random_k"]),
                    "--annotations", annotations,
                    "--traceback-max-reviews", str(CONTROLLED_DEFAULTS["rq2"]["traceback_max_reviews"]),
                    "--out", out_root, "--fig-dir", fig_dir]
            all_ok &= run_step("rq2 (metrics)", argv, runs)
        else:
            print("RQ2: sampling done. Fill the 'category'/'evidence' columns of\n"
                  f"     {app_out / 'rq2' / 'sample.csv'} (or drop a copy as sample_annotated.csv),\n"
                  "     then re-run with --rq rq2 to compute metrics, trace-back and figures.")
            runs["rq2 (metrics)"] = {"skipped": "no annotations yet"}

    if "rq3" in selected:
        argv = [py, RQ_SCRIPTS["rq3"],
                "--app-id", str(args.app_id),
                "--split", str(CONTROLLED_DEFAULTS["rq3"]["split"]),
                "--n-trials", str(args.rq3_n_trials),
                "--eval-days", str(args.rq3_eval_days),
                "--n-jobs", str(args.rq3_n_jobs),
                "--out", out_root, "--fig-dir", fig_dir]
        all_ok &= run_step("rq3", argv, runs)

    # ---- manifest ----------------------------------------------------------
    reports = {}
    for rq in selected:
        report = app_out / rq / "report.json"
        if report.exists():
            reports[rq] = str(report)
    manifest["reports"] = reports
    manifest_path = app_out / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print(f"Outputs: {app_out}   Figures: {fig_dir}")
    if not all_ok:
        print("Some steps FAILED — see the log above.")
        return 1
    print("All selected RQ studies completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
