import datetime
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Final, Generator, Optional

from dotenv import load_dotenv
from pandas import DataFrame, Series

load_dotenv()
DB_ROOT: Final = os.getenv("STARAPTOR_DB_ROOT", "~/.staraptor/")

import duckdb


def _resolve_db_path(app_id: int) -> Path:
    """
    Resolves the DuckDB file path for an app, falling back to defaults if unset.
    :param app_id: The Steam unique identifier for the application.
    :return: The resolved Path to the app's .db file.
    """
    if DB_ROOT is None:
        print("W: .env is missing or is not set up correctly. Forcing defaults... (~/.staraptor/)")
        default_folder = Path("~/.staraptor/").expanduser()
        default_folder.mkdir(parents=True, exist_ok=True)
        return default_folder / f"{app_id}.db"

    db_folder = Path(DB_ROOT).expanduser()

    if db_folder.exists() and not db_folder.is_dir():
        raise NotADirectoryError(f"{DB_ROOT} is not a valid folder.")

    if not db_folder.exists():
        db_folder.mkdir(parents=True, exist_ok=True)

    return db_folder / f"{app_id}.db"


@contextmanager
def db_connection(app_id: int) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Opens a DuckDB connection for an app's database, closing it on exit.
    :param app_id: The Steam unique identifier for the application.
    :return: A context manager yielding an open DuckDBPyConnection.
    """
    db_path = _resolve_db_path(app_id)

    conn = duckdb.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


def get_existing_review_ids(app_id: int) -> set[str]:
    """
    Fetches the set of review ids already stored for an app.
    :param app_id: The Steam unique identifier for the application.
    :return: A set of recommendationid strings (empty if the table doesn't exist yet).
    """
    with db_connection(app_id) as conn:
        try:
            cursor = conn.execute("SELECT recommendationid FROM reviews")
            return {str(row[0]) for row in cursor.fetchall()}
        except duckdb.CatalogException:  # Table doesn't exist yet
            return set()


def _init_reviews_db(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Creates the reviews table if it doesn't already exist.
    :param conn: An open DuckDB connection.
    :return: None.
    """
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS reviews
                 (
                     recommendationid VARCHAR PRIMARY KEY,
                     review VARCHAR,
                     language VARCHAR,
                     timestamp_updated INTEGER,
                     steam_purchase BOOLEAN,
                     received_for_free BOOLEAN,
                     voted_up BOOLEAN,
                     weighted_vote_score DOUBLE,
                     playtime_at_review INTEGER,
                     primarily_steam_deck BOOLEAN,
                     has_hardware_info BOOLEAN,
                     hw_resolution VARCHAR,
                     hw_ram VARCHAR,
                     hw_vram VARCHAR,
                     hw_gpu VARCHAR,
                     hw_os VARCHAR,
                     hw_cpu_vendor VARCHAR,
                     isSpam BOOLEAN,
                     multiplayer INTEGER,
                     immersion INTEGER,
                     community INTEGER,
                     replayability INTEGER,
                     story INTEGER,
                     monetizationModel INTEGER,
                     gameplay INTEGER,
                     controls INTEGER,
                     graphics INTEGER,
                     customization INTEGER,
                     audio INTEGER,
                     difficulty INTEGER,
                     isUseful BOOLEAN,
                     containsPositiveAspects BOOLEAN,
                     containsNegativeAspects BOOLEAN,
                     actuallyRecommendsTitle BOOLEAN,
                     containsBugDescription BOOLEAN
                 )
    """)

def insert_reviews(app_id: int, new_reviews: list[dict] | str) -> None:
    """
    Stores new reviews into the app's database, ignoring duplicates by primary key.
    :param app_id: The Steam unique identifier for the application.
    :param new_reviews: A list of review dicts, or a JSON string encoding one.
    :return: None.
    """
    if not new_reviews:
        return
    if isinstance(new_reviews, str):
        if new_reviews in ("[]", ""):
            return
        reviews_list = json.loads(new_reviews)
    else:
        reviews_list = new_reviews

    if not reviews_list:
        return

    columns = list(reviews_list[0].keys())

    # Placeholders let DuckDB bind dict keys directly (e.g. $recommendationid, $review, ...)
    placeholders = ", ".join([f"${col}" for col in columns])

    query = f"""
            INSERT OR IGNORE INTO reviews ({", ".join(columns)}) 
            VALUES ({placeholders})
        """

    with db_connection(app_id) as conn:
        _init_reviews_db(conn)

        try:
            conn.executemany(query, reviews_list)

        except Exception as e:
            print(f"E: An error has occurred! {e}")
            import traceback
            traceback.print_exc()


def export(app_id: int, path:Path):
    """
    Exports the reviews table for an app to a CSV file.
    :param app_id: The Steam unique identifier for the application.
    :param path: The target directory; the file is named "{app_id}.csv".
    :return: None.
    """
    with db_connection(app_id) as conn:
        try:
            conn.execute(f"COPY reviews TO '{str(path.expanduser().joinpath((str(app_id)+'.csv')))}' (HEADER, DELIMITER ',')")
        except Exception as e:
            print(f"E: An error has occurred! {e}")


def _init_kpi_db(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Creates the daily_kpis table if it doesn't already exist.
    :param conn: An open DuckDB connection.
    :return: None.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_kpis (
            date DATE PRIMARY KEY,
            mean_concurrent DOUBLE,
            median_concurrent DOUBLE,
            peak_concurrent INTEGER,
            min_concurrent INTEGER,
            pct_change DOUBLE,
            trend VARCHAR
        )
    """)


def insert_daily_kpis(app_id: int, kpi_df: DataFrame) -> None:
    """
    Stores or updates daily KPI records in DuckDB.
    :param app_id: The Steam unique identifier for the application.
    :param kpi_df: A DataFrame matching the daily_kpis schema.
    :return: None.
    """
    if kpi_df.empty:
        return

    with db_connection(app_id) as conn:
        _init_kpi_db(conn)
        try:
            conn.execute("INSERT OR REPLACE INTO daily_kpis SELECT * FROM kpi_df")
        except Exception as e:
            print(f"E: Failed to store KPIs into DuckDB: {e}")


def get_daily_kpis_map(app_id: int) -> dict[datetime.date, str]:
    """
    Retrieves a map of date -> retention trend for transaction labeling.
    :param app_id: The Steam unique identifier for the application.
    :return: A dict mapping date to trend label (empty if the table doesn't exist yet).
    """
    with db_connection(app_id) as conn:
        try:
            cursor = conn.execute("SELECT date, trend FROM daily_kpis")
            return {row[0]: row[1] for row in cursor.fetchall()}
        except duckdb.CatalogException:
            return {}

def get_existing_kpi_dates(app_id: int) -> set[datetime.date]:
    """
    Returns the set of dates already stored in the daily_kpis table.
    :param app_id: The Steam unique identifier for the application.
    :return: A set of dates (empty if the table doesn't exist yet).
    """
    with db_connection(app_id) as conn:
        try:
            cursor = conn.execute("SELECT date FROM daily_kpis")
            return {row[0] for row in cursor.fetchall()}
        except duckdb.CatalogException:
            return set()


def _init_rules_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Creates the mining_runs and association_rules tables if they don't already exist.
    :param conn: An open DuckDB connection.
    :return: None.
    """
    # Table 1: run metadata
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS mining_runs
                 (
                     run_id
                     VARCHAR
                     PRIMARY
                     KEY,
                     timestamp
                     TIMESTAMP,
                     window_size_days
                     INTEGER,
                     total_reviews
                     INTEGER,
                     min_support
                     DOUBLE,
                     min_confidence
                     DOUBLE,
                     min_lift
                     DOUBLE,
                     deduplicate_mode
                     VARCHAR
                 )
                 """)

    # Table 2: mined association rules
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS association_rules
                 (
                     rule_id VARCHAR PRIMARY KEY,
                     run_id VARCHAR REFERENCES mining_runs(run_id),
                     antecedents VARCHAR,
                     consequents VARCHAR,
                     antecedent_support DOUBLE,
                     consequent_support DOUBLE,
                     support DOUBLE,
                     confidence DOUBLE,
                     lift DOUBLE,
                     leverage DOUBLE,
                     conviction DOUBLE
                     )
                 """)


def save_mining_results(
        app_id: int,
        rules_df: DataFrame,
        metadata: dict,
        params: dict
) -> str:
    """
    Stores a mining run and its generated rules into DuckDB.
    :param app_id: The Steam unique identifier for the application.
    :param rules_df: The DataFrame of mined association rules.
    :param metadata: Run metadata; may include "timestamp_end" for historical backfills, or "window_size_days"/"total_reviews".
    :param params: The mining parameters used (min_support, min_confidence, min_lift, deduplicate_mode).
    :return: The generated run_id, or "" if rules_df was empty.
    """
    if rules_df.empty:
        print("W: Rules DataFrame is empty. Skipping database insertion.")
        return ""

    # The run's timestamp must represent the date the rule REFERS TO, not the moment mining
    # was launched: Analyzer.run_historical() sets metadata["timestamp_end"] (the end of the
    # historical sliding window). If absent — as in Analyzer.run()'s daily cycle, which
    # reflects "as of today" — now() is used instead. Without this distinction, a historical
    # backfill (many runs launched in quick succession) would stack every run_id/timestamp on
    # the same execution instant instead of spreading across their real historical dates —
    # and matrix_generator.py, which joins association_rules to KPIs on this column, would
    # only see non-zero rule_lift_* features on the single day of the backfill, zeroed out
    # everywhere else.
    timestamp_end = metadata.get("timestamp_end")
    if timestamp_end is not None:
        now = datetime.datetime.fromtimestamp(timestamp_end, tz=datetime.timezone.utc)
    else:
        now = datetime.datetime.now(datetime.timezone.utc)

    run_id = generate_human_run_id(now)

    # noinspection PyUnusedLocal
    run_record = DataFrame([{
        "run_id": run_id,
        "timestamp": now,
        "window_size_days": metadata.get("window_size_days", 0),
        "total_reviews": metadata.get("total_reviews", 0),
        "min_support": params.get("min_support", 0.0),
        "min_confidence": params.get("min_confidence", 0.0),
        "min_lift": params.get("min_lift", 0.0),
        "deduplicate_mode": params.get("deduplicate_mode", "both")
    }])

    # noinspection PyUnusedLocal
    df_to_save = rules_df.copy()
    df_to_save["run_id"] = run_id
    df_to_save["rule_id"] = [str(uuid.uuid4()) for _ in range(len(df_to_save))]

    columns_map = {
        "antecedent support": "antecedent_support",
        "consequent support": "consequent_support"
    }
    df_to_save = df_to_save.rename(columns=columns_map)

    rules_columns = [
        "rule_id", "run_id", "antecedents", "consequents",
        "antecedent_support", "consequent_support", "support",
        "confidence", "lift", "leverage", "conviction"
    ]

    # Fill missing metrics if any (e.g. leverage/conviction depending on mlxtend version)
    for col in rules_columns:
        if col not in df_to_save.columns:
            df_to_save[col] = 0.0

    df_to_save = df_to_save[rules_columns]

    with db_connection(app_id) as conn:
        _init_rules_tables(conn)
        # NB: 'run_record' and 'df_to_save' are not passed explicitly to conn.execute() —
        # DuckDB resolves FROM table names by looking up variables with that name in the
        # calling Python scope (relation API). They are referenced by name inside the SQL
        # string, not unused; some IDEs flag this as a false positive.
        conn.execute("INSERT INTO mining_runs SELECT * FROM run_record")
        conn.execute("INSERT INTO association_rules SELECT * FROM df_to_save")

    return run_id

def list_mining_runs(app_id: int) -> DataFrame:
    """
    Returns a clean overview of all mining runs sorted by most recent.
    :param app_id: The Steam unique identifier for the application.
    :return: A DataFrame with run metadata and total rule counts.
    """
    with db_connection(app_id) as conn:
        query = """
            SELECT 
                m.run_id,
                m.timestamp,
                m.total_reviews,
                COUNT(r.rule_id) AS total_rules,
                m.min_support,
                m.min_confidence,
                m.min_lift
            FROM mining_runs m
            LEFT JOIN association_rules r ON m.run_id = r.run_id
            GROUP BY m.run_id, m.timestamp, m.total_reviews, m.min_support, m.min_confidence, m.min_lift
            ORDER BY m.timestamp DESC
        """
        return conn.execute(query).df()

def generate_human_run_id(timestamp: datetime.datetime) -> str:
    """
    Generates a human-readable run id from a timestamp.
    :param timestamp: The timestamp the run id should encode.
    :return: An id like "run_20260724_105626_a3f1".
    """
    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
    short_hash = uuid.uuid4().hex[:4]
    return f"run_{time_str}_{short_hash}"

def get_latest_run_id(app_id: int) -> str | None:
    """
    Fetches the run_id of the most recent mining run.
    :param app_id: The Steam unique identifier for the application.
    :return: The latest run_id, or None if no runs exist.
    """
    with db_connection(app_id) as conn:
        try:
            row = conn.execute(
                "SELECT run_id FROM mining_runs ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None
        except duckdb.CatalogException:
            return None


def export_all_rules(app_id: int, output_dir: str | Path | None = None) -> int:
    """
    Exports the association rules of every mining run in the DB to separate CSV files.
    :param app_id: The Steam unique identifier for the application.
    :param output_dir: The directory to export CSVs into; defaults to DB_ROOT/exports.
    :return: The number of runs successfully exported.
    """
    if output_dir is None:
        export_path = Path(DB_ROOT).expanduser() / "exports"
    else:
        export_path = Path(output_dir).expanduser()

    export_path.mkdir(parents=True, exist_ok=True)

    with db_connection(app_id) as conn:
        try:
            runs = conn.execute("SELECT run_id FROM mining_runs").fetchall()
        except duckdb.CatalogException:
            print(f"W: No rules table found for App ID {app_id}.")
            return 0

        if not runs:
            print(f"[*] No runs found in the database for App ID {app_id}.")
            return 0

        exported_count = 0
        for (run_id,) in runs:
            target_file = export_path / f"{app_id}_rules_{run_id}.csv"
            clean_path_str = str(target_file.resolve()).replace("'", "''")

            # No '?' parameter binding for the path here, since COPY TO needs a literal target
            export_query = f"""
                COPY (
                    SELECT antecedents, consequents, support, confidence, lift, leverage, conviction
                    FROM association_rules 
                    WHERE run_id = '{run_id}'
                    ORDER BY lift DESC
                ) TO '{clean_path_str}' (HEADER, DELIMITER ',');
            """

            try:
                conn.execute(export_query)
                print(f"[OK] Exported run '{run_id}' to:\n    {target_file}")
                exported_count += 1
            except Exception as e:
                print(f"E: Error exporting CSV for run {run_id}: {e}")

        return exported_count

def clear_mining_rules(app_id: int) -> None:
    """
    Deletes all association rules and mining runs recorded for an app.
    :param app_id: The Steam unique identifier for the application.
    :return: None.
    """
    with db_connection(app_id) as conn:
        try:
            conn.execute("DELETE FROM association_rules")
            conn.execute("DELETE FROM mining_runs")
            print(f"[OK] Cleanup complete: cleared all rules and runs for App ID {app_id}.")
        except duckdb.CatalogException:
            print(f"[*] No rules/runs table found to clean up for App ID {app_id}.")
        except Exception as e:
            print(f"E: Error while deleting rules: {e}")


def _init_tuning_db(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Creates the hyperparameter_tuning table if it doesn't already exist.
    :param conn: An open DuckDB connection.
    :return: None.
    """
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS hyperparameter_tuning
                 (
                     tuning_id VARCHAR PRIMARY KEY,
                     tier_size INTEGER,
                     dataset_size_at_tuning INTEGER,
                     timestamp TIMESTAMP,
                     params VARCHAR,
                     score DOUBLE,
                     mape_multistep_pct DOUBLE,
                     mape_1step_pct DOUBLE,
                     mape_multistep_std DOUBLE,
                     n_trials INTEGER,
                     n_folds INTEGER
                 )
                 """)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    """
    Converts a value to float, explicitly handling None (float(None) would raise TypeError).
    :param value: The value to convert.
    :param default: The value to return if `value` is None.
    :return: The converted float, or `default` if `value` was None.
    """
    return float(value) if value is not None else default


def serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes a hyperparameter dict for DuckDB/JSON storage by stripping non-serializable callables.
    :param params: A dict with "xgb_params" and "decay_params" keys.
    :return: A dict with the same structure, with any callable "objective" removed from xgb_params.
    """
    xgb_params = dict(params.get("xgb_params", {}))
    decay_params = dict(params.get("decay_params", {}))

    if "objective" in xgb_params and callable(xgb_params["objective"]):
        xgb_params.pop("objective", None)

    return {
        "xgb_params": xgb_params,
        "decay_params": decay_params,
    }


def save_tuned_params(
        app_id: int,
        tier_size: int,
        dataset_size_at_tuning: int,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        n_trials: int,
        n_folds: int,
) -> str:
    """
    Saves a tuned hyperparameter record into DuckDB, replacing any prior record for the same tier.
    :param app_id: The Steam unique identifier for the application.
    :param tier_size: The dataset-size tier these parameters were tuned for.
    :param dataset_size_at_tuning: The full dataset size at the time tuning was run.
    :param params: The hyperparameter dict (see serialize_params for the expected structure).
    :param metrics: The tuning result metrics (score, mape_multistep_pct, mape_1step_pct, mape_multistep_std).
    :param n_trials: The number of Optuna trials run.
    :param n_folds: The number of walk-forward folds used.
    :return: The generated tuning_id.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    time_str = now.strftime("%Y%m%d_%H%M%S")
    tuning_id = f"tune_{time_str}_{uuid.uuid4().hex[:4]}"

    sanitized_params = serialize_params(params)

    record = DataFrame([{
        "tuning_id": tuning_id,
        "tier_size": tier_size,
        "dataset_size_at_tuning": dataset_size_at_tuning,
        "timestamp": now,
        "params": json.dumps(sanitized_params),
        "score": _safe_float(metrics.get("score")),
        "mape_multistep_pct": _safe_float(metrics.get("mape_multistep_pct")),
        "mape_1step_pct": _safe_float(metrics.get("mape_1step_pct")),
        "mape_multistep_std": _safe_float(metrics.get("mape_multistep_std")),
        "n_trials": n_trials,
        "n_folds": n_folds,
    }])

    with db_connection(app_id) as conn:
        _init_tuning_db(conn)
        conn.execute("DELETE FROM hyperparameter_tuning WHERE tier_size = ?", [tier_size])
        conn.execute("INSERT INTO hyperparameter_tuning SELECT * FROM record")

    return tuning_id


def list_tuning_history(app_id: int) -> DataFrame:
    """
    Returns the full tuning history for an app, ordered by increasing tier size.
    :param app_id: The Steam unique identifier for the application.
    :return: A DataFrame of tuning records (empty if the table doesn't exist yet).
    """
    with db_connection(app_id) as conn:
        try:
            return conn.execute(
                "SELECT * FROM hyperparameter_tuning ORDER BY tier_size ASC"
            ).df()
        except duckdb.CatalogException:
            return DataFrame()


def get_latest_tuning_record(app_id: int) -> Optional["Series"]:
    """
    Returns the most recent tuning record by timestamp, regardless of tier size.
    :param app_id: The Steam unique identifier for the application.
    :return: A single-row pandas Series, or None if no tuning history exists.
    """
    history = list_tuning_history(app_id)
    if history.empty:
        return None
    return history.loc[history["timestamp"].idxmax()]


def get_params_for_size(app_id: int, n_samples: int) -> Optional[Dict[str, Any]]:
    """
    Fetches the best-fit tuned hyperparameters for a given dataset size.
    :param app_id: The Steam unique identifier for the application.
    :param n_samples: The current dataset size to find parameters for.
    :return: The parsed hyperparameter dict for the largest tuned tier not exceeding n_samples (or the smallest tier available if none qualify), or None if no tuning history exists.
    """
    history = list_tuning_history(app_id)
    if history.empty:
        return None

    eligible = history[history["tier_size"] <= n_samples]
    row = eligible.loc[eligible["tier_size"].idxmax()] if not eligible.empty \
        else history.loc[history["tier_size"].idxmin()]

    return json.loads(str(row["params"]))