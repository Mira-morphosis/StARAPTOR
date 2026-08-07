import datetime
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Final, Generator, Optional, Tuple

from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()
DB_ROOT: Final = os.getenv("STARAPTOR_DB_ROOT", "~/.staraptor/")

import duckdb


def _resolve_db_path(app_id: int) -> Path:
    """Resolves DB Path and implements fallbacks"""
    if DB_ROOT is None:
        print("W: .env is missing or is not set up correctly. Forcing defaults... (~/.staraptor/)")
        default_folder = Path("~/.staraptor/").expanduser()
        default_folder.mkdir(parents=True, exist_ok=True)
        return default_folder / f"{app_id}.db"

    db_folder = Path(DB_ROOT).expanduser()

    if db_folder.exists() and not db_folder.is_dir():
        # Fatal error raised for the handler
        raise NotADirectoryError(f"{DB_ROOT} is not a valid folder.")

    if not db_folder.exists():
        db_folder.mkdir(parents=True, exist_ok=True)

    return db_folder / f"{app_id}.db"


@contextmanager
def db_connection(app_id: int) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    # If raises NotADirectoryError, the exception is passed to the caller.
    db_path = _resolve_db_path(app_id)

    conn = duckdb.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


def get_existing_review_ids(app_id: int) -> set[str]:
    with db_connection(app_id) as conn:
        try:
            cursor = conn.execute("SELECT recommendationid FROM reviews")
            return {str(row[0]) for row in cursor.fetchall()}
        except duckdb.CatalogException:  #CatalogException = Database is created already
            return set()


def _init_reviews_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Initializes the review table"""
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
    Stores the JSON reviews into the app database.
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

        # We extract keys from the first element of the list
    columns = list(reviews_list[0].keys())

    # Placeholders are prepared so that DuckDB can dynamically build the query (e.g., $recommendationid, $review, ...)
    placeholders = ", ".join([f"${col}" for col in columns])

    query = f"""
            INSERT OR IGNORE INTO reviews ({", ".join(columns)}) 
            VALUES ({placeholders})
        """

    with db_connection(app_id) as conn:
        _init_reviews_db(conn)

        try:
            # DuckDB automatically maps the $column keys
            conn.executemany(query, reviews_list)

        except Exception as e:
            print(f"E: An error has occurred! {e}")
            import traceback
            traceback.print_exc()


def export(app_id: int, path:Path):
    with db_connection(app_id) as conn:
        try:
            conn.execute(f"COPY reviews TO '{str(path.expanduser().joinpath((str(app_id)+'.csv')))}' (HEADER, DELIMITER ',')")
        except Exception as e:
            print(f"E: An error has occurred! {e}")


def _init_kpi_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Initializes the daily KPIs table."""
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
    """Stores or updates daily KPI records in DuckDB."""
    if kpi_df.empty:
        return

    with db_connection(app_id) as conn:
        _init_kpi_db(conn)
        try:
            # DuckDB natively ingests pandas DataFrames directly
            conn.execute("INSERT OR REPLACE INTO daily_kpis SELECT * FROM kpi_df")
        except Exception as e:
            print(f"E: Failed to store KPIs into DuckDB: {e}")


def get_daily_kpis_map(app_id: int) -> dict[datetime.date, str]:
    """Retrieves a map of date -> retention trend for transaction labeling."""
    with db_connection(app_id) as conn:
        try:
            cursor = conn.execute("SELECT date, trend FROM daily_kpis")
            return {row[0]: row[1] for row in cursor.fetchall()}
        except duckdb.CatalogException:
            return {}

def get_existing_kpi_dates(app_id: int) -> set[datetime.date]:
    """Returns the set of dates already stored in the daily_kpis table."""
    with db_connection(app_id) as conn:
        try:
            cursor = conn.execute("SELECT date FROM daily_kpis")
            return {row[0] for row in cursor.fetchall()}
        except duckdb.CatalogException:
            return set()


def _init_rules_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Initializes tables for association rule execution history and rule results."""
    # Table 1: Run Metadata
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

    # Table 2: Mined Association Rules
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
    Returns the generated run_id.
    """
    if rules_df.empty:
        print("W: Rules DataFrame is empty. Skipping database insertion.")
        return ""

    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = generate_human_run_id(now)

    # Prepare Run Metadata record
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

    # Prepare Rules records linked to run_id
    df_to_save = rules_df.copy()
    df_to_save["run_id"] = run_id
    df_to_save["rule_id"] = [str(uuid.uuid4()) for _ in range(len(df_to_save))]

    # Ensure column naming matches DuckDB schema
    columns_map = {
        "antecedent support": "antecedent_support",
        "consequent support": "consequent_support"
    }
    df_to_save = df_to_save.rename(columns=columns_map)

    # Select only relevant columns
    rules_columns = [
        "rule_id", "run_id", "antecedents", "consequents",
        "antecedent_support", "consequent_support", "support",
        "confidence", "lift", "leverage", "conviction"
    ]

    # Fill missing metrics if any (e.g., leverage/conviction depending on mlxtend version)
    for col in rules_columns:
        if col not in df_to_save.columns:
            df_to_save[col] = 0.0

    df_to_save = df_to_save[rules_columns]

    with db_connection(app_id) as conn:
        _init_rules_tables(conn)
        # Insert Run metadata
        conn.execute("INSERT INTO mining_runs SELECT * FROM run_record")
        # Insert batch of Rules directly from Pandas DataFrame
        conn.execute("INSERT INTO association_rules SELECT * FROM df_to_save")

    return run_id

def list_mining_runs(app_id: int) -> DataFrame:
    """Returns a clean overview of all mining runs sorted by most recent."""
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
    """Generates an ID like: run_20260724_105626_a3f1"""
    time_str = timestamp.strftime("%Y%m%d_%H%M%S")
    short_hash = uuid.uuid4().hex[:4]
    return f"run_{time_str}_{short_hash}"

def get_latest_run_id(app_id: int) -> str | None:
    """Fetches the run_id of the most recent mining run."""
    with db_connection(app_id) as conn:
        try:
            row = conn.execute(
                "SELECT run_id FROM mining_runs ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None


def export_all_rules(app_id: int, output_dir: str | Path | None = None) -> int:
    """
    Esporta le regole di TUTTE le run presenti nel DB in file CSV distinti.
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
            print(f"W: Nessuna tabella delle regole trovata per App ID {app_id}.")
            return 0

        if not runs:
            print(f"[*] Nessuna run trovata nel database per App ID {app_id}.")
            return 0

        exported_count = 0
        for (run_id,) in runs:
            target_file = export_path / f"{app_id}_rules_{run_id}.csv"
            clean_path_str = str(target_file.resolve()).replace("'", "''")

            # Formattazione corretta della query COPY TO senza parametro '?' per il path
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
                print(f"[✓] Esportata run '{run_id}' in:\n    {target_file}")
                exported_count += 1
            except Exception as e:
                print(f"E: Errore durante l'esportazione CSV della run {run_id}: {e}")

        return exported_count

def clear_mining_rules(app_id: int) -> None:
    """
    Cancella tutte le regole di associazione e le esecuzioni registrate per un determinato App ID.
    """
    with db_connection(app_id) as conn:
        try:
            conn.execute("DELETE FROM association_rules")
            conn.execute("DELETE FROM mining_runs")
            print(f"[✓] Pulizia completata: cancellate tutte le regole e run precedenti per App ID {app_id}.")
        except duckdb.CatalogException:
            print(f"[*] Nessuna tabella regole/run da ripulire trovata per App ID {app_id}.")
        except Exception as e:
            print(f"E: Errore durante la cancellazione delle regole: {e}")


# =====================================================================================
# HYPERPARAMETER TUNING STORAGE — analoga a mining_runs/association_rules: una tabella
# per-app_id nello stesso file .db, che tiene traccia degli iperparametri XGBoost
# ottimizzati per ciascun tier di dimensione dataset (vedi analyzer/tuner.py).
# =====================================================================================

def _init_tuning_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Inizializza la tabella degli iperparametri tunati per tier di dimensione dataset."""
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
    Salva (o sostituisce) gli iperparametri migliori per un dato tier di dimensione
    dataset. Un solo record per tier viene mantenuto: se il tier esiste già,
    il vecchio tuning viene sovrascritto da quello nuovo.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    time_str = now.strftime("%Y%m%d_%H%M%S")
    tuning_id = f"tune_{time_str}_{uuid.uuid4().hex[:4]}"

    record = DataFrame([{
        "tuning_id": tuning_id,
        "tier_size": tier_size,
        "dataset_size_at_tuning": dataset_size_at_tuning,
        "timestamp": now,
        "params": json.dumps(params),
        "score": float(metrics.get("score", float("nan"))),
        "mape_multistep_pct": float(metrics.get("mape_multistep_pct", float("nan"))),
        "mape_1step_pct": float(metrics.get("mape_1step_pct", float("nan"))),
        "mape_multistep_std": float(metrics.get("mape_multistep_std", float("nan"))),
        "n_trials": n_trials,
        "n_folds": n_folds,
    }])

    with db_connection(app_id) as conn:
        _init_tuning_db(conn)
        # Un solo record per tier: rimuove il tuning precedente sullo stesso tier, se esiste.
        conn.execute("DELETE FROM hyperparameter_tuning WHERE tier_size = ?", [tier_size])
        conn.execute("INSERT INTO hyperparameter_tuning SELECT * FROM record")

    return tuning_id


def list_tuning_history(app_id: int) -> DataFrame:
    """Ritorna lo storico completo dei tuning per l'App ID, ordinato per tier crescente."""
    with db_connection(app_id) as conn:
        try:
            return conn.execute(
                "SELECT * FROM hyperparameter_tuning ORDER BY tier_size ASC"
            ).df()
        except duckdb.CatalogException:
            return DataFrame()


def get_latest_tuning_record(app_id: int) -> Optional["DataFrame"]:
    """Ritorna il record di tuning più recente per timestamp (indipendentemente dal tier)."""
    history = list_tuning_history(app_id)
    if history.empty:
        return None
    return history.loc[history["timestamp"].idxmax()]


def get_params_for_size(app_id: int, n_samples: int) -> Optional[Dict[str, Any]]:
    """
    Seleziona gli iperparametri del tier più adatto a n_samples righe correnti.
    Usa il tier più grande tunato che non supera n_samples; se n_samples è sotto
    il tier minimo mai tunato, ricade sul tier più piccolo disponibile (cold-start).
    Ritorna None se non esiste alcun tuning storicizzato per l'App ID.
    """
    history = list_tuning_history(app_id)
    if history.empty:
        return None

    eligible = history[history["tier_size"] <= n_samples]
    row = eligible.loc[eligible["tier_size"].idxmax()] if not eligible.empty \
        else history.loc[history["tier_size"].idxmin()]

    return json.loads(row["params"])