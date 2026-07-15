import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Generator

from dotenv import load_dotenv

load_dotenv()
DB_ROOT: Final = os.getenv("STARAPTOR_DB_ROOT")

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


def _init_db(conn: duckdb.DuckDBPyConnection) -> None:
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
        _init_db(conn)

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
