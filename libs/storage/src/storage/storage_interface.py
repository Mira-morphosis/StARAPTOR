import datetime
from evaluator.evaluator_interface import ReviewProcessor
from fetcher.fetcher_interface import fetch_review_history, fetch_kpi
from pandas import DataFrame
from storage.storage_utils import get_existing_review_ids, insert_reviews, insert_daily_kpis, get_existing_kpi_dates


async def review_updater(
        app_id: int,
        max_days: int | None = None,
        num_reviews: int = 250,
        top_n: int = 125,
        delay: float = 2.0,
        verbose: bool = False
):
    """
    Fetches reviews from the last max_days days, runs them through the evaluator, and stores the results.
    :param app_id: The Steam unique identifier for the application.
    :param max_days: The maximum number of days to go back to.
    :param num_reviews: The maximum number of reviews to fetch.
    :param top_n: The operation will select the top_n best reviews and (num_reviews - top_n) random reviews.
    :param delay: The delay in seconds between each request.
    :param verbose: Regulates logging output.
    :return: None.
    """
    try:
        stored_reviews = get_existing_review_ids(app_id)
    except NotADirectoryError as e:
        print(f"FATAL: {e}")
        exit(1)

    processor = ReviewProcessor()

    time_start = datetime.datetime.now()
    if verbose:
        print(f"Starting historical fetch for App ID {app_id}...")

    all_raw_reviews = await fetch_review_history(
        app_id=app_id,
        max_days=max_days,
        existing_ids=stored_reviews,
        num_reviews=num_reviews,
        top_n=top_n,
        delay=delay,
        verbose=verbose
    )

    if not all_raw_reviews:
        if verbose:
            print("No new historical reviews found to process.")
        return

    if verbose:
        print(f"Processing total history of {len(all_raw_reviews)} items with the evaluator")

    processed_reviews = await processor.process_batch(all_raw_reviews)

    if processed_reviews:
        insert_reviews(app_id, processed_reviews)
        if verbose:
            print(f"Successfully stored {len(processed_reviews)} filtered reviews into DuckDB.")
            print(f"Total time required: {(datetime.datetime.now() - time_start).total_seconds()}")
    elif verbose:
        print("All fetched reviews were skipped after spam filtering.")


async def kpi_updater(
    app_id: int,
    max_days: int | None = None,
    verbose: bool = False
):
    """
    Fetches historical KPI data from SteamCharts and inserts any missing daily records into DuckDB.
    :param app_id: The Steam unique identifier for the application.
    :param max_days: The maximum number of days to go back to (capped at 140).
    :param verbose: Regulates logging output.
    :return: None.
    """

    # SteamCharts only exposes this much history through the current API.
    HARD_LIMIT_DAYS = 140
    if max_days is None or max_days > HARD_LIMIT_DAYS:
        max_days = HARD_LIMIT_DAYS

    try:
        stored_dates = get_existing_kpi_dates(app_id)
    except NotADirectoryError as e:
        print(f"FATAL: {e}")
        exit(1)

    time_start = datetime.datetime.now()
    if verbose:
        print(f"Starting historical KPI fetch for App ID {app_id}...")

    kpi_df: DataFrame = await fetch_kpi(app_id)

    if kpi_df.empty:
        if verbose:
            print("No KPI data returned from fetcher.")
        return

    if stored_dates:
        kpi_df = kpi_df[~kpi_df["date"].isin(stored_dates)]

    if max_days is not None and not kpi_df.empty:
        kpi_df = kpi_df.sort_values("date").tail(max_days)

    if kpi_df.empty:
        if verbose:
            print("KPI database is already up to date. No new records to insert.")
        return

    if verbose:
        print(f"Storing {len(kpi_df)} missing daily KPI snapshots...")

    insert_daily_kpis(app_id, kpi_df)

    if verbose:
        elapsed = (datetime.datetime.now() - time_start).total_seconds()
        print(f"Successfully stored {len(kpi_df)} daily KPI records into DuckDB.")
        print(f"Total time required: {elapsed:.2f}s")