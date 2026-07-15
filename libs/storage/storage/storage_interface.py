import datetime

from fetcher.fetcher_interface import fetch_review_history
from storage.storage_utils import get_existing_review_ids, insert_reviews
from evaluator.evaluator_interface import ReviewProcessor


async def review_updater(
        app_id: int,
        max_days: int | None = None,
        num_reviews: int = 250,
        top_n: int = 125,
        delay: float = 2.0,
        verbose: bool = False
):
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