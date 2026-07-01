from fetcher.interface import fetch_periodically
from storage.storage_utils import get_existing_review_ids, insert_reviews


async def storage_updater(
    app_id: int,
    delay: float = 2.0,
    max_batches: int|None = None,
    verbose: bool = False
):
    try:
        stored_reviews = get_existing_review_ids(app_id)
    except NotADirectoryError as e:
        print(f"FATAL: {e}")
        exit(1)

    new_reviews = await fetch_periodically(
        app_id=app_id,
        delay=delay,
        existing_ids=stored_reviews,
        max_batches= max_batches,
        verbose = verbose
    )
    if new_reviews:
        insert_reviews(app_id, new_reviews)
        if verbose:
            print(f"Successfully stored {len(new_reviews)} reviews.")