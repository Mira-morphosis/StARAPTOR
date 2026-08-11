import asyncio
import datetime
import random
from typing import AsyncGenerator
from pandas import DataFrame, to_datetime

import httpx

from fetcher.fetcher_singleton import fetcher


async def fetch(fetcher_instance:httpx.AsyncClient, app_id: int, cursor: str = "*") -> tuple[dict, dict, str]:
    """
    Fetches Reviews from Steam's store API
    :param fetcher_instance: The client to use
    :param app_id: The game's appid
    :param cursor: (Optional) Continues retrieval from where it stopped.
    :return: The tuple (JSON query summary, JSON review batch, cursor) as [dict, dict, str]
    """

    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "json": 1,
        "filter": "recent",
        "language": "all",
        "num_per_page": 100,
        "cursor": cursor
    }
    response = await fetcher_instance.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    summary = data.get("query_summary", [])
    reviews = data.get("reviews", [])
    next_cursor = data.get("cursor", "")
    return summary, reviews, next_cursor


async def fetch_periodically(
        app_id: int,
        cursor: str = "*",
        delay: float = 2.0,
        max_batches: int | None = None,
        existing_ids: set[str] | None = None,
        verbose: bool = False
) -> AsyncGenerator[list[dict], None]:
    """
    Yields cleaned batches incrementally as they are fetched.
    Legacy function, as the project needs dynamic sample generation.
    Use fetch_today().
    """
    batches_fetched: int = 0
    known_ids = existing_ids or set()
    stop_fetching = False

    while (max_batches is None or batches_fetched < max_batches) and not stop_fetching:
        summary, reviews, cursor = await fetch(fetcher.get(), app_id, cursor)

        if not reviews:
            if verbose:
                print("Nothing to show")
            break

        cleaned_batch = []
        for r in reviews:
            cleaned = clean_and_flatten_review(r)
            review_id = cleaned.get('recommendationid')

            if review_id in known_ids:
                if verbose:
                    print(f"{review_id} is already in the set. Stopping fetch...")
                stop_fetching = True
                break

            cleaned_batch.append(cleaned)

        if cleaned_batch:
            yield cleaned_batch  # Stream current batch immediately

        batches_fetched += 1

        if not stop_fetching and (max_batches is None or batches_fetched < max_batches):
            await asyncio.sleep(delay)

    await fetcher.close_client()

async def fetch_single_day(
        app_id: int,
        target_date: datetime.date = datetime.date.today(),
        existing_ids: set[str] | None = None,
        num_reviews: int = 250,
        top_n: int = 125,
        delay: float = 2.0,
        verbose: bool = False,
        cursor: str = "*"
) -> tuple[list[dict], str]:
    """
    Yields a sample of num_reviews cleaned reviews or less from a date that contains the top top_n reviews
    plus random reviews.
    :returns: A tuple consisting of the list of reviews and a cursor
    :param app_id: The Steam unique identifier for the application.
    :param target_date: The day targeted by the operation.
    :param existing_ids: The set of ids of already collected reviews.
    :param num_reviews: The maximum number of reviews to fetch.
    :param top_n: The operation will select the top_n best reviews and (num_reviews - top_n) random reviews.
    :param delay: The delay in seconds between each request
    :param verbose: Regulates logging output
    :param cursor: You may pass here a cursor from a precedent function call to avoid fetching the same reviews.
    """

    known_ids = existing_ids or set()
    stop_fetching = False

    date_start = datetime.datetime.combine(
        target_date, datetime.time.min, tzinfo=datetime.timezone.utc
    )
    date_end = datetime.datetime.combine(
        target_date, datetime.time.max, tzinfo=datetime.timezone.utc
    )

    date_start_timestamp = int(date_start.timestamp())
    date_end_timestamp = int(date_end.timestamp())

    all_fetched_reviews = []

    while not stop_fetching:
        summary, reviews, cursor = await fetch(fetcher.get(), app_id, cursor)

        if not reviews:
            if verbose:
                print("Nothing to show")
            break

        current_batch_ended = False
        for r in reviews:
            timestamp = int(r.get("timestamp_created", 0))
            if timestamp > date_end_timestamp:
                continue
            if timestamp < date_start_timestamp:
                stop_fetching = True
                current_batch_ended = True
                break

            cleaned = clean_and_flatten_review(r)
            review_id = cleaned.get('recommendationid')

            if review_id in known_ids:
                if verbose:
                    print(f"{review_id} is already in the set. Skipping duplicate...")
                continue

            all_fetched_reviews.append(cleaned)

        if current_batch_ended:
            break

        if not stop_fetching:
            await asyncio.sleep(delay)

    if len(all_fetched_reviews) <= num_reviews:
        return all_fetched_reviews, cursor

    indexed_reviews = list(enumerate(all_fetched_reviews))
    indexed_reviews.sort(
        key=lambda x: x[1].get("weighted_vote_score", 0.0),
        reverse=True
    )

    top_indices = {idx for idx, _ in indexed_reviews[:top_n]}
    remaining_indexed = [item for item in indexed_reviews if item[0] not in top_indices]

    random_sample_size = min(num_reviews - top_n, len(remaining_indexed))
    random_indices = {idx for idx, _ in random.sample(remaining_indexed, random_sample_size)}

    selected_indices = top_indices.union(random_indices)
    return [all_fetched_reviews[i] for i in range(len(all_fetched_reviews)) if i in selected_indices],cursor

async def fetch_today(
        app_id: int,
        existing_ids: set[str] | None = None,
        num_reviews: int = 250,
        top_n: int = 125,
        delay: float = 2.0,
        verbose: bool = False
) -> list[dict]:
    res, cursor = await fetch_single_day(app_id, datetime.date.today(), existing_ids, num_reviews, top_n, delay, verbose)
    await fetcher.close_client()
    return res

async def fetch_review_history(
        app_id: int,
        max_days: int|None = None,
        existing_ids: set[str] | None = None,
        num_reviews: int = 250,
        top_n: int = 125,
        delay: float = 2.0,
        verbose: bool = False
)-> list[dict]:
    """
    Loops backward day by day, fetching reviews for each day up to max_days,
    aggregating them into a single list.
    :returns: A tuple consisting of the list of reviews and a cursor
    :param app_id: The Steam unique identifier for the application.
    :param max_days: The maximum number of days to go back to.
    :param existing_ids: The set of ids of already collected reviews.
    :param num_reviews: The maximum number of reviews to fetch.
    :param top_n: The operation will select the top_n best reviews and (num_reviews - top_n) random reviews.
    :param delay: The delay in seconds between each request
    :param verbose: Regulates logging output.
    """

    #Until a better KPI API is found, this is a limit we have to bear with.
    HARD_LIMIT_DAYS = 140
    if max_days is None or max_days > HARD_LIMIT_DAYS:
        max_days = HARD_LIMIT_DAYS

    all_reviews = []
    known_ids = existing_ids or set()
    history_cursor = "*"

    # Starts the loop from today's date
    current_target_date = datetime.date.today()
    days_fetched = 0

    while True:
        # Checks condition for breaking if max_days is set
        if max_days is not None and days_fetched >= max_days:
            if verbose:
                print(f"Reached max_days limit ({max_days}). Stopping.")
            break

        if verbose:
            print(f"[{days_fetched + 1}] Fetching reviews for: {current_target_date}")

        # Calls single-day fetcher
        day_reviews, history_cursor = await fetch_single_day(
            app_id=app_id,
            target_date=current_target_date,
            existing_ids=known_ids,
            num_reviews=num_reviews,
            top_n=top_n,
            delay=delay,
            verbose=verbose,
            cursor=history_cursor
        )

        if not day_reviews and history_cursor == "":
            if verbose:
                print("Steam returned an empty cursor. No more historical reviews exist. Stopping.")
            break

        if day_reviews:
            all_reviews.extend(day_reviews)
            # Update our tracking set to avoid duplicates across days
            for r in day_reviews:
                if 'recommendationid' in r:
                    known_ids.add(r['recommendationid'])

        # Shift target to the day before
        current_target_date -= datetime.timedelta(days=1)
        days_fetched += 1

    await fetcher.close_client()
    return all_reviews


def clean_and_flatten_review(review_dict: dict) -> dict:
    """
    Cleans "noisy" parameters from reviews
    :param review_dict: The review to be cleaned
    """
    author_data = review_dict.get("author", {})
    hardware_data = review_dict.get("hardware", {})

    # Initializes to defaults if data is empty
    cleaned = {
        "recommendationid": str(review_dict.get("recommendationid", "")),  # Converti in stringa per matchare VARCHAR
        "review": review_dict.get("review", ""),
        "language": review_dict.get("language", ""),
        "timestamp_updated":int(review_dict.get("timestamp_updated", datetime.datetime.today().timestamp())),
        "steam_purchase": bool(review_dict.get("steam_purchase", False)),
        "received_for_free": bool(review_dict.get("received_for_free", False)),
        "voted_up": bool(review_dict.get("voted_up", False)),
        "weighted_vote_score": float(review_dict.get("weighted_vote_score", 0.0)),
        "playtime_at_review": int(author_data.get("playtime_at_review", 0)),
        "primarily_steam_deck": bool(review_dict.get("primarily_steam_deck", False)),
        "has_hardware_info": False,
        "hw_resolution": "N/A",
        "hw_ram": "N/A",
        "hw_vram": "N/A",
        "hw_gpu": "N/A",
        "hw_os": "N/A",
        "hw_cpu_vendor": "N/A"
    }
    if isinstance(hardware_data, dict) and hardware_data:
        cleaned["has_hardware_info"] = True

        try:
            ram_mb = int(hardware_data.get("system_ram", 0))
            ram_gb = f"{round(ram_mb / 1024)}GB" if ram_mb > 0 else "Unknown"
        except (ValueError, TypeError):
            ram_gb = "Unknown"

        try:
            vram_mb = int(hardware_data.get("vram_size", 0))
            vram_gb = f"{round(vram_mb / 1024)}GB" if vram_mb > 0 else "Unknown"
        except (ValueError, TypeError):
            vram_gb = "Unknown"

        width = hardware_data.get("screen_width")
        height = hardware_data.get("screen_height")
        resolution = f"{width}x{height}" if width and height else "Unknown"

        cleaned.update({
            "hw_os": str(hardware_data.get("os", "Unknown")),
            "hw_gpu": str(hardware_data.get("dx_video_card", "Unknown")),
            "hw_cpu_vendor": str(hardware_data.get("cpu_vendor", "Unknown")),
            "hw_ram": ram_gb,
            "hw_vram": vram_gb,
            "hw_resolution": resolution
        })

    return cleaned

async def fetch_kpi(app_id:int) -> DataFrame:
    url = f"https://steamcharts.com/app/{app_id}/chart-data.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        raw_data = response.json()  # Expected format: [[timestamp_ms, player_count], ...]
        if not raw_data:
            return DataFrame()
        df = DataFrame(raw_data, columns=["timestamp", "players"])
        df["date"] = to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
        daily = df.groupby("date")["players"].agg(
            mean_concurrent="mean",
            median_concurrent="median",
            peak_concurrent="max",
            min_concurrent="min"
        ).reset_index()
        daily = daily.sort_values("date").reset_index(drop=True)
        daily["pct_change"] = daily["peak_concurrent"].pct_change() * 100.0
        daily["pct_change"] = daily["pct_change"].fillna(0.0)

        def categorize_trend(pct: float) -> str:
            if pct >= 5.0:
                return "retention_growth"
            elif pct <= -5.0:
                return "retention_drop"
            else:
                return "retention_stable"

        daily["trend"] = daily["pct_change"].apply(categorize_trend)
        return daily