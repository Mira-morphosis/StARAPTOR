import asyncio

import httpx

from fetcher_singleton import fetcher


async def fetch(fetcher_instance:httpx.AsyncClient, app_id: int, cursor: str = "*") -> tuple[dict, dict, str]:
    """
    Fetches Reviews from Steam's store API
    :param fetcher_instance: The client to use
    :param app_id: The game's appid
    :param cursor: (Optional) pass this parameter to continue retrieval from where it stopped.
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
        max_batches: int|None = None,
        verbose:bool = False
) -> list[dict]:
    all_reviews = []
    batches_fetched: int = 0
    while max_batches is None or batches_fetched < max_batches:
        summary, reviews, cursor = await fetch(fetcher.get(), app_id, cursor)

        if not reviews:
            if verbose:
                print("Nothing to show")
            break

        if verbose:
            print(reviews)
        all_reviews.extend(reviews)
        batches_fetched += 1

        await asyncio.sleep(delay)
    await fetcher.close_client()
    return all_reviews


async def test():
    try:
        await fetch_periodically(app_id=730, max_batches=5, verbose=True)
    finally:
        print("Fetcher is closing...")


if __name__ == "__main__":
    asyncio.run(test())
