import asyncio

from fetcher_singleton import fetcher


async def fetch(app_id: int, cursor: str = "*") -> tuple[dict, dict, str]:
    """
    Fetches Reviews from Steam's store API
    :param app_id: The game's appid
    :param cursor: (Optional) pass this parameter to continue retrieval from where it stopped.
    :return: The tuple (JSON query summary, JSON review batch, cursor) as [dict, dict, str]
    """
    client = fetcher.get()

    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "json": 1,
        "filter": "recent",
        "language": "all",
        "num_per_page": 100,
        "cursor": cursor
    }
    response = await client.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    summary = data.get("query_summary", [])
    reviews = data.get("reviews", [])
    next_cursor = data.get("cursor", "")
    return summary, reviews, next_cursor

async def test():
    try:
        res = await fetch(730) #CS2
        print(res[0])
        print(res[1])
        print(res[2])
    finally:
        await fetcher.close_client()
        print("Fetcher is closing...")

if __name__ == "__main__":
    asyncio.run(test())