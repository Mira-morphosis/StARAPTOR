import asyncio

import httpx

from fetcher.fetcher_singleton import fetcher


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

        cleaned_batch = [clean_and_flatten_review(r) for r in reviews]
        all_reviews.extend(cleaned_batch)
        batches_fetched += 1

        await asyncio.sleep(delay)
    await fetcher.close_client()
    # TODO: Discretization
    return all_reviews


def clean_and_flatten_review(review_dict: dict) -> dict:
    """
    Cleans "noisy" parameters from reviews
    :param review_dict: The review to be cleaned
    """
    author_data = review_dict.get("author", {})
    hardware_data = review_dict.get("hardware", {})

    cleaned = {
        # review body
        "review": review_dict.get("review", ""),
        "language": review_dict.get("language", ""),

        # metadata
        "voted_up": review_dict.get("voted_up"),
        "steam_purchase": review_dict.get("steam_purchase"),
        "received_for_free": review_dict.get("received_for_free"),
        "primarily_steam_deck": review_dict.get("primarily_steam_deck"),
        "hardware": review_dict.get("hardware", "Unknown"),

        # Numeric data
        "playtime_at_review": author_data.get("playtime_at_review", 0),
    }
    # Hardware info
    if isinstance(hardware_data, dict) and hardware_data:
        # ram to GB
        try:
            ram_mb = int(hardware_data.get("system_ram", 0))
            ram_gb = f"{round(ram_mb / 1024)}GB" if ram_mb > 0 else "Unknown"
        except (ValueError, TypeError):
            ram_gb = "Unknown"

        # VRAM to GB
        try:
            vram_mb = int(hardware_data.get("vram_size", 0))
            vram_gb = f"{round(vram_mb / 1024)}GB" if vram_mb > 0 else "Unknown"
        except (ValueError, TypeError):
            vram_gb = "Unknown"

        # Resolution
        width = hardware_data.get("screen_width")
        height = hardware_data.get("screen_height")
        resolution = f"{width}x{height}" if width and height else "Unknown"

        # Add cleaned and flattened hardware data
        cleaned.update({
            "has_hardware_info": True,
            "hw_os": hardware_data.get("os", "Unknown"),
            "hw_gpu": hardware_data.get("dx_video_card", "Unknown"),
            "hw_cpu_vendor": hardware_data.get("cpu_vendor", "Unknown"),
            "hw_ram": ram_gb,
            "hw_vram": vram_gb,
            "hw_resolution": resolution
        })
    else:
        cleaned.update({
            "has_hardware_info": False,
            "hw_os": "N/A",
            "hw_gpu": "N/A",
            "hw_cpu_vendor": "N/A",
            "hw_ram": "N/A",
            "hw_vram": "N/A",
            "hw_resolution": "N/A"
        })

    return cleaned