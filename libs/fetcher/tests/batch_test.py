import asyncio
import csv
from pathlib import Path

from fetcher.interface import fetch_periodically

async def test():
    try:
        reviews:list[dict] = await fetch_periodically(app_id=730, max_batches=5, verbose=True)

        if not reviews:
            print("No reviews found to save.")
            return

        output_path = Path("out/result.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_keys = set()
        for review in reviews:
            all_keys.update(review.keys())
        fieldnames = list(all_keys)

        with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reviews)

        print(f"Successfully saved {len(reviews)} reviews to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Fetcher is closing...")

if __name__ == "__main__":
    asyncio.run(test())