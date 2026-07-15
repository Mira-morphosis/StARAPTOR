import asyncio
from pathlib import Path

from storage.storage_interface import review_updater
from storage.storage_utils import export


async def test():
    await review_updater(app_id=730, max_days=1, verbose=True)
    export(730, path=Path('out/'))

if __name__ == "__main__":
    asyncio.run(test())