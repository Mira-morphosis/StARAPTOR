import asyncio
from pathlib import Path

from storage.storage_interface import review_updater
from storage.storage_utils import export


async def test():
    #we'll try app_id=1086940 next
    await review_updater(app_id=1086940, max_days=90, verbose=True)
    export(1086940, path=Path('out/'))

if __name__ == "__main__":
    asyncio.run(test())