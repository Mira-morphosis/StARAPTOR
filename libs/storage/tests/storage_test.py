import asyncio
import os
from pathlib import Path

from storage.main import storage_updater
from storage.storage_utils import export


async def test():
    await storage_updater(
        app_id=730,
        max_batches=5,
        verbose=True
    )
    export(730, path=Path('out/'))

if __name__ == "__main__":
    asyncio.run(test())