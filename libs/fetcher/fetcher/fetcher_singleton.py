import httpx
from httpx import AsyncClient


class ReviewFetcher:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    def get(self) -> AsyncClient:
        if self._client is None:
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            self._client = httpx.AsyncClient(limits=limits, follow_redirects=True)
        # This could never return None
        # noinspection PyTypeChecker
        return self._client

    async def close_client(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None


fetcher: ReviewFetcher = ReviewFetcher()
