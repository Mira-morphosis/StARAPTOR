import httpx
from httpx import AsyncClient


class ReviewFetcher:
    """Lazily-created singleton wrapper around a shared httpx AsyncClient."""

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    def get(self) -> AsyncClient:
        """
        Returns the shared AsyncClient, creating it on first use.
        :return: The shared httpx AsyncClient instance.
        """
        if self._client is None:
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            self._client = httpx.AsyncClient(limits=limits, follow_redirects=True)
        # This could never return None
        # noinspection PyTypeChecker
        return self._client

    async def close_client(self):
        """Closes and discards the shared client, if one was created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


fetcher: ReviewFetcher = ReviewFetcher()