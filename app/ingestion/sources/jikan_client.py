"""
Async Jikan API client with rate limiting and retry logic.

"""

from __future__ import annotations

import asyncio
import time

from collections.abc import AsyncGenerator
from typing import Any, cast

import httpx

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings

settings = get_settings()


class JikanClient:
    """Async client for Jikan API (MyAnimeList data) with built-in safeguards."""

    def __init__(self, limit_per_page: int = 25):
        from app.core.logging import get_logger

        self.logger = get_logger(__name__)
        self.base_url = settings.JIKAN_API_URL.rstrip("/")
        self.rate_limit_per_sec = settings.JIKAN_RATE_LIMIT_PER_SECOND
        self.limit_per_page = min(limit_per_page, 25)  # Jikan max is 25
        self._last_request_time = 0.0
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> JikanClient:
        self._client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": f"{settings.APP_NAME}/{settings.APP_VERSION}",
                "Accept": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()

    async def _enforce_rate_limit(self) -> None:
        """Sleep to respect rate limit between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 1.0 / self.rate_limit_per_sec

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self._last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        reraise=True,
    )
    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make rate-limited GET request with retries."""
        await self._enforce_rate_limit()

        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context manager.")

        url = f"{self.base_url}/{endpoint}"
        self.logger.debug(f"Fetching: {url} with params {params}")

        response = await self._client.get(url, params=params)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    async def iterate_top_anime(
        self, max_entries: int = 10
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Iterate through top anime entries with pagination.

        Args:
            max_entries: Maximum number of anime to fetch (default: 10 for minimal pipeline)

        Yields:
            Anime data dictionaries from Jikan API
        """
        page = 1
        entries_fetched = 0

        while entries_fetched < max_entries:
            try:
                self.logger.info(
                    f"Fetching page {page} from Jikan API (entries {entries_fetched}/{max_entries})..."
                )
                response = await self._get(
                    "top/anime", params={"page": page, "limit": self.limit_per_page}
                )

                anime_list = response.get("data", [])
                if not anime_list:
                    self.logger.warning("No anime data returned from API")
                    break

                for anime in anime_list:
                    if entries_fetched >= max_entries:
                        return

                    # Validate required fields per spec
                    if "mal_id" not in anime or "title" not in anime:
                        self.logger.warning(
                            f"Skipping malformed anime entry: {anime.get('mal_id', 'unknown')}"
                        )
                        continue

                    yield anime
                    entries_fetched += 1

                # Check pagination
                pagination = response.get("pagination", {})
                if not pagination.get("has_next_page", False):
                    self.logger.info("Reached end of pagination")
                    break

                page += 1
                await asyncio.sleep(0.5)  # Additional buffer between pages

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    self.logger.warning("Rate limited by Jikan API, sleeping for 60 seconds...")
                    await asyncio.sleep(60)
                    continue
                elif e.response.status_code == 404:
                    self.logger.warning(f"Page {page} not found, stopping pagination")
                    break
                else:
                    self.logger.error(
                        f"HTTP {e.response.status_code} on page {page}: {e.response.text}"
                    )
                    raise
            except Exception as e:
                self.logger.error(f"Unexpected error fetching page {page}: {e}", exc_info=True)
                raise
