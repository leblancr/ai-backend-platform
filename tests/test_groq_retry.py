import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from app.services.groq_retry import post_with_retry


@pytest.mark.parametrize("status_code", [200, 201, 400, 500])
def test_returns_non_rate_limit_response_without_retry(status_code):
    async def run():
        response = httpx.Response(
            status_code, request=httpx.Request("POST", "https://example.test")
        )
        client = AsyncMock()
        client.post.return_value = response

        result = await post_with_retry(client, "https://example.test")

        assert result is response
        client.post.assert_awaited_once_with("https://example.test")

    asyncio.run(run())


def test_retries_using_retry_after_header(monkeypatch):
    async def run():
        limited = httpx.Response(
            429,
            headers={"retry-after": "2.5"},
            request=httpx.Request("POST", "https://example.test"),
        )
        success = httpx.Response(
            200, request=httpx.Request("POST", "https://example.test")
        )
        client = AsyncMock()
        client.post.side_effect = [limited, success]
        sleep = AsyncMock()
        monkeypatch.setattr("app.services.groq_retry.asyncio.sleep", sleep)

        result = await post_with_retry(client, "https://example.test")

        assert result is success
        assert client.post.await_count == 2
        sleep.assert_awaited_once_with(3.0)

    asyncio.run(run())


def test_retries_using_wait_time_in_response_text(monkeypatch):
    async def run():
        limited = httpx.Response(
            429,
            text="Rate limited; try again in 3.2s",
            request=httpx.Request("POST", "https://example.test"),
        )
        success = httpx.Response(
            200, request=httpx.Request("POST", "https://example.test")
        )
        client = AsyncMock()
        client.post.side_effect = [limited, success]
        sleep = AsyncMock()
        monkeypatch.setattr("app.services.groq_retry.asyncio.sleep", sleep)

        result = await post_with_retry(client, "https://example.test")

        assert result is success
        sleep.assert_awaited_once_with(3.7)

    asyncio.run(run())


def test_caps_retry_wait(monkeypatch):
    async def run():
        limited = httpx.Response(
            429,
            headers={"retry-after": "60"},
            request=httpx.Request("POST", "https://example.test"),
        )
        success = httpx.Response(
            200, request=httpx.Request("POST", "https://example.test")
        )
        client = AsyncMock()
        client.post.side_effect = [limited, success]
        sleep = AsyncMock()
        monkeypatch.setattr("app.services.groq_retry.asyncio.sleep", sleep)

        result = await post_with_retry(client, "https://example.test")

        assert result is success
        sleep.assert_awaited_once_with(15.5)

    asyncio.run(run())


def test_stops_after_configured_number_of_attempts(monkeypatch):
    async def run():
        limited = httpx.Response(
            429, request=httpx.Request("POST", "https://example.test")
        )
        client = AsyncMock()
        client.post.return_value = limited
        sleep = AsyncMock()
        monkeypatch.setattr("app.services.groq_retry.asyncio.sleep", sleep)

        result = await post_with_retry(
            client, "https://example.test", attempts=3
        )

        assert result is limited
        assert client.post.await_count == 3
        assert sleep.await_count == 2

    asyncio.run(run())
