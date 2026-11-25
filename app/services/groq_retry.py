import asyncio
import re

import httpx


async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    attempts: int = 4,
    **kwargs,
) -> httpx.Response:
    response = await client.post(url, **kwargs)

    for _ in range(attempts - 1):
        if response.status_code != 429:
            break

        wait = 5.0
        header = response.headers.get("retry-after")

        if header:
            try:
                wait = float(header)
            except ValueError:
                pass
        else:
            found = re.search(r"try again in ([\d.]+)s", response.text)
            if found:
                wait = float(found.group(1))

        await asyncio.sleep(min(wait, 15.0) + 0.5)
        response = await client.post(url, **kwargs)

    return response
