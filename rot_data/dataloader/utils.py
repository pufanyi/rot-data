import asyncio
import os
from typing import Final

import aiohttp
from anyio import Path

CHUNK_SIZE: Final[int] = 1 << 16
PART_EXTENSION: Final[str] = ".part"
USER_AGENT: Final[str] = "rot-data/1.0"


class DownloadError(RuntimeError):
    """Raised when a download fails after exhausting retries."""


def _parse_total_size(content_range: str | None) -> int | None:
    if not content_range:
        return None
    if "/" not in content_range:
        return None
    _, _, total_str = content_range.partition("/")
    total_str = total_str.strip()
    if total_str == "*":
        return None
    try:
        return int(total_str)
    except ValueError:
        return None


def _expected_total_size(
    response: aiohttp.ClientResponse, resume_from: int
) -> int | None:
    total = _parse_total_size(response.headers.get("Content-Range"))
    if total is not None:
        return total
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    try:
        length = int(content_length)
    except ValueError:
        return None
    if response.status == 206:
        return resume_from + length
    return length


async def _write_body(
    response: aiohttp.ClientResponse, destination: Path, append: bool
) -> int:
    bytes_written = 0
    mode = "ab" if append else "wb"
    async with await destination.open(mode) as file_obj:
        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
            if not chunk:
                continue
            bytes_written += await file_obj.write(chunk)
    return bytes_written


class _RetryableDownloadError(Exception):
    pass


async def download_file(
    url: str, path: str | os.PathLike, num_retries: int = 5, timeout: float = 10
) -> Path:
    """Download ``url`` to ``path`` with HTTP range resume support."""

    target_path = await (await Path(path).expanduser()).absolute()

    if await target_path.exists():
        if await target_path.is_dir():
            raise DownloadError(f"Destination {target_path} is a directory")
        return target_path

    partial_path = target_path.parent / f"{target_path.name}{PART_EXTENSION}"
    await partial_path.parent.mkdir(parents=True, exist_ok=True)

    timeout_config = aiohttp.ClientTimeout(
        total=None, sock_connect=timeout, sock_read=timeout
    )
    backoff = 0.5

    async with aiohttp.ClientSession(
        timeout=timeout_config, raise_for_status=False
    ) as session:
        for attempt in range(1, num_retries + 1):
            partial_stat = None
            if await partial_path.exists():
                partial_stat = await partial_path.stat()
            resume_from = partial_stat.st_size if partial_stat else 0
            headers = {"User-Agent": USER_AGENT}
            if resume_from:
                headers["Range"] = f"bytes={resume_from}-"

            try:
                async with session.get(url, headers=headers) as response:
                    if response.status in {200, 206}:
                        if response.status == 200 and resume_from:
                            await partial_path.unlink(missing_ok=True)
                            partial_stat = None
                            resume_from = 0
                        written = await _write_body(
                            response, partial_path, append=resume_from > 0
                        )
                        expected_total = _expected_total_size(response, resume_from)
                        total_so_far = resume_from + written
                        if expected_total is not None and total_so_far < expected_total:
                            raise _RetryableDownloadError(
                                f"Connection closed early. Expected {expected_total} "
                                f"bytes, got {total_so_far}."
                            )
                        await partial_path.replace(target_path)
                        return target_path

                    if response.status == 416 and resume_from:
                        expected_total = _parse_total_size(
                            response.headers.get("Content-Range")
                        )
                        if (
                            expected_total is not None
                            and partial_stat is not None
                            and partial_stat.st_size >= expected_total
                        ):
                            await partial_path.replace(target_path)
                            return target_path
                        await partial_path.unlink(missing_ok=True)
                        raise _RetryableDownloadError(
                            "Server rejected range; restarting download"
                        )

                    if 500 <= response.status < 600:
                        raise _RetryableDownloadError(
                            f"Server returned {response.status} while downloading {url}"
                        )

                    body_snippet = (await response.text())[:200]
                    raise DownloadError(
                        f"Unexpected status {response.status} while downloading {url}: "
                        f"{body_snippet}"
                    )

            except _RetryableDownloadError as retryable:
                error: Exception = retryable
            except (TimeoutError, aiohttp.ClientError) as client_error:
                error = client_error
            else:
                continue

            if attempt == num_retries:
                raise DownloadError(
                    f"Failed to download {url} after {num_retries} attempts"
                ) from error

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)

    raise DownloadError(f"Failed to download {url} after {num_retries} attempts")
