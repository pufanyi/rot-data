import os
import time
from collections.abc import Mapping
from http.client import HTTPResponse
from pathlib import Path
from typing import Final
from urllib import error, request

from loguru import logger

from .progress import get_progress_manager

CHUNK_SIZE: Final[int] = 1 << 16
PART_EXTENSION: Final[str] = ".part"
USER_AGENT: Final[str] = "rot-data/1.0"
MAX_BACKOFF: Final[float] = 8.0


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
    headers: Mapping[str, str] | None, status: int, resume_from: int
) -> int | None:
    if headers is None:
        return None
    total = _parse_total_size(headers.get("Content-Range"))
    if total is not None:
        return total
    content_length = headers.get("Content-Length")
    if content_length is None:
        return None
    try:
        length = int(content_length)
    except ValueError:
        return None
    if status == 206:
        return resume_from + length
    return length


def _write_body(
    response: HTTPResponse,
    destination: Path,
    append: bool,
    task_id: int | None = None,
) -> int:
    bytes_written = 0
    mode = "ab" if append else "wb"
    manager = get_progress_manager()
    with destination.open(mode) as file_obj:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            file_obj.write(chunk)
            bytes_written += len(chunk)
            if task_id is not None and manager.is_active:
                manager.update(task_id, advance=len(chunk))
    return bytes_written


class _RetryableDownloadError(Exception):
    pass


def _decode_snippet(data: bytes) -> str:
    return data.decode("utf-8", "replace")


def download_file(
    url: str,
    path: str | os.PathLike,
    num_retries: int = 5,
    timeout: float = 10,
    description: str | None = None,
) -> Path:
    """
    Download ``url`` to ``path`` with HTTP range resume support.

    Args:
        url: URL to download from
        path: Local path to save the file
        num_retries: Number of retry attempts
        timeout: Request timeout in seconds
        description: Optional description for progress bar (defaults to filename)
    """

    target_path = Path(path).expanduser().resolve()
    if target_path.exists():
        if target_path.is_dir():
            raise DownloadError(f"Destination {target_path} is a directory")
        display_name = description or target_path.name
        logger.info(f"File already exists: {display_name}")
        return target_path

    display_name = description or target_path.name
    logger.info(f"Starting download: {display_name}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = target_path.parent / f"{target_path.name}{PART_EXTENSION}"

    backoff = 0.5
    manager = get_progress_manager()

    for attempt in range(1, num_retries + 1):
        partial_stat = partial_path.stat() if partial_path.exists() else None
        resume_from = partial_stat.st_size if partial_stat else 0
        headers = {"User-Agent": USER_AGENT}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
            logger.debug(f"Resuming download from byte {resume_from}")

        request_obj = request.Request(url, headers=headers, method="GET")

        try:
            with request.urlopen(request_obj, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                if status not in {200, 206}:
                    body_snippet = _decode_snippet(response.read(200))
                    raise DownloadError(
                        f"Unexpected status {status} while downloading {url}: "
                        f"{body_snippet}"
                    )

                if status == 200 and resume_from:
                    partial_path.unlink(missing_ok=True)
                    partial_stat = None
                    resume_from = 0

                expected_total = _expected_total_size(
                    response.headers, status, resume_from
                )

                task_id = None
                if manager.is_active:
                    task_id = manager.add_download_task(
                        f"[cyan]Downloading {display_name}",
                        total=expected_total if expected_total else None,
                        completed=resume_from,
                    )

                written = _write_body(
                    response,
                    partial_path,
                    append=resume_from > 0,
                    task_id=task_id,
                )

                total_so_far = resume_from + written
                if expected_total is not None and total_so_far < expected_total:
                    if task_id is not None and manager.is_active:
                        manager.remove_task(task_id)
                    raise _RetryableDownloadError(
                        f"Connection closed early. Expected {expected_total} "
                        f"bytes, got {total_so_far}."
                    )

                if task_id is not None and manager.is_active:
                    manager.remove_task(task_id)
                partial_path.replace(target_path)
                logger.success(f"Download completed: {display_name}")
                return target_path

        except _RetryableDownloadError as retryable:
            error_to_raise: Exception = retryable
            logger.warning(f"Retry attempt {attempt}/{num_retries}: {retryable}")
        except error.HTTPError as http_error:
            status = http_error.code
            if status == 416 and resume_from:
                expected_total = _parse_total_size(
                    http_error.headers.get("Content-Range")
                )
                if (
                    expected_total is not None
                    and partial_stat is not None
                    and partial_stat.st_size >= expected_total
                ):
                    partial_path.replace(target_path)
                    logger.success(f"Download completed: {display_name}")
                    return target_path
                partial_path.unlink(missing_ok=True)
                error_to_raise = _RetryableDownloadError(
                    "Server rejected range; restarting download"
                )
                logger.warning(
                    f"Retry attempt {attempt}/{num_retries}: Range rejected, restarting"
                )
            elif 500 <= status < 600:
                error_to_raise = _RetryableDownloadError(
                    f"Server returned {status} while downloading {url}"
                )
                logger.warning(
                    f"Retry attempt {attempt}/{num_retries}: Server error {status}"
                )
            else:
                body_snippet = _decode_snippet(http_error.read()[:200])
                logger.error(f"Download failed with status {status}: {body_snippet}")
                raise DownloadError(
                    f"Unexpected status {status} while downloading {url}: "
                    f"{body_snippet}"
                ) from http_error
        except (error.URLError, TimeoutError, OSError) as network_error:
            error_to_raise = network_error
            logger.warning(
                f"Retry attempt {attempt}/{num_retries}: "
                f"Network error - {network_error}"
            )
        else:
            continue

        if attempt == num_retries:
            logger.error(f"Download failed after {num_retries} attempts: {url}")
            raise DownloadError(
                f"Failed to download {url} after {num_retries} attempts"
            ) from error_to_raise

        time.sleep(backoff)
        backoff = min(backoff * 2, MAX_BACKOFF)

    raise DownloadError(f"Failed to download {url} after {num_retries} attempts")
