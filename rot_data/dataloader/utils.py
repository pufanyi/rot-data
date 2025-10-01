import os
import time
import zipfile
from collections.abc import Mapping
from http.client import HTTPResponse
from pathlib import Path
from typing import Final
from urllib import error, request

from loguru import logger
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    TextColumn,
)

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
    progress: Progress | None = None,
    task_id: int | None = None,
) -> int:
    bytes_written = 0
    mode = "ab" if append else "wb"
    with destination.open(mode) as file_obj:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            file_obj.write(chunk)
            bytes_written += len(chunk)
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=len(chunk))
    return bytes_written


class _RetryableDownloadError(Exception):
    pass


def _decode_snippet(data: bytes) -> str:
    return data.decode("utf-8", "replace")


def download_file(
    url: str, path: str | os.PathLike, num_retries: int = 5, timeout: float = 10
) -> Path:
    """Download ``url`` to ``path`` with HTTP range resume support."""

    target_path = Path(path).expanduser().resolve()
    if target_path.exists():
        if target_path.is_dir():
            raise DownloadError(f"Destination {target_path} is a directory")
        logger.info(f"File already exists: {target_path.name}")
        return target_path

    logger.info(f"Starting download: {url}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = target_path.parent / f"{target_path.name}{PART_EXTENSION}"

    backoff = 0.5

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
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
                            f"Unexpected status {status} while downloading {url}: {body_snippet}"
                        )

                    if status == 200 and resume_from:
                        partial_path.unlink(missing_ok=True)
                        partial_stat = None
                        resume_from = 0

                    expected_total = _expected_total_size(
                        response.headers, status, resume_from
                    )
                    
                    task_id = progress.add_task(
                        f"[cyan]Downloading {target_path.name}",
                        total=expected_total if expected_total else None,
                        completed=resume_from,
                    )

                    written = _write_body(
                        response, partial_path, append=resume_from > 0, progress=progress, task_id=task_id
                    )
                    
                    total_so_far = resume_from + written
                    if expected_total is not None and total_so_far < expected_total:
                        raise _RetryableDownloadError(
                            f"Connection closed early. Expected {expected_total} bytes, "
                            f"got {total_so_far}."
                        )

                    partial_path.replace(target_path)
                    logger.success(f"Download completed: {target_path.name}")
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
                        logger.success(f"Download completed: {target_path.name}")
                        return target_path
                    partial_path.unlink(missing_ok=True)
                    error_to_raise = _RetryableDownloadError(
                        "Server rejected range; restarting download"
                    )
                    logger.warning(f"Retry attempt {attempt}/{num_retries}: Range rejected, restarting")
                elif 500 <= status < 600:
                    error_to_raise = _RetryableDownloadError(
                        f"Server returned {status} while downloading {url}"
                    )
                    logger.warning(f"Retry attempt {attempt}/{num_retries}: Server error {status}")
                else:
                    body_snippet = _decode_snippet(http_error.read()[:200])
                    logger.error(f"Download failed with status {status}: {body_snippet}")
                    raise DownloadError(
                        f"Unexpected status {status} while downloading {url}: {body_snippet}"
                    ) from http_error
            except (error.URLError, TimeoutError, OSError) as network_error:
                error_to_raise = network_error
                logger.warning(f"Retry attempt {attempt}/{num_retries}: Network error - {network_error}")
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


def unzip_file(
    archive_path: str | os.PathLike, destination: str | os.PathLike
) -> Path:
    """Extract ``archive_path`` into ``destination`` and return the extracted path."""

    archive = Path(archive_path).expanduser().resolve()
    output_dir = Path(destination).expanduser().resolve()

    if not archive.exists():
        raise FileNotFoundError(f"Archive {archive} does not exist")

    logger.info(f"Extracting archive: {archive.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zip_file:
        members = zip_file.infolist()
        total_files = len(members)
        
        # Security check
        for member in members:
            resolved_member = (output_dir / member.filename).resolve()
            if not resolved_member.is_relative_to(output_dir):
                raise DownloadError("Archive contains unsafe paths outside destination")
        
        # Extract with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total} files)"),
        ) as progress:
            task_id = progress.add_task(
                f"[green]Extracting {archive.name}",
                total=total_files,
            )
            
            for member in members:
                zip_file.extract(member, output_dir)
                progress.update(task_id, advance=1)

    logger.success(f"Extraction completed: {total_files} files extracted to {output_dir}")
    return output_dir
