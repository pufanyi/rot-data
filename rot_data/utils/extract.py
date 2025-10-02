import os
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class ZipEntry:
    """Represents an entry in a ZIP archive."""
    
    pathname: str
    is_file: bool
    is_dir: bool


class ZipReader:
    """Read ZIP archives using the unzip command."""
    
    def __init__(self, zip_path: str | os.PathLike):
        self.zip_path = Path(zip_path).expanduser().resolve()
        if not self.zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {self.zip_path}")
    
    def list_entries(self) -> list[ZipEntry]:
        """List all entries in the ZIP archive."""
        try:
            # Use unzip -Z1 to list all files (one per line)
            result = subprocess.run(
                ["unzip", "-Z1", str(self.zip_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            
            entries = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                # Directories end with /
                is_dir = line.endswith("/")
                is_file = not is_dir
                entries.append(ZipEntry(
                    pathname=line.rstrip("/"),
                    is_file=is_file,
                    is_dir=is_dir,
                ))
            
            return entries
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list ZIP entries: {e.stderr}")
            raise RuntimeError(f"Failed to list ZIP entries: {e.stderr}") from e
    
    def read_file(self, filename: str) -> bytes:
        """Read a specific file from the ZIP archive into memory."""
        try:
            # Use unzip -p to extract file to stdout
            result = subprocess.run(
                ["unzip", "-p", str(self.zip_path), filename],
                capture_output=True,
                check=True,
            )
            return result.stdout
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to read file '{filename}' from ZIP: {e.stderr}")
            raise RuntimeError(
                f"Failed to read file '{filename}' from ZIP: {e.stderr}"
            ) from e
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def extract_zip(
    zip_path: str | os.PathLike,
    dest_dir: str | os.PathLike,
    file_pattern: str | None = None,
) -> Path:
    """
    Extract a ZIP archive to a destination directory.
    
    Args:
        zip_path: Path to the ZIP file
        dest_dir: Destination directory for extraction
        file_pattern: Optional glob pattern to extract specific files only
    
    Returns:
        Path to the extraction directory
    """
    zip_path = Path(zip_path).expanduser().resolve()
    dest_dir = Path(dest_dir).expanduser().resolve()
    
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        cmd = ["unzip", "-q", "-o", str(zip_path), "-d", str(dest_dir)]
        if file_pattern:
            cmd.append(file_pattern)
        
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        logger.debug(f"Extracted {zip_path.name} to {dest_dir}")
        return dest_dir
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract ZIP archive: {e.stderr}")
        raise RuntimeError(f"Failed to extract ZIP archive: {e.stderr}") from e


def iter_zip_entries(zip_path: str | os.PathLike) -> Iterator[tuple[ZipEntry, bytes]]:
    """
    Iterate over all file entries in a ZIP archive, yielding (entry, content) tuples.
    
    Args:
        zip_path: Path to the ZIP file
    
    Yields:
        Tuple of (ZipEntry, file_content_bytes) for each file in the archive
    """
    with ZipReader(zip_path) as reader:
        entries = reader.list_entries()
        for entry in entries:
            if entry.is_file:
                content = reader.read_file(entry.pathname)
                yield entry, content

