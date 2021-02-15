# stdlib
from pathlib import Path
import urllib.request


def download_if_needed(path: Path, url: str) -> None:
    """
    Helper for downloading a file, if it is now already on the disk.
    """
    if path.exists():
        return

    urllib.request.urlretrieve(url, path)
