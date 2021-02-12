# stdlib
from pathlib import Path
import urllib.request


def download_if_needed(path: Path, url: str) -> None:
    if path.exists():
        return

    urllib.request.urlretrieve(url, path)
