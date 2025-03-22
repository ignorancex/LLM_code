"""Utility module for downloading and extracting GitHub repository archives."""

import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

REPO_OWNER = "google"
REPO_NAME = "fonts"
COMMIT_HASH = "c06520efccd2c99d970b536d4f62cb4d95b4e6b2"
TARGET_DIR = "fonts"


def construct_archive_url(repo_owner: str, repo_name: str, commit_hash: str) -> str:
    """Construct a GitHub archive URL."""
    return f"https://github.com/{repo_owner}/{repo_name}/archive/{commit_hash}.zip"


def download_file(url: str, dest_path: Path, timeout: int = 10) -> None:
    """Download a file from the given URL."""
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    with dest_path.open("wb") as dest_file:
        for chunk in response.iter_content(chunk_size=8192):
            dest_file.write(chunk)


def extract_zip(archive_path: Path, extract_to: Path) -> None:
    """Extract a ZIP file."""
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def move_files(src_dir: Path, target_dir: Path) -> None:
    """Move files from the source directory to the target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        target_path = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target_path)


def download_google_fonts(
    repo_owner: str,
    repo_name: str,
    commit_hash: str,
    target_dir: str,
) -> None:
    """Download and extract a specific commit from a GitHub repository."""
    archive_url = construct_archive_url(repo_owner, repo_name, commit_hash)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        archive_path = temp_dir_path / f"{commit_hash}.zip"
        extract_dir = temp_dir_path / "extracted"

        download_file(archive_url, archive_path)
        extract_zip(archive_path, extract_dir)

        extracted_commit_dir = extract_dir / f"{repo_name}-{commit_hash}"
        if extracted_commit_dir.exists():
            move_files(extracted_commit_dir, Path(target_dir))
        else:
            msg = f"Extracted directory not found: {extracted_commit_dir}"
            raise FileNotFoundError(msg)


if __name__ == "__main__":
    download_google_fonts(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        commit_hash=COMMIT_HASH,
        target_dir=TARGET_DIR,
    )
