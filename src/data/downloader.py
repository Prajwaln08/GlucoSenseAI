"""
Google Drive downloader for GlucoSense AI datasets.

Downloads both datasets from Google Drive to data/raw/ using gdown.
Implements a local cache check — files already present are never re-downloaded.

Usage:
    python -m src.data.downloader --dataset nature_paper
    python -m src.data.downloader --dataset cgmacros
    python -m src.data.downloader --dataset all

Requires: gdown >= 5.2.0  (pip install gdown)
The Google Drive folders must be shared as "Anyone with the link can view".
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from src.config import (
    DATA_RAW_DIR,
    NP_DATA_FOLDER_ID,
    CGMACROS_FOLDER_ID,
    NP_DEMOGRAPHICS_FILE_ID,
    CGMACROS_BIO_FILE_ID,
    NP_TRAINING_USERS,
    NP_DB_USERS,
    CGMACROS_TRAINING_USERS,
    NP_FILE_PREFIXES,
)
from src.utils import get_logger

log = get_logger(__name__)

# ── Local cache paths ─────────────────────────────────────────────────────────
NP_RAW_DIR       = DATA_RAW_DIR / "nature_paper"
CGMACROS_RAW_DIR = DATA_RAW_DIR / "cgmacros"
_CGMACROS_ID_CACHE = CGMACROS_RAW_DIR / ".csv_file_ids.json"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _gdown_import():
    """Lazy import gdown with a clear error if not installed."""
    try:
        import gdown
        return gdown
    except ImportError:
        log.error(
            "gdown is not installed. Run:  pip install gdown==5.2.0"
        )
        sys.exit(1)


def _download_file(file_id: str, output_path: Path, force: bool = False) -> bool:
    """
    Download a single Google Drive file.
    Returns True if downloaded, False if skipped (cache hit).
    """
    gdown = _gdown_import()

    if output_path.exists() and not force:
        log.info(f"Cache hit — skipping: {output_path.name}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading → {output_path.name} ...")

    for attempt in range(1, 4):
        try:
            gdown.download(id=file_id, output=str(output_path), quiet=False)
            if output_path.exists() and output_path.stat().st_size > 0:
                log.success(f"Downloaded: {output_path.name}")
                return True
            raise RuntimeError("Downloaded file is empty.")
        except Exception as exc:
            log.warning(f"Attempt {attempt}/3 failed: {exc}")
            if attempt < 3:
                time.sleep(5 * attempt)

    log.error(f"Failed to download {output_path.name} after 3 attempts.")
    return False


def _download_folder(
    folder_id: str,
    output_dir: Path,
    force: bool = False,
) -> bool:
    """
    Download a Google Drive folder recursively.
    Uses resume=True so already-downloaded files are skipped automatically.
    Returns True if gdown ran, False if no attempt was made.
    """
    gdown = _gdown_import()

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading folder → {output_dir} ...")

    for attempt in range(1, 4):
        try:
            gdown.download_folder(
                id=folder_id,
                output=str(output_dir),
                quiet=False,
                use_cookies=False,
                resume=True,
                remaining_ok=True,
            )
            log.success(f"Folder downloaded: {output_dir}")
            return True
        except Exception as exc:
            log.warning(f"Attempt {attempt}/3 failed: {exc}")
            if attempt < 3:
                time.sleep(10 * attempt)

    log.error(f"Failed to download folder {folder_id} after 3 attempts.")
    return False


# ── Public API ────────────────────────────────────────────────────────────────

def download_np_demographics(force: bool = False) -> Path:
    """Download Nature's Paper Demographics.csv."""
    out = NP_RAW_DIR / "Demographics.csv"
    _download_file(NP_DEMOGRAPHICS_FILE_ID, out, force=force)
    return out


def download_cgmacros_bio(force: bool = False) -> Path:
    """Download CGMacros bio.csv."""
    out = CGMACROS_RAW_DIR / "bio.csv"
    _download_file(CGMACROS_BIO_FILE_ID, out, force=force)
    return out


def download_np_user(user_id: str, force: bool = False) -> Path:
    """
    Download a single Nature's Paper user folder from Google Drive.

    The NP data folder contains subfolders named '001', '002', etc.
    gdown.download_folder is called on the ROOT folder; if the user subfolder
    already exists we skip. For selective per-user download, call this after
    a full folder download and it will return immediately (cache hit).
    """
    user_dir = NP_RAW_DIR / user_id

    # Fast path: all 8 expected files already present
    if not force and _np_user_complete(user_dir, user_id):
        log.info(f"NP user {user_id}: all files cached — skipping download.")
        return user_dir

    # Download the entire NP data folder once; subfolders are preserved
    log.info(
        f"NP user {user_id} not fully cached. Downloading full NP data folder "
        f"(this only happens once — subsequent calls are instant cache hits)."
    )
    _download_folder(
        folder_id=NP_DATA_FOLDER_ID,
        output_dir=NP_RAW_DIR,
        force=force,
    )
    return user_dir


def _get_cgmacros_csv_ids(force: bool = False) -> dict[str, str]:
    """
    Build a {user_id: drive_file_id} mapping for all CGMacros CSV files.
    Enumerates the Drive folder once (skip_download=True) and caches the result
    locally so subsequent calls are instant. Photos subfolders are never downloaded.
    """
    gdown = _gdown_import()

    if not force and _CGMACROS_ID_CACHE.exists():
        return json.loads(_CGMACROS_ID_CACHE.read_text())

    log.info("Scanning CGMacros Drive folder for CSV file IDs (one-time setup)...")
    CGMACROS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    entries = gdown.download_folder(
        id=CGMACROS_FOLDER_ID,
        skip_download=True,
        remaining_ok=True,
        use_cookies=False,
        quiet=True,
    ) or []

    mapping: dict[str, str] = {}
    for entry in entries:
        # entry.path is like "CGMacros-001/CGMacros-001.csv"
        m = re.search(r"CGMacros-(\d{3})\.csv$", entry.path)
        if m:
            mapping[m.group(1)] = entry.id

    _CGMACROS_ID_CACHE.write_text(json.dumps(mapping, indent=2))
    log.info(f"Discovered {len(mapping)} CGMacros CSV file IDs.")
    return mapping


def download_cgmacros_user(user_id: str, force: bool = False) -> Path:
    """
    Download a single CGMacros user's CSV from Google Drive.
    Only downloads the CSV file — photos subfolders are skipped entirely.
    """
    subfolder_name = f"CGMacros-{user_id}"
    user_dir = CGMACROS_RAW_DIR / subfolder_name
    expected_csv = user_dir / f"{subfolder_name}.csv"

    if not force and expected_csv.exists() and expected_csv.stat().st_size > 0:
        log.info(f"CGMacros user {user_id}: CSV cached — skipping download.")
        return user_dir

    csv_ids = _get_cgmacros_csv_ids()
    if user_id not in csv_ids:
        raise FileNotFoundError(
            f"CGMacros user {user_id} CSV not found on Drive.\n"
            f"Run: python -m src.data.downloader --dataset cgmacros --users {user_id}"
        )

    user_dir.mkdir(parents=True, exist_ok=True)
    _download_file(csv_ids[user_id], expected_csv, force=force)
    return user_dir


def download_np_data(
    users: list[str] | None = None,
    include_demographics: bool = True,
    force: bool = False,
) -> dict[str, Path]:
    """
    Download Nature's Paper data for a list of users.

    Args:
        users: list of user IDs (e.g. ['003','004']). None → all training users.
        include_demographics: also download Demographics.csv.
        force: re-download even if cached.

    Returns:
        dict mapping user_id → local folder Path
    """
    if users is None:
        users = NP_TRAINING_USERS + NP_DB_USERS

    paths = {}
    for uid in users:
        paths[uid] = download_np_user(uid, force=force)

    if include_demographics:
        download_np_demographics(force=force)

    _report_np_cache(users)
    return paths


def download_cgmacros_data(
    users: list[str] | None = None,
    include_bio: bool = True,
    force: bool = False,
) -> dict[str, Path]:
    """
    Download CGMacros data for a list of users.

    Args:
        users: list of user IDs. None → all 45 training users.
        include_bio: also download bio.csv.
        force: re-download even if cached.

    Returns:
        dict mapping user_id → local folder Path
    """
    if users is None:
        users = CGMACROS_TRAINING_USERS

    paths = {}
    for uid in users:
        paths[uid] = download_cgmacros_user(uid, force=force)

    if include_bio:
        download_cgmacros_bio(force=force)

    _report_cgmacros_cache(users)
    return paths


# ── Cache reporting ───────────────────────────────────────────────────────────

def _np_user_complete(user_dir: Path, user_id: str) -> bool:
    """True if all 8 expected NP CSV files are present and non-empty."""
    if not user_dir.exists():
        return False
    for prefix in NP_FILE_PREFIXES:
        f = user_dir / f"{prefix}_{user_id}.csv"
        if not f.exists() or f.stat().st_size == 0:
            return False
    return True


def _report_np_cache(users: list[str]) -> None:
    complete = [u for u in users if _np_user_complete(NP_RAW_DIR / u, u)]
    missing  = [u for u in users if u not in complete]
    log.info(
        f"NP cache report: {len(complete)}/{len(users)} users complete "
        f"| missing: {missing or 'none'}"
    )


def _report_cgmacros_cache(users: list[str]) -> None:
    complete, missing = [], []
    for uid in users:
        csv = CGMACROS_RAW_DIR / f"CGMacros-{uid}" / f"CGMacros-{uid}.csv"
        (complete if (csv.exists() and csv.stat().st_size > 0) else missing).append(uid)
    log.info(
        f"CGMacros cache report: {len(complete)}/{len(users)} users complete "
        f"| missing: {missing or 'none'}"
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GlucoSense AI datasets from Google Drive")
    parser.add_argument(
        "--dataset",
        choices=["nature_paper", "cgmacros", "all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--users",
        nargs="+",
        default=None,
        help="Specific user IDs (e.g. 003 004). Default: all training users.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already cached.",
    )
    args = parser.parse_args()

    if args.dataset in ("nature_paper", "all"):
        download_np_data(users=args.users, force=args.force)

    if args.dataset in ("cgmacros", "all"):
        download_cgmacros_data(users=args.users, force=args.force)
