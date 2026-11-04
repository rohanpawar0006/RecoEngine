"""
Create clean, comprehensive submission ZIP archive for Google Drive upload.
Excludes large raw datasets (.zip, raw csvs), .venv, .git, and temporary cache directories.
Includes all code, tests, docs, reports, demo, artifacts, and dev data.
"""

import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = PROJECT_ROOT.parent / "RecoEngine_Submission.zip"
# Also save a copy inside project root or report location if desired
ALT_OUTPUT_ZIP = PROJECT_ROOT / "RecoEngine_Submission.zip"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "htmlcov",
    ".idea",
    ".vscode",
    "data/raw",  # Raw 1.5GB MovieLens download files
}

EXCLUDE_FILES = {
    "dump.rdb",
    "RecoEngine_Submission.zip",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".swp",
}


def should_include(rel_path: Path) -> bool:
    parts = rel_path.parts
    # Check directory exclusions
    for ex in EXCLUDE_DIRS:
        ex_parts = Path(ex).parts
        if len(parts) >= len(ex_parts) and parts[:len(ex_parts)] == ex_parts:
            return False

    # Check file exclusions
    if rel_path.name in EXCLUDE_FILES or rel_path.suffix in EXCLUDE_EXTENSIONS:
        return False

    # Exclude raw 20M train.parquet to save ~150MB if desired, but keep dev & test
    # Actually, let's keep processed dev & test & val and vocabularies
    if rel_path.name == "train.parquet" and "dev" not in parts:
        return False  # train.parquet is 138MB; dev/train.parquet is 4MB

    return True


def create_zip():
    print(f"Creating submission zip: {ALT_OUTPUT_ZIP} ...")
    count = 0
    total_bytes = 0

    with zipfile.ZipFile(ALT_OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            rel_dir = Path(root).relative_to(PROJECT_ROOT)
            if not should_include(rel_dir):
                dirs[:] = []  # Don't recurse
                continue

            for f in files:
                file_path = Path(root) / f
                rel_file = file_path.relative_to(PROJECT_ROOT)
                if should_include(rel_file):
                    zf.write(file_path, arcname=str(rel_file))
                    count += 1
                    total_bytes += file_path.stat().st_size

    zip_size_mb = ALT_OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    orig_size_mb = total_bytes / (1024 * 1024)
    print("=================================================================")
    print(f"ZIP Created Successfully: {ALT_OUTPUT_ZIP}")
    print(f"Files Packaged: {count}")
    print(f"Original Size: {orig_size_mb:.2f} MB")
    print(f"Compressed ZIP Size: {zip_size_mb:.2f} MB")
    print("=================================================================")


if __name__ == "__main__":
    create_zip()
