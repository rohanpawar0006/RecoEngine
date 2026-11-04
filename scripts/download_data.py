"""
MovieLens 25M Download Helper Script.
Downloads and extracts the MovieLens 25M dataset into data/raw/.
Run only when directed.
"""

import logging
import os
import urllib.request
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ZIP_PATH = os.path.join(TARGET_DIR, "ml-25m.zip")


def download_and_extract():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    if os.path.exists(os.path.join(TARGET_DIR, "ml-25m", "ratings.csv")):
        logger.info("MovieLens 25M dataset already exists at %s", os.path.join(TARGET_DIR, "ml-25m"))
        return

    logger.info("Downloading MovieLens 25M from %s to %s ...", MOVIELENS_25M_URL, ZIP_PATH)
    urllib.request.urlretrieve(MOVIELENS_25M_URL, ZIP_PATH)
    logger.info("Download complete. Extracting files...")

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)
        
    logger.info("Extraction complete. Raw files placed in %s", TARGET_DIR)


if __name__ == "__main__":
    download_and_extract()

