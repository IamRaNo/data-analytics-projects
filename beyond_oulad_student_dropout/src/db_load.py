import os,sys
import time
import logging
import pandas as pd
from sqlalchemy import create_engine
import kagglehub


# add project root to PYTHONPATH
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from config.settings import DB_CONFIG, KAGGLE_DATASET

from config.paths import (
    LOG_DIR,
    RAW_DATA_DIR,
    UPLOAD_LOG_PATH
)

# ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# logging config
logging.basicConfig(
    filename=UPLOAD_LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Downloading dataset from Kaggle")
start = time.time()

path = kagglehub.dataset_download(KAGGLE_DATASET)

logging.info(
    f"Kaggle download completed in {round(time.time() - start, 2)} seconds"
)

# database engine
engine = create_engine(
    f"{DB_CONFIG['dialect']}+{DB_CONFIG['driver']}://"
    f"{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/"
    f"{DB_CONFIG['database']}"
)

# load CSV files (recursive-safe)
for root, _, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            table_name = file.replace(".csv", "")
            file_path = os.path.join(root, file)

            try:
                df = pd.read_csv(file_path)
                logging.info(f"Loading {table_name} | {df.shape}")

                start = time.time()
                df.to_sql(
                    table_name,
                    engine,
                    index=False,
                    if_exists="replace",
                    chunksize=10_000
                )

                logging.info(
                    f"{table_name} loaded in "
                    f"{round(time.time() - start, 2)} seconds"
                )

            except Exception as e:
                logging.error(f"Failed loading {table_name}: {e}")
