import os

# project root (one level above config/)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# directories
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RAW_DATA_DIR = os.path.join(DATASET_DIR, "raw_data")

# files
UPLOAD_LOG_PATH = os.path.join(LOG_DIR, "data_upload.log")
