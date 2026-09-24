"""Shared paths and constants for the fake hotel review experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_URL = (
    "https://raw.githubusercontent.com/MichiganNLP/"
    "multilingual_reviews_deception/master/data/all_data.csv"
)
RAW_DATA = ROOT / "data" / "raw" / "all_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.25
LANGUAGE = "English"

# Label coding: 1 = AI-generated (fake), 0 = written by a human (genuine).
LABEL_COL = "label"
TEXT_COL = "text"
SENTIMENT_COL = "sentiment"
