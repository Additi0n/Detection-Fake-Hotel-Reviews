# 路径和常量
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

URL = "https://raw.githubusercontent.com/MichiganNLP/multilingual_reviews_deception/master/data/all_data.csv"
RAW = os.path.join(BASE, "data", "raw", "all_data.csv")
TRAIN = os.path.join(BASE, "data", "processed", "train.csv")
TEST = os.path.join(BASE, "data", "processed", "test.csv")

TAB_DIR = os.path.join(BASE, "results", "tables")
FIG_DIR = os.path.join(BASE, "results", "figures")
MOD_DIR = os.path.join(BASE, "results", "models")

SEED = 42
