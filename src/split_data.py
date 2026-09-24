"""筛选英文评论，并按 75/25 分层划分训练集和测试集。
"""
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_URL, LABEL_COL, LANGUAGE, PROCESSED_DIR, RANDOM_STATE,
                    RAW_DATA, SENTIMENT_COL, TABLES_DIR, TEST_FILE, TEST_SIZE,
                    TEXT_COL, TRAIN_FILE)


def download_if_missing():
    if not RAW_DATA.exists():
        RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, RAW_DATA)


def load_english_reviews():
    df = pd.read_csv(RAW_DATA, index_col=0)
    df = df[df["Review_Language"] == LANGUAGE].reset_index(drop=True)

    # Booking.com 的评论分为优点（upside）和缺点（downside）两段，
    # 这里把两段拼接成一个文档。
    upside = df["Upside_Review"].fillna("").str.strip()
    downside = df["Downside_Review"].fillna("").str.strip()
    text = (upside + " " + downside).str.strip()

    # 模型只使用评论文本作为输入。评分、酒店和城市保留下来供参考，但不作为特征。
    return pd.DataFrame({
        "review_id": df.index,
        TEXT_COL: text,
        LABEL_COL: df["source"].astype(int),
        SENTIMENT_COL: df["Sentiment"],
        "upside": upside,
        "downside": downside,
        "review_score": df["Review_Score"],
        "city": df["City Name"],
        "hotel": df["Hotel Name"],
    })


def summarise(train, test):
    rows = []
    for name, part in (("train", train), ("test", test)):
        counts = part.groupby([LABEL_COL, SENTIMENT_COL]).size()
        for (label, sentiment), n in counts.items():
            rows.append({"split": name, LABEL_COL: label,
                         SENTIMENT_COL: sentiment, "n": n})
    return pd.DataFrame(rows)


def main():
    download_if_missing()
    reviews = load_english_reviews()
    assert len(reviews) == 2000, f"expected 2000 English reviews, got {len(reviews)}"
    assert not reviews[TEXT_COL].eq("").any(), "found reviews with empty text"

    # 按"标签 × 正负面"分层：训练集和测试集的类别分布相同，
    # 每个类别内部的正负面比例也相同（比较正面和负面评论上的准确率时需要）。
    strata = reviews[LABEL_COL].astype(str) + "_" + reviews[SENTIMENT_COL]
    train, test = train_test_split(reviews, test_size=TEST_SIZE,
                                   stratify=strata, random_state=RANDOM_STATE)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    train.sort_values("review_id").to_csv(TRAIN_FILE, index=False)
    test.sort_values("review_id").to_csv(TEST_FILE, index=False)

    summary = summarise(train, test)
    summary.to_csv(TABLES_DIR / "split_summary.csv", index=False)
    print(summary.pivot_table(index="split", columns=[LABEL_COL, SENTIMENT_COL],
                              values="n"))
    print(f"train: {len(train)} reviews -> {TRAIN_FILE}")
    print(f"test:  {len(test)} reviews -> {TEST_FILE}")


if __name__ == "__main__":
    main()
