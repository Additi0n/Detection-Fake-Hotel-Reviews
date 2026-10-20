# 取出英文评论，按 75/25 分层划分训练集和测试集
# 用法: python src/split_data.py
import os
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RAW, SEED, TAB_DIR, TEST, TRAIN, URL


def main():
    if not os.path.exists(RAW):
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        print("downloading", URL)
        urllib.request.urlretrieve(URL, RAW)

    df = pd.read_csv(RAW, index_col=0)
    df = df[df["Review_Language"] == "English"].reset_index(drop=True)

    # 评论分成优点(upside)和缺点(downside)两段，可能有一段为空，拼成一个文本
    up = df["Upside_Review"].fillna("").str.strip()
    down = df["Downside_Review"].fillna("").str.strip()

    # 只用文本做特征，评分/酒店/城市只是留着参考
    out = pd.DataFrame({
        "id": df.index,
        "text": (up + " " + down).str.strip(),
        "label": df["source"].astype(int),  # 1 = AI 生成, 0 = 人写
        "sentiment": df["Sentiment"],
        "upside": up,
        "downside": down,
        "score": df["Review_Score"],
        "city": df["City Name"],
        "hotel": df["Hotel Name"],
    })

    # 按 标签 x 正负面 分层，这样测试集里正面和负面评论也是平衡的
    grp = out["label"].astype(str) + "_" + out["sentiment"]
    tr, te = train_test_split(out, test_size=0.25, stratify=grp, random_state=SEED)

    os.makedirs(os.path.dirname(TRAIN), exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    tr.sort_values("id").to_csv(TRAIN, index=False)
    te.sort_values("id").to_csv(TEST, index=False)

    tr = tr.assign(split="train")
    te = te.assign(split="test")
    cnt = pd.crosstab(pd.concat([tr, te])["split"], [out["label"], out["sentiment"]])
    cnt.to_csv(os.path.join(TAB_DIR, "split_summary.csv"))
    print(cnt)


if __name__ == "__main__":
    main()
