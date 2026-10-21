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

    up = df["Upside_Review"].fillna("").str.strip()
    down = df["Downside_Review"].fillna("").str.strip()

    out = pd.DataFrame({
        "id": df.index,
        "text": (up + " " + down).str.strip(),
        "label": df["source"].astype(int),  
        "sentiment": df["Sentiment"],
        "upside": up,
        "downside": down,
        "score": df["Review_Score"],
        "city": df["City Name"],
        "hotel": df["Hotel Name"],
    })

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
