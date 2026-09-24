"""加载处理好的训练集和测试集，并划分验证集。

测试集只用于最终评估选定的 5 个模型。模型开发全部在训练集上进行，
可以用一次性的留出验证集 (train_val_split)
也可以用分层 k 折交叉验证 (cv_folds)。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import (LABEL_COL, RANDOM_STATE, SENTIMENT_COL, TEST_FILE,
                    TEXT_COL, TRAIN_FILE)


class ReviewData:
    VAL_SIZE = 0.2
    N_FOLDS = 5

    def __init__(self, df):
        df = df.reset_index(drop=True)
        self.ids = df["review_id"].to_numpy()
        self.text = df[TEXT_COL]
        self.y = df[LABEL_COL].to_numpy()
        self.sentiment = df[SENTIMENT_COL].to_numpy()
        self._df = df

    def __len__(self):
        return len(self.y)

    @classmethod
    def _read(cls, path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python src/split_data.py` first")
        return cls(pd.read_csv(path, keep_default_na=False))

    @classmethod
    def load_train(cls):
        return cls._read(TRAIN_FILE)

    @classmethod
    def load_test(cls):
        """只在最终评估选定模型时调用。"""
        return cls._read(TEST_FILE)

    @property
    def strata(self):
        """按标签 x 正负面分组，用于分层划分。"""
        return np.char.add(self.y.astype(str), self.sentiment.astype(str))

    def subset(self, idx):
        return ReviewData(self._df.iloc[idx])

    def train_val_split(self, val_size=VAL_SIZE, random_state=RANDOM_STATE):
        """划分为训练部分和验证部分。

        默认 val_size = 0.2 时 ,1500 条训练评论分成 1200 条用于拟合、
        300 条用于验证，按"标签 * 正负面"分层。
        """
        train_idx, val_idx = train_test_split(np.arange(len(self)), test_size=val_size,
                                              stratify=self.strata,
                                              random_state=random_state)
        return self.subset(train_idx), self.subset(val_idx)

    def cv_folds(self, n_splits=N_FOLDS, random_state=RANDOM_STATE):
        """分层 k 折交叉验证的索引，按"标签 * 正负面"分层。

        返回的列表可以直接作为 cv 传给 GridSearchCV 或
        LogisticRegressionCV, 让所有模型都在同一组折上调参。
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(skf.split(np.zeros(len(self)), self.strata))


if __name__ == "__main__":
    train = ReviewData.load_train()
    fit, val = train.train_val_split()
    for name, part in (("train (full)", train), ("fit", fit), ("validation", val)):
        counts = pd.crosstab(part.y, part.sentiment)
        print(f"{name}: {len(part)} reviews\n{counts}\n")
    for k, (tr, va) in enumerate(train.cv_folds()):
        print(f"fold {k}: {len(tr)} train / {len(va)} validation, "
              f"AI share in validation = {train.y[va].mean():.2f}")
