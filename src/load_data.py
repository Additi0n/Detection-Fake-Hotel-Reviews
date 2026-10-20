# 读取训练集/测试集，划分验证集和交叉验证的折
# 测试集只在最后评估模型时用一次，调参只用训练集
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import SEED, TEST, TRAIN


class ReviewData:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.X = self.df["text"]
        self.y = self.df["label"].values
        self.sent = self.df["sentiment"].values
        # 标签 x 正负面，分层用
        self.grp = self.df["label"].astype(str) + "_" + self.df["sentiment"]

    @staticmethod
    def load(name):
        # name 是 "train" 或 "test"
        path = TRAIN if name == "train" else TEST
        return ReviewData(pd.read_csv(path, keep_default_na=False))

    def split(self, size=0.2):
        # 留出一部分做验证集，默认 1200 / 300
        tr, va = train_test_split(self.df, test_size=size, stratify=self.grp,
                                  random_state=SEED)
        return ReviewData(tr), ReviewData(va)

    def folds(self, k=5):
        # 分层 k 折，所有模型调参都用同一组折
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        return list(skf.split(self.X, self.grp))
