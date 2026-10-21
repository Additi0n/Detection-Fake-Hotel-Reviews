# Lasso 逻辑回归 (L1 正则)
# 词袋 -> LogisticRegression(penalty="l1")
# 调参: C = 1/lambda，5 折交叉验证
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config import SEED
from preprocessing import TextPreprocessor

CS = np.logspace(-2, 2, 17)


def tune(d):
    pipe = Pipeline([
        ("vec", TextPreprocessor().make_vec()),
        # penalty 和 l1_ratio 都写上：sklearn < 1.8 只看 penalty（只写 l1_ratio=1 会悄悄变成 L2），
        # 1.8 以后只看 l1_ratio
        ("clf", LogisticRegression(penalty="l1", l1_ratio=1, solver="liblinear", max_iter=5000,
                                   random_state=SEED)),
    ])
    # penalty 在 1.8 以后会报弃用警告；单进程跑才能把每次拟合的警告都屏蔽掉，LR 很快
    gs = GridSearchCV(pipe, {"clf__C": CS}, cv=d.folds(), scoring="accuracy", n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        gs.fit(d.X, d.y)

    cv = gs.cv_results_
    res = pd.DataFrame({
        "C": list(cv["param_clf__C"]),
        "acc": cv["mean_test_score"],
        "sd": cv["std_test_score"],
    })
    return gs.best_estimator_, gs.best_params_, res


def top(m, n=5):
    # 系数 > 0 偏向 AI (label=1)，< 0 偏向人写
    names = m["vec"].get_feature_names_out()
    w = m["clf"].coef_[0]
    idx = np.argsort(w)

    rows = []
    for i in idx[::-1][:n]:
        rows.append([names[i], w[i], "AI"])
    for i in idx[:n]:
        rows.append([names[i], w[i], "human"])
    return pd.DataFrame(rows, columns=["feature", "score", "class"])


def plot(res, path):
    b = res.loc[res["acc"].idxmax()]

    plt.figure(figsize=(6, 4))
    plt.errorbar(res["C"], res["acc"], yerr=res["sd"], marker="o", ms=4, capsize=3)
    plt.plot(b["C"], b["acc"], "r*", ms=14, label=f"selected (C={b['C']:.3g})")
    plt.xscale("log")
    plt.xlabel("C (= 1/lambda)")
    plt.ylabel("5-fold CV accuracy")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
