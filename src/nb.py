# 多项式朴素贝叶斯
# 词袋 -> 卡方检验选前 k 个词 -> MultinomialNB
# 调参: k 和平滑参数 alpha，5 折交叉验证
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocessing import TextPreprocessor

KS = [10, 25, 50, 100, 200, 300, 400, 500, 750, 1000, "all"]
ALPHAS = [0.1, 0.5, 1.0]


def tune(d):
    pipe = Pipeline([
        ("vec", TextPreprocessor().make_vec()),
        ("sel", SelectKBest(chi2)),
        ("clf", MultinomialNB()),
    ])
    grid = {"sel__k": KS, "clf__alpha": ALPHAS}
    gs = GridSearchCV(pipe, grid, cv=d.folds(), scoring="accuracy", n_jobs=-1)
    gs.fit(d.X, d.y)

    cv = gs.cv_results_
    res = pd.DataFrame({
        "k": list(cv["param_sel__k"]),
        "alpha": list(cv["param_clf__alpha"]),
        "acc": cv["mean_test_score"],
        "sd": cv["std_test_score"],
    })
    return gs.best_estimator_, gs.best_params_, res


def top(m, n=5):
    # 两类的对数条件概率之差 log P(w|AI) - log P(w|human)
    # 大于 0 偏向 AI，小于 0 偏向人写
    names = m["vec"].get_feature_names_out()[m["sel"].get_support()]
    lp = m["clf"].feature_log_prob_
    s = lp[1] - lp[0]
    idx = np.argsort(s)

    rows = []
    for i in idx[::-1][:n]:
        rows.append([names[i], s[i], "AI"])
    for i in idx[:n]:
        rows.append([names[i], s[i], "human"])
    return pd.DataFrame(rows, columns=["feature", "score", "class"])


def plot(res, path):
    res = res.copy()
    res["k"] = res["k"].replace("all", -1)
    res = res.sort_values("k")
    b = res.loc[res["acc"].idxmax()]

    plt.figure(figsize=(6, 4))
    # 三个 alpha 的结果几乎重合，左右错开一点
    for a, dx in zip(ALPHAS, [0.94, 1.0, 1.06]):
        r = res[(res["alpha"] == a) & (res["k"] != -1)]
        plt.errorbar(r["k"] * dx, r["acc"], yerr=r["sd"], marker="o", ms=4, capsize=2, label=f"alpha={a}")
    plt.axhline(res[res["k"] == -1]["acc"].max(), color="gray", ls="--", label="all words")
    plt.plot(b["k"], b["acc"], "r*", ms=14, label=f"selected (k={b['k']}, alpha={b['alpha']})")
    plt.xscale("log")
    plt.xlabel("k (number of selected words)")
    plt.ylabel("5-fold CV accuracy")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
