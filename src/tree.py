# 单棵分类树
# 词袋 -> DecisionTreeClassifier
# 调参: 代价复杂度剪枝参数 ccp_alpha (rpart 里的 cp)，5 折交叉验证
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from config import SEED
from preprocessing import TextPreprocessor

ALPHAS = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]


def tune(d):
    pipe = Pipeline([
        ("vec", TextPreprocessor().make_vec()),
        ("clf", DecisionTreeClassifier(random_state=SEED)),
    ])
    gs = GridSearchCV(pipe, {"clf__ccp_alpha": ALPHAS}, cv=d.folds(), scoring="accuracy", n_jobs=-1)
    gs.fit(d.X, d.y)

    cv = gs.cv_results_
    res = pd.DataFrame({
        "alpha": list(cv["param_clf__ccp_alpha"]),
        "acc": cv["mean_test_score"],
        "sd": cv["std_test_score"],
    })
    return gs.best_estimator_, gs.best_params_, res


def top(m, n=5):
    # 基尼重要性 (每个词带来的不纯度下降之和)
    names = m["vec"].get_feature_names_out()
    imp = m["clf"].feature_importances_
    idx = np.argsort(imp)[::-1][:n]
    return pd.DataFrame({"feature": names[idx], "score": imp[idx]})


def plot(res, path):
    # alpha 有 0，用 symlog 画
    plt.figure(figsize=(6, 4))
    plt.errorbar(res["alpha"], res["acc"], yerr=res["sd"], marker="o", capsize=3)
    plt.xscale("symlog", linthresh=0.0005)
    plt.xlabel("ccp_alpha")
    plt.ylabel("5-fold CV accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
