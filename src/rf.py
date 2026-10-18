# 随机森林
# 词袋 -> RandomForestClassifier
# 调参: 树的数量和每次分裂考虑的特征数量，5 折交叉验证
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config import SEED
from preprocessing import TextPreprocessor

N_TREES = [100, 300, 500]
MAX_FEATURES = ["sqrt", 0.1, 0.3]


def tune(d):
    pipe = Pipeline([
        ("vec", TextPreprocessor().make_vec()),
        ("clf", RandomForestClassifier(random_state=SEED, n_jobs=1)),
    ])
    grid = {
        "clf__n_estimators": N_TREES,
        "clf__max_features": MAX_FEATURES,
    }
    # 并行交叉验证；refit=True 会在完整训练集上重新训练最优模型
    gs = GridSearchCV(pipe, grid, cv=d.folds(), scoring="accuracy", n_jobs=-1, refit=True, verbose=1,
                      error_score="raise")
    gs.fit(d.X, d.y)

    cv = gs.cv_results_
    res = pd.DataFrame({
        "n_estimators": list(cv["param_clf__n_estimators"]),
        "max_features": list(cv["param_clf__max_features"]),
        "acc": cv["mean_test_score"],
        "sd": cv["std_test_score"],
    })
    return gs.best_estimator_, gs.best_params_, res


def top(m, n=5):
    # 不纯度下降重要性：只有大小，不表示偏向 AI 还是人写
    names = m["vec"].get_feature_names_out()
    imp = m["clf"].feature_importances_
    idx = np.argsort(imp)[::-1][:n]
    return pd.DataFrame({"feature": names[idx], "score": imp[idx]})


def plot(res, path):
    # 每条曲线固定 max_features，只改变树的数量
    plt.figure(figsize=(7, 4))
    for f in res["max_features"].unique():
        r = res[res["max_features"] == f].sort_values("n_estimators")
        plt.errorbar(r["n_estimators"], r["acc"], yerr=r["sd"], marker="o", capsize=3, label=f"max_features={f}")
    plt.xlabel("Number of trees")
    plt.ylabel("5-fold CV accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
