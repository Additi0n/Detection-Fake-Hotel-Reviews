# 梯度提升
# 词袋 -> GradientBoostingClassifier
# 调参: 树的数量、学习率和树深度，5 折交叉验证
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config import SEED
from preprocessing import TextPreprocessor

N_TREES = [100, 200]
LEARNING_RATES = [0.03, 0.1]
DEPTHS = [1, 2, 3]


def tune(d):
    pipe = Pipeline([
        ("vec", TextPreprocessor().make_vec()),
        ("clf", GradientBoostingClassifier(random_state=SEED)),
    ])
    grid = {
        "clf__n_estimators": N_TREES,
        "clf__learning_rate": LEARNING_RATES,
        "clf__max_depth": DEPTHS,
    }
    # 并行交叉验证；refit=True 会在完整训练集上重新训练最优模型
    gs = GridSearchCV(pipe, grid, cv=d.folds(), scoring="accuracy",
                      n_jobs=-1, refit=True, verbose=1, error_score="raise")
    gs.fit(d.X, d.y)

    cv = gs.cv_results_
    res = pd.DataFrame({
        "n_estimators": list(cv["param_clf__n_estimators"]),
        "learning_rate": list(cv["param_clf__learning_rate"]),
        "max_depth": list(cv["param_clf__max_depth"]),
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
    # 每个子图固定树深度，每条曲线固定学习率
    depths = sorted(res["max_depth"].unique())
    fig, axes = plt.subplots(1, len(depths), figsize=(5 * len(depths), 4),sharey=True, squeeze=False)
    for ax, depth in zip(axes[0], depths):
        sub = res[res["max_depth"] == depth]
        for rate in sorted(sub["learning_rate"].unique()):
            r = sub[sub["learning_rate"] == rate].sort_values("n_estimators")
            ax.errorbar(r["n_estimators"], r["acc"], yerr=r["sd"],
                        marker="o", capsize=3, label=f"learning_rate={rate}")
        ax.set_title(f"max_depth={depth}")
        ax.set_xlabel("Number of trees")
        ax.legend()
    axes[0, 0].set_ylabel("5-fold CV accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
