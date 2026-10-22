import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config import SEED
from preprocessing import TextPreprocessor

N_TREES = [100, 200, 400, 800]
LEARNING_RATES = [0.03, 0.1, 0.3]
DEPTHS = [1, 2, 3, 4, 5]


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
    names = m["vec"].get_feature_names_out()
    imp = m["clf"].feature_importances_
    idx = np.argsort(imp)[::-1][:n]
    return pd.DataFrame({"feature": names[idx], "score": imp[idx]})


def plot(res, path):
    b = res.loc[res["acc"].idxmax()]
    fig, axes = plt.subplots(1, len(DEPTHS), figsize=(13, 3.3), sharey=True)
    for ax, depth in zip(axes, DEPTHS):
        sub = res[res["max_depth"] == depth]
        for rate in LEARNING_RATES:
            r = sub[sub["learning_rate"] == rate].sort_values("n_estimators")
            ax.errorbar(r["n_estimators"], r["acc"], yerr=r["sd"], marker="o", ms=4, capsize=3,
                        label=f"learning_rate={rate}")
        if depth == b["max_depth"]:
            ax.plot(b["n_estimators"], b["acc"], "r*", ms=14, label="selected")
        ax.set_title(f"max_depth={depth}")
        ax.set_xscale("log")
        ax.set_xticks(N_TREES, [str(n) for n in N_TREES])
        ax.minorticks_off()
        ax.set_xlabel("Number of trees")
    axes[0].set_ylabel("5-fold CV accuracy")
    h, l = axes[list(DEPTHS).index(b["max_depth"])].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
