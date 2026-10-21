
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

        ("clf", LogisticRegression(penalty="l1", l1_ratio=1, solver="liblinear", max_iter=5000,
                                   random_state=SEED)),
    ])
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
