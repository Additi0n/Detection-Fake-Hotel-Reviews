import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config import SEED
from preprocessing import TextPreprocessor

N_TREES = [100, 300, 500, 1000]
MAX_FEATURES = [1, 3, "log2", "sqrt", 0.1, 0.3]


def tune(d):
    pipe = Pipeline([
        ("vec", TextPreprocessor().make_vec()),
        ("clf", RandomForestClassifier(random_state=SEED, n_jobs=1)),
    ])
    grid = {
        "clf__n_estimators": N_TREES,
        "clf__max_features": MAX_FEATURES,
    }
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
    names = m["vec"].get_feature_names_out()
    imp = m["clf"].feature_importances_
    idx = np.argsort(imp)[::-1][:n]
    return pd.DataFrame({"feature": names[idx], "score": imp[idx]})


def plot(res, path):
    b = res.loc[res["acc"].idxmax()]

    plt.figure(figsize=(6, 4))
    for f, dx in zip(MAX_FEATURES, [-30, -18, -6, 6, 18, 30]):
        r = res[res["max_features"].astype(str) == str(f)].sort_values("n_estimators")
        plt.errorbar(r["n_estimators"] + dx, r["acc"], yerr=r["sd"], marker="o", ms=4, capsize=3,
                     label=f"max_features={f}")
    plt.plot(b["n_estimators"], b["acc"], "r*", ms=14,
             label=f"selected ({b['n_estimators']} trees, {b['max_features']})")
    plt.xticks(N_TREES)
    plt.xlabel("Number of trees")
    plt.ylabel("5-fold CV accuracy")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
