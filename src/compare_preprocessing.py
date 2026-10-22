import os
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import SEED, TAB_DIR
from load_data import ReviewData
from preprocessing import TextPreprocessor

SETTINGS = [
    ("min_df=1", dict(min_df=1)),
    ("default (min_df=5)", dict()),
    ("min_df=10", dict(min_df=10)),
    ("stop words removed", dict(stop=True)),
    ("stemming", dict(stem=True)),
    ("stop words removed + stemming", dict(stop=True, stem=True)),
    ("unigrams + bigrams", dict(ngram=(1, 2))),
]


def main():
    d = ReviewData.load("train")
    folds = d.folds()

    rows = []
    for name, kw in SETTINGS:
        n_feat = len(TextPreprocessor(**kw).make_vec().fit(d.X).vocabulary_)

        nb = Pipeline([("vec", TextPreprocessor(**kw).make_vec()), ("clf", MultinomialNB())])
        lr = Pipeline([
            ("vec", TextPreprocessor(**kw).make_vec()),
            ("clf", LogisticRegressionCV(Cs=10, penalty="l1", solver="liblinear",
                                         scoring="accuracy", cv=5, max_iter=5000, random_state=SEED)),
        ])
        nb_acc = cross_val_score(nb, d.X, d.y, cv=folds, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            lr_acc = cross_val_score(lr, d.X, d.y, cv=folds, n_jobs=1)

        rows.append([name, n_feat, nb_acc.mean(), nb_acc.std(), lr_acc.mean(), lr_acc.std()])
        print(name, n_feat, "nb %.3f  lr %.3f" % (nb_acc.mean(), lr_acc.mean()))

    res = pd.DataFrame(rows, columns=["setting", "n_features", "nb_acc", "nb_sd", "lr_acc", "lr_sd"])
    os.makedirs(TAB_DIR, exist_ok=True)
    res.to_csv(os.path.join(TAB_DIR, "preprocessing.csv"), index=False)


if __name__ == "__main__":
    main()
