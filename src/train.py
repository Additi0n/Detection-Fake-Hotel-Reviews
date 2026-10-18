# 在训练集上调参，用最优参数在整个训练集上重新训练并保存模型
# 用法: python src/train.py            (全部模型)
#       python src/train.py nb lr     (只跑某几个)
# 输出:
#   results/models/{name}.joblib        最终模型
#   results/tables/cv_{name}.csv        调参结果
#   results/tables/feat_{name}.csv      前 5 个重要特征
#   results/figures/cv_{name}.png       调参曲线
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import lr
import nb
import tree
import rf
import gb
from config import FIG_DIR, MOD_DIR, TAB_DIR
from load_data import ReviewData

MODELS = {"nb": nb, "lr": lr, "tree": tree, "rf": rf, "gb": gb, }


def main():
    names = sys.argv[1:] or list(MODELS)
    for p in [TAB_DIR, FIG_DIR, MOD_DIR]:
        os.makedirs(p, exist_ok=True)

    d = ReviewData.load("train")

    for name in names:
        mod = MODELS[name]
        print("====", name)

        m, best, res = mod.tune(d)
        print("best params:", best)
        print("best cv acc: %.4f" % res["acc"].max())

        res.to_csv(os.path.join(TAB_DIR, "cv_%s.csv" % name), index=False)
        mod.plot(res, os.path.join(FIG_DIR, "cv_%s.png" % name))
        joblib.dump(m, os.path.join(MOD_DIR, "%s.joblib" % name))

        feat = mod.top(m)
        feat.to_csv(os.path.join(TAB_DIR, "feat_%s.csv" % name), index=False)
        print(feat.to_string(index=False))


if __name__ == "__main__":
    main()
