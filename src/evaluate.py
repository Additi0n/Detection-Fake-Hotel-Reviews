# 在最终测试集上评估五个模型，并完成作业需要的 McNemar 检验
#
# 运行顺序：
#   python src/train.py
#   python src/evaluate.py
#
# label = 1 表示 AI 生成评论，label = 0 表示人类评论。

import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

from config import MOD_DIR, TAB_DIR
from load_data import ReviewData


MODELS = ["nb", "lr", "tree", "rf", "gb"]

# 只比较作业问题真正需要的模型，不做所有模型的两两比较。
COMPARISONS = [
    ("nb", "lr"),
    ("tree", "rf"),
    ("tree", "gb"),
]


def model_metrics(y_true, y_pred):
    """计算一个模型的四个主要评价指标。"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def mcnemar_table(y_true, pred_a, pred_b):
    """制作 McNemar 检验使用的 2 x 2 正确/错误表。"""
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    both_correct = int((a_correct & b_correct).sum())
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    both_wrong = int((~a_correct & ~b_correct).sum())

    return [[both_correct, a_only], [b_only, both_wrong]]


def main():
    os.makedirs(TAB_DIR, exist_ok=True)
    test = ReviewData.load("test")

    # 保存每条测试评论的真实标签和各模型预测，方便复查。
    predictions = pd.DataFrame({
        "id": test.df["id"],
        "label": test.y,
        "sentiment": test.sent,
    })

    metric_rows = []

    for name in MODELS:
        path = os.path.join(MOD_DIR, name + ".joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "找不到模型文件 %s。请先运行 python src/train.py" % path
            )

        model = joblib.load(path)
        pred = model.predict(test.X)
        predictions[name] = pred

        row = {"model": name}
        row.update(model_metrics(test.y, pred))
        metric_rows.append(row)

        # 行是真实类别，列是预测类别；0=human，1=AI。
        cm = confusion_matrix(test.y, pred, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["true_human", "true_ai"],
            columns=["pred_human", "pred_ai"],
        )
        cm_df.to_csv(os.path.join(TAB_DIR, "confusion_%s.csv" % name))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(os.path.join(TAB_DIR, "test_metrics.csv"), index=False)
    predictions.to_csv(
        os.path.join(TAB_DIR, "test_predictions.csv"), index=False
    )

    # McNemar 检验：零假设是两个模型犯错的概率相同。
    test_rows = []
    for name_a, name_b in COMPARISONS:
        table = mcnemar_table(
            test.y,
            predictions[name_a].to_numpy(),
            predictions[name_b].to_numpy(),
        )

        # exact=True 使用精确检验，避免依赖大样本近似。
        result = mcnemar(table, exact=True)

        test_rows.append({
            "model_a": name_a,
            "model_b": name_b,
            "both_correct": table[0][0],
            "a_correct_b_wrong": table[0][1],
            "a_wrong_b_correct": table[1][0],
            "both_wrong": table[1][1],
            "statistic": result.statistic,
            "p_value": result.pvalue,
            "significant_at_0.01": result.pvalue < 0.01,
        })

    tests = pd.DataFrame(test_rows)
    tests.to_csv(os.path.join(TAB_DIR, "mcnemar_tests.csv"), index=False)

    print("\nFinal test-set metrics")
    print(metrics.round(4).to_string(index=False))
    print("\nMcNemar tests (alpha = 0.01)")
    print(tests.to_string(index=False))
    print("\nResults saved in:", TAB_DIR)


if __name__ == "__main__":
    main()
