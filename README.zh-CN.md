# AI 生成酒店评论检测

[English](README.md) | 中文

本项目的任务是判断一条英文酒店评论出自人类（Booking.com 上的真实评论）还是 GPT-4。数据来自 MAiDE-up 语料库（Ignat et al., NAACL 2025 Findings）中的英文部分。

## 目录结构

```
project1/
├── assignment/               作业说明（PDF）
├── data/
│   ├── raw/all_data.csv      原始数据，来自 MichiganNLP/multilingual_reviews_deception
│   └── processed/            train.csv（1500 条）/ test.csv（500 条），由 split_data.py 生成
├── src/
│   ├── config.py             路径、随机种子、列名
│   ├── split_data.py         筛选英文评论，按 75/25 分层划分训练集和测试集
│   ├── load_data.py          ReviewData 类：加载数据，划分训练/验证集，生成交叉验证折
│   ├── preprocessing.py      （待写）文本清洗与词袋表示
│   ├── tune_models.py        （待写）5 个模型的交叉验证 / OOB 调参
│   ├── evaluate.py           （待写）测试集指标与 McNemar 检验
│   ├── feature_importance.py （待写）每个模型最重要的 5 个特征
│   └── main.py               （待写）一次跑完整个流程
├── results/
│   ├── tables/               报告表格所用的 CSV 结果
│   └── figures/              报告所用的图
├── report/                   LaTeX 源文件（JMLR 文档类，单栏）
├── requirements.txt
├── README.md                 英文说明
└── README.zh-CN.md           中文说明
```

## 环境配置

开发时使用的是 Python 3.13。

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Linux 和 macOS 上把 `.venv/Scripts/python` 换成 `.venv/bin/python`。

## 运行

### 1. 划分数据集

```bash
python src/split_data.py
```

如果 `data/raw/all_data.csv` 不存在，脚本会先下载它，然后做三件事：

1. 保留 2000 条英文评论，其中人写的 1000 条（`source = 0`），GPT-4 生成的 1000 条（`source = 1`）；
2. 把每条评论的 upside（优点）和 downside（缺点）两段拼成一个 `text` 字段；
3. 按"标签 × 正负面"分层，把数据 75/25 分成训练集和测试集，`random_state = 42`。

| 数据集 | 人写 NEG | 人写 POS | AI NEG | AI POS | 合计 |
|--------|----------|----------|--------|--------|------|
| train  | 375      | 375      | 375    | 375    | 1500 |
| test   | 125      | 125      | 125    | 125    | 500  |

同样的计数也保存在 `results/tables/split_summary.csv`。

### 2. 加载数据

```python
from load_data import ReviewData

train = ReviewData.load_train()      # 1500 条
fit, val = train.train_val_split()   # 1200 / 300，返回的仍是 ReviewData
folds = train.cv_folds()             # 5 折分层索引，可直接传给 GridSearchCV(cv=folds)
test = ReviewData.load_test()        # 500 条，只在最终评估时使用
```

`ReviewData` 对象有以下属性：`ids`、`text`、`y`（1 表示 AI，0 表示人写）、`sentiment`，以及用于分层的 `strata`（标签 × 正负面）。验证集划分和交叉验证折都按 `strata` 分层。

运行 `python src/load_data.py` 会打印各部分的类别分布，可以用来检查划分是否正确。

## 数据列说明（processed 文件）

| 列名 | 说明 |
|------|------|
| `review_id` | 评论在英文子集中的行号 |
| `text` | upside + downside 拼接后的文本，是模型唯一的输入 |
| `label` | 1 表示 AI 生成（假评论），0 表示人写（真评论） |
| `sentiment` | `POS` / `NEG`，用于按正负面分组评估测试集 |
| `upside`、`downside` | 原始的两段评论 |
| `review_score`、`city`、`hotel` | 元数据，不作为特征使用 |

## 设计说明

- **测试集只用一次**：`load_test()` 只在最终评估 5 个选定模型时调用，调参和模型选择都只用训练集。
- **按标签 × 正负面分层**：保证训练集和测试集的类别分布一致，同时让测试集中正面、负面评论数量平衡，方便比较模型在两类评论上的准确率。
- **不使用评分等元数据**：AI 评论的评分分布与真实评论明显不同（负面评论平均 3.0 对 3.7，正面评论平均 8.4 对 9.4）。使用评分会让模型依赖元数据而不是文本，所以只用评论文本作为输入。
