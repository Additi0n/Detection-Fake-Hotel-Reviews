# AI 生成酒店评论检测

[English](README.md) | 中文

本项目的任务是判断一条英文酒店评论出自人类（Booking.com 上的真实评论）还是 GPT-4。数据来自 MAiDE-up 语料库（Ignat et al., NAACL 2025 Findings）中的英文部分。

## 目录结构

```
project1/
├── assignment/            作业说明（PDF）
├── data/
│   ├── raw/all_data.csv   原始数据，来自 MichiganNLP/multilingual_reviews_deception
│   └── processed/         train.csv（1500 条）/ test.csv（500 条），由 split_data.py 生成
├── src/
│   ├── config.py          路径和随机种子
│   ├── split_data.py      筛选英文评论，按 75/25 分层划分训练集和测试集
│   ├── load_data.py       ReviewData 类：读取数据、划分验证集、生成交叉验证折
│   ├── preprocessing.py   TextPreprocessor 类：分词和词袋向量化器
│   ├── nb.py              多项式朴素贝叶斯（加卡方 top-k 特征选择）
│   ├── lr.py              Lasso（L1）逻辑回归
│   ├── tree.py            单棵分类树（代价复杂度剪枝）
│   ├── rf.py              随机森林（树的数量、每次分裂考虑的特征数）
│   ├── gb.py              梯度提升（树的数量、学习率、树深度）
│   └── train.py           调参，用最优参数在全部训练集上重新训练，保存结果
├── results/
│   ├── models/            训练好的模型（.joblib），不进 git
│   ├── tables/            报告表格所用的 CSV 结果
│   └── figures/           报告所用的图
├── report/                LaTeX 源文件（JMLR 文档类，单栏）
├── requirements.txt
├── README.md              英文说明
└── README.zh-CN.md        中文说明
```

还没写的：`evaluate.py`（测试集指标、McNemar 检验、随机森林在正面和负面评论上的比较），见文末“后续评估”。

## 环境配置

开发时使用的是 Python 3.13。

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Linux 和 macOS 上把 `.venv/Scripts/python` 换成 `.venv/bin/python`。

## 运行

```bash
python src/split_data.py     # 1. 生成 train.csv / test.csv
python src/train.py          # 2. 调参并训练全部 5 个模型：nb、lr、tree、rf、gb
python src/train.py nb lr    #    也可以只跑其中几个
python src/train.py rf gb    #    只跑随机森林和梯度提升
```

### 1. 划分数据集（`split_data.py`）

如果 `data/raw/all_data.csv` 不存在，脚本会先下载它，然后做三件事：

1. 保留 2000 条英文评论，其中人写的 1000 条（`source = 0`），GPT-4 生成的 1000 条（`source = 1`）；
2. 把每条评论的 upside（优点）和 downside（缺点）两段拼成一个 `text` 字段；
3. 按"标签 × 正负面"分层，把数据 75/25 分成训练集和测试集，随机种子为 42。

| 数据集 | 人写 NEG | 人写 POS | AI NEG | AI POS | 合计 |
|--------|----------|----------|--------|--------|------|
| train  | 375      | 375      | 375    | 375    | 1500 |
| test   | 125      | 125      | 125    | 125    | 500  |

同样的计数也保存在 `results/tables/split_summary.csv`。

### 2. 训练模型（`train.py`）

`train.py` 对每个模型在训练集上做网格搜索，用 5 折分层交叉验证（所有模型用同一组折），选出最优参数后在全部 1500 条训练数据上重新训练，输出以下文件：

| 文件 | 内容 |
|------|------|
| `results/models/{name}.joblib` | 最终训练好的 Pipeline |
| `results/tables/cv_{name}.csv` | 每组参数的交叉验证准确率（均值和标准差） |
| `results/figures/cv_{name}.png` | 调参曲线 |
| `results/tables/feat_{name}.csv` | 前 5 个重要特征 |

这一步不使用测试集。

| 模型 | Pipeline | 参数网格 | 选出的参数 | 交叉验证准确率 |
|------|----------|----------|-----------|----------------|
| `nb` | 词袋 → SelectKBest(chi2) → MultinomialNB | k ∈ {10, …, 1000, all}，alpha ∈ {0.1, 0.5, 1} | k = 750，alpha = 0.5 | 0.903 |
| `lr` | 词袋 → LogisticRegression(l1_ratio = 1) | C 从 10^-2 到 10^2，共 17 个值 | C = 1（1351 个权重中 203 个非零） | 0.915 |
| `tree` | 词袋 → DecisionTreeClassifier | ccp_alpha ∈ {0, …, 0.05} | ccp_alpha = 0.002（53 个叶节点） | 0.818 |
| `rf` | 词袋 → RandomForestClassifier | n_estimators ∈ {100, 300, 500}，max_features ∈ {sqrt, 0.1, 0.3} | n_estimators = 300，max_features = sqrt | 0.914 |
| `gb` | 词袋 → GradientBoostingClassifier | n_estimators ∈ {100, 200}，learning_rate ∈ {0.03, 0.1}，max_depth ∈ {1, 2, 3} | n_estimators = 200，learning_rate = 0.1，max_depth = 3 | 0.909 |

所有模型都用 `GridSearchCV` 在 `d.folds()` 给出的同一组 5 折上调参，并固定 `random_state = 42`，其余参数用库的默认值。随机森林共 9 组参数 × 5 折 = 45 次训练，梯度提升共 12 组 × 5 折 = 60 次，都不含最后在全部训练集上的那一次重新训练。

随机森林和梯度提升的参数含义：

- **n_estimators**：随机森林里是树的数量，梯度提升里是迭代的轮数（每轮加一棵树）。
- **max_features**：随机森林每次分裂时考虑的特征数。`sqrt` 是特征总数的平方根，0.1 和 0.3 是约 10% 和 30% 的特征。
- **learning_rate**：梯度提升每加一棵新树时，这棵树的贡献权重。
- **max_depth**：梯度提升里每棵树的最大深度。

梯度提升选出的参数（200 轮、学习率 0.1、深度 3）落在网格的边界上，扩大网格后交叉验证准确率可能还会提高。如果要调整网格，只能根据训练集上的交叉验证结果来决定。

`cv_{name}.csv` 是交叉验证结果，不能当作测试集准确率。调参图的画法：

- **nb**：横轴是 k，每条曲线对应一个 alpha。
- **lr**、**tree**：横轴分别是 C 和 ccp_alpha。
- **rf**：横轴是树的数量，每条曲线对应一个 max_features。
- **gb**：每个子图对应一个树深度，每条曲线对应一个学习率，横轴是树的数量。

所有图的误差线都是 5 折结果的标准差，不是置信区间。

重要特征的计算方法：

- **nb**：log P(词 | AI) − log P(词 | 人写)，大于 0 偏向 AI，小于 0 偏向人写。
- **lr**：回归系数，大于 0 偏向 AI，小于 0 偏向人写。
- **tree**、**rf**、**gb**：`feature_importances_`，即基于不纯度下降的重要性，在所有树上累加。

树模型的重要性只有大小，没有方向，光看分数判断不了一个词偏向 AI 还是人写。相关的词会互相分摊重要性，这种指标本身也有偏差，所以报告里要结合词义和训练数据谨慎解释。

### 模块用法

```python
from load_data import ReviewData
from preprocessing import TextPreprocessor

d = ReviewData.load("train")     # 1500 条；d.X 是文本，d.y 是标签，d.sent 是正负面
tr, va = d.split()               # 留出验证集，1200 / 300
folds = d.folds()                # 5 折分层索引，传给 GridSearchCV(cv=folds)
te = ReviewData.load("test")     # 500 条，只在最终评估时使用

vec = TextPreprocessor().make_vec()   # 未拟合的 CountVectorizer，放进 Pipeline 使用
```

验证集划分和交叉验证折都按"标签 × 正负面"（`d.grp`）分层。

每个模型文件（`nb.py`、`lr.py`、`tree.py`、`rf.py`、`gb.py`）都提供同样三个函数，`train.py` 通过 `MODELS` 字典调用它们：

| 函数 | 作用 | 返回值 / 输出 |
|------|------|--------------|
| `tune(d)` | 建 Pipeline，交叉验证调参，用最优参数重新训练 | 最终模型、最优参数、调参结果表 |
| `top(m, n=5)` | 取最重要的 n 个词 | `feature`、`score` 两列（线性模型还有 `class` 列）的 DataFrame |
| `plot(res, path)` | 画调参曲线 | 保存 PNG 图片 |

## 文本预处理

1. 转小写，把弯引号统一成直引号，去掉重音符号；
2. 只保留字母组成的词，保留 wasn't 这类缩写，丢弃数字和标点；
3. **保留停用词，不做词干提取，只用单词（unigram）**；
4. 用 `min_df = 5` 删掉出现在少于 5 条评论里的稀疏词，训练集词表从 4737 个词降到 1351 个。

向量化器是每个 Pipeline 的第一步，所以交叉验证时每一折只用自己的训练部分建词表。

其他设置可以用来做比较，例如 `TextPreprocessor(stop=True, stem=True, ngram=(1, 2), min_df=10)`。确定默认设置之前，我们在训练集上用 5 折交叉验证粗略比较过几种设置（准确率，模型未调参）：

| 设置 | 特征数 | 朴素贝叶斯 | Lasso 逻辑回归 |
|------|--------|-----------|----------------|
| 不删稀疏词（min_df = 1） | 4737 | 0.894 | 0.911 |
| **采用的设置（min_df = 5）** | **1351** | **0.901** | **0.907** |
| min_df = 10 | 845 | 0.895 | 0.907 |
| 去停用词 | 1138 | 0.897 | 0.895 |
| 词干提取 | 1186 | 0.895 | 0.905 |
| 去停用词 + 词干 | 1003 | 0.885 | 0.902 |
| 单词 + 二元词组 | 3665 | 0.908 | 0.909 |

各设置之间的差别都在 1 到 2 个百分点以内，和交叉验证本身的波动差不多。选择理由：

- **保留停用词**：去掉停用词后 Lasso 的准确率下降约 1 个百分点。I、we、the 这类虚词反映写作风格，而人写和 AI 生成的评论恰好在风格上不同。
- **不做词干提取**：准确率没有提高，而且词干（如 servic、lobbi）会让重要特征更难解释。
- **min_df = 5**：特征数从 4737 降到 1351，准确率基本不变。
- **只用单词**：加入二元词组后特征数接近原来的 3 倍，提升也不明显。

## 数据列说明（processed 文件）

| 列名 | 说明 |
|------|------|
| `id` | 评论在英文子集中的行号 |
| `text` | upside + downside 拼接后的文本，是模型唯一的输入 |
| `label` | 1 表示 AI 生成（假评论），0 表示人写（真评论） |
| `sentiment` | `POS` / `NEG`，用于按正负面分组评估测试集 |
| `upside`、`downside` | 原始的两段评论 |
| `score`、`city`、`hotel` | 元数据，不作为特征使用 |

## 设计说明

- **测试集只用一次**：`ReviewData.load("test")` 只在最终评估 5 个选定模型时调用，调参和模型选择都只用训练集。
- **按标签 × 正负面分层**：保证训练集和测试集的类别分布一致，同时让测试集中正面、负面评论数量平衡，方便比较模型在两类评论上的准确率。
- **不使用评分等元数据**：AI 评论的评分分布与真实评论明显不同（负面评论平均 3.0 对 3.7，正面评论平均 8.4 对 9.4）。使用评分会让模型依赖元数据而不是文本，所以只用评论文本作为输入。

## 后续评估

5 个模型都选定之后，`evaluate.py` 还要完成：

1. 在同一批 500 条测试评论上计算 accuracy、precision、recall、F1，以 AI 生成（`label = 1`）为正类。
2. 朴素贝叶斯和逻辑回归比较；随机森林、梯度提升分别和单棵树比较。用 McNemar 检验，显著性水平 0.01，并报告每次检验的 2×2 表。
3. 用同一个最终随机森林，分别计算它在正面（POS）和负面（NEG）测试评论上的准确率。两组是不同的评论，要用适合独立样本的检验。
4. 保存每条测试评论的 ID、真实标签、正负面和各模型的预测，保证不同模型的预测能逐条对齐。
