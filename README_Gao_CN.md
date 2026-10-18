# AI 生成酒店评论检测：随机森林与梯度提升

说明：随机森林（Random Forest）和梯度提升（Gradient Boosting）。
两个模型复用项目已有的数据划分、文本预处理和训练入口，判断英文酒店评论是人写的还是 AI 生成的。

## 目录结构

以下列出本模块使用的文件及输出位置。

| 路径 | 说明                          |
|------|-----------------------------|
| `src/rf.py` | 随机森林：调参、重要特征提取、绘图           |
| `src/gb.py` | 梯度提升：调参、重要特征提取、绘图           |
| `src/train.py` | 统一训练入口，保存模型和结果，已增加rf和gb模型训练 | 

测试集指标与统计检验由后续统一评估流程负责。

## 环境配置

使用项目统一的 Python 环境和 `requirements.txt`。在运行代码的同一个解释器中安装依赖：

```bash
python -m pip install -r requirements.txt
```

本模块及公共代码使用 scikit-learn、pandas、NumPy、Matplotlib、NLTK 和 joblib。


## 在控制台运行

```bash
python src/train.py rf gb    # 调参并训练随机森林、梯度提升
python src/train.py rf       # 只训练随机森林
python src/train.py gb       # 只训练梯度提升
```


### 1. 读取数据

复用现有的 `train.csv` 和 `test.csv`，不为这两个模型重新随机划分数据。

| 数据集 | 人写 NEG | 人写 POS | AI NEG | AI POS | 合计 |
|--------|----------|----------|--------|--------|------|
| train | 375 | 375 | 375 | 375 | 1500 |
| test | 125 | 125 | 125 | 125 | 500 |

模型只使用 `text` 列。`sentiment` 用于分层和后续分组评估，评分、酒店、城市等字段不作为模型特征。

### 2. 训练模型（`train.py`）

两个模型均使用 `GridSearchCV`，在训练集上进行 5 折分层交叉验证，所有模型使用 `d.folds()` 提供的同一组折。根据平均验证准确率选择最优参数，再在全部 1500 条训练评论上重新训练。

| 模型 | Pipeline | 参数网格 | 交叉验证训练次数 |
|------|----------|----------|------------------|
| `rf` | 词袋 → RandomForestClassifier | n_estimators ∈ {100, 300, 500}；max_features ∈ {sqrt, 0.1, 0.3} | 9 组 × 5 折 = 45 次 |
| `gb` | 词袋 → GradientBoostingClassifier | n_estimators ∈ {100, 200}；learning_rate ∈ {0.03, 0.1}；max_depth ∈ {1, 2, 3} | 12 组 × 5 折 = 60 次 |

以上次数不包含最后在完整训练集上的一次重新训练。参数网格是初始搜索范围，不代表已经覆盖所有可能的最优设置；如需调整，只根据训练集交叉验证结果决定。

参数含义：

- **n_estimators**：随机森林中的树数量；梯度提升中的迭代阶段数。
- **max_features**：随机森林每次分裂考虑的特征数量。`sqrt` 表示特征总数的平方根，0.1 和 0.3 表示约 10% 和 30%。
- **learning_rate**：梯度提升每次加入新树时的贡献权重。
- **max_depth**：梯度提升中每棵树允许的最大深度。

两个模型固定 `random_state=SEED`，公共随机种子为 42；其他未指定参数沿用所安装库版本的默认值。

### 3. 保存结果

`train.py` 自动创建输出目录并保存以下文件，其中 `{name}` 为 `rf` 或 `gb`：

| 文件 | 内容 |
|------|------|
| `results/models/{name}.joblib` | 最终训练好的 Pipeline，包含向量化器和分类器 |
| `results/tables/cv_{name}.csv` | 每组参数的交叉验证准确率均值 `acc` 和标准差 `sd` |
| `results/figures/cv_{name}.png` | 调参曲线 |
| `results/tables/feat_{name}.csv` | 最终模型的前 5 个重要词语及分数 |

重新训练会覆盖同名输出文件。最佳参数也会打印在控制台中，可以结合 CV 表保存实验记录。

`cv` 是交叉验证结果，不能当作测试集准确率。`feat` 是特征重要性结果，列名为 `feature` 和 `score`。

调参图的绘制方式：

- **rf**：横轴为树数量，每条曲线固定一个 `max_features`。
- **gb**：每个子图固定一个树深度，每条曲线固定一个学习率，横轴为树数量。
- 图中的误差线表示五折结果的标准差，不是置信区间。

### 模块用法

两个模块保持与现有 `nb.py`、`lr.py` 和 `tree.py` 相同的接口：

| 函数 | 作用 | 返回值／输出 |
|------|------|-------------|
| `tune(d)` | 构建 Pipeline，交叉验证调参并重新训练 | 最终模型、最佳参数、调参结果表 |
| `top(m, n=5)` | 提取最重要的 n 个词语 | 包含 `feature` 和 `score` 的 DataFrame |
| `plot(res, path)` | 绘制调参结果 | 保存 PNG 图片 |

模块内部使用公共接口：

```python
from load_data import ReviewData
from preprocessing import TextPreprocessor

d = ReviewData.load("train")         # d.X 是文本，d.y 是真假标签
folds = d.folds()                    # 统一的 5 折分层索引
vec = TextPreprocessor().make_vec()  # 未拟合的向量化器，放进 Pipeline
```

## 文本预处理

沿用公共预处理的默认设置：

1. 转小写，统一撇号，去掉重音符号；
2. 提取字母组成的词，保留 wasn't 这类缩写；
3. 保留停用词，不做词干提取，只用 unigram；
4. 使用词频表示，`min_df=5` 删除出现在少于 5 条训练评论中的词。

向量化器位于 Pipeline 内部。交叉验证时，每一折只用自己的训练部分建立词表，再转换该折的验证部分，避免提前利用验证集或测试集信息。

## 重要特征

两个模型均使用分类器的 `feature_importances_`，按基于不纯度下降的重要性从高到低选出五个词。

重要性越高，表示该特征在模型的树分裂中贡献越大，但它没有正负方向，不能仅凭分数判断这个词偏向 AI 还是人写。相关词语可能分担重要性，这种指标也存在偏差，因此报告中需要结合词语含义和训练数据作谨慎解释。

## 后续评估

模型选择完成后，统一评估流程还需要完成：

1. 在相同的 500 条测试评论上计算 accuracy、precision、recall、F1，以 AI 生成（`label=1`）为正类。
2. 将随机森林、梯度提升分别与最终单棵树进行比较，使用 McNemar 检验，显著性水平为 0.01，并报告对应的 2×2 表。
3. 使用同一个最终随机森林，分别评估 POS 和 NEG 测试评论的真假识别准确率，并对两个不同样本组采用适当的统计检验。
4. 保存每条评论的 ID、真实标签、情感和预测结果，保证不同模型的预测可以逐条对齐。

