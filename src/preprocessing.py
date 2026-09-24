"""文本预处理与词袋表示。

每条评论依次经过以下步骤：
1. 规范化：转小写，统一弯引号（’ ‘ ´ `）为直引号 '，去掉重音符号；
2. 分词：只保留由字母组成的词，允许带撇号的缩写（如 wasn't、hotel's）
   数字和标点全部丢弃；
3. 计数：CountVectorizer 统计词频，可选加入二元词组（bigram），
   并用 min_df 删掉出现文档数过少的稀疏词。

向量化器必须放进 sklearn 的 Pipeline 里，交叉验证时每一折只在自己的训练部分上
拟合词表，避免验证集信息泄漏。
"""
import re
import unicodedata

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer


class TextPreprocessor:
    """分词并构建词袋向量化器。

    默认设置：保留停用词、不做词干提取、只用单词、删掉出现在少于 5 条评论里的词。
    实例可以被 pickle，所以 GridSearchCV 并行时也能使用。
    """

    TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")
    QUOTES = str.maketrans({"’": "'", "‘": "'", "´": "'", "`": "'"})

    def __init__(self, stop_words=False, stem=False, ngram_range=(1, 1), min_df=5):
        self.stop_words = stop_words
        self.stem = stem
        self.ngram_range = ngram_range
        self.min_df = min_df
        self._stemmer = SnowballStemmer("english") if stem else None

    @classmethod
    def normalize(cls, text):
        text = text.lower().translate(cls.QUOTES)
        text = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in text if not unicodedata.combining(ch))

    def tokenize(self, text):
        tokens = self.TOKEN_RE.findall(self.normalize(text))
        # 停用词表里没有带撇号的形式，所以 wasn't、don't 这类缩写会保留下来
        if self.stop_words:
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        if self._stemmer is not None:
            tokens = [self._stemmer.stem(t) for t in tokens]
        return tokens

    def build_vectorizer(self):
        """返回一个未拟合的词频向量化器。

        min_df 为整数时表示最少出现的文档数，为小数时表示最少出现的文档比例。
        """
        return CountVectorizer(tokenizer=self.tokenize, lowercase=False,
                               token_pattern=None, ngram_range=self.ngram_range,
                               min_df=self.min_df)
