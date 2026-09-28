# 文本预处理 + 词袋
# 1. 小写，弯引号换成 '，去掉重音 (türk -> turk)
# 2. 只留字母组成的词，保留 wasn't 这种缩写，数字和标点丢掉
# 3. 可选: 去停用词 / 词干 / 加 bigram (默认都不用)
# 4. CountVectorizer 数词频，min_df=5 去掉稀疏词
#
# 向量化器要放进 Pipeline 里，这样交叉验证时每一折只用自己的训练部分建词表
import re
import unicodedata

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer


class TextPreprocessor:
    def __init__(self, stop=False, stem=False, ngram=(1, 1), min_df=5):
        self.stop = stop
        self.stem = stem
        self.ngram = ngram
        self.min_df = min_df
        self.stemmer = SnowballStemmer("english")

    def tokenize(self, s):
        s = s.lower()
        for q in ["’", "‘", "´", "`"]:
            s = s.replace(q, "'")
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))

        toks = re.findall(r"[a-z]+(?:'[a-z]+)*", s)
        if self.stop:
            toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
        if self.stem:
            toks = [self.stemmer.stem(t) for t in toks]
        return toks

    def make_vec(self):
        return CountVectorizer(tokenizer=self.tokenize, lowercase=False, token_pattern=None,
                               ngram_range=self.ngram, min_df=self.min_df)
