from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
from typing_extensions import override

from mlfeel.tokenizer import BaseTokenizer
from mlfeel.vocabulary import VocabIndexer


class Vectorizer(ABC):
    dim: int
    indexer: VocabIndexer

    def __init__(self):
        self.dim = 0
        self.indexer = VocabIndexer(BaseTokenizer())

    @abstractmethod
    def transform(self, corpus: list[str]) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self, corpus: list[str]) -> None:
        pass


class TF_IDF_Vectorizer(Vectorizer):
    @override
    def fit(self, corpus: list[str]) -> None:
        self.indexer.indexing(corpus)
        self.word_to_index: dict[str, int] = self.indexer.word_to_index
        self.dim: int = len(self.word_to_index)
        N = len(corpus)
        if N == 0 or self.dim == 0:
            self.idf: np.ndarray = np.array([], dtype=np.float32)
            return
        corpus_sets = [set(self.indexer.tokenizer.tokenize_text(doc)) for doc in corpus]
        df_counts = np.zeros(self.dim, dtype=np.float32)
        for word, index in self.word_to_index.items():
            # On compte combien de documents contiennent le mot
            doc_count = sum(1 for doc_set in corpus_sets if word in doc_set)
            df_counts[index] = float(doc_count)
        self.idf = np.log((1.0 + N) / (1.0 + df_counts)) + 1.0

    @override
    def transform(self, corpus: list[str]):
        result: np.ndarray = np.zeros((len(corpus), self.dim), dtype=np.float32)
        for doc_idx, doc in enumerate(corpus):
            tokens_list = self.indexer.tokenizer.tokenize_text(doc)
            if not tokens_list:
                continue
            counts = Counter(tokens_list)
            total_words = len(tokens_list)
            for word, count in counts.items():
                if word in self.word_to_index:
                    word_idx = self.word_to_index[word]
                    result[doc_idx, word_idx] = count / total_words
        return result * self.idf
