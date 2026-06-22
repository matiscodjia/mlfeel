import pytest

from mlfeel.tokenizer import BaseTokenizer
from mlfeel.vocabulary import VocabIndexer


@pytest.fixture
def tokenizer() -> BaseTokenizer:
    return BaseTokenizer()


@pytest.fixture
def indexer(tokenizer: BaseTokenizer) -> VocabIndexer:
    return VocabIndexer(tokenizer)


def test_vocab_indexer_indexing(indexer: VocabIndexer):
    corpus = ["Hello world! Hello NLP."]
    indexer.indexing(corpus)

    assert len(indexer.index_to_word) == 3
    assert "hello" in indexer.index_to_word.values()
    assert "world" in indexer.index_to_word.values()
    assert "nlp" in indexer.index_to_word.values()


def test_vocab_indexer_id_mapping(indexer: VocabIndexer):
    corpus = ["apple banana apple"]
    indexer.indexing(corpus)

    words = list(indexer.index_to_word.values())
    assert len(words) == 2
    assert "apple" in words
    assert "banana" in words


def test_vocab_indexer_empty_corpus(indexer: VocabIndexer):
    indexer.indexing([""])
    assert indexer.index_to_word == {}


def test_vocab_indexer_persistence(indexer: VocabIndexer):
    indexer.indexing(["cat dog"])
    assert len(indexer.index_to_word) == 2

    indexer.indexing(["bird"])
    assert len(indexer.index_to_word) == 1
    assert 0 in indexer.index_to_word
    assert indexer.index_to_word[0] == "bird"


def test_vocab_indexer_bidirectional_mapping(indexer: VocabIndexer):
    corpus = ["apple banana"]
    indexer.indexing(corpus)

    assert len(indexer.word_to_index) == 2
    assert indexer.word_to_index["apple"] == 0
    assert indexer.word_to_index["banana"] == 1

    for word, idx in indexer.word_to_index.items():
        assert indexer.index_to_word[idx] == word


def test_vocab_indexer_word_to_index_persistence(indexer: VocabIndexer):
    indexer.indexing(["cat dog"])
    indexer.indexing(["bird"])

    assert "bird" in indexer.word_to_index
    assert indexer.word_to_index["bird"] == 0
    assert "cat" not in indexer.word_to_index
