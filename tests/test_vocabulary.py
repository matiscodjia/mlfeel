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
    corpus = "Hello world! Hello NLP."
    indexer.indexing(corpus)

    assert len(indexer.index) == 3
    assert "hello" in indexer.index.values()
    assert "world" in indexer.index.values()
    assert "nlp" in indexer.index.values()


def test_vocab_indexer_id_mapping(indexer: VocabIndexer):
    corpus = "apple banana apple"
    indexer.indexing(corpus)

    words = list(indexer.index.values())
    assert len(words) == 2
    assert "apple" in words
    assert "banana" in words


def test_vocab_indexer_empty_corpus(indexer: VocabIndexer):
    indexer.indexing("")
    assert indexer.index == {}


def test_vocab_indexer_persistence(indexer: VocabIndexer):
    indexer.indexing("cat dog")
    assert len(indexer.index) == 2

    indexer.indexing("bird")
    assert len(indexer.index) == 1
    assert 0 in indexer.index
    assert indexer.index[0] == "bird"
