import pytest

from mlfeel.tokenizer import BaseTokenizer
from mlfeel.vectorizer import Vectorizer  # À adapter selon ton chemin d'import
from mlfeel.vocabulary import VocabIndexer


@pytest.fixture
def vectorizer() -> Vectorizer:
    return Vectorizer()


@pytest.fixture
def populated_indexer() -> VocabIndexer:
    tokenizer = BaseTokenizer()
    indexer = VocabIndexer(tokenizer)
    indexer.indexing("apple banana cherry")
    return indexer


def test_vectorizer_basic_mapping(
    vectorizer: Vectorizer, populated_indexer: VocabIndexer
):
    words = ["apple", "cherry"]
    vectors = vectorizer.vectorize(words, populated_indexer.word_to_index)

    assert vectors == [0, 2]
    assert vectorizer.dim == 3


def test_vectorizer_empty_list(vectorizer: Vectorizer, populated_indexer: VocabIndexer):
    vectors = vectorizer.vectorize([], populated_indexer.word_to_index)
    assert vectors == []
    assert vectorizer.dim == 3


def test_vectorizer_key_error_handling(
    vectorizer: Vectorizer, populated_indexer: VocabIndexer
):
    words = ["apple", "unknown"]
    with pytest.raises(KeyError):
        vectorizer.vectorize(words, populated_indexer.word_to_index)
