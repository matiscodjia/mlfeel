import pytest

from mlfeel.vectorizer import TF_IDF_Vectorizer


@pytest.mark.model
def test_tfidf_vectorizer_pipeline() -> None:
    # 1. Initialisation des composants
    vectorizer = (
        TF_IDF_Vectorizer()
    )  # En supposant que vous passez l'indexer au constructeur

    # Corpus d'entraînement (N = 2 documents)
    train_corpus = ["Le film est excellent.", "Un navet complet, le pire film."]

    # 2. Phase d'apprentissage (Fit)
    vectorizer.fit(train_corpus)

    # Assertions sur le vocabulaire
    assert vectorizer.dim > 0
    assert "film" in vectorizer.word_to_index
    assert "excellent" in vectorizer.word_to_index

    # 3. Phase de transformation (Transform)
    train_matrix = vectorizer.transform(train_corpus)

    # Invariant géométrique : la forme doit être (N_documents, V_vocabulaire)
    assert train_matrix.shape == (2, vectorizer.dim)

    # 4. Test de robustesse face à l'inférence (Mots inconnus en production)
    # "incroyable" et "chef d'oeuvre" n'existent pas dans le train_corpus
    prod_corpus = ["Un film incroyable, un chef d oeuvre."]

    # Cette ligne ne doit pas lever de KeyError
    prod_matrix = vectorizer.transform(prod_corpus)

    # Invariant de production : la forme doit être (1, V_vocabulaire)
    assert prod_matrix.shape == (1, vectorizer.dim)

    # 5. Invariant mathématique : "le" est dans les deux documents, "excellent" dans un seul.
    # L'IDF de "excellent" doit donc être strictement supérieur à l'IDF de "le"
    idx_le = vectorizer.word_to_index["le"]
    idx_excellent = vectorizer.word_to_index["excellent"]

    assert (
        vectorizer.idf[idx_excellent] > vectorizer.idf[idx_le]
    ), "L'IDF n'écrase pas les mots fréquents."
