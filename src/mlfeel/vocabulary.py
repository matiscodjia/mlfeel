from mlfeel.tokenizer import BaseTokenizer


class VocabIndexer:
    tokenizer: BaseTokenizer

    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer
        self.index_to_word: dict[int, str] = {}
        self.word_to_index: dict[str, int] = {}

    def indexing(self, corpus: list[str]) -> None:
        """
        Parcourt une liste de documents, extrait l'ensemble des tokens uniques
        et construit la double table d'indexation.
        """
        all_tokens: list[str] = []

        # 1. On accumule les tokens de TOUS les documents
        for doc in corpus:
            all_tokens.extend(self.tokenizer.tokenize_text(doc))

        # 2. Extraction des termes uniques triés pour garantir le déterminisme
        unique_words = sorted(list(set(all_tokens)))

        if not unique_words:
            self.index_to_word = {}
            self.word_to_index = {}
            return

        # 3. Création des correspondances bijectives
        self.index_to_word = {i: word for i, word in enumerate(unique_words)}
        self.word_to_index = {word: i for i, word in enumerate(unique_words)}
