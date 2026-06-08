from mlfeel.tokenizer import BaseTokenizer


class VocabIndexer:
    tokenizer: BaseTokenizer

    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer
        self.index_to_word: dict[int, str] = {}
        self.word_to_index: dict[str, int] = {}

    def indexing(self, corpus: str):
        """
        Does an indexing of all the words tokinzed by the BaseTokenizer.

         Steps (apply in this exact order):
           1. Tokenize the corpus
           2. Retain only unique words
           3. Index them in a dictionnary
                  Parameters
         ----------
         text : str
             Raw input string.

         Returns
         -------
         str
           VocabIndexer
        """
        tokens = self.tokenizer.tokenize_text(corpus)
        print("tokens: ", tokens)
        unique_words = {t for t in tokens if t != ""}
        unique_words = sorted(set(tokens))
        print(unique_words)
        if not unique_words:
            self.index_to_word = {}
            self.word_to_index
            return
        self.index_to_word = {i: word for i, word in enumerate(unique_words)}
        self.word_to_index = {word: i for i, word in enumerate(unique_words)}
