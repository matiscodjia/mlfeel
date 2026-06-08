from mlfeel.tokenizer import BaseTokenizer


class VocabIndexer:
    tokenizer: BaseTokenizer

    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer
        self.index: dict[int, str] = {}

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
        unique_words = set(tokens)
        print(unique_words)
        if not unique_words:
            self.index = {}
            return
        self.index = {i: word for i, word in enumerate(unique_words)}
