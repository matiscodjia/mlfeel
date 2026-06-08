class Vectorizer:
    dim: int

    def __init__(self):
        self.dim = 0

    def vectorize(self, words_list: list[str], word_to_index: dict[str, int]):
        self.dim = len(word_to_index)
        return [word_to_index[word] for word in words_list]
