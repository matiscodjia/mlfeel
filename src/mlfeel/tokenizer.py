import re


class BaseTokenizer:

    def preprocess_text(self, text: str) -> str:
        """
        Apply a minimal, reproducible normalisation pipeline to a single string.

        Steps (apply in this exact order):
          1. Lowercase all characters
          2. Remove HTML tags (e.g. <br />, <p>, </div>)
          3. Replace any character that is not a letter, digit, or space
             with a single space
          4. Collapse multiple consecutive whitespace characters into one space
          5. Strip leading and trailing whitespace

        Parameters
        ----------
        text : str
            Raw input string.

        Returns
        -------
        str
            Normalised string.

        Examples
        --------
        >>> preprocess_text("Hello,   World! <br/>")
        'hello world'
        >>> preprocess_text("  NLP is FUN!!!  ")
        'nlp is fun'
        >>> preprocess_text("<p>U.S. stocks rose 2.3%</p>")
        'u s stocks rose 2 3'
        """
        cleaned = re.sub(r"<[^>]+>", "", string=text.lower())
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
        cleaned = " ".join(cleaned.split())
        return cleaned.strip()
