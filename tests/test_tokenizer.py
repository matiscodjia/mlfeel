from mlfeel.tokenizer import BaseTokenizer

my_tokenizer = BaseTokenizer()


def test_html_punctuation():
    assert my_tokenizer.preprocess_text("Hello,   World! <br/>") == "hello world"


def test_spaces_punctuation():
    assert my_tokenizer.preprocess_text("  NLP is FUN!!!  ") == "nlp is fun"


def test_html_mixed_punctuation():
    assert (
        my_tokenizer.preprocess_text("<p>U.S. stocks rose 2.3%</p>")
        == "u s stocks rose 2 3"
    )


def test_html_punctuation_to_list():
    assert my_tokenizer.tokenize_text("Hello,   World! <br/>") == ["hello", "world"]


def test_spaces_punctuation_to_list():
    assert my_tokenizer.tokenize_text("  NLP is FUN!!!  ") == ["nlp", "is", "fun"]


def test_html_mixed_punctuation_to_list():
    assert my_tokenizer.tokenize_text("<p>U.S. stocks rose 2.3%</p>") == [
        "u",
        "s",
        "stocks",
        "rose",
        "2",
        "3",
    ]
