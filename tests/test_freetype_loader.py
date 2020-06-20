import ganji.datasets


def test_find_codepoints():
    assert ord("あ") in ganji.datasets.find_codepoints("hiragana")
