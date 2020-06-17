import ganji.freetype_loader as ft


def test_find_codepoints():
    assert ord("あ") in ft.find_codepoints("hiragana")
