from course_generator.flashcards import flashcards_to_anki_tsv
from course_generator.html_export import html_section_to_plain


def test_html_section_to_plain_strips_tags():
    html = "<section><h2>Title</h2><p>Body text</p><ul><li>One</li></ul></section>"
    plain = html_section_to_plain(html)
    assert "Title" in plain
    assert "Body text" in plain
    assert "<" not in plain


def test_flashcards_to_anki_tsv():
    tsv = flashcards_to_anki_tsv([{"front": "Q", "back": "A", "source": "glossary"}])
    assert "Q\tA\tglossary" in tsv
