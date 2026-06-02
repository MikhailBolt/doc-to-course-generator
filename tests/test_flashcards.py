from course_generator.flashcards import build_flashcards


def test_build_flashcards_from_glossary_and_lessons():
    outline = {
        "glossary": [{"term": "RAG", "definition": "Retrieval augmented generation"}],
        "lessons": [
            {"title": "Intro", "goal": "Learn basics", "key_points": ["point one", "point two"]},
        ],
    }
    payloads = [{"summary": "Basics summary", "key_takeaways": ["remember X"]}]
    cards = build_flashcards(outline, payloads)
    fronts = [c["front"] for c in cards]
    assert any("RAG" in f for f in fronts)
    assert any("point one" in f for f in fronts)
    assert any("remember X" in c["back"] for c in cards)
