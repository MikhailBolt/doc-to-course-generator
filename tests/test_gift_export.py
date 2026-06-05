from course_generator.gift_export import combined_gift_export, questions_to_gift


def test_questions_to_gift_format():
    gift = questions_to_gift(
        [
            {
                "question": "What is RAG?",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
            }
        ],
        category="Test",
    )
    assert "$CATEGORY: Test" in gift
    assert "=A" in gift
    assert "~B" in gift


def test_combined_gift_export():
    text = combined_gift_export(
        [{"question": "P?", "options": ["True", "False"], "correct_answer": "True", "type": "true_false"}],
        [],
    )
    assert "Pre-test" in text
