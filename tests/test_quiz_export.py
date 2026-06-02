from course_generator.quiz_export import quiz_rows_for_csv, combined_quiz_csv_text


def test_quiz_rows_for_csv_pads_options():
    rows = quiz_rows_for_csv(
        [{"question": "Q1?", "type": "single_choice", "options": ["A", "B"], "correct_answer": "A", "explanation": "e"}],
        quiz_name="final",
    )
    assert rows[0]["option_a"] == "A"
    assert rows[0]["option_c"] == ""
    assert rows[0]["quiz"] == "final"


def test_combined_quiz_csv_has_bom_header():
    text = combined_quiz_csv_text(
        [{"question": "P?", "options": ["True", "False"], "correct_answer": "True", "type": "true_false"}],
        [],
    )
    assert "quiz,number,question" in text
    assert "pretest" in text
