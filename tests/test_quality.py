from argparse import Namespace

from course_generator.quality import compute_quality_score


def test_quality_score_basic_outline():
    outline = {
        "course_title": "Test Course",
        "course_description": "A short test course.",
        "target_audience": "Developers",
        "prerequisites": ["Basic Python"],
        "learning_outcomes": ["Understand RAG", "Use FAISS", "Run Ollama"],
        "glossary": [
            {"term": "RAG", "definition": "Retrieval augmented generation"},
            {"term": "FAISS", "definition": "Vector index"},
            {"term": "Chunk", "definition": "Text segment"},
        ],
        "lessons": [
            {
                "title": "Lesson 1",
                "goal": "Intro",
                "key_points": ["a", "b", "c"],
            },
            {
                "title": "Lesson 2",
                "goal": "Practice",
                "key_points": ["d", "e", "f"],
            },
        ],
    }
    lesson_payloads = [
        {"summary": "s1", "key_takeaways": ["a", "b"], "sources": [{"document_name": "d.pdf"}]},
        {"summary": "s2", "key_takeaways": ["c", "d"], "sources": [{"document_name": "d.pdf"}]},
    ]
    args = Namespace(
        min_lessons=2,
        max_lessons=5,
        skip_pretest=True,
        skip_final_quiz=True,
        skip_outline_rag=False,
        disable_review_pass=False,
        quiz_questions=0,
        pretest_questions=0,
    )
    result = compute_quality_score(outline, lesson_payloads, [], [], args, outline_rag_used=True)
    assert result["overall_score"] >= 50
    assert result["grade"] in {"A", "B", "C", "D"}
    assert isinstance(result.get("recommendations"), list)
