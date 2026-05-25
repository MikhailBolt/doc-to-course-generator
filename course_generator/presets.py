"""Named generation presets for CLI flags and Streamlit UI."""

from typing import Any, Dict

PRESETS: Dict[str, Dict[str, Any]] = {
    "Quick draft": {
        "min_lessons": 3,
        "max_lessons": 5,
        "quiz_questions": 5,
        "pretest_questions": 3,
        "skip_pretest": True,
        "skip_final_quiz": False,
        "disable_review_pass": True,
        "skip_outline_rag": False,
        "top_k": 4,
    },
    "Full course": {
        "min_lessons": 5,
        "max_lessons": 8,
        "quiz_questions": 12,
        "pretest_questions": 6,
        "skip_pretest": False,
        "skip_final_quiz": False,
        "disable_review_pass": False,
        "skip_outline_rag": False,
        "include_source_excerpts": True,
        "top_k": 6,
    },
    "Outline only (fast)": {
        "min_lessons": 4,
        "max_lessons": 6,
        "quiz_questions": 0,
        "pretest_questions": 0,
        "skip_pretest": True,
        "skip_final_quiz": True,
        "disable_review_pass": True,
        "skip_outline_rag": False,
        "top_k": 4,
    },
}

PRESET_NAMES = ["Custom"] + list(PRESETS.keys())
