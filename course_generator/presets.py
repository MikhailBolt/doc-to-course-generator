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

# CLI slugs: python main.py --preset quick
PRESET_CLI_SLUGS = {
    "quick": "Quick draft",
    "full": "Full course",
    "outline": "Outline only (fast)",
}


def apply_cli_preset(args) -> None:
    """Override argparse Namespace fields from a CLI preset slug."""
    slug = getattr(args, "preset", None)
    if not slug:
        return
    preset_name = PRESET_CLI_SLUGS.get(slug)
    if not preset_name:
        raise ValueError(f"Unknown preset '{slug}'. Choose from: {', '.join(PRESET_CLI_SLUGS)}")
    for key, value in PRESETS[preset_name].items():
        setattr(args, key, value)
