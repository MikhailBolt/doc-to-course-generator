from argparse import Namespace
from typing import Any, Dict, List

from course_generator.utils import ensure_minimum_quiz_coverage


def _grade_from_score(score: int) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    return "D"


def compute_quality_score(
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    pretest_data: List[Dict[str, Any]],
    quiz_data: List[Dict[str, Any]],
    args: Namespace,
    *,
    outline_rag_used: bool,
) -> Dict[str, Any]:
    """Heuristic quality score (0–100) for generated course artifacts."""
    checks: List[Dict[str, Any]] = []
    total = 0
    max_total = 100

    def add(check_id: str, label: str, earned: int, maximum: int, passed: bool, detail: str = "") -> None:
        nonlocal total
        earned = max(0, min(earned, maximum))
        total += earned
        checks.append({
            "id": check_id,
            "label": label,
            "passed": passed,
            "score": earned,
            "max": maximum,
            "detail": detail,
        })

    title = str(outline.get("course_title", "")).strip()
    description = str(outline.get("course_description", "")).strip()
    audience = str(outline.get("target_audience", "")).strip()
    outcomes = outline.get("learning_outcomes", []) or []
    glossary = outline.get("glossary", []) or []
    lessons = outline.get("lessons", []) or []

    outline_core_ok = bool(title and description and audience)
    add(
        "outline_core",
        "Outline core fields",
        10 if outline_core_ok else 0,
        10,
        outline_core_ok,
        "title, description, target_audience",
    )

    outcomes_ok = len([x for x in outcomes if str(x).strip()]) >= 3
    add(
        "learning_outcomes",
        "Learning outcomes (≥3)",
        5 if outcomes_ok else max(0, len(outcomes)),
        5,
        outcomes_ok,
        f"count={len(outcomes)}",
    )

    glossary_ok = len(glossary) >= 3
    add(
        "glossary",
        "Glossary items (≥3)",
        5 if glossary_ok else min(5, len(glossary)),
        5,
        glossary_ok,
        f"count={len(glossary)}",
    )

    lesson_count = len(lessons)
    min_lessons = int(getattr(args, "min_lessons", 1))
    max_lessons = int(getattr(args, "max_lessons", 99))
    lessons_range_ok = min_lessons <= lesson_count <= max_lessons
    lessons_min_ok = lesson_count >= min_lessons
    lesson_pts = 10 if lessons_range_ok else (7 if lessons_min_ok else min(7, lesson_count * 2))
    add(
        "lesson_count",
        "Lesson count in range",
        lesson_pts,
        10,
        lessons_range_ok,
        f"lessons={lesson_count}, expected {min_lessons}–{max_lessons}",
    )

    if lessons:
        complete_lessons = 0
        for lesson, payload in zip(lessons, lesson_payloads):
            pts_ok = isinstance(lesson.get("key_points"), list) and len(lesson.get("key_points", [])) >= 3
            payload_ok = bool(str(payload.get("summary", "")).strip()) and len(payload.get("key_takeaways", []) or []) >= 2
            if str(lesson.get("title", "")).strip() and str(lesson.get("goal", "")).strip() and pts_ok and payload_ok:
                complete_lessons += 1
        ratio = complete_lessons / len(lessons)
        add(
            "lesson_completeness",
            "Lesson completeness",
            round(25 * ratio),
            25,
            ratio >= 0.8,
            f"{complete_lessons}/{len(lessons)} lessons complete",
        )
    else:
        add("lesson_completeness", "Lesson completeness", 0, 25, False, "no lessons")

    if not getattr(args, "skip_pretest", False):
        expected = int(getattr(args, "pretest_questions", 0))
        got = len(pretest_data)
        pretest_ok = got >= max(1, expected) if expected else got > 0
        add(
            "pretest",
            "Pre-test questions",
            8 if pretest_ok else min(8, got * 2),
            8,
            pretest_ok,
            f"got={got}, expected≈{expected}",
        )
    else:
        add("pretest", "Pre-test questions", 8, 8, True, "skipped by config")

    if not getattr(args, "skip_final_quiz", False):
        lesson_titles = [str(l.get("title", "")).strip() for l in lessons]
        coverage = ensure_minimum_quiz_coverage(quiz_data, lesson_titles)
        ratio = float(coverage.get("coverage_ratio", 0))
        expected_q = int(getattr(args, "quiz_questions", 0))
        got_q = len(quiz_data)
        quiz_count_ok = got_q >= max(1, expected_q) if expected_q else got_q > 0
        coverage_pts = round(12 * ratio)
        count_pts = 5 if quiz_count_ok else min(5, got_q)
        add(
            "quiz_coverage",
            "Quiz lesson coverage",
            coverage_pts,
            12,
            ratio >= 0.7,
            f"coverage={ratio:.0%}, missing={coverage.get('missing_lessons', [])}",
        )
        add(
            "quiz_count",
            "Final quiz question count",
            count_pts,
            5,
            quiz_count_ok,
            f"got={got_q}, expected≈{expected_q}",
        )
    else:
        add("quiz_coverage", "Quiz lesson coverage", 12, 12, True, "skipped by config")
        add("quiz_count", "Final quiz question count", 5, 5, True, "skipped by config")

    rag_pts = 5 if outline_rag_used else 0
    add(
        "outline_rag",
        "Outline grounded with RAG",
        rag_pts,
        5,
        outline_rag_used,
        "enabled" if outline_rag_used else "disabled or empty context",
    )

    if lesson_payloads:
        with_sources = sum(1 for p in lesson_payloads if p.get("sources"))
        src_ratio = with_sources / len(lesson_payloads)
        llm_count = sum(
            1 for p in lesson_payloads if p.get("generation_mode") in ("llm", "llm_retry")
        )
        llm_ratio = llm_count / len(lesson_payloads)
        add(
            "lesson_sources",
            "Lessons with retrieved sources",
            round(3 * src_ratio),
            3,
            src_ratio >= 0.8,
            f"{with_sources}/{len(lesson_payloads)} lessons",
        )
        add(
            "lesson_llm",
            "Lessons from LLM (not fallback)",
            round(2 * llm_ratio),
            2,
            llm_ratio >= 0.9,
            f"{llm_count}/{len(lesson_payloads)} llm-generated",
        )
    else:
        add("lesson_sources", "Lessons with retrieved sources", 0, 3, False, "no lesson payloads")
        add("lesson_llm", "Lessons from LLM (not fallback)", 0, 2, False, "no lesson payloads")

    overall = round(100 * total / max_total) if max_total else 0
    overall = max(0, min(100, overall))

    recommendations = _build_recommendations(checks, overall, args, outline_rag_used, lesson_payloads)

    return {
        "overall_score": overall,
        "grade": _grade_from_score(overall),
        "checks": checks,
        "summary": f"Score {overall}/100 (grade {_grade_from_score(overall)})",
        "recommendations": recommendations,
    }


def _build_recommendations(
    checks: List[Dict[str, Any]],
    overall: int,
    args: Namespace,
    outline_rag_used: bool,
    lesson_payloads: List[Dict[str, Any]],
) -> List[str]:
    tips: List[str] = []
    failed = {c["id"]: c for c in checks if not c.get("passed")}

    if "outline_rag" in failed and getattr(args, "skip_outline_rag", False):
        tips.append("Enable outline RAG (uncheck skip / remove --skip-outline-rag) for better topic grounding.")
    elif not outline_rag_used and "outline_rag" in failed:
        tips.append("Outline RAG returned empty context — try --rebuild or add more source text.")

    if "lesson_count" in failed:
        tips.append("Adjust min/max lessons or add richer source documents so the model can plan more lessons.")

    if "lesson_completeness" in failed:
        tips.append("Lessons look thin — increase --top-k or enable review pass for fuller lesson sections.")

    if lesson_payloads:
        fallback_n = sum(1 for p in lesson_payloads if p.get("generation_mode") == "fallback")
        if fallback_n:
            tips.append(
                f"{fallback_n} lesson(s) used fallback content (LLM JSON failed) — try a stronger model or --ollama-timeout."
            )

    if "quiz_coverage" in failed:
        tips.append("Final quiz misses lessons — raise --quiz-questions or rerun with review pass enabled.")

    if "pretest" in failed and not getattr(args, "skip_pretest", False):
        tips.append("Pre-test is short — increase --pretest-questions or simplify the outline.")

    if "glossary" in failed:
        tips.append("Glossary is small — ensure source docs define terms; review pass may help.")

    if overall < 70 and getattr(args, "disable_review_pass", False):
        tips.append("Enable review pass for clearer outline and quiz wording.")

    if not tips and overall < 85:
        tips.append("Good baseline — try Full course preset or include_source_excerpts for richer output.")

    return tips[:6]
