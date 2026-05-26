import json
from typing import Any, Dict

from langchain_ollama import OllamaLLM

from course_generator.generation import get_language_instruction
from course_generator.utils import call_llm


def llm_quality_review(
    llm: OllamaLLM,
    outline: Dict[str, Any],
    quality: Dict[str, Any],
    language: str,
) -> str:
    """Short narrative review of course quality (not a second scoring pass)."""
    failed = [c for c in quality.get("checks", []) if not c.get("passed")]
    prompt = f"""
You are reviewing an AI-generated training course for instructional quality.

{get_language_instruction(language)}

Heuristic score: {quality.get("overall_score", 0)}/100 (grade {quality.get("grade", "?")})
Failed checks: {json.dumps(failed, ensure_ascii=False)}

Course outline (JSON):
{json.dumps(outline, ensure_ascii=False, indent=2)}

Write 4–6 concise bullet points:
- what is strong
- what is weak or risky
- 2–3 concrete improvements for the next generation run

Use plain text bullets starting with "- ". No markdown headings. No JSON.
"""
    text = call_llm(llm, prompt).strip()
    return text[:4000]
