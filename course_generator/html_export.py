import html
import json
from datetime import datetime
from typing import Any, Dict, List

from course_generator.utils import estimate_duration_minutes


def build_question_card_html(question: Dict[str, Any], idx: int, prefix: str, language: str) -> str:
    qid = f"{prefix}-q-{idx}"
    question_text = html.escape(str(question.get("question", "")))
    explanation = html.escape(str(question.get("explanation", "")))
    options = question.get("options", [])
    correct_answer = str(question.get("correct_answer", ""))
    explanation_label = "Показать объяснение" if language == "ru" else "Show explanation"

    options_html = []
    for opt_idx, option in enumerate(options):
        option_escaped = html.escape(str(option))
        input_id = f"{qid}-opt-{opt_idx}"
        options_html.append(
            f"""
<label class="quiz-option" for="{input_id}">
  <input type="radio" name="{qid}" id="{input_id}" value="{option_escaped}" data-correct="{html.escape(correct_answer)}" />
  <span>{option_escaped}</span>
</label>
"""
        )

    lesson_badge = ""
    if question.get("lesson_title"):
        lesson_badge = f'<div class="question-lesson">{html.escape(str(question["lesson_title"]))}</div>'

    return f"""
<div class="quiz-card" data-question="{qid}">
  {lesson_badge}
  <div class="quiz-question">{idx}. {question_text}</div>
  <div class="quiz-options">
    {''.join(options_html)}
  </div>
  <details class="quiz-explanation">
    <summary>{explanation_label}</summary>
    <p>{explanation}</p>
  </details>
</div>
"""


def build_glossary_html(glossary: List[Dict[str, Any]]) -> str:
    if not glossary:
        return "<p class='muted'>No glossary items generated.</p>"

    items = []
    for entry in glossary:
        term = html.escape(str(entry.get("term", "")).strip())
        definition = html.escape(str(entry.get("definition", "")).strip())
        if not term or not definition:
            continue
        items.append(
            f"""
<div class="glossary-item">
  <h3>{term}</h3>
  <p>{definition}</p>
</div>
"""
        )

    return "\n".join(items) if items else "<p class='muted'>No glossary items generated.</p>"


def build_markdown_summary(outline: Dict[str, Any], docs_info: List[Dict[str, Any]], lesson_payloads: List[Dict[str, Any]], language: str) -> str:
    is_ru = language == "ru"
    title = outline.get("course_title", "Generated Course")
    description = outline.get("course_description", "")
    target = outline.get("target_audience", "")
    prerequisites = outline.get("prerequisites", [])
    learning_outcomes = outline.get("learning_outcomes", [])
    lessons = outline.get("lessons", [])

    lines = [f"# {title}", "", description, "", f"**{'Целевая аудитория' if is_ru else 'Target audience'}:** {target}", ""]

    if prerequisites:
        lines.append(f"## {'Требования' if is_ru else 'Prerequisites'}")
        lines.extend([f"- {item}" for item in prerequisites])
        lines.append("")

    if learning_outcomes:
        lines.append(f"## {'Результаты обучения' if is_ru else 'Learning outcomes'}")
        lines.extend([f"- {item}" for item in learning_outcomes])
        lines.append("")

    lines.append(f"## {'Документы' if is_ru else 'Documents'}")
    lines.extend([f"- {doc['name']} ({doc['pages']} sections, {doc.get('type', 'unknown')})" for doc in docs_info])
    lines.append("")

    lines.append(f"## {'Уроки' if is_ru else 'Lessons'}")
    for idx, lesson in enumerate(lessons, start=1):
        lines.append(f"### {idx}. {lesson.get('title', '')}")
        lines.append(lesson.get("goal", ""))
        key_points = lesson.get("key_points", [])
        if key_points:
            lines.extend([f"- {item}" for item in key_points])
        summary = lesson_payloads[idx - 1].get("summary", "") if idx - 1 < len(lesson_payloads) else ""
        if summary:
            lines.append("")
            lines.append(summary)
        lines.append("")

    return "\n".join(lines)


def build_course_html(
    outline: Dict[str, Any],
    lesson_payloads: List[Dict[str, Any]],
    docs_info: List[Dict[str, Any]],
    pretest_data: List[Dict[str, Any]],
    final_quiz_data: List[Dict[str, Any]],
    language: str,
    include_source_excerpts: bool,
) -> str:
    is_ru = language == "ru"

    labels = {
        "overview": "Обзор" if is_ru else "Overview",
        "prerequisites": "Требования" if is_ru else "Prerequisites",
        "pretest": "Входной тест" if is_ru else "Pre-test",
        "glossary": "Глоссарий" if is_ru else "Glossary",
        "final_quiz": "Итоговый тест" if is_ru else "Final quiz",
        "learning_outcomes": "Результаты обучения" if is_ru else "Learning outcomes",
        "documents_used": "Использованные документы" if is_ru else "Documents used",
        "target_audience": "Целевая аудитория" if is_ru else "Target audience",
        "estimated_duration": "Оценочная длительность" if is_ru else "Estimated duration",
        "lessons": "Уроки" if is_ru else "Lessons",
        "documents_count": "Документы" if is_ru else "Documents used",
        "final_questions": "Вопросы финального теста" if is_ru else "Final quiz questions",
        "pretest_intro": "Пройдите короткий диагностический тест перед изучением уроков." if is_ru else "Use this short diagnostic quiz before starting the lessons.",
        "final_intro": "Пройдите итоговый тест, чтобы проверить усвоение материала." if is_ru else "Complete the final quiz to check knowledge gained through the course.",
        "check_pretest": "Проверить входной тест" if is_ru else "Check pre-test",
        "check_final": "Проверить итоговый тест" if is_ru else "Check final quiz",
        "reset": "Сбросить" if is_ru else "Reset",
        "sources_used": "Использованные источники" if is_ru else "Sources used",
        "source_excerpts": "Фрагменты источников" if is_ru else "Source excerpts",
        "generated_on": "Сгенерировано" if is_ru else "Generated on",
        "minutes": "минут" if is_ru else "minutes",
        "ai_generated_course": "Курс, сгенерированный ИИ" if is_ru else "AI-generated course",
        "from_source_docs": "Курс создан на основе исходных документов" if is_ru else "AI-generated course from source documents",
        "score": "Результат" if is_ru else "Score",
    }

    course_title = html.escape(outline.get("course_title", "Generated Course"))
    course_description = html.escape(outline.get("course_description", ""))
    target_audience = html.escape(outline.get("target_audience", ""))
    learning_outcomes = outline.get("learning_outcomes", [])
    prerequisites = outline.get("prerequisites", [])
    glossary = outline.get("glossary", [])

    overview_items = "\n".join(f"<li>{html.escape(str(item))}</li>" for item in learning_outcomes) or "<li>No learning outcomes generated.</li>"
    prerequisites_items = "\n".join(f"<li>{html.escape(str(item))}</li>" for item in prerequisites) or "<li>No prerequisites specified.</li>"
    docs_items = "\n".join(
        f"<li><strong>{html.escape(doc['name'])}</strong> — {doc['pages']} sections — {html.escape(doc.get('type', 'unknown'))}</li>"
        for doc in docs_info
    ) or "<li>No documents listed.</li>"

    toc_items = [
        f'<li><a href="#overview">{labels["overview"]}</a></li>',
        f'<li><a href="#prerequisites">{labels["prerequisites"]}</a></li>',
    ]
    if pretest_data:
        toc_items.append(f'<li><a href="#pretest">{labels["pretest"]}</a></li>')
    for i, lesson in enumerate(outline.get("lessons", []), start=1):
        title = html.escape(lesson.get("title", f"Lesson {i}"))
        toc_items.append(f'<li><a href="#lesson-{i}">{title}</a></li>')
    toc_items.append(f'<li><a href="#glossary">{labels["glossary"]}</a></li>')
    if final_quiz_data:
        toc_items.append(f'<li><a href="#final-quiz">{labels["final_quiz"]}</a></li>')
    toc_html = "\n".join(toc_items)

    lesson_sections = []
    for i, payload in enumerate(lesson_payloads, start=1):
        raw_section = payload["lesson_html"]
        section_with_anchor = raw_section.replace('<section class="lesson-section">', f'<section class="lesson-section" id="lesson-{i}">', 1)

        sources = payload.get("sources", [])
        source_items = "\n".join(
            f"<li>{html.escape(str(src['document_name']))} — page {html.escape(str(src['page']))}, chunk {html.escape(str(src['chunk_id']))}</li>"
            for src in sources
        )
        if source_items:
            section_with_anchor += f"""
<div class="lesson-sources">
  <h3>{labels['sources_used']}</h3>
  <ul>{source_items}</ul>
</div>
"""

        excerpts = payload.get("source_excerpts", []) if include_source_excerpts else []
        if excerpts:
            excerpt_items = "".join(
                f"<div class=\"source-excerpt\"><strong>{html.escape(str(e['document_name']))}</strong> — page {html.escape(str(e['page']))}, chunk {html.escape(str(e['chunk_id']))}<p>{html.escape(str(e['excerpt']))}</p></div>"
                for e in excerpts
            )
            section_with_anchor += f"""
<div class="lesson-excerpts">
  <h3>{labels['source_excerpts']}</h3>
  {excerpt_items}
</div>
"""
        lesson_sections.append(section_with_anchor)

    lesson_sections_html = "\n\n".join(lesson_sections)
    pretest_html = "".join(build_question_card_html(q, idx + 1, "pretest", language) for idx, q in enumerate(pretest_data))
    final_quiz_html = "".join(build_question_card_html(q, idx + 1, "final", language) for idx, q in enumerate(final_quiz_data))
    glossary_html = build_glossary_html(glossary)

    lesson_count = len(outline.get("lessons", []))
    estimated_minutes = estimate_duration_minutes(outline)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pretest_section = f"""
      <section class="card" id="pretest">
        <h2>{labels['pretest']}</h2>
        <p class="muted">{labels['pretest_intro']}</p>
        <div class="quiz-actions">
          <button type="button" onclick="gradeQuiz('pretest')">{labels['check_pretest']}</button>
          <button type="button" class="secondary" onclick="resetQuiz('pretest')">{labels['reset']}</button>
        </div>
        <div class="progress-wrap"><div class="progress-bar" id="pretest-progress"></div></div>
        <div class="quiz-results" id="pretest-results"></div>
        {pretest_html}
      </section>
""" if pretest_data else ""

    final_quiz_section = f"""
      <section class="card" id="final-quiz">
        <h2>{labels['final_quiz']}</h2>
        <p class="muted">{labels['final_intro']}</p>
        <div class="quiz-actions">
          <button type="button" onclick="gradeQuiz('final')">{labels['check_final']}</button>
          <button type="button" class="secondary" onclick="resetQuiz('final')">{labels['reset']}</button>
        </div>
        <div class="progress-wrap"><div class="progress-bar" id="final-progress"></div></div>
        <div class="quiz-results" id="final-results"></div>
        {final_quiz_html}
      </section>
""" if final_quiz_data else ""

    return f"""<!DOCTYPE html>
<html lang="{language}">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{course_title}</title>
  <style>
    :root {{ --bg:#0f172a; --panel:#ffffff; --text:#1f2937; --muted:#64748b; --line:#e2e8f0; --primary:#2563eb; --primary-soft:#dbeafe; --shadow:0 10px 30px rgba(15,23,42,0.08); --radius:18px; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Arial, Helvetica, sans-serif; background:#f8fafc; color:var(--text); line-height:1.65; }}
    .layout {{ display:grid; grid-template-columns:280px 1fr; min-height:100vh; }}
    .sidebar {{ position:sticky; top:0; align-self:start; height:100vh; overflow:auto; background:var(--bg); color:#fff; padding:24px 18px; }}
    .sidebar h2 {{ margin:0 0 8px; font-size:1.25rem; }}
    .sidebar p {{ color:#cbd5e1; margin-top:0; font-size:0.95rem; }}
    .sidebar ol {{ padding-left:18px; margin:18px 0 0; }}
    .sidebar li {{ margin-bottom:10px; }}
    .sidebar a {{ color:#bfdbfe; text-decoration:none; }}
    .sidebar a:hover {{ text-decoration:underline; }}
    .content {{ padding:28px; max-width:1100px; width:100%; margin:0 auto; }}
    .card {{ background:var(--panel); border-radius:var(--radius); padding:24px; box-shadow:var(--shadow); margin-bottom:20px; }}
    .hero {{ padding:28px; }}
    .hero h1 {{ margin:8px 0 12px; font-size:2.2rem; line-height:1.2; color:#0f172a; }}
    .badge {{ display:inline-block; background:var(--primary-soft); color:#1d4ed8; padding:6px 10px; border-radius:999px; font-size:0.9rem; margin-right:8px; margin-bottom:8px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:14px; margin-top:16px; }}
    .stat {{ border:1px solid var(--line); border-radius:14px; padding:14px; background:#fff; }}
    .stat-label {{ color:var(--muted); font-size:0.9rem; }}
    .stat-value {{ margin-top:6px; font-size:1.2rem; font-weight:700; color:#0f172a; }}
    h2, h3 {{ color:#0f172a; }}
    ul, ol {{ padding-left:20px; }}
    .muted {{ color:var(--muted); }}
    .lesson-section {{ background:#fff; border-radius:var(--radius); padding:24px; box-shadow:var(--shadow); margin-bottom:12px; }}
    .lesson-sources, .lesson-excerpts {{ margin-top:18px; padding-top:16px; border-top:1px solid var(--line); }}
    .source-excerpt {{ border:1px solid var(--line); padding:12px; border-radius:12px; margin-bottom:10px; background:#f8fafc; }}
    .glossary-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:14px; }}
    .glossary-item {{ border:1px solid var(--line); border-radius:14px; padding:16px; background:#fff; }}
    .quiz-actions {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:18px; }}
    button {{ border:0; background:var(--primary); color:#fff; padding:10px 14px; border-radius:12px; cursor:pointer; font-size:0.95rem; }}
    button.secondary {{ background:#334155; }}
    .quiz-card {{ border:1px solid var(--line); border-radius:16px; padding:18px; margin-bottom:14px; background:#fff; }}
    .question-lesson {{ display:inline-block; margin-bottom:10px; font-size:0.8rem; background:#eef2ff; color:#4338ca; padding:4px 10px; border-radius:999px; }}
    .quiz-question {{ font-weight:700; margin-bottom:10px; }}
    .quiz-options {{ display:grid; gap:10px; }}
    .quiz-option {{ display:flex; align-items:center; gap:10px; border:1px solid var(--line); border-radius:12px; padding:10px 12px; cursor:pointer; transition:0.2s ease; }}
    .quiz-option:hover {{ background:#f8fafc; }}
    .quiz-option.correct {{ border-color:#86efac; background:#f0fdf4; }}
    .quiz-option.incorrect {{ border-color:#fca5a5; background:#fef2f2; }}
    .quiz-explanation {{ margin-top:12px; color:var(--muted); }}
    .quiz-results {{ margin-top:14px; padding:14px; border-radius:14px; background:#eff6ff; color:#1e3a8a; display:none; }}
    .progress-wrap {{ height:10px; border-radius:999px; background:#e2e8f0; overflow:hidden; margin-top:12px; }}
    .progress-bar {{ height:100%; width:0%; background:linear-gradient(90deg, #2563eb, #38bdf8); transition:width 0.3s ease; }}
    .footer {{ margin-top:28px; color:var(--muted); font-size:0.95rem; text-align:center; }}
    @media (max-width:980px) {{ .layout {{ grid-template-columns:1fr; }} .sidebar {{ position:relative; height:auto; }} .content {{ padding:18px; }} }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h2>{course_title}</h2>
      <p>{labels['from_source_docs']}</p>
      <ol>{toc_html}</ol>
    </aside>
    <main class="content">
      <section class="card hero" id="overview">
        <span class="badge">{labels['ai_generated_course']}</span>
        <span class="badge">HTML</span>
        <span class="badge">{lesson_count} {labels['lessons'].lower()}</span>
        <h1>{course_title}</h1>
        <p>{course_description}</p>
        <p><strong>{labels['target_audience']}:</strong> {target_audience}</p>
        <div class="grid">
          <div class="stat"><div class="stat-label">{labels['estimated_duration']}</div><div class="stat-value">{estimated_minutes} {labels['minutes']}</div></div>
          <div class="stat"><div class="stat-label">{labels['lessons']}</div><div class="stat-value">{lesson_count}</div></div>
          <div class="stat"><div class="stat-label">{labels['documents_count']}</div><div class="stat-value">{len(docs_info)}</div></div>
          <div class="stat"><div class="stat-label">{labels['final_questions']}</div><div class="stat-value">{len(final_quiz_data)}</div></div>
        </div>
      </section>
      <section class="card" id="prerequisites"><h2>{labels['prerequisites']}</h2><ul>{prerequisites_items}</ul></section>
      <section class="card"><h2>{labels['learning_outcomes']}</h2><ul>{overview_items}</ul></section>
      <section class="card"><h2>{labels['documents_used']}</h2><ul>{docs_items}</ul></section>
      {pretest_section}
      {lesson_sections_html}
      <section class="card" id="glossary"><h2>{labels['glossary']}</h2><div class="glossary-grid">{glossary_html}</div></section>
      {final_quiz_section}
      <div class="footer">{labels['generated_on']} {generated_at}</div>
    </main>
  </div>
  <script>
    const SCORE_LABEL = {json.dumps(labels['score'])};
    function getQuizCards(prefix) {{ return Array.from(document.querySelectorAll(`[data-question^="${{prefix}}-q-"]`)); }}
    function updateProgress(prefix) {{ const cards = getQuizCards(prefix); const answered = cards.filter(card => card.querySelector('input[type="radio"]:checked')).length; const total = cards.length || 1; const percent = Math.round((answered / total) * 100); const bar = document.getElementById(`${{prefix}}-progress`); if (bar) bar.style.width = `${{percent}}%`; }}
    function gradeQuiz(prefix) {{ const cards = getQuizCards(prefix); let correct = 0; cards.forEach(card => {{ const selected = card.querySelector('input[type="radio"]:checked'); const options = card.querySelectorAll('.quiz-option'); options.forEach(opt => opt.classList.remove('correct','incorrect')); if (!selected) return; const correctAnswer = selected.getAttribute('data-correct'); const optionLabels = card.querySelectorAll('.quiz-option'); optionLabels.forEach(label => {{ const input = label.querySelector('input'); if (!input) return; if (input.value === correctAnswer) label.classList.add('correct'); if (input.checked && input.value !== correctAnswer) label.classList.add('incorrect'); }}); if (selected.value === correctAnswer) correct += 1; }}); const results = document.getElementById(`${{prefix}}-results`); if (results) {{ const total = cards.length; const percent = total ? Math.round((correct / total) * 100) : 0; results.style.display = 'block'; results.innerHTML = `<strong>${{SCORE_LABEL}}:</strong> ${{correct}} / ${{total}} (${{percent}}%)`; }} updateProgress(prefix); }}
    function resetQuiz(prefix) {{ const cards = getQuizCards(prefix); cards.forEach(card => {{ card.querySelectorAll('input[type="radio"]').forEach(input => input.checked = false); card.querySelectorAll('.quiz-option').forEach(opt => opt.classList.remove('correct','incorrect')); }}); const results = document.getElementById(`${{prefix}}-results`); if (results) {{ results.style.display = 'none'; results.innerHTML = ''; }} updateProgress(prefix); }}
    document.addEventListener('change', (event) => {{ const target = event.target; if (target.matches('input[type="radio"]')) {{ const name = target.name || ''; if (name.startsWith('pretest-')) updateProgress('pretest'); if (name.startsWith('final-')) updateProgress('final'); }} }});
    updateProgress('pretest'); updateProgress('final');
  </script>
</body>
</html>
"""
