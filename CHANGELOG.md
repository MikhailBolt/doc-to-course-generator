# Changelog

## [1.5.0] — 2026-05-29

### Added
- `course_full.md` extended Markdown export with lesson bodies and quizzes
- `flashcards_anki.txt` for Anki import
- `--config` JSON configuration file (`config.example.json`)
- HTML course dark/light theme toggle (localStorage)
- Quality check for LLM vs fallback lesson generation
- Streamlit: flashcard preview, safer download buttons

## [1.4.0] — 2026-05-29

### Added
- `flashcards.json` study deck from glossary and lesson key points
- `quizzes.csv` export (UTF-8 BOM) for pre-test and final quiz
- `--validate-outline` CLI check
- `--max-files` limit for large document folders
- Lesson `generation_mode` tracking (`llm` vs `fallback`) in reports
- HTML: generator version in footer and back-to-top link

### Changed
- `DocCollection` reports truncation when `max_files` applies

## [1.3.0] — 2026-05-29

### Added
- **Checkpoints** (`--checkpoint`, `--resume-checkpoint`) to save/resume long lesson runs
- **`.docx` input** documents (python-docx)
- **`--list-runs`** — recent `generation_report.json` summary in CLI
- **`--ollama-timeout`** / `OLLAMA_TIMEOUT` for LLM requests
- Course HTML: print stylesheet and sidebar scroll-spy
- Streamlit: lesson progress bar, checkpoint options

### Changed
- `collect_source_files` raises `DocumentSourceError` instead of `sys.exit` (better for UI/tests)

## [1.2.0] — 2026-05-29

### Added
- PDF export (`--export-pdf`, Streamlit checkbox) via `course_summary.pdf`
- Ollama preflight before generation with clear setup hints
- `delivery_manifest.txt` inside `course_delivery.zip`
- CLI `--open` to open `course.html` in the browser after a successful run
- Dry-run: per-file sizes and total size

### Changed
- `--check-ollama` and Streamlit health check use the same friendly messages

## [1.1.0]

- Dry-run, outline-only, resume-from-outline, Streamlit saved settings

## [1.0.0]

- Initial versioned release: package layout, Streamlit UI, quality score, delivery ZIP
