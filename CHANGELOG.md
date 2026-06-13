# Changelog

## [1.10.0] — 2026-05-29

### Added
- **`--list-bundles`** — list recent `course_bundle.json` files in output dir
- Streamlit **From bundle** and **Rebuild exports (no LLM)** workflows
- HTML course: **Print / PDF** sidebar button
- Improved `@media print` styles (page breaks, quiz explanations visible)

### Changed
- Removed dead imports from `pipeline.py` after export refactor
- Bundle-related keys supported in `--config` JSON

## [1.9.0] — 2026-05-29

### Added
- **`--from-bundle`** — continue or rebuild from `course_bundle.json`
- **`--artifacts-only`** — re-export HTML/ZIP from bundle (no LLM)
- **`--regenerate-lessons`** — regenerate selected lessons by number
- **`--validate-bundle`** — validate bundle JSON
- **`pyproject.toml`** — `pip install -e .` and `doc-to-course` entry point
- HTML: **#** copy-link buttons on lesson headings

### Changed
- Export/save logic consolidated in `export_phase.py`

## [1.8.0] — 2026-05-29

### Added
- `--inspect-docs` document inventory (JSON)
- `--regenerate-fallback` to retry lessons that used fallback HTML
- `--min-quality-score` quality gate for CI (exit code 2)
- `run_manifest.json` compact run summary
- HTML course: lesson checkboxes, progress bar, in-page search (localStorage)

## [1.7.0] — 2026-05-29

### Added
- **Batch mode** (`--batch-dir`) for sequential multi-config runs
- `--check-embeddings` to verify HuggingFace embedding model
- `--diff-reports` to compare two `generation_report.json` files
- `--json-result` machine-readable success payload for scripts/CI
- Post-run **output audit** warnings for missing artifacts
- Streamlit: table of **saved checkpoints**
- Example batch config: `batch_configs/example-quick.json`

## [1.6.0] — 2026-05-29

### Added
- Moodle **GIFT** export (`quizzes.gift`, `--export-gift`)
- `OUTPUT_INDEX.md` describing all artifacts (included in delivery ZIP)
- `--init-dirs` to scaffold project folders and sample doc
- `--print-config` to dump effective settings as JSON
- `python -m course_generator` entry point
- Dry-run **estimated runtime** (minutes)
- Streamlit: explore saved FAISS index without running full pipeline

### Changed
- Lessons: second LLM attempt (`llm_retry`) before fallback HTML

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
