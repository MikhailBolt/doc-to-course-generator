# Changelog

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
