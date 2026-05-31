# Changelog

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
