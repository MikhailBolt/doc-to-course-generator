# 📚 Doc-to-Course Generator

**Version 1.7.0**

Generate a structured **HTML training course** and **assessment quizzes** from documents using **local LLMs**, **RAG**, **FAISS**, and **Ollama**.

```bash
python main.py --version
```

---

## 🚀 Features

- Generate full **HTML course**
- Generate **pre-test** and **final quiz**
- Supports **PDF / TXT / MD / DOCX**
- Local **RAG pipeline**
- FAISS vector search
- Ollama (local LLM)
- Auto rebuild index
- Optional review pass
- Bundle export (`course_bundle.json`)
- **Streamlit Web UI** with live progress
- **Heuristic quality score** in `generation_report.json`
- **Ollama health check** (`--check-ollama` / UI button)
- **Delivery ZIP** (`course_delivery.zip`) with all artifacts
- **Streamlit presets** (Quick draft / Full course / Outline only)
- **Quality recommendations** after each run
- **Optional DOCX / PDF export** and **LLM quality narrative**
- **Ollama preflight** with actionable error messages
- **Dry-run** shows file sizes; delivery ZIP includes `delivery_manifest.txt`
- **Checkpoints** to resume interrupted runs (`--checkpoint`, `--resume-checkpoint`)
- **`--list-runs`** — view recent generation reports from CLI
- **`flashcards.json`** and **`quizzes.csv`** (LMS-friendly) on each full run
- **`--validate-outline`** — check outline JSON before generation
- **`--max-files`** — cap how many source files are indexed
- **`--config`** — JSON file with generation defaults (`config.example.json`)
- **`course_full.md`** — full course text + quiz reference
- **`flashcards_anki.txt`** — Anki import (TSV)
- HTML course: **dark/light theme** toggle (saved in browser)
- **`quizzes.gift`** — import quizzes into Moodle
- **`OUTPUT_INDEX.md`** — guide to all generated files
- **`--init-dirs`** / **`python -m course_generator`** — project bootstrap
- **Lesson LLM retry** before fallback content
- Streamlit: **search existing FAISS index** without full generation
- **`--batch-dir`** — run multiple JSON configs sequentially (`batch_configs/`)
- **`--check-embeddings`** / **`--diff-reports`** / **`--json-result`** for automation
- Streamlit: list **saved checkpoints** for resume
- **CLI presets** (`--preset quick|full|outline`)
- **Recursive docs** (`--recursive-docs` / `DOCS_RECURSIVE` / Streamlit “Scan subfolders”)
- Relative source labels in RAG for nested files (fewer name collisions)

---

## 🏗 Pipeline

Documents → Chunking → Embeddings → FAISS → RAG → LLM → Course + Quiz

---

## ▶️ Usage

Basic:

```
python main.py --docs-path docs
```

Try the included sample document:

```
python main.py --docs-path docs/sample-topic.txt
```

Web UI (Streamlit):

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Check Ollama before a long run:

```
python main.py --check-ollama --model llama3
python main.py --list-models
```

Use a preset:

```
python main.py --docs-path docs --preset full
python main.py --docs-path docs --preset quick --export-docx --quality-llm-review
```

Include supported files in subfolders when `docs-path` is a directory:

```
python main.py --docs-path docs --recursive-docs
python main.py --docs-path docs --no-recursive-docs
```

Advanced:

```
python main.py \
  --docs-path docs \
  --quiz-questions 12 \
  --pretest-questions 5 \
  --skip-pretest \
  --skip-final-quiz
```

Workflow helpers:

```
# List files and estimated LLM work (no Ollama, no index build)
python main.py --dry-run

# Save outline only (builds FAISS, 1–2 LLM calls)
python main.py --outline-only
python main.py --preset outline

# Continue from a saved outline JSON
python main.py --from-outline output/course_outline.json

# Save progress and resume after interruption
python main.py --checkpoint
python main.py --resume-checkpoint output/.checkpoints/default/checkpoint.json

# Recent runs
python main.py --list-runs

# Batch: one run per JSON in batch_configs/
python main.py --batch-dir batch_configs

# Automation-friendly JSON summary
python main.py --json-result
```

---

## ⚙️ Env config

Create `.env` file:

```
DOCS_PATH=docs
LLM_MODEL=llama3
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

QUIZ_QUESTIONS=10
PRETEST_QUESTIONS=5
DIFFICULTY=medium
RETRIEVAL_TYPE=similarity
LANGUAGE=en

# Optional: collect PDF/TXT/MD from subfolders when docs-path is a directory
DOCS_RECURSIVE=false

# Optional: outline grounding via FAISS retrieval
OUTLINE_RAG_MAX_CHUNKS=28
OUTLINE_RAG_MAX_CHARS=12000
```

---

## 📦 Output

- course.html  
- course_outline.json  
- quiz.json  
- pretest.json  
- course_bundle.json  
- `generation_report.json` (includes **quality** breakdown)
- `course_delivery.zip` (all of the above in one archive)
- `course_summary.docx` (optional, `--export-docx`)
- `course_summary.pdf` (optional, `--export-pdf`)
- `flashcards.json` / `quizzes.csv` (on by default; `--no-export-flashcards`, `--no-export-quiz-csv`)
- `delivery_manifest.txt` (inside `course_delivery.zip`)

---

## 💡 Why this project

- RAG pipeline implementation
- Local LLM usage (Ollama)
- Automated course generation
- EdTech automation use-case

---

## 🧪 Tests

```
pip install -r requirements-dev.txt
pytest tests/ -q
```

Push/PR to `main`: workflow `.github/workflows/ci.yml` installs deps and runs `pytest`.

## 🛣 Future improvements

- Multi-agent generation pipeline
