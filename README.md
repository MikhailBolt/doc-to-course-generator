# 📚 Doc-to-Course Generator

Generate a structured **HTML training course** and **assessment quizzes** from local documents using **local LLMs**, **RAG**, **FAISS**, and **Ollama**.

This version extends the original project by supporting not only **PDF**, but also **TXT** and **Markdown** files, plus optional skipping of assessments and a bundled JSON export.

## 🚀 Features

- Generate a complete **course in HTML**
- Generate **pre-test** and **final quiz** in JSON
- Supports:
  - one file or many files
  - `.pdf`, `.txt`, `.md`
- Local **RAG pipeline** with FAISS
- Local **LLM inference** via Ollama
- Automatic index rebuild when source files change
- Optional **review pass** for outline and quiz
- Optional **prefixed output files**
- Optional **source excerpts** inside lesson artifacts
- Optional **skip pre-test** / **skip final quiz**
- Generates:
  - `course.html`
  - `course_outline.json`
  - `pretest.json`
  - `quiz.json`
  - `lesson_summaries.json`
  - `course_summary.md`
  - `course_metadata.json`
  - `generation_report.json`
  - `course_bundle.json`

## 🏗 Architecture

```text
Local source files
   ↓
Text extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS vector store
   ↓
RAG retrieval per lesson
   ↓
LLM generates:
  - course outline
  - lesson HTML
  - pre-test
  - final quiz
   ↓
Review pass
   ↓
Saved outputs
```

## 📦 Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama
- PyPDF
- python-dotenv

## 📁 Project Structure

```text
doc-to-course-generator/
│
├── main.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── docs/
├── vectorstore/
├── logs/
└── output/
```

## ⚙️ Installation

### 1. Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/doc-to-course-generator.git
cd doc-to-course-generator
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama

```bash
ollama run llama3
```

## ▶️ Usage

Multiple files:

```bash
python main.py --docs-path docs --model llama3 --difficulty medium --quiz-questions 12 --pretest-questions 5 --retrieval-type mmr
```

Single file:

```bash
python main.py --docs-path docs/my_document.pdf
```

TXT/Markdown sources:

```bash
python main.py --docs-path docs --include-source-excerpts
```

With output prefix:

```bash
python main.py --docs-path docs --output-prefix ds_course_v2
```

Disable review pass:

```bash
python main.py --docs-path docs --disable-review-pass
```

Skip assessments:

```bash
python main.py --docs-path docs --skip-pretest --skip-final-quiz
```

## 🛠 CLI Options

```bash
python main.py \
  --docs-path docs \
  --model llama3 \
  --difficulty medium \
  --quiz-questions 12 \
  --pretest-questions 5 \
  --retrieval-type mmr \
  --top-k 6 \
  --chunk-size 1200 \
  --chunk-overlap 200 \
  --language en \
  --min-lessons 4 \
  --max-lessons 7 \
  --include-source-excerpts \
  --output-prefix ds_course_v2
```

## ⚙️ Environment Variables

Example `.env.example`:

```env
DOCS_PATH=docs
DB_FAISS_PATH=vectorstore/db_faiss
MANIFEST_FILE=vectorstore/index_manifest.json
OUTPUT_DIR=output
LOG_DIR=logs

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=llama3

CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=6
QUIZ_QUESTIONS=10
PRETEST_QUESTIONS=5
DIFFICULTY=medium
RETRIEVAL_TYPE=similarity
LANGUAGE=en
MAX_PREVIEW_CHARS_PER_FILE=6000
OUTPUT_PREFIX=
MIN_LESSONS=4
MAX_LESSONS=7
```

## 📤 Generated Outputs

After execution, the `output/` folder contains:

- `course.html`
- `course_outline.json`
- `pretest.json`
- `quiz.json`
- `lesson_summaries.json`
- `course_summary.md`
- `course_metadata.json`
- `generation_report.json`
- `course_bundle.json`

## 💡 What this version improves

Compared to the uploaded version, this update adds:

- support for `.txt` and `.md` alongside `.pdf`
- `--skip-pretest`
- `--skip-final-quiz`
- `--include-source-excerpts`
- configurable `--min-lessons` and `--max-lessons`
- bundled export in `course_bundle.json`
- richer lesson summaries with optional excerpts

## 🛣 Future Improvements

- export course to DOCX or PDF
- support image extraction from PDFs
- add question quality scoring
- add API and web UI
- multilingual course generation
- incremental vector DB updates
