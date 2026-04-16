# 📚 Doc-to-Course Generator

Generate a structured **HTML training course** and **assessment quizzes** from PDF documents using **local LLMs**, **RAG**, **FAISS**, and **Ollama**.

This project takes one PDF or a folder of PDFs, builds a local vector index, generates a course outline, creates lesson content in HTML, produces a diagnostic pre-test and final quiz, and saves all outputs locally.

## 🚀 Features

- Generate a complete **course in HTML**
- Generate **pre-test** and **final quiz** in JSON
- Supports **one PDF or multiple PDFs**
- Local **RAG pipeline** with FAISS
- Local **LLM inference** via Ollama
- Automatic index rebuild when documents change
- Optional **review pass** for outline and quiz
- Optional **prefixed output files**
- Generates:
  - `course.html`
  - `course_outline.json`
  - `pretest.json`
  - `quiz.json`
  - `lesson_summaries.json`
  - `course_summary.md`
  - `course_metadata.json`
  - `generation_report.json`

## 🏗 Architecture

```text
PDF documents
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

Multiple PDFs:

```bash
python main.py --docs-path docs --model llama3 --difficulty medium --quiz-questions 12 --pretest-questions 5 --retrieval-type mmr
```

Single PDF:

```bash
python main.py --docs-path docs/my_document.pdf
```

With output prefix:

```bash
python main.py --docs-path docs --output-prefix ds_course_v2
```

Disable review pass:

```bash
python main.py --docs-path docs --disable-review-pass
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

## 💡 Why this project is useful

This project demonstrates:

- practical use of **RAG for educational content generation**
- document understanding with local LLMs
- structured content generation
- automated assessment generation
- local, privacy-friendly AI workflows
- production-style output artifacts and reporting

## 🛣 Future Improvements

- export course to DOCX or PDF
- add difficulty-specific lesson variants
- support image extraction from PDFs
- add question quality scoring
- add API and web UI
- multilingual course generation
