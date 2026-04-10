# 📚 Doc-to-Course Generator

Generate a structured **HTML training course** and **assessment quizzes** from PDF documents using **local LLMs**, **RAG**, **FAISS**, and **Ollama**.

This project takes one PDF or a folder of PDFs, builds a local vector index, generates a course outline, creates lesson content in HTML, produces a diagnostic pre-test and final quiz, and saves all outputs locally.

---

## 🚀 Features

- Generate a complete **course in HTML**
- Generate **pre-test** and **final quiz** in JSON
- Supports **one PDF or multiple PDFs**
- Local **RAG pipeline** with FAISS
- Local **LLM inference** via Ollama
- Automatic index rebuild when documents change
- Course sections include:
  - overview
  - prerequisites
  - learning outcomes
  - glossary
  - lesson content
  - pre-test
  - final quiz
- Interactive quiz inside generated HTML
- Source-aware generation using retrieved chunks
- Saves metadata, lesson summaries, and generation report

---

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
Saved outputs:
  - course.html
  - pretest.json
  - quiz.json
  - lesson_summaries.json
  - course_metadata.json
  - generation_report.json
```

---

## 📦 Tech Stack

- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- Ollama
- PyPDF
- python-dotenv

---

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
├── docs/                 # source PDF files
├── vectorstore/          # FAISS index
├── logs/                 # execution logs
└── output/               # generated outputs
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/doc-to-course-generator.git
cd doc-to-course-generator
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama

Download Ollama and make sure the model is available locally.

```bash
ollama run llama3
```

---

## 🧾 Requirements

Recommended `requirements.txt`:

```txt
langchain
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-ollama
faiss-cpu
sentence-transformers
pypdf
python-dotenv
```

---

## ▶️ Usage

Place your PDF files in the `docs/` folder, then run:

```bash
python main.py --docs-path docs --model llama3 --difficulty medium --quiz-questions 12 --pretest-questions 5 --retrieval-type mmr
```

To use a single PDF file:

```bash
python main.py --docs-path docs/my_document.pdf
```

---

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
  --chunk-overlap 200
```

---

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
```

---

## 📤 Generated Outputs

After execution, the `output/` folder contains:

- `course.html` — full generated course
- `pretest.json` — diagnostic quiz
- `quiz.json` — final assessment quiz
- `lesson_summaries.json` — lesson summaries and takeaways
- `course_metadata.json` — metadata and outline
- `generation_report.json` — generation statistics

---

## 📖 Example Workflow

1. Add source documents to `docs/`
2. Run the generator
3. Open `output/course.html` in your browser
4. Review generated quiz files in JSON
5. Iterate by changing documents or parameters

---

## 💡 Why this project is useful

This project demonstrates:

- practical use of **RAG for educational content generation**
- document understanding with local LLMs
- structured content generation
- automated test creation
- local, privacy-friendly AI workflows
- production-style outputs and reporting

---

## 🛣 Future Improvements

- export course to DOCX or PDF
- add difficulty-specific lesson variants
- support image extraction from PDFs
- add question quality scoring
- generate flashcards
- add API and web UI
- multilingual course generation

---

## 👨‍💻 Author

Mikhail B

AI Engineer & Researcher  
Python, LangChain, and Applied Machine Learning

---

## 📜 License

MIT
