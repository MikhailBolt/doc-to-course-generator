# 📚 Doc-to-Course Generator

Generate a structured **HTML training course** and **assessment quizzes** from documents using **local LLMs**, **RAG**, **FAISS**, and **Ollama**.

---

## 🚀 Features

- Generate full **HTML course**
- Generate **pre-test** and **final quiz**
- Supports **PDF / TXT / MD**
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
- **Optional DOCX export** and **LLM quality narrative**
- **CLI presets** (`--preset quick|full|outline`)

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

Advanced:

```
python main.py \
  --docs-path docs \
  --quiz-questions 12 \
  --pretest-questions 5 \
  --skip-pretest \
  --skip-final-quiz
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

---

## 💡 Why this project

- RAG pipeline implementation
- Local LLM usage (Ollama)
- Automated course generation
- EdTech automation use-case

---

## 🧪 Tests

```
py -3 -m pytest tests/ -q
```

## 🛣 Future improvements

- Multi-agent generation pipeline
- Export to PDF
- Persisted user presets in the UI
