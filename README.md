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

---

## 🏗 Pipeline

Documents → Chunking → Embeddings → FAISS → RAG → LLM → Course + Quiz

---

## ▶️ Usage

Basic:

```
python main.py --docs-path docs
```

Web UI (Streamlit):

```
pip install -r requirements.txt
streamlit run streamlit_app.py
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

---

## 💡 Why this project

- RAG pipeline implementation
- Local LLM usage (Ollama)
- Automated course generation
- EdTech automation use-case

---

## 🛣 Future improvements

- Web UI (Streamlit)
- Course quality scoring
- Multi-agent generation pipeline
- Export to PDF / DOCX
