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
DEFAULT_INTERVIEW_QUESTIONS=5
DEFAULT_QUIZ_QUESTIONS=5
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
