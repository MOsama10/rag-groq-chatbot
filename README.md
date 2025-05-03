
```markdown
# 🔍 RAG + GROQ Chatbot

This project is a **FastAPI-powered intelligent chatbot** that can:
- 📊 Answer questions from a structured **user activity dataset**
- 🌍 Handle general knowledge questions using **GROQ LLM API**

It uses **semantic search (RAG)** to detect data-related queries and falls back to **GROQ** when necessary.

---

## 🚀 Features

- 🧠 Embedding-based query understanding (via `sentence-transformers`)
- 🧾 Structured CSV data reading
- 🗣️ GROQ LLM integration for general answers
- 🧪 Auto-routing of user messages to data or model
- 🧱 Modular FastAPI backend
- 🐍 `.env` config for security

---

## 📁 Project Structure

```

chatbot\_project/
│
├── main.py                  # FastAPI entry point
├── chat\_router.py           # Route handler for /chat
├── rag.py                   # RAG logic using embeddings
├── groq\_api.py              # GROQ API wrapper with prompt engineering
│
├── data.csv                 # Your structured dataset
├── .env                     # Stores API key and model
├── requirements.txt         # All dependencies
└── README.md                # This file

````

---

## ⚙️ Setup Instructions

### 1. 🔧 Clone & Create Environment

```bash
git clone <repo-url>
cd chatbot_project
python -m venv .venv_chatbot_project
.venv_chatbot_project\Scripts\activate
````

### 2. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

> Ensure your Python version is 3.9–3.11 for full compatibility.

### 3. 🔐 Configure `.env`

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=mistral-7b-instruct
```

### 4. 🏁 Run the Server

```bash
uvicorn main:app --reload
```

Go to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 💬 Example Questions

### ✅ Data Questions

* "What is the total login count?"
* "How many different PCs were accessed?"
* "What is the user's engagement risk score?"

### 🌍 General Questions

* "Who is Elon Musk?"
* "Explain how black holes form."
* "What is cloud computing?"

---

## 🧪 Testing Logic

1. If the query semantically matches any dataset column → answer using `data.csv`
2. Otherwise → answer using GROQ API (LLM)

---

## 🛠️ TODO / Enhancements

* [ ] Add a web frontend (e.g., React)
* [ ] Store chat history in a database
* [ ] Use FAISS for fast vector search
* [ ] Role-based permissions for data access

---



