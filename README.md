
# 🧠 AskVineet AI — RAG Chatbot

> *Chat with any document, instantly*

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://rag-chatbot-nzuynzjvn4c2iwj6tfm9dr.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-vin1bun-181717?style=for-the-badge&logo=github)](https://github.com/vin1bun)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Vineet%20Prakash-0077B5?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/vineetprakash03)

---


---

## 📌 Problem Statement

Large language models hallucinate — they generate confident answers that are simply wrong. For document-specific queries like HR policies, research papers, or resumes, this is unacceptable.

AskVineet AI solves this using RAG (Retrieval Augmented Generation) — every answer is grounded in the uploaded document with the exact source page shown. No hallucination. No guessing.

---

## 🚀 Live Demo

🔗 [https://rag-chatbot-nzuynzjvn4c2iwj6tfm9dr.streamlit.app/](https://rag-chatbot-nzuynzjvn4c2iwj6tfm9dr.streamlit.app/)

Upload any PDF → Ask any question → Get grounded answers instantly.

---

## ⚙️ How It Works

```
Upload PDF
    ↓
Split into chunks (500 chars, 50 overlap)
    ↓
Convert chunks to vectors (HuggingFace embeddings)
    ↓
Store vectors in FAISS
    ↓
User asks a question
    ↓
Question converted to vector
    ↓
FAISS retrieves top 3 similar chunks
    ↓
Chunks + Question sent to LLaMA 3 via Groq
    ↓
Answer streamed back with source attribution
```

---

## ✨ Features

- 📂 Multiple PDF support — chat across documents simultaneously
- ⚡ Streaming responses — answers type out live like ChatGPT
- 🧠 Chat history — remembers last 3 exchanges for contextual conversations
- 📄 Source attribution — every answer shows exactly which page it came from
- 🔐 Secure API key management via Streamlit Secrets

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain (LCEL Pipeline) |
| Embeddings | HuggingFace — all-MiniLM-L6-v2 |
| Vector Database | FAISS (Facebook AI Similarity Search) |
| LLM | LLaMA 3.1 8B via Groq API |
| PDF Loader | PyPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py                        ← Full Streamlit application
├── requirements.txt              ← All dependencies
└── RAG_Chatbot_Experiment.ipynb  ← Research & experimentation notebook
```

---

## 🔧 Run Locally

**1 — Clone the repo**
```bash
git clone https://github.com/vin1bun/rag-chatbot.git
cd rag-chatbot
```

**2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**3 — Add your Groq API key**

Create a `.streamlit/secrets.toml` file →
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Get your free Groq API key at → https://console.groq.com

**4 — Run the app**
```bash
streamlit run app.py
```

---

## 🧠 Key Engineering Decisions

**HuggingFace over Gemini Embeddings**
Gemini embeddings had persistent API v1beta compatibility issues with LangChain. Switched to HuggingFace sentence-transformers — works offline, no rate limits, zero cost, and widely used in production RAG systems.

**Groq over OpenAI**
Groq runs LLaMA 3 on custom LPU hardware at extremely low latency for free. No per-token cost makes it ideal for an open-access portfolio app.

**LCEL Pipeline**
Used LangChain Expression Language instead of the deprecated chains module — the modern, production-recommended approach for building RAG pipelines.

**Sliding Window Chat History**
Kept only the last 3 exchanges in context to balance conversation memory with token limits — a real production constraint at scale.

---

## 🔮 Future Improvements

- Add reranking using a cross-encoder model for better retrieval quality
- Support Word documents, web URLs and YouTube transcripts as input
- Add user authentication so each user has their own document space
- Integrate as an AI agent in a larger e-commerce platform

---

## 👤 About

**Vineet Prakash** — Data Scientist

- 🌐 [Live App](https://rag-chatbot-nzuynzjvn4c2iwj6tfm9dr.streamlit.app/)
- 💻 [GitHub](https://github.com/vin1bun)
- 🔗 [LinkedIn](https://linkedin.com/in/vineetprakash03)

> *"AI can build, but only humans can innovate."*


---

Copy this entire block and paste it directly into your GitHub README ✅
