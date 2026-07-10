# 🤖 GitHub Repo Chatbot

> Chat with any GitHub repository — ask questions, get explanations, and understand codebases instantly.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://langchain-learning-cworwtcwztrpphjbnydrpc.streamlit.app/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=langchain)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLM_API-F55036?style=for-the-badge)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)

### 🌐 [👉 Try the Live App Here](https://langchain-learning-cworwtcwztrpphjbnydrpc.streamlit.app/)

---

## 📌 What is This?

GitHub Repo Chatbot lets you paste any public GitHub repo URL and instantly start chatting with it. It clones the repo, indexes every file into a vector store, and answers your questions using retrieved code chunks as context — no hallucination, just actual repo data.

---

## ⚙️ How It Works

### Full Pipeline

```
GitHub URL
    │
    ▼
┌─────────────────────────────────┐
│         Git Clone               │
│  Clones the repo locally        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│      File Loader + Chunker      │
│  Reads every code/text/readme   │
│  file using TextLoader          │
│                                 │
│  Language-aware splitting:      │
│  .py → Python splitter          │
│  .js → JS splitter              │
│  .md → Markdown splitter        │
│  others → Generic splitter      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     Embedding + Vector Store    │
│  all-MiniLM-L6-v2 (HuggingFace)│
│  → Chroma DB (persisted)        │
└────────────────┬────────────────┘
                 │
                 ▼
         [Ready to Chat]
                 │
    ┌────────────┴────────────┐
    │       User Query        │
    └────────────┬────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│         Router Model            │
│  (Structured Output via Pydantic│
│   RouterStructure)              │
│                                 │
│  Generates:                     │
│  • 3 expanded retrieval queries │
│  • keywords (incl. readme)      │
│  • k  → number of chunks needed │
│  • fetch_k → MMR pool size      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       MMR Retriever             │
│  Searches Chroma DB using       │
│  Maximal Marginal Relevance     │
│  → returns k diverse chunks     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│         Main LLM                │
│  Receives:                      │
│  • User query                   │
│  • Retrieved context (chunks)   │
│  • Chat history (trimmed)       │
│  • Repo metadata (file list)    │
│                                 │
│  Generates final response       │
└────────────────┬────────────────┘
                 │
                 ▼
         [Response to User]
```

### Key Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Embedding Model | `all-MiniLM-L6-v2` | Lightweight, runs on CPU, good for code+text |
| Vector Store | Chroma DB | Easy local persistence, LangChain native |
| Search Type | MMR (Maximal Marginal Relevance) | Avoids redundant chunks, better coverage |
| Chunking | Language-aware `RecursiveCharacterTextSplitter` | Respects code structure per language |
| Router | Structured Output (Pydantic) | Dynamic k/fetch_k based on query complexity |
| Chat Memory | `trim_messages` (last 4000 tokens) | Keeps context window manageable |
| LLM Fallbacks | `.with_fallbacks()` chain | Auto-switches if primary model fails |

---

## 🧠 Models Used

**Main LLM** (with fallback chain):
1. `llama-3.3-70b-versatile` ← primary
2. `openai/gpt-oss-120b`
3. `qwen/qwen3.6-27b`
4. `meta-llama/llama-4-scout-17b-16e-instruct`

**Router Model** (with fallback chain):
1. `openai/gpt-oss-20b` ← primary
2. `qwen/qwen3-32b`
3. `llama-3.1-8b-instant`

All served via **Groq API** for ultra-fast inference.

---

## 🚀 Features

- 🔗 Clone and index any public GitHub repo in one click
- 🧩 Language-aware chunking for 20+ languages (Python, JS, Java, Go, Rust, C++, etc.)
- 🔍 MMR retrieval for diverse, non-redundant context
- 🧭 Smart router that adjusts retrieval depth based on query complexity
- 💬 Multi-turn chat with trimmed history
- 📋 Expander panel to inspect what the router generated
- ⚡ Fast responses via Groq inference
- 🛡️ LLM fallback chains — no single point of failure

---

## 🛠️ Tech Stack

- **[LangChain](https://langchain.com)** — chains, prompts, retrievers, message trimming
- **[Groq](https://groq.com)** — fast LLM inference
- **[Chroma DB](https://trychroma.com)** — local vector store
- **[HuggingFace](https://huggingface.co)** — `all-MiniLM-L6-v2` embedding model
- **[Streamlit](https://streamlit.io)** — web UI
- **[Pydantic](https://docs.pydantic.dev)** — structured router output

---

## 📦 Setup (Run Locally)

### 1. Clone this repo

```bash
git clone https://github.com/Dwarkesh-code/<repo-name>.git
cd <repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API keys

Create a `.env` file or set Streamlit secrets:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 💡 Usage

1. Paste a public GitHub repo URL in the sidebar
2. Click **"Clone & Index Repo"** — wait for it to clone and build the vector store
3. Ask anything in the chat:
   - *"Explain this repo"*
   - *"How does the authentication work?"*
   - *"What does the `extchecker` function do?"*
   - *"Rate this repo"*
   - *"Summarize the main project"*

---

## ⚠️ Limitations

- Only works with **public** GitHub repos
- Very large repos may take longer to index
- Accuracy depends on how well the code is documented
- Not a replacement for GitHub Copilot — built as a learning project

---

## 📚 What I Learned Building This

- Vector stores and embeddings (Chroma DB + HuggingFace)
- Language-aware text splitting with `RecursiveCharacterTextSplitter`
- MMR retrieval strategy
- Structured LLM output with Pydantic + `.with_structured_output()`
- LLM fallback chains with `.with_fallbacks()`
- Chat history trimming with `trim_messages`
- Deploying LangChain apps on Streamlit Cloud

---

## 👨‍💻 Developer

Built by **[Dwarkesh Code](https://github.com/Dwarkesh-code)** — a self-taught developer from Sujangarh, Rajasthan, currently learning the LangChain framework.

- 🐙 GitHub: [Dwarkesh-code](https://github.com/Dwarkesh-code)
- 💼 LinkedIn: [dwarkesh-code](https://www.linkedin.com/in/dwarkesh-code)
- 🚀 Streamlit: [GithubRepo\_ChatBot](https://langchain-learning-cworwtcwztrpphjbnydrpc.streamlit.app/)

---

> *"Built to learn, not to compete with Copilot."* 
