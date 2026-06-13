# 🎥 Smart YouTube RAG Assistant

An AI-powered application that lets you **summarize** and **ask questions** about any YouTube video using Retrieval-Augmented Generation (RAG).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-youtube-rag-hxj2sanspwntv2plxefnvp.streamlit.app/)

---

## 🚀 Live Demo

👉 **[Try it here](https://smart-youtube-rag-hxj2sanspwntv2plxefnvp.streamlit.app/)**

---

## ✨ Features

- 📄 **Video Summarization** — Get a structured, markdown-formatted summary of any YouTube video
- 💬 **Q&A Chat** — Ask anything about the video and get accurate answers from the transcript
- 🌍 **Multi-language Support** — Works with videos in Hindi, English, and other languages with available captions
- ⚡ **Smart Caching** — Already processed videos are loaded instantly from Pinecone without re-indexing
- 🤖 **Multiple Free LLMs** — Choose from several free AI models via OpenRouter
- ⬇️ **Download Summary** — Save the generated summary as a markdown file

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | OpenRouter (Free Models) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | Pinecone |
| RAG Framework | LangChain |
| Transcript | youtube-transcript-api |

---

## 📁 Project Structure

```
Smart-YouTube-RAG/
├── app.py                          # Entry point
├── requirements.txt                # Dependencies
├── .env.example                    # Environment variables template
└── youtube_rag/
    ├── main.py                     # Main Streamlit orchestrator
    ├── config/
    │   ├── settings.py             # Centralized config
    │   └── uiconfigfile.ini        # UI configuration
    ├── components/
    │   ├── transcript_loader.py    # YouTube transcript fetching
    │   ├── text_processor.py       # Text chunking
    │   ├── embeddings.py           # HuggingFace embeddings
    │   ├── vector_store.py         # Pinecone vector operations
    │   ├── llm.py                  # OpenRouter LLM setup
    │   └── rag_chain.py            # Q&A and summarization chains
    └── ui/
        ├── streamlit_ui.py         # UI components and sidebar
        └── display_result.py       # Result rendering
```

---

## ⚙️ Getting Started Locally

### 1. Clone the repository

```bash
git clone https://github.com/Souradeep-ghosh/Smart-YouTube-RAG.git
cd Smart-YouTube-RAG
```

### 2. Create and activate conda environment

```bash
conda create -n YouTubeRAG python=3.10
conda activate YouTubeRAG
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```env
OPENROUTER_API_KEY=your_openrouter_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=youtube-rag
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## 🔑 API Keys Required

| Service | Where to Get | Cost |
|---|---|---|
| OpenRouter | [openrouter.ai](https://openrouter.ai/settings/keys) | Free |
| Pinecone | [app.pinecone.io](https://app.pinecone.io) | Free tier available |

---

## 📖 How to Use

1. **Enter API Keys** in the sidebar (OpenRouter + Pinecone)
2. **Paste a YouTube URL** in the input field
3. **Click "Process Video"** — the transcript is fetched, chunked, and indexed into Pinecone
4. **Choose a tab:**
   - 📄 **Summarize** — Click "Generate Summary" for a structured overview
   - 💬 **Ask Questions** — Type any question about the video in the chat

---

## 🧠 How It Works

```
YouTube URL
    ↓
Transcript Fetched (youtube-transcript-api)
    ↓
Text Chunked (LangChain RecursiveCharacterTextSplitter)
    ↓
Chunks Embedded (HuggingFace all-MiniLM-L6-v2)
    ↓
Stored in Pinecone (namespaced by video ID)
    ↓
User Question → Semantic Search → Top K Chunks Retrieved
    ↓
LLM (OpenRouter) generates answer from retrieved context
```

---

## 🌐 Deploying on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set the main file as `app.py`
4. Add secrets in the dashboard (**Edit Secrets**):

```toml
OPENROUTER_API_KEY = "your-key"
PINECONE_API_KEY = "your-key"
PINECONE_INDEX_NAME = "youtube-rag"
```

---

## 👨‍💻 Author

**Souradeep Ghosh**

[![GitHub](https://img.shields.io/badge/GitHub-Souradeep--ghosh-black?logo=github)](https://github.com/Souradeep-ghosh)

---

## ⭐ Support

If you found this project useful, please consider giving it a **star** on GitHub! ⭐