# KnowledgeNexus 🧠

A multi-agent AI knowledge platform built with LangGraph and RAG architecture. Ingest documents (PDF, TXT, CSV, Excel), perform semantic retrieval using ChromaDB vector database, generate citation-aware responses, and visualize data with interactive charts.

## Features

- **Multi-File Support**: PDF, TXT, CSV, and Excel (XLSX) document ingestion
- **Semantic Retrieval**: Vector-based similarity search with MMR reranking
- **Citation-Aware Responses**: Every claim is attributed to its source
- **Data Visualization**: Interactive charts (Bar, Line, Pie, Scatter, Histogram) with data labels
- **Multi-Agent Workflow**: LangGraph orchestration for intelligent routing
- **Modern UI**: Streamlit-based chat interface with custom styling

## Architecture

```
┌─────────────────────────────────────────┐
│         Streamlit Web UI                │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│      LangGraph Agent Hub                │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ Ingestion│ │ Retrieval│ │Response │ │
│  │  Agent   │ │  Agent   │ │Generator│ │
│  └──────────┘ └──────────┘ └─────────┘ │
└─────────────────────────────────────────┘
                  │
┌─────────────────────────────────────────┐
│      ChromaDB Vector Store             │
└─────────────────────────────────────────┘
```

## Tech Stack

- **LLM**: Groq (Llama 3.1 70B) via OpenAI-compatible API
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **Orchestration**: LangGraph
- **UI**: Streamlit

## Setup

### 1. Create Virtual Environment with uv

```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# or on Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Configure Environment

Edit `.env` and add your Groq API key:

```
GROQ_API_KEY=your-groq-api-key-here
```

Get your Groq API key from [groq.com](https://groq.com).

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

## Usage

1. **Upload Documents**: Use the sidebar to upload PDF, TXT, CSV, or Excel files
2. **Process**: Click "Process Documents" to ingest and index
3. **Query**: Ask questions in the chat interface
4. **Review Citations**: Check source badges for attribution
5. **Visualize Data**: Expand "Data Visualization & Analysis" section to generate charts

## Project Structure

```
├── app.py                 # Streamlit UI
├── src/
│   ├── agents.py          # LangGraph multi-agent workflow
│   ├── document_processor.py  # PDF/TXT/CSV/Excel parsing
│   ├── vector_store.py   # ChromaDB vector storage
│   └── prompts.py        # Agent prompt templates
├── styles/
│   └── main.css          # Custom CSS styling
├── requirements.txt      # Python dependencies
├── SPEC.md              # Detailed specification
└── README.md            # This file
```

## Requirements

- Python 3.11+
- Groq API key (from groq.com)
- 4GB+ RAM for vector storage

## License

MIT