# rag-complaint-insights
RAG Complaint Insights
A production-grade Retrieval-Augmented Generation (RAG) system for analyzing customer complaints using semantic search, local LLMs, and interactive evaluation — fully containerized and modular.


Project Overview
RAG Complaint Insights is a complete pipeline for answering natural language questions about customer complaints. It combines:

Semantic retrieval over complaint documents

Local LLMs (Gemma, Mistral, LLaMA2) via Ollama

Streamlit assistant interface

Evaluation engine with semantic scoring

Dashboard for model comparison and PDF reporting

Dockerized for reproducible deployment

System Architecture
User ↔ Streamlit UI
        ↓
   RAG Pipeline
   ├── Retrieve top-k chunks (ChromaDB)
   ├── Build prompt (context + question)
   └── Generate answer (Ollama LLM)
        ↓
   Evaluation Engine (optional)
        ↓
   Dashboard + PDF Reporting
Tech Stack & Justification:
Component	Tool	Why It Was Chosen
Vector DB	ChromaDB	Fast, local, and Python-native
Embeddings	SentenceTransformers	High-quality semantic similarity (MiniLM, MPNet)
LLM Inference	Ollama	Lightweight local LLMs with multi-model support
UI	Streamlit	Rapid prototyping and interactive dashboards
Evaluation	Cosine similarity + PDF	Quantitative scoring and exportable summaries
Containerization	Docker	Reproducible, portable, and deployment-ready
RAG Pipeline Design:
1. Embedding & Indexing
Documents embedded using SentenceTransformers

Stored in ChromaDB with metadata for traceability

2. Retrieval
Top-k chunks retrieved using cosine similarity

Semantic relevance check filters weak context

3. Prompt Construction
Context + question formatted into a flexible prompt

Encourages reasoning even when the context is incomplete

4. Answer Generation
Prompt passed to Ollama LLM (Gemma, Mistral, etc.)

Fallback logic triggers if the context is irrelevant

Streamlit Assistant
Ask natural language questions

View grounded answers with retrieved sources

Provide feedback:

Download Q&A logs

Evaluation Engine:
Batch evaluation of questions across models

Semantic similarity scoring (MiniLM)

Flags weak answers based on a confidence threshold

Outputs CSV with scores, chunks, and metadata

Evaluation Dashboard: 
Upload evaluation results

Set confidence thresholds

View weak vs. strong answers

Score distribution histograms

Model leaderboard

Export PDF summary reports

Docker Setup
# Build the image
docker build -t rag-insights .

# Run the container
docker run -p 8501:8501 rag-insights
Testing:
pytest tests/
Includes tests for:

RAG pipeline output structure

Evaluation scoring

Embedding wrapper compatibility

Branches:
Branch	Description
main	Stable baseline
assistant-ui	Streamlit assistant interface
evaluation-engine	Evaluation logic and notebook
evaluation-dashboard	Dashboard with charts, leaderboard, and PDF export
embedding-wrapper-fix	ChromaDB compatibility fix
docker-setup	Dockerfile and containerization
rag-fallback-enhancement	Smarter fallback logic with semantic checks
How to Use: 
Clone this  repo

Install dependencies or build Docker image

Start Ollama and pull a model (e.g. ollama run gemma:2b)

Run the app:
streamlit run interface/streamlit_app.py
Ask questions, evaluate models, and export results

Future Enhancements:
Hybrid retrieval (keyword + vector)

Feedback loop for fine-tuning

Scheduled evaluation jobs with email reports

Streamlit login/auth for multi-user access

Acknowledgments:
Ollama for local LLMs

ChromaDB for blazing-fast vector search

SentenceTransformers for powerful embeddings

Streamlit for rapid UI development
