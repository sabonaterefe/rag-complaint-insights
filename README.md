# rag-complaint-insights

RAG Complaint Insights
A modular, production-ready pipeline for semantic search over consumer complaint narratives, built using Retrieval-Augmented Generation (RAG) principles, SentenceTransformers, and ChromaDB. This project enables regulatory teams, analysts, and product owners to extract meaningful insights from unstructured complaint data.

Business Motivation
The Consumer Financial Protection Bureau (CFPB) collects thousands of consumer complaints about financial products and services. These narratives are rich with signals about systemic issues, customer pain points, and regulatory risks — but they are buried in unstructured text.

Objective: Design a scalable system that enables semantic search and exploration of complaint narratives to support:

Regulatory investigations

Trend and root cause analysis

Customer experience insights

Internal QA and escalation workflows

Solution Overview
This project implements a Retrieval-Augmented Generation (RAG)-inspired architecture that transforms raw complaint narratives into a searchable vector database using modern NLP techniques.

Pipeline Architecture
Raw CSV → Preprocessing → Chunking → Embedding → ChromaDB Index → Semantic Retrieval

Stage	Description
Preprocessing	Cleans and filters complaint narratives
Chunking	Splits long texts into semantically coherent chunks
Embedding	Converts chunks into dense vectors using transformers
Indexing	Stores vectors and metadata in ChromaDB
Retrieval	Performs semantic search over indexed chunks
Why This Approach?
Challenge	Solution	Benefit
Long, noisy narratives	Recursive chunking with overlap	Preserves semantic context
Keyword search limitations	SentenceTransformer embeddings	Captures meaning, not just words
Scalable storage	ChromaDB vector store	Fast, persistent, and lightweight
Traceability	Metadata with each chunk	Enables filtering and auditability
Key Findings
Semantic search surfaces more relevant and nuanced results than keyword-based methods.

Many complaints share latent themes such as billing confusion and poor communication across companies and products.

Embedding-based retrieval enables clustering, trend detection, and future LLM-based summarization.

Project Structure
rag-complaint-insights/
├── data/
│   └── interim/
│       └── filtered_complaints.csv        # (excluded from Git)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   └── 02_retrieve_chunks.ipynb
├── reports/
│   └── eda_summary.json
├── src/
│   ├── data_preprocessing.py
│   ├── chunking.py
│   ├── embed_store.py
│   └── retrieve.py
├── vector_store/                          # (excluded from Git)
├── requirements.txt
└── README.md
How to Run
Install dependencies

pip install -r requirements.txt
Preprocess the data

python src/data_preprocessing.py
Embed and index

python src/embed_store.py
Query semantically

python src/retrieve.py
Example Use Case
Query: "Find complaints about unauthorized credit card charges"

The system retrieves semantically similar chunks such as:

"I noticed a charge I didn’t make..."

"My card was used without my permission..."

"The company refused to reverse the fraudulent transaction..."

Even if the exact phrase "unauthorized charge" is not used.

Tools and Technologies
SentenceTransformers

ChromaDB

LangChain (for chunking)

Python 3.10+

Modular, script-based architecture

Future Enhancements
Integrate LLMs for answer generation (Task 3)

Add clustering and topic modeling

Deploy as a web-based semantic search interface

Fine-tune embeddings for domain specificity

Author
Sabona Terefe Machine Learning Engineer | NLP Practitioner | Deployment-Focused Builder GitHub: https://github.com/sabonaterefe
