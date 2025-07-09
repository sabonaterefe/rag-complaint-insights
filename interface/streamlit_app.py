import os
import sys
import streamlit as st
from datetime import datetime
import pandas as pd
import requests
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.retrieve import retrieve_top_k_chunks
from src.prompt_template import build_prompt
from src.generator import generate_answer
from src.evaluation import run_evaluation

# --- 🔧 Page Config ---
st.set_page_config(page_title="Complaint Insights Assistant", page_icon="📊", layout="wide")

# --- 🧠 Session State Defaults ---
st.session_state.setdefault("qa_history", [])
st.session_state.setdefault("theme", "light")
st.session_state.setdefault("model", "gemma:2b")

# --- 🎯 Sidebar ---
with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.markdown("Built with [Ollama](https://ollama.com) + ChromaDB")
    st.markdown("---")
    st.session_state.model = st.selectbox("🧠 Choose Model", ["gemma:2b", "llama2", "mistral"], index=["gemma:2b", "llama2", "mistral"].index(st.session_state.model))
    theme_choice = st.radio("🌓 Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "light" else 1)
    st.session_state.theme = theme_choice.lower()
    st.markdown("📦 **Vector Store**: ChromaDB")
    st.markdown("🕒 **Session**: " + datetime.now().strftime("%Y-%m-%d %H:%M"))

    if st.session_state.qa_history:
        df = pd.DataFrame(st.session_state.qa_history)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Q&A Log", data=csv, file_name="qa_log.csv", mime="text/csv")

# --- 📑 Tabs ---
tab1, tab2 = st.tabs(["💬 Assistant", "📈 Evaluation Dashboard"])

# --- 💬 Assistant Tab ---
with tab1:
    st.markdown(
        f"<h1 style='text-align: center; color: {'white' if st.session_state.theme == 'dark' else '#333'};'>📊 Complaint Insights Assistant</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align: center; font-size: 18px; color: {'#ccc' if st.session_state.theme == 'dark' else '#555'};'>Ask a question about customer complaints. The assistant will retrieve relevant excerpts and generate a grounded answer.</p>",
        unsafe_allow_html=True
    )

    query = st.text_input("🔍 Enter your question:", placeholder="e.g., What are common complaints about credit card billing?", max_chars=200)

    if query:
        with st.spinner("🔎 Retrieving and generating answer..."):
            chunks, metadatas = retrieve_top_k_chunks(query, k=3)
            prompt = build_prompt(chunks, query)
            answer = generate_answer(prompt, st.session_state.model)

        st.markdown("### 📘 Answer")
        if answer.startswith("⚠️"):
            st.error(answer)
        else:
            st.success(answer)

        col1, col2 = st.columns([1, 1])
        feedback = "none"
        with col1:
            if st.button("👍 Helpful"):
                st.toast("Thanks for your feedback! ✅")
                feedback = "positive"
        with col2:
            if st.button("👎 Needs Improvement"):
                st.toast("Thanks — we’ll use this to improve. 🛠️")
                feedback = "negative"

        st.markdown("### 📎 Retrieved Sources")
        for i, chunk in enumerate(chunks):
            with st.expander(f"Source {i+1}"):
                st.code(chunk.strip()[:1000], language="text")

        st.session_state.qa_history.append({
            "question": query,
            "answer": answer,
            "feedback": feedback,
            "model": st.session_state.model
        })

# --- 📈 Evaluation Dashboard Tab ---
with tab2:
    st.markdown("## 📈 RAG Evaluation Dashboard")
    st.markdown("Upload evaluation results or run evaluation directly from the UI.")

    with st.expander("⚙️ Run Evaluation Now"):
        default_questions = [
            "What are common complaints about credit card billing?",
            "How do customers describe poor communication?",
            "What issues are raised about account closures?",
            "Are there complaints about debt collection practices?",
            "Do customers report unauthorized charges?",
            "What are common issues with loan servicing?",
            "How do customers describe delays in fund transfers?",
            "Are there complaints about overdraft fees?",
            "Do customers mention being misled about interest rates?",
            "What problems are reported with mortgage servicing?"
        ]

        selected_models = st.multiselect("🧠 Choose Models to Compare", ["gemma:2b", "llama2", "mistral"], default=["gemma:2b"])
        if st.button("🚀 Run Evaluation"):
            for model in selected_models:
                output_path = f"reports/eval_{model.replace(':', '_')}.csv"
                run_evaluation(default_questions, output_path=output_path, method="semantic", model=model)
                st.success(f"✅ Evaluation complete for {model}")

    uploaded_files = st.file_uploader("📥 Upload one or more evaluation CSVs", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.75, 0.01)

        leaderboard = []

        for file in uploaded_files:
            df = pd.read_csv(file)
            model_name = file.name.replace("eval_", "").replace(".csv", "").replace("_", ":")

            st.markdown(f"### 📊 Results for `{model_name}`")

            total = len(df)
            scored = df["similarity_score"].notna().sum()
            passed = (df["similarity_score"] >= threshold).sum()
            avg_score = df["similarity_score"].mean()

            leaderboard.append({"Model": model_name, "Avg Score": round(avg_score, 3)})

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Questions", total)
            col2.metric("Evaluated", scored)
            col3.metric("Above Threshold", passed)

            st.markdown("#### 📈 Score Distribution")
            st.bar_chart(df["similarity_score"])

            st.markdown("#### 📉 Score Trend")
            df_sorted = df.sort_values("similarity_score", ascending=False).reset_index()
            st.line_chart(df_sorted["similarity_score"])

            weak_df = df[df["similarity_score"] < threshold]
            strong_df = df[df["similarity_score"] >= threshold]

            st.markdown("#### ⚠️ Weak Answers")
            st.dataframe(weak_df[["question", "generated_answer", "similarity_score"]])

            st.markdown("#### ✅ Strong Answers")
            st.dataframe(strong_df[["question", "generated_answer", "similarity_score"]])

            st.download_button(f"⬇️ Download Weak Answers ({model_name})", weak_df.to_csv(index=False), f"weak_{model_name}.csv", "text/csv")
            st.download_button(f"⬇️ Download Strong Answers ({model_name})", strong_df.to_csv(index=False), f"strong_{model_name}.csv", "text/csv")

        # --- 🏆 Leaderboard ---
        if leaderboard:
            st.markdown("## 🏆 Model Leaderboard")
            leaderboard_df = pd.DataFrame(leaderboard).sort_values("Avg Score", ascending=False)
            st.dataframe(leaderboard_df)

            # --- 📄 Export PDF Summary ---
            if st.button("📄 Export Summary as PDF"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="RAG Evaluation Summary", ln=True, align="C")
                pdf.ln(10)

                for row in leaderboard_df.itertuples(index=False):
                    pdf.cell(200, 10, txt=f"{row.Model}: Avg Score = {row._2}", ln=True)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    pdf.output(tmp.name)
                    with open(tmp.name, "rb") as f:
                        st.download_button("⬇️ Download PDF Summary", f.read(), file_name="rag_eval_summary.pdf", mime="application/pdf")
