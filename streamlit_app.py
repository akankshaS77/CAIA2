import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import os
import numpy as np
import pdfplumber
import nltk
import streamlit as st
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch
import sys
import io
import re

# Load Open-Source Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts):
    """Generate embeddings using SentenceTransformer."""
    return embedding_model.encode(texts, convert_to_tensor=True).cpu().numpy()

# Load Financial Data from PDF with Sentence Tokenization
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF using pdfplumber (better than PyPDF2)."""
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def clean_extracted_text(text):
    """Fix merged words and normalize whitespace."""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)  # Add space between merged words
    text = re.sub(r"\s+", " ", text)  # Normalize excessive whitespace
    return text.strip()

def load_financial_data(uploaded_file, chunk_size=1):
    raw_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_extracted_text(raw_text)
    sentences = sent_tokenize(cleaned_text)
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

# Implement Guardrail (Filtering Non-Financial Queries)
def filter_query(query):
    financial_keywords = [
        "revenue", "net income", "gross profit", "operating profit", "earnings", 
        "cash flow", "assets", "liabilities", "equity", "balance sheet", "income statement", 
        "financial position", "operating expenses", "depreciation", "amortization", "net earnings",
        "stock price", "market capitalization", "shareholder equity", "earnings per share", "dividend",
        "stock buyback", "trading symbol", "New York Stock Exchange", "risk factors", 
        "regulatory compliance", "SEC filings", "Sarbanes-Oxley Act", "audit report", "debt maturity",
        "loan covenants", "leverage ratio", "credit facility", "advertising revenue", "digital revenue",
        "podcast revenue", "sponsorship revenue", "event revenue", "network revenue", "operating margin",
        "advertising expenses", "marketing budget", "loss"
    ]
    return any(word in query.lower() for word in financial_keywords)

# Preprocessing (Tokenization for BM25)
def preprocess(texts):
    return [text.lower().split() for text in texts]

# Multi-Stage Retrieval: BM25 + Embeddings + Re-ranking
def multi_stage_retrieval(query, bm25_corpus, bm25, embeddings, original_texts):
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[-10:][::-1]
    query_embedding = embed_text([query])
    similarities = np.dot(embeddings, query_embedding.T).squeeze()
    top_embedding_indices = np.argsort(similarities)[-5:][::-1]
    retrieved_indices = list(set(top_bm25_indices) | set(top_embedding_indices))
    ranked_results = sorted(retrieved_indices, key=lambda i: (bm25_scores[i] + similarities[i]), reverse=True)
    return [original_texts[i] for i in ranked_results[:3]]

# Confidence Calculation with Weighted Scores
def calculate_confidence(query, bm25, bm25_corpus, embeddings):
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    query_embedding = embed_text([query])
    similarities = np.dot(embeddings, query_embedding.T).squeeze()
    top_bm25_scores = np.sort(bm25_scores)[-5:]
    top_embedding_scores = np.sort(similarities)[-5:]
    if np.max(bm25_scores) > 0:
        top_bm25_scores /= np.max(bm25_scores)
    if np.max(similarities) > 0:
        top_embedding_scores /= np.max(similarities)
    confidence_score = (0.7 * np.mean(top_bm25_scores)) + (0.3 * np.mean(top_embedding_scores))
    return f"High Confidence: {confidence_score:.2f}" if confidence_score > 0.9 else f"Low Confidence: {confidence_score:.2f}"

# Capture print statements inside Streamlit
class StreamlitLogger(io.StringIO):
    def write(self, message):
        super().write(message)
        st.session_state["log"] += message  # Append logs to session state

# Streamlit UI
def main():
    st.set_page_config(page_title="Financial QA System", layout="wide")
    
    # Centered Title & Description
    st.markdown(
        """
        <h1 style="text-align: center;">📊 Financial Document Q&A</h1>
        <p style="text-align: center; font-size:18px; color:gray;">
            A professional AI-powered financial question-answering system for investors.
        </p>
        """, 
        unsafe_allow_html=True
    )

    # Initialize log storage
    if "log" not in st.session_state:
        st.session_state["log"] = ""

    # Capture print outputs
    logger = StreamlitLogger()
    sys.stdout = logger

    # Layout: Left (Upload & Query) | Right (Output & Logs)
    col1, col2 = st.columns([1, 1])  # Two equal columns

    with col1:
        uploaded_file = st.file_uploader("📂 Upload a financial statement (PDF)", type=["pdf"])
        if uploaded_file:
            with st.spinner("🔄 Processing document..."):
                texts = load_financial_data(uploaded_file, chunk_size=1)
                tokenized_corpus = preprocess(texts)
                bm25 = BM25Okapi(tokenized_corpus)
                embeddings = embed_text(texts)
                st.success("✅ Document processed successfully!")

        query = st.text_input("🔍 Enter your financial question:")
        if st.button("Get Answer"):
            print(f"User Query: {query}")  # Log the query
            if uploaded_file:
                if filter_query(query):
                    confidence = calculate_confidence(query, bm25, tokenized_corpus, embeddings)
                    retrieved_texts = multi_stage_retrieval(query, tokenized_corpus, bm25, embeddings, texts)
                    response = "\n\n".join(retrieved_texts)

                    print(f"Confidence Level: {confidence}")  # Log confidence level
                    print(f"Retrieved Text: {response}")  # Log retrieved response

                    # Format response for better readability
                    formatted_response = "\n\n".join(f"- {line.strip()}" for line in response.split("\n") if line.strip())

                    st.success(f"*Confidence: {confidence}*")
                    st.markdown(f"*Extracted Information:*\n\n{formatted_response}", unsafe_allow_html=True)
                else:
                    st.error("⚠ Invalid query. Please ask about financial topics.")
            else:
                st.error("⚠ Please upload a document first!")

    with col2:
        st.markdown("### 📜 Debug Logs")
        st.text_area("Live Logs", value=st.session_state["log"], height=300)

if __name__ == "__main__":
    main()
