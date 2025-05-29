import os
import sqlite3
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from typing import List, Dict
import pickle
from datetime import datetime

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Database interface
class FinancialDataDB:
    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)

    def get_stock_data(self, symbol: str) -> Dict:
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT close, sma_50, ema_20, rsi, volatility 
                FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            if row:
                return {
                    "close": row[0],
                    "sma_50": row[1],
                    "ema_20": row[2],
                    "rsi": row[3],
                    "volatility": row[4]
                }
            else:
                print(f"No data found for symbol {symbol}")
                return {}
        except Exception as e:
            print(f"Error querying database for {symbol}: {e}")
            return {}

    def __del__(self):
        self.conn.close()

# RAG System
class RAGApplication:
    def __init__(self, documents_dir="Documents", faiss_index_dir="faiss_index"):
        self.documents_dir = documents_dir
        self.faiss_index_dir = faiss_index_dir
        self.embedding_dim = 384
        self.index = None
        self.chunks = []
        self.metadata = []

        if not os.path.exists(self.faiss_index_dir):
            os.makedirs(self.faiss_index_dir)

        self.load_faiss_index()
        if self.index is None or not self.chunks:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.process_documents()
            self.save_faiss_index()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        words = text.split()
        chunks, current_chunk, current_length = [], [], 0
        for word in words:
            current_length += len(word) + 1
            current_chunk.append(word)
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [], 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def process_documents(self):
        if not os.path.exists(self.documents_dir):
            print(f"Documents directory {self.documents_dir} not found.")
            return

        for doc_file in os.listdir(self.documents_dir):
            if doc_file.endswith(".pdf"):
                doc_path = os.path.join(self.documents_dir, doc_file)
                print(f"Processing document: {doc_path}")
                text = self.extract_text_from_pdf(doc_path)
                if not text:
                    print(f"No text extracted from {doc_path}. Skipping.")
                    continue
                chunks = self.chunk_text(text)
                embeddings = np.array(embedder.encode(chunks), dtype=np.float32)
                self.index.add(embeddings)
                self.chunks.extend(chunks)
                self.metadata.extend([{"source": doc_file} for _ in chunks])
                print(f"Stored {len(chunks)} chunks from {doc_file}")

    def save_faiss_index(self):
        faiss.write_index(self.index, os.path.join(self.faiss_index_dir, "index.faiss"))
        with open(os.path.join(self.faiss_index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(self.faiss_index_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        print("Saved FAISS index and metadata.")

    def load_faiss_index(self):
        try:
            index_path = os.path.join(self.faiss_index_dir, "index.faiss")
            chunks_path = os.path.join(self.faiss_index_dir, "chunks.pkl")
            metadata_path = os.path.join(self.faiss_index_dir, "metadata.pkl")
            if os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(metadata_path):
                self.index = faiss.read_index(index_path)
                with open(chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)
                with open(metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                print("Loaded FAISS index and metadata.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = np.array([embedder.encode(query)], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "source": self.metadata[idx]["source"]
                })
        return results

    def generate_response(self, query: str) -> tuple[str, List[Dict]]:
        relevant_chunks = self.retrieve_relevant_chunks(query)
        if not relevant_chunks:
            return "No relevant information found.", []

        context = "\n".join([chunk["text"] for chunk in relevant_chunks])
        try:
            summary = summarizer(context, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            response = f"Document Insights:\n{summary}"
        except Exception as e:
            print(f"Summarization error: {e}")
            response = context[:500]  # fallback

        sources = [{"type": "document", "source": c["source"], "text": c["text"]} for c in relevant_chunks]
        return response, sources


# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("Initializing Financial Data and RAG...")
    financial_db = FinancialDataDB()
    rag_app = RAGApplication()

    sample_queries = [
        "What were Apple’s total revenues and net income for Q2 2025?",
        "Break down Apple’s Q2 2025 revenue by product category.",
        "What are Apple’s major operating expenses for Q2 2025?",
        "How did Apple’s services revenue grow compared to Q2 2024?",
        "What regions contributed the most to Apple’s revenue in Q2 2025?",
        "Summarize Apple’s cash flow activities for the first six months of FY25.",
        "What is Apple’s financial position in terms of assets and liabilities as of Q2 2025?",
        "What were Apple’s major financing activities in the first half of FY25?",
        "How much did Apple spend on share repurchases in H1 FY25?",
        "What does Apple’s earnings per share reveal about their Q2 2025 performance?",

        "What was Alphabet’s total revenue and net income for Q1 2025?",
        "How did Google Cloud perform financially in Q1 2025?",
        "What revenue segments does Alphabet report, and how did they trend in Q1 2025?",
        "What geographic regions drove Alphabet’s Q1 2025 revenue growth?",
        "How much revenue backlog does Google Cloud have?",
        "What forward-looking statements did Alphabet include in their 10-Q?",
        "What are Alphabet’s major cost drivers for Q1 2025?",
        "What were the significant changes in Alphabet’s cash flow in Q1 2025?",
        "How much stock did Alphabet repurchase in Q1 2025?",
        "What trends or risks are highlighted in Alphabet’s Q1 10-Q filing?"
    ]

    log_file_path = "query_log.txt"
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== Query Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for query in sample_queries:
            print(f"\nQuery: {query}")
            response, sources = rag_app.generate_response(query)
            print(f"Response:\n{response}")
            print("Sources:")
            for source in sources:
                print(f"- {source['source']}: {source['text'][:100]}...")

            log_file.write(f"\nQuery: {query}\nResponse:\n{response}\nSources:\n")
            for source in sources:
                excerpt = source['text'][:100].replace('\n', ' ').replace('\r', ' ')
                log_file.write(f"- {source['source']}: {excerpt}...\n")
            log_file.write("-" * 80 + "\n")

    print(f"\nQuery results saved to {log_file_path}")
