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

# Initialize embedding model (free Hugging Face model)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize LLM (free Hugging Face model for summarization)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Class for financial data database
class FinancialDataDB:
    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)

    def get_stock_data(self, symbol: str) -> Dict:
        """Fetch the latest data for a given symbol from the database."""
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

class RAGApplication:
    def __init__(self, documents_dir: str = "Documents", faiss_index_dir: str = "faiss_index"):
        """Initialize the RAG system with a directory of financial documents and FAISS index directory."""
        self.documents_dir = documents_dir
        self.faiss_index_dir = faiss_index_dir
        self.embedding_dim = 384  # Dimension of embeddings from all-MiniLM-L6-v2
        self.index = None
        self.chunks = []
        self.metadata = []

        # Create FAISS index directory if it doesn't exist
        if not os.path.exists(self.faiss_index_dir):
            os.makedirs(self.faiss_index_dir)

        # Load FAISS index and metadata if they exist, otherwise process documents
        self.load_faiss_index()
        if self.index is None or not self.chunks:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.process_documents()
            self.save_faiss_index()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
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
        """Split text into chunks for embedding."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            current_length += len(word) + 1
            current_chunk.append(word)
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def process_documents(self):
        """Process all PDFs in the documents directory and store embeddings in FAISS."""
        if not os.path.exists(self.documents_dir):
            print(f"Documents directory {self.documents_dir} not found.")
            return

        for doc_file in os.listdir(self.documents_dir):
            if doc_file.endswith(".pdf"):
                doc_path = os.path.join(self.documents_dir, doc_file)
                print(f"Processing document: {doc_path}")
                
                # Extract text and chunk it
                text = self.extract_text_from_pdf(doc_path)
                if not text:
                    print(f"No text extracted from {doc_path}. Skipping.")
                    continue
                
                chunks = self.chunk_text(text)
                
                # Generate embeddings
                embeddings = embedder.encode(chunks)
                
                # Store embeddings in FAISS and keep track of chunks/metadata
                embeddings = np.array(embeddings, dtype=np.float32)
                self.index.add(embeddings)
                self.chunks.extend(chunks)
                self.metadata.extend([{"source": doc_file} for _ in chunks])
                
                print(f"Stored {len(chunks)} chunks from {doc_file}")

    def save_faiss_index(self):
        """Save the FAISS index and metadata to disk."""
        # Save the FAISS index
        faiss.write_index(self.index, os.path.join(self.faiss_index_dir, "index.faiss"))
        
        # Save chunks and metadata using pickle
        with open(os.path.join(self.faiss_index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(self.faiss_index_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Saved FAISS index and metadata to {self.faiss_index_dir}")

    def load_faiss_index(self):
        """Load the FAISS index and metadata from disk if they exist."""
        index_path = os.path.join(self.faiss_index_dir, "index.faiss")
        chunks_path = os.path.join(self.faiss_index_dir, "chunks.pkl")
        metadata_path = os.path.join(self.faiss_index_dir, "metadata.pkl")

        if os.path.exists(index_path) and os.path.exists(chunks_path) and os.path.exists(metadata_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)
                with open(metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded FAISS index and metadata from {self.faiss_index_dir}")
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                self.index = None
                self.chunks = []
                self.metadata = []

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant document chunks for a given query using FAISS."""
        # Embed the query
        query_embedding = embedder.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search for the top_k nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve the corresponding chunks and metadata
        retrieved = []
        for idx in indices[0]:
            if idx < len(self.chunks):  # Ensure index is valid
                retrieved.append({
                    "text": self.chunks[idx],
                    "source": self.metadata[idx]["source"]
                })
        return retrieved

    def generate_response(self, query: str) -> tuple[str, List[Dict]]:
        """Generate a contextual response for the query using retrieved documents."""
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query)
        if not relevant_chunks:
            return "No relevant information found in the documents.", []

        # Combine chunks into context
        context = "\n".join([chunk["text"] for chunk in relevant_chunks])
        
        # Summarize context using LLM
        try:
            summary = summarizer(context, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            response = f"Document Insights:\n{summary}"
        except Exception as e:
            print(f"Error generating summary: {e}")
            response = "Error generating response. Retrieved context:\n" + context

        # Prepare sources for attribution
        sources = [
            {"type": "document", "source": chunk["source"], "text": chunk["text"]}
            for chunk in relevant_chunks
        ]
        return response, sources

# Main execution for testing the RAG application and database
if __name__ == "__main__":
    # Test the database connection (Part 1)
    print("Testing database connection...")
    financial_db = FinancialDataDB(db_path="market_data.db")
    symbols = ['AAPL', 'GOOGL', 'BTC-USD', 'ETH-USD', 'MSFT']
    for symbol in symbols:
        data = financial_db.get_stock_data(symbol)
        print(f"Data for {symbol}: {data}")

    # Initialize and test the RAG application (Part 2)
    print("\nTesting RAG application...")
    rag_app = RAGApplication(documents_dir="Documents", faiss_index_dir="faiss_index")

    # Sample queries to test the RAG system
    sample_queries = [
        "What does Apple's latest earnings report say about their revenue?",
        "What are the key points in Apple's 2024 10-K filing?",
        "How does the financial research paper describe market trends?"
    ]

    # Open a text file to log queries and responses
    log_file_path = "query_log.txt"
    with open(log_file_path, "a") as log_file:
        log_file.write(f"\n=== Query Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        
        # Test each query and log the results
        for query in sample_queries:
            print(f"\nQuery: {query}")
            response, sources = rag_app.generate_response(query)
            print(f"Response:\n{response}")
            print("\nSources:")
            for source in sources:
                print(f"- Document: {source['source']}\n  Excerpt: {source['text'][:100]}...")
            
            # Log to file
            log_file.write(f"\nQuery: {query}\n")
            log_file.write(f"Response:\n{response}\n")
            log_file.write("Sources:\n")
            for source in sources:
                log_file.write(f"- Document: {source['source']}\n  Excerpt: {source['text'][:100]}...\n")
            log_file.write("-" * 80 + "\n")
            
            print("-" * 80)
    
    print(f"Query results saved to {log_file_path}")
