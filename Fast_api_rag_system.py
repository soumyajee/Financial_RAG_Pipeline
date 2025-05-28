import os
import sqlite3
import numpy as np
import faiss
import pdfplumber
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2
from transformers import pipeline
from typing import List, Dict
import pickle
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
import shutil
import logging
from collections import defaultdict
import psutil
import requests

# Set up logging
logging.basicConfig(
    filename="api_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize embedding model and LLM
embedder = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Custom Rate Limiter
class CustomRateLimiter:
    def __init__(self):
        self.requests = defaultdict(lambda: defaultdict(lambda: {"count": 0, "reset_time": datetime.now()}))
        self.limits = {
            "/health": 20,
            "/market-data/{symbol}": 10,
            "/upload-document": 5,
            "/query": 10
        }
        self.time_window = 60

    def check_limit(self, endpoint: str, client_ip: str) -> bool:
        now = datetime.now()
        request_info = self.requests[endpoint][client_ip]
        if now >= request_info["reset_time"]:
            request_info["count"] = 0
            request_info["reset_time"] = now + timedelta(seconds=self.time_window)
        limit = self.limits.get(endpoint, 10)
        if request_info["count"] >= limit:
            logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}")
            return False
        request_info["count"] += 1
        return True

# FinancialDataDB
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
                logger.warning(f"No data found for symbol {symbol}")
                return {}
        except Exception as e:
            logger.error(f"Error querying database for {symbol}: {e}")
            return {}

    def __del__(self):
        self.conn.close()

# RAGApplication
class RAGApplication:
    def __init__(self, documents_dir: str = "Documents", faiss_index_dir: str = "faiss_index"):
        self.documents_dir = documents_dir
        self.faiss_index_dir = faiss_index_dir
        self.embedding_dim = 384
        self.index = None
        self.chunks = []
        self.metadata = []
        self.processed_files = set()
        if not os.path.exists(self.faiss_index_dir):
            os.makedirs(self.faiss_index_dir)
        self.load_faiss_index()
        if self.index is None or not self.chunks:
            self.index = IndexFlatL2(self.embedding_dim)
            self.process_documents()
            self.save_faiss_index()

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        try:
            pages = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    pages.append({
                        "text": page_text,
                        "metadata": {"source": os.path.basename(pdf_path), "page_number": page_num}
                    })
                    if not page_text:
                        logger.warning(f"No text extracted from page {page_num} of {pdf_path}")
            return pages
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
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
        if not os.path.exists(self.documents_dir):
            logger.warning(f"Documents directory {self.documents_dir} not found.")
            return
        for doc_file in os.listdir(self.documents_dir):
            if doc_file.endswith(".pdf") and doc_file not in self.processed_files:
                doc_path = os.path.join(self.documents_dir, doc_file)
                logger.info(f"Processing document: {doc_path}")
                pages = self.extract_text_from_pdf(doc_path)
                if pages:
                    self.process_document(doc_path, pages)
                    self.processed_files.add(doc_file)

    def save_faiss_index(self):
        faiss.write_index(self.index, os.path.join(self.faiss_index_dir, "index.faiss"))
        with open(os.path.join(self.faiss_index_dir, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(self.faiss_index_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved FAISS index and metadata to {self.faiss_index_dir}")

    def load_faiss_index(self):
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
                logger.info(f"Loaded FAISS index and metadata from {self.faiss_index_dir}")
            except Exception as e:
                logger.error(f"Error loading FAISS index: {e}")
                self.index = None
                self.chunks = []
                self.metadata = []

    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = embedder.encode([query])[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        retrieved = []
        seen_texts = set()
        for idx in indices[0]:
            if idx < len(self.chunks) and self.chunks[idx] not in seen_texts:
                if "page_number" not in self.metadata[idx]:
                    logger.warning(f"Missing page_number for chunk index {idx} from {self.metadata[idx]['source']}")
                retrieved.append({
                    "text": self.chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "page_number": str(self.metadata[idx].get("page_number", "N/A"))
                })
                seen_texts.add(self.chunks[idx])
        return retrieved

    def process_document(self, file_path: str, pages: List[Dict]):
        for page in pages:
            chunks = self.chunk_text(page["text"])
            if not chunks:
                continue
            embeddings = embedder.encode(chunks)
            embeddings = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings)
            self.chunks.extend(chunks)
            self.metadata.extend([page["metadata"] for _ in chunks])
        self.save_faiss_index()
        logger.info(f"Processed and added document {file_path} with {len(chunks)} chunks")

    def generate_response(self, query: str, financial_db: FinancialDataDB) -> tuple[str, List[Dict]]:
        symbols = ['AAPL', 'GOOGL', 'BTC-USD', 'ETH-USD', 'MSFT']
        symbol_in_query = None
        for symbol in symbols:
            if symbol.lower() in query.lower():
                symbol_in_query = symbol
                break

        market_data = {}
        if symbol_in_query:
            market_data = financial_db.get_stock_data(symbol_in_query)
            if market_data:
                market_info = (
                    f"Market Data for {symbol_in_query}:\n"
                    f"Close: {market_data['close']}\n"
                    f"SMA 50: {market_data['sma_50']}\n"
                    f"EMA 20: {market_data['ema_20']}\n"
                    f"RSI: {market_data['rsi']}\n"
                    f"Volatility: {market_data['volatility']}\n"
                )
            else:
                market_info = f"No market data available for {symbol_in_query}.\n"
        else:
            market_info = ""

        relevant_chunks = self.retrieve_relevant_chunks(query)
        if not relevant_chunks:
            response = f"{market_info}No relevant information found in the documents."
            sources = []
            return response, sources

        context_chunks = relevant_chunks
        if "revenue" in query.lower() or "sales" in query.lower():
            revenue_chunks = [chunk for chunk in relevant_chunks if "net sales" in chunk["text"].lower() or "revenue" in chunk["text"].lower()]
            context_chunks = revenue_chunks if revenue_chunks else relevant_chunks

        context = "\n".join([chunk["text"] for chunk in context_chunks])
        try:
            summary = summarizer(context, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            response = f"{market_info}Document Insights:\n{summary}"
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            response = f"{market_info}Error generating response. Retrieved context:\n{context}"

        sources = [
            {
                "type": "document",
                "source": chunk["source"],
                "page_number": str(chunk["page_number"]),
                "text": chunk["text"]
            }
            for chunk in context_chunks
        ]
        if market_info:
            sources.append({"type": "market_data", "source": f"market_data.db - {symbol_in_query}", "text": market_info.strip()})
        return response, sources

# Initialize FastAPI app
app = FastAPI(
    title="Financial Intelligence RAG System API",
    description="API for real-time financial data, document upload, and intelligent querying.",
    version="1.0.0"
)

# Initialize rate limiter, database, and RAG system
rate_limiter = CustomRateLimiter()
financial_db = FinancialDataDB(db_path="market_data.db")
rag_app = RAGApplication(documents_dir="Documents", faiss_index_dir="faiss_index")

# Directory for uploaded documents
UPLOAD_DIR = "Documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class MarketDataResponse(BaseModel):
    symbol: str
    close: float
    sma_50: float
    ema_20: float
    rsi: float
    volatility: float
    data_source: str
    response_time: float

class QueryResponse(BaseModel):
    query: str
    response: str
    sources: List[Dict[str, str]]
    response_time: float

class HealthResponse(BaseModel):
    status: str
    uptime: float
    documents_processed: int
    memory_usage: float
    external_api_status: str
    faiss_index_status: str
    processing_rate: float
    response_time: float

# Track server start time
start_time = time.time()
processed_chunks = 0  # Track total chunks processed

# Helper functions for health checks
def check_external_api():
    try:
        # Replace with your actual external API (e.g., Yahoo Finance, Alpha Vantage)
        response = requests.get("https://query1.finance.yahoo.com/v1/test/getcrumb", timeout=5)
        return "operational" if response.status_code == 200 else "down"
    except requests.RequestException:
        return "down"

def check_faiss_index():
    try:
        return "active" if rag_app.index is not None and rag_app.index.ntotal > 0 else "inactive"
    except Exception as e:
        logger.error(f"Error checking FAISS index: {e}")
        return "inactive"

def calculate_processing_rate():
    # Calculate processing rate based on chunks processed over uptime
    uptime = time.time() - start_time
    return processed_chunks / uptime if uptime > 0 else 0.0

# Middleware for rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    endpoint = request.url.path
    client_ip = request.client.host
    if not rate_limiter.check_limit(endpoint, client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    response = await call_next(request)
    return response

# Endpoint 1: Health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    start = time.time()
    try:
        conn = sqlite3.connect("market_data.db")
        conn.close()
        doc_count = len([f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")])
        response_time = time.time() - start
        return {
            "status": "healthy",
            "uptime": time.time() - start_time,
            "documents_processed": doc_count,
            "memory_usage": psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            "external_api_status": check_external_api(),
            "faiss_index_status": check_faiss_index(),
            "processing_rate": calculate_processing_rate(),
            "response_time": response_time
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Endpoint 2: Retrieve market data
@app.get("/market-data/{symbol}", response_model=MarketDataResponse)
async def get_market_data(symbol: str):
    start = time.time()
    try:
        data = financial_db.get_stock_data(symbol.upper())
        if not data:
            raise HTTPException(status_code=404, detail=f"No market data found for symbol: {symbol}")
        response = {
            "symbol": symbol.upper(),
            "close": data["close"],
            "sma_50": data["sma_50"],
            "ema_20": data["ema_20"],
            "rsi": data["rsi"],
            "volatility": data["volatility"],
            "data_source": "market_data.db",  # Updated to match Streamlit expectation
            "response_time": time.time() - start
        }
        logger.info(f"Market data retrieved for {symbol}: {data}")
        return response
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

# Endpoint 3: Upload a financial document
@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    global processed_chunks
    start = time.time()
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        if file.filename in rag_app.processed_files:
            raise HTTPException(status_code=400, detail=f"Document {file.filename} already processed")
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        pages = rag_app.extract_text_from_pdf(file_path)
        if not pages:
            raise HTTPException(status_code=500, detail=f"Failed to extract text from {file.filename}")
        # Track chunks processed
        chunk_count = 0
        for page in pages:
            chunks = rag_app.chunk_text(page["text"])
            chunk_count += len(chunks)
        rag_app.process_document(file_path, pages)
        rag_app.processed_files.add(file.filename)
        processed_chunks += chunk_count
        response_time = time.time() - start
        logger.info(f"Document {file.filename} uploaded and processed in {response_time:.2f} seconds")
        return JSONResponse(
            content={
                "message": f"Document {file.filename} uploaded and processed successfully",
                "response_time": response_time
            },
            status_code=201
        )
    except Exception as e:
        logger.error(f"Error uploading document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

# Endpoint 4: Intelligent query
@app.post("/query", response_model=QueryResponse)
async def intelligent_query(request: QueryRequest):
    start = time.time()
    try:
        query = request.query
        response_text, sources = rag_app.generate_response(query, financial_db)
        response_time = time.time() - start
        logger.info(f"Query processed: {query} in {response_time:.2f} seconds")
        return {
            "query": query,
            "response": response_text,
            "sources": sources,
            "response_time": response_time
        }
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)