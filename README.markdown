# Financial Intelligence RAG System

A production-ready AI system for real-time financial market data analysis and document-based insights using RAG (Retrieval-Augmented Generation).

## Setup Instructions

### Prerequisites
- Python 3.9+
- SQLite (included with Python)
- Git
- Internet connection for initial dependency installation

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/financial-rag-system.git
   cd financial-rag-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include: `fastapi`, `uvicorn`, `streamlit`, `plotly`, `pandas`, `requests`, `sentence-transformers`, `transformers`, `faiss-cpu`, `PyPDF2`, `psutil`, `numpy`.

3. **Prepare Sample Data**:
   - Place sample PDF financial documents (e.g., SEC filings) in the `Documents/` directory.
   - Ensure `market_data.db` exists with market data for symbols: AAPL, GOOGL, BTC-USD, ETH-USD, MSFT. Sample data is provided in `sample_data/`.

4. **Run the FastAPI Backend**:
   ```bash
   uvicorn Fast_api_rag_system:app --host 0.0.0.0 --port 8000
   ```

5. **Run the Streamlit Web Interface**:
   ```bash
   streamlit run streamlit_ui.py
   ```
   Access at `http://localhost:8501`.

### Directory Structure
- `main.py`: FastAPI backend for data retrieval, document processing, and querying.
- `app.py`: Streamlit web interface for user interaction.
- `Documents/`: Directory for uploaded PDF documents.
- `faiss_index/`: FAISS vector index and metadata.
- `market_data.db`: SQLite database for market data.
- `api_log.txt`: System logs with rotation.
- `docs/`: Documentation files (API, demo, technical report).
- `sample_data/`: Sample datasets for testing.

### Notes
- All components use free resources (Yahoo Finance, Hugging Face models, FAISS, SQLite).
- Ensure the FastAPI backend is running before starting the Streamlit app.
- Logs are stored in `api_log.txt` with JSON format and 10MB rotation.
