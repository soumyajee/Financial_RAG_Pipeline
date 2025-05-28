# Demo Instructions

## Overview
This guide demonstrates the key features of the Financial Intelligence RAG System using the web interface and API.

## Prerequisites
- System setup as per `README.md`.
- FastAPI backend running (`uvicorn main:app --host 0.0.0.0 --port 8000`).
- Streamlit app running (`streamlit run app.py`).
- Sample PDFs in `Documents/` (e.g., `sample_earnings.pdf`).
- `market_data.db` with data for AAPL, GOOGL, BTC-USD, ETH-USD, MSFT.

## Demo Steps

### 1. View Market Data
1. Open `http://localhost:8501` and select "Dashboard" from the sidebar.
2. Choose a symbol (e.g., AAPL) from the dropdown.
3. Observe the technical indicators and price trend chart.
   - **Expected Output**: Close price, SMA, EMA, RSI, volatility, and a line chart.

### 2. Upload a Document
1. Navigate to "Document Upload" in the sidebar.
2. Upload `sample_earnings.pdf` from the `Documents/` directory.
3. Verify the success message and processing time.
   - **Expected Output**: "Document sample_earnings.pdf uploaded and processed successfully".

### 3. Submit Queries
1. Go to "Query Interface" in the sidebar.
2. Enter one of the following sample queries:
   - "What is Apple's stock performance compared to their latest earnings guidance?"
   - "How does Google's RSI compare to its volatility?"
   - "Summarize the latest earnings report for Microsoft."
   - "What is the current price trend for BTC-USD?"
   - "Compare ETH-USD volatility with its EMA."
3. Review the response and sources.
   - **Expected Output**: Response with market data and document insights, plus source attribution.

### 4. Check System Status
1. Select "System Status" in the sidebar.
2. Verify health metrics (status, uptime, memory usage, etc.).
   - **Expected Output**: Green "healthy" status if all components are functional.

### 5. API Testing
1. Use `curl` to test endpoints (see `docs/api.md` for examples).
2. Example:
   ```bash
   curl http://localhost:8000/market-data/AAPL
   curl -X POST -H "Content-Type: application/json" -d '{"query":"What is Apple stock performance?"}' http://localhost:8000/query
   ```

## Notes
- Check `api_log.txt` for detailed logs if errors occur.
- Sample data is in `sample_data/` for offline testing.
- Ensure the FastAPI backend is running before using the web interface.