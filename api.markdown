# API Documentation

## Overview
The Financial Intelligence RAG System API provides endpoints for real-time market data, document uploads, intelligent querying, and system health checks.

**Base URL**: `http://localhost:8000`

## Endpoints

### 1. Health Check
- **Endpoint**: `GET /health`
- **Description**: Returns system status and performance metrics.
- **Response**:
  ```json
  {
    "status": "healthy",
    "uptime": 3600.12,
    "documents_processed": 5,
    "response_time": 0.02,
    "memory_usage": 150.5,
    "external_api_status": "connected",
    "faiss_index_status": "healthy",
    "processing_rate": 10.5
  }
  ```
- **Example**:
  ```bash
  curl http://localhost:8000/health
  ```

### 2. Get Market Data
- **Endpoint**: `GET /market-data/{symbol}`
- **Description**: Retrieves market data and technical indicators for a symbol.
- **Parameters**:
  - `symbol`: Stock or crypto ticker (e.g., AAPL, BTC-USD).
- **Response**:
  ```json
  {
    "symbol": "AAPL",
    "close": 145.32,
    "sma_50": 140.25,
    "ema_20": 142.10,
    "rsi": 55.4,
    "volatility": 0.02,
    "response_time": 0.03,
    "data_source": "database"
  }
  ```
- **Example**:
  ```bash
  curl http://localhost:8000/market-data/AAPL
  ```

### 3. Upload Document
- **Endpoint**: `POST /upload-document`
- **Description**: Uploads a PDF financial document for processing.
- **Request**:
  - Content-Type: `multipart/form-data`
  - Body: `{ "file": <PDF file> }`
- **Response**:
  ```json
  {
    "message": "Document earnings.pdf uploaded and processed successfully",
    "response_time": 1.25
  }
  ```
- **Example**:
  ```bash
  curl -X POST -F "file=@earnings.pdf" http://localhost:8000/upload-document
  ```

### 4. Intelligent Query
- **Endpoint**: `POST /query`
- **Description**: Submits a query combining market data and document insights.
- **Request**:
  ```json
  {
    "query": "What is Apple's stock performance compared to their latest earnings guidance?"
  }
  ```
- **Response**:
  ```json
  {
    "query": "What is Apple's stock performance compared to their latest earnings guidance?",
    "response": "Market Data for AAPL: Close: 145.32, SMA 50: 140.25, EMA 20: 142.10, RSI: 55.4, Volatility: 0.02\nDocument Insights: Apple's Q2 earnings reported strong revenue growth...",
    "sources": [
      {
        "type": "document",
        "source": "earnings.pdf",
        "page_number": "2",
        "text": "Apple's Q2 earnings..."
      },
      {
        "type": "market_data",
        "source": "market_data.db - AAPL",
        "text": "Market Data for AAPL: ..."
      }
    ],
    "response_time": 0.45
  }
  ```
- **Example**:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"query":"What is Apple stock performance?"}' http://localhost:8000/query
  ```

## Notes
- Rate limiting: 20 requests/min for `/health`, 10 for `/market-data`, 5 for `/upload-document`, 10 for `/query`.
- Errors return HTTP status codes (e.g., 400, 429, 500) with descriptive messages.