import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# FastAPI backend URL
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(page_title="Financial Intelligence RAG System", layout="wide")

# Title and description
st.title("Financial Intelligence RAG System")
st.markdown("Upload financial documents, view real-time market data, and query financial insights.")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page", ["Dashboard", "Document Upload", "Query Interface", "System Status"])

# Function to fetch market data
def fetch_market_data(symbol: str):
    try:
        response = requests.get(f"{API_BASE_URL}/market-data/{symbol}")
        response.raise_for_status()
        data = response.json()
        # Debug: Uncomment to inspect the API response
        # print("Market Data API Response:", data)
        return data
    except requests.RequestException as e:
        st.error(f"Error fetching market data for {symbol}: {e}")
        return None

# Function to upload document
def upload_document(file):
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload-document", files=files)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error uploading document: {e}")
        return None

# Function to query
def run_query(query: str):
    try:
        response = requests.post(f"{API_BASE_URL}/query", json={"query": query})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error processing query: {e}")
        return None

# Function to fetch health status
def fetch_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        data = response.json()
        # Debug: Print the health response to inspect its structure
        print("Health API Response:", data)
        return data
    except requests.RequestException as e:
        st.error(f"Error fetching system health: {e}")
        return None

# Dashboard Page
if page == "Dashboard":
    st.header("Market Data Dashboard")
    symbols = ["AAPL", "GOOGL", "BTC-USD", "ETH-USD", "MSFT"]
    selected_symbol = st.selectbox("Select Stock/Crypto", symbols)

    if selected_symbol:
        data = fetch_market_data(selected_symbol)
        if data and isinstance(data, dict):
            st.subheader(f"Market Data for {data.get('symbol', selected_symbol)}")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Technical Indicators**")
                st.write(f"Close: {data.get('close', 0):.2f}")
                st.write(f"SMA 50: {data.get('sma_50', 0):.2f}")
                st.write(f"EMA 20: {data.get('ema_20', 0):.2f}")
                st.write(f"RSI: {data.get('rsi', 0):.2f}")
                st.write(f"Volatility: {data.get('volatility', 0):.2f}")
                st.write(f"Data Source: {data.get('data_source', 'N/A')}")
                st.write(f"Response Time: {data.get('response_time', 0):.2f} seconds")

            with col2:
                df = pd.DataFrame({
                    "Time": [datetime.now() - timedelta(minutes=i) for i in range(10, -1, -1)],
                    "Price": [data.get('close', 0) + i * 0.1 for i in range(11)]
                })
                fig = px.line(df, x="Time", y="Price", title=f"{data.get('symbol', selected_symbol)} Price Trend")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"No valid data returned for {selected_symbol}")

# Document Upload Page
elif page == "Document Upload":
    st.header("Upload Financial Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            result = upload_document(uploaded_file)
            if result:
                st.success(result.get("message", "Document processed successfully"))
                st.write(f"Processing Time: {result.get('response_time', 0):.2f} seconds")

# Query Interface Page
elif page == "Query Interface":
    st.header("Query Financial Insights")
    query = st.text_input("Enter your query (e.g., 'What is Apple's stock performance compared to their latest earnings guidance?')")
    
    if st.button("Submit Query"):
        if query:
            with st.spinner("Processing query..."):
                result = run_query(query)
                if result:
                    st.subheader("Response")
                    st.write(result.get("response", "No response available"))
                    st.subheader("Sources")
                    for source in result.get("sources", []):
                        st.write(f"- **{source.get('type', 'Unknown').title()}**: {source.get('source', 'N/A')} (Page: {source.get('page_number', 'N/A')})")
                    st.write(f"Response Time: {result.get('response_time', 0):.2f} seconds")
        else:
            st.error("Please enter a valid query.")

# System Status Page
elif page == "System Status":
    st.header("System Status")
    health = fetch_health()
    if health and isinstance(health, dict):
        st.write("**System Health**")
        status_color = "green" if health.get("status") == "healthy" else "orange"
        st.markdown(f"Status: <span style='color:{status_color}'>{health.get('status', 'unknown').title()}</span>", unsafe_allow_html=True)
        st.write(f"Uptime: {health.get('uptime', 0) / 3600:.2f} hours")
        st.write(f"Documents Processed: {health.get('documents_processed', 0)}")
        st.write(f"Memory Usage: {health.get('memory_usage', 0):.2f} MB")
        st.write(f"External API Status: {health.get('external_api_status', 'unknown').title()}")
        st.write(f"FAISS Index Status: {health.get('faiss_index_status', 'unknown').title()}")
        st.write(f"Processing Rate: {health.get('processing_rate', 0):.2f} chunks/second")
        st.write(f"Response Time: {health.get('response_time', 0):.2f} seconds")
    else:
        st.error("Failed to retrieve system health data. Please check if the backend is running.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Financial Intelligence RAG System | 2025")