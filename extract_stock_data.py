import yfinance as yf
import pandas as pd
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def fetch_realtime_data(symbols):
    """
    Fetch real-time market data for given symbols from Yahoo Finance.
    Args:
        symbols (list): List of stock/crypto tickers (e.g., ['AAPL', 'GOOGL']).
    Returns:
        pd.DataFrame: Flat DataFrame with columns ['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume'].
    """
    try:
        # Fetch 1-day data with 1-minute intervals
        data = yf.download(
            tickers=symbols,
            period="1d",
            interval="1m",
            group_by='ticker',
            auto_adjust=True,
            threads=True
        )
        
        if data.empty:
            logger.error("No data retrieved for any symbols")
            raise ValueError("No data retrieved for any symbols")

        # Handle multi-level column index
        if len(symbols) > 1:
            # Create a list to store DataFrames for each symbol
            dfs = []
            for symbol in symbols:
                try:
                    # Extract data for each symbol
                    symbol_data = data[symbol].copy()
                    symbol_data['symbol'] = symbol
                    symbol_data.reset_index(inplace=True)
                    symbol_data = symbol_data.rename(columns={'Datetime': 'timestamp'})
                    dfs.append(symbol_data)
                except KeyError:
                    logger.warning(f"No data available for {symbol}")
                    continue
            # Concatenate all symbol DataFrames
            if not dfs:
                logger.error("No valid data for any symbols")
                raise ValueError("No valid data for any symbols")
            result = pd.concat(dfs, ignore_index=True)
        else:
            # Single symbol case
            result = data.copy()
            result['symbol'] = symbols[0]
            result.reset_index(inplace=True)
            result = result.rename(columns={'Datetime': 'timestamp'})
        
        logger.info(f"Fetched data for {len(symbols)} symbols")
        return result[['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise

def validate_data(df):
    """
    Validate market data and handle missing values.
    Args:
        df (pd.DataFrame): DataFrame with columns ['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume'].
    Returns:
        pd.DataFrame: Validated and cleaned DataFrame.
    """
    try:
        # Ensure required columns exist
        required_columns = ['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")

        # Validate timestamp format
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].isnull().any():
            logger.warning("Invalid timestamps found, dropping affected rows")
            df = df.dropna(subset=['timestamp'])

        # Handle missing values in numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_data = df[numeric_cols].isnull().any()
        if missing_data.any():
            logger.warning(f"Missing values in columns: {missing_data[missing_data].index.tolist()}")
            # Fill missing values with forward fill, then backward fill
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')

        # Validate positive prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        if (df[price_cols] <= 0).any().any():
            logger.warning("Negative or zero prices found, dropping affected rows")
            df = df[(df[price_cols] > 0).all(axis=1)]

        # Ensure non-empty DataFrame
        if df.empty:
            logger.error("DataFrame is empty after validation")
            raise ValueError("DataFrame is empty after validation")

        logger.info("Data validated successfully")
        return df[required_columns]
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    symbols = ['AAPL', 'GOOGL', 'BTC-USD', 'ETH-USD', 'MSFT']
    try:
        raw_data = fetch_realtime_data(symbols)
        valid_data = validate_data(raw_data)
        print(valid_data.head())
        # Save to CSV for inspection
        valid_data.to_csv("valid_market_data.csv", index=False)
        logger.info("Saved validated data to data/valid_market_data.csv")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Error: {str(e)}")
