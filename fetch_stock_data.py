import yfinance as yf
import pandas as pd
import time
import logging
from logger import setup_logger

logger = setup_logger()

def fetch_realtime_data(symbols, max_retries=3, delay=5) -> pd.DataFrame:
    """
    Fetch real-time market data for given symbols from Yahoo Finance with retries.
    Args:
        symbols (list): List of stock/crypto tickers (e.g., ['AAPL', 'GOOGL']).
        max_retries (int): Maximum number of retries for failed fetches.
        delay (int): Delay between retries in seconds.
    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume'].
    """
    dfs = []
    successful_symbols = 0
    required_columns = ['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']

    for symbol in symbols:
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    tickers=symbol,
                    period="1d",
                    interval="1m",
                    auto_adjust=True,
                    threads=False
                )
                # --- FIX: Flatten MultiIndex columns if present ---
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = ['_'.join([str(c) for c in col if c]) for col in data.columns]
                logger.debug(f"Columns after yf.download for {symbol}: {list(data.columns)}")
                if data.empty:
                    logger.warning(f"No data retrieved for {symbol} on attempt {attempt + 1}")
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts")
                    time.sleep(delay)
                    continue

                # Reset index and rename Datetime to timestamp
                data.reset_index(inplace=True)
                logger.debug(f"Columns after reset_index for {symbol}: {list(data.columns)}")
                # Some yfinance versions use 'Datetime', others 'index'
                if 'Datetime' in data.columns:
                    data = data.rename(columns={'Datetime': 'timestamp'})
                elif 'index' in data.columns:
                    data = data.rename(columns={'index': 'timestamp'})
                else:
                    data['timestamp'] = data.index

                logger.debug(f"Columns after rename for {symbol}: {list(data.columns)}")

                # Standardize column names: map columns like 'Open_AAPL' to 'Open'
                for expected_col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    matching_cols = [col for col in data.columns if col.lower().startswith(expected_col.lower())]
                    if matching_cols:
                        data[expected_col] = data[matching_cols[0]]
                    else:
                        data[expected_col] = pd.NA
                logger.debug(f"Columns after standardizing for {symbol}: {list(data.columns)}")

                # Add symbol column
                data['symbol'] = symbol

                # Ensure all required columns are present
                for col in required_columns:
                    if col not in data.columns:
                        logger.warning(f"Missing column {col} for {symbol}, adding with NaN")
                        data[col] = pd.NA
                logger.debug(f"Columns after adding missing for {symbol}: {list(data.columns)}")

                # Select only the required columns
                data = data[required_columns]
                dfs.append(data)
                successful_symbols += 1
                logger.info(f"Successfully fetched data for {symbol}")
                break
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol} on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch data for {symbol} after {max_retries} attempts: {str(e)}")
                time.sleep(delay)

    if not dfs:
        logger.warning("No valid data for any symbols, returning empty DataFrame")
        return pd.DataFrame(columns=required_columns)

    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Fetched data for {successful_symbols} out of {len(symbols)} symbols")
    logger.debug(f"Type of fetch_realtime_data result: {type(result)}")
    logger.debug(f"Columns of final result: {list(result.columns)}")
    return result[required_columns]

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate market data and handle missing values.
    Args:
        df (pd.DataFrame): DataFrame with columns ['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume'].
    Returns:
        pd.DataFrame: Validated and cleaned DataFrame.
    """
    try:
        logger.debug(f"Type of input to validate_data: {type(df)}")
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"

        required_columns = ['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
        if df.empty:
            logger.warning("Received an empty DataFrame, returning as-is")
            return pd.DataFrame(columns=required_columns)

        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")

        # Validate timestamp format
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].isnull().any():
            logger.warning("Invalid timestamps found, dropping affected rows")
            df = df.dropna(subset=['timestamp'])

        # Check for duplicate timestamps per symbol
        duplicates = df.duplicated(subset=['symbol', 'timestamp'], keep=False)
        if duplicates.any():
            logger.warning("Duplicate timestamps found, keeping latest")
            df = df.sort_values(by=['symbol', 'timestamp']).drop_duplicates(subset=['symbol', 'timestamp'], keep='last')

        # Handle missing values in numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_data = df[numeric_cols].isnull().any()
        if missing_data.any():
            logger.warning(f"Missing values in columns: {missing_data[missing_data].index.tolist()}")
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')

        # Validate positive prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        if (df[price_cols] <= 0).any().any():
            logger.warning("Negative or zero prices found, dropping affected rows")
            df = df[(df[price_cols] > 0).all(axis=1)]

        if df.empty:
            logger.warning("DataFrame is empty after validation, returning empty DataFrame")
            return pd.DataFrame(columns=required_columns)

        logger.info("Data validated successfully")
        return df[required_columns]
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise

def fetch_and_validate(symbols) -> pd.DataFrame:
    """
    Fetch and validate market data for given symbols.
    Args:
        symbols (list): List of stock/crypto tickers.
    Returns:
        pd.DataFrame: Validated DataFrame.
    """
    raw_data = fetch_realtime_data(symbols)
    logger.debug(f"Type of raw_data in fetch_and_validate: {type(raw_data)}")
    assert isinstance(raw_data, pd.DataFrame), f"Expected DataFrame, got {type(raw_data)}"
    valid_data = validate_data(raw_data)
    return valid_data

if __name__ == "__main__":
    symbols = ['AAPL', 'GOOGL', 'BTC-USD', 'ETH-USD', 'MSFT']
    try:
        valid_data = fetch_and_validate(symbols)
        valid_data.to_csv("valid_market_data.csv", index=False)
        logger.info("Saved validated data to valid_market_data.csv")
        print(valid_data.head())
    except Exception as e:
        logger.error(f"Error in fetch_stock_data: {str(e)}")
        print(f"Error: {str(e)}")
