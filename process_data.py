import pandas as pd
import ta
import logging
from logger import setup_logger

logger = setup_logger()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators (SMA, EMA, RSI, Volatility) for each symbol.
    Args:
        df (pd.DataFrame): DataFrame with columns ['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume'].
    Returns:
        pd.DataFrame: DataFrame with additional columns for indicators.
    """
    try:
        # Debug: Confirm input type
        logger.debug(f"Type of input to calculate_technical_indicators: {type(df)}")
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"

        # Handle empty DataFrame
        required_columns = ['timestamp', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
        indicator_columns = ['SMA_50', 'EMA_20', 'RSI', 'Volatility']
        all_columns = required_columns + indicator_columns
        if df.empty:
            logger.warning("Received an empty DataFrame, returning empty DataFrame with all columns")
            return pd.DataFrame(columns=all_columns)

        grouped = df.groupby('symbol')
        result_dfs = []

        for symbol, group in grouped:
            group = group.sort_values('timestamp')
            group['SMA_50'] = ta.trend.SMAIndicator(group['Close'], window=50).sma_indicator()
            group['EMA_20'] = ta.trend.EMAIndicator(group['Close'], window=20).ema_indicator()
            group['RSI'] = ta.momentum.RSIIndicator(group['Close'], window=14).rsi()
            group['Volatility'] = group['Close'].pct_change().rolling(window=20).std() * 100
            result_dfs.append(group)

        result = pd.concat(result_dfs, ignore_index=True)

        # Handle NaNs introduced by indicators
        indicator_cols = ['SMA_50', 'EMA_20', 'RSI', 'Volatility']
        missing_indicators = result[indicator_cols].isnull().any()
        if missing_indicators.any():
            logger.warning(f"Missing indicator values in columns: {missing_indicators[missing_indicators].index.tolist()}")
            result[indicator_cols] = result[indicator_cols].fillna(method='ffill').fillna(method='bfill')

        logger.info("Calculated technical indicators")
        return result
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        raise

def process_data(input_path="valid_market_data.csv", output_path="processed_market_data.csv"):
    """
    Load validated data, calculate indicators, and save the result.
    Args:
        input_path (str): Path to the validated data CSV.
        output_path (str): Path to save the processed data CSV.
    Returns:
        pd.DataFrame: Processed DataFrame with indicators.
    """
    try:
        df = pd.read_csv(input_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"Loaded validated data from {input_path}")

        processed_data = calculate_technical_indicators(df)

        processed_data.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        return processed_data
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        processed_data = process_data()
        print(processed_data.head())
    except Exception as e:
        logger.error(f"Error in process_data script: {str(e)}")
        print(f"Error: {str(e)}")