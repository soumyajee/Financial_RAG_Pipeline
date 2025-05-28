from apscheduler.schedulers.background import BackgroundScheduler
import time
import pandas as pd
from fetch_stock_data import fetch_and_validate
from process_data import calculate_technical_indicators
from database import MarketDatabase
import logging
from logger import setup_logger
logger = setup_logger()

class DataScheduler:
    def __init__(self, symbols: list, db: MarketDatabase):
        self.symbols = symbols
        self.db = db
        self.scheduler = BackgroundScheduler()
        self.buffer = {}  # In-memory buffer for batching

    def fetch_and_process(self):
        """Fetch data, validate, calculate indicators, and store in buffer."""
        try:
            # Fetch and validate data
            raw_data = fetch_and_validate(self.symbols)
            logger.debug(f"Type of raw_data in fetch_and_process: {type(raw_data)}")
            assert isinstance(raw_data, pd.DataFrame), f"Expected DataFrame, got {type(raw_data)}"

            # Calculate indicators
            processed_data = calculate_technical_indicators(raw_data)
            logger.debug(f"Type of processed_data in fetch_and_process: {type(processed_data)}")
            assert isinstance(processed_data, pd.DataFrame), f"Expected DataFrame, got {type(processed_data)}"

            # Skip buffering if DataFrame is empty
            if processed_data.empty:
                logger.warning("Processed data is empty, skipping buffer update")
                return

            # Store in buffer
            for symbol, group in processed_data.groupby('symbol'):
                rows = group.to_dict(orient="records")
                self.buffer[symbol] = self.buffer.get(symbol, []) + rows
            logger.info("Fetched, processed, and buffered market data")
        except Exception as e:
            logger.error(f"Error in fetch_and_process: {str(e)}")

    def flush_buffer(self):
        """Save buffered data to database and clear buffer."""
        try:
            for symbol, rows in self.buffer.items():
                if rows:
                    buffer_df = pd.DataFrame(rows)
                    self.db.save_market_data(buffer_df)
            self.buffer.clear()
            logger.info("Flushed buffer to database")
        except Exception as e:
            logger.error(f"Error flushing buffer: {str(e)}")

    def start(self):
        """Start scheduling data fetches and buffer flushes."""
        self.scheduler.add_job(self.fetch_and_process, "interval", minutes=1)
        self.scheduler.add_job(self.flush_buffer, "interval", minutes=5)
        self.scheduler.start()
        logger.info("Data scheduler started")

def store_initial_data(db, input_path="processed_market_data.csv"):
    """
    Load processed data and store it in SQLite.
    """
    try:
        df = pd.read_csv(input_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        db.save_market_data(df)
        logger.info(f"Stored initial processed data from {input_path}")
    except Exception as e:
        logger.error(f"Error storing initial data: {str(e)}")
        raise

if __name__ == "__main__":
    symbols = ['AAPL', 'GOOGL', 'BTC-USD', 'ETH-USD', 'MSFT']
    db = MarketDatabase()

    # Store initial processed data (from process_data.py)
    try:
        store_initial_data(db)
    except Exception as e:
        logger.error(f"Error storing initial data: {str(e)}")
        print(f"Error: {str(e)}")

    # Start scheduler for continuous updates
    scheduler = DataScheduler(symbols, db)
    try:
        scheduler.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scheduler.scheduler.shutdown()
        logger.info("Scheduler stopped")