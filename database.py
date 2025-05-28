import sqlite3
import pandas as pd
import logging
from logger import setup_logger

logger = setup_logger()

class MarketDatabase:
    def __init__(self, db_path="market_data.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_table()

    def create_table(self):
        """Create market_data table with indices for efficient querying."""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT,
                    timestamp TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    sma_50 REAL,
                    ema_20 REAL,
                    rsi REAL,
                    volatility REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON market_data (symbol)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON market_data (timestamp)")
            self.conn.commit()
            logger.info("Market data table and indices created or verified")
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")

    def save_market_data(self, data: pd.DataFrame):
        """Save market data to SQLite with bulk insert."""
        try:
            cursor = self.conn.cursor()
            for _, row in data.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO market_data (
                        symbol, timestamp, open, high, low, close, volume,
                        sma_50, ema_20, rsi, volatility
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['symbol'],
                    row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume'],
                    row['SMA_50'],
                    row['EMA_20'],
                    row['RSI'],
                    row['Volatility']
                ))
            self.conn.commit()
            logger.info(f"Saved {len(data)} rows to database")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")

    def query_market_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Query market data by symbol and optional date range."""
        try:
            query = "SELECT * FROM market_data WHERE symbol = ?"
            params = [symbol]
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            df = pd.read_sql_query(query, self.conn, params=params)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Queried data for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error querying data for {symbol}: {str(e)}")
            return pd.DataFrame()
