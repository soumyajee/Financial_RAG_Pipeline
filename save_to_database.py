import pandas as pd
import sqlite3
import os
import logging
from datetime import datetime
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVLoader:
    """Load existing CSV files into database"""
    
    def __init__(self, db_path: str = "financial_data.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create market_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                adj_close REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON market_data(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON market_data(symbol)')
        
        # Create technical_indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                sma_20 REAL,
                sma_50 REAL,
                ema_12 REAL,
                ema_26 REAL,
                rsi_14 REAL,
                volatility REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database setup complete")
    
    def load_csv_file(self, csv_file_path: str, symbol: str = None):
        """Load a single CSV file"""
        try:
            logger.info(f"Loading CSV file: {csv_file_path}")
            
            # Read CSV
            df = pd.read_csv(csv_file_path)
            logger.info(f"CSV has {len(df)} rows and columns: {list(df.columns)}")
            
            # Auto-detect symbol from filename if not provided
            if symbol is None:
                filename = os.path.basename(csv_file_path)
                symbol = filename.replace('.csv', '').split('_')[0].upper()
                logger.info(f"Auto-detected symbol: {symbol}")
            
            # Handle column names
            column_mapping = {
                'timestamp': 'Date',
                'Timestamp': 'Date',
                'Open': 'Open',
                'High': 'High', 
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume',
                'Adj Close': 'Adj_Close',
                'Adj_Close': 'Adj_Close'
            }
            
            # Standardize column names
            for old_col in df.columns:
                if old_col in column_mapping:
                    df.rename(columns={old_col: column_mapping[old_col]}, inplace=True)
            
            # Check if we have the required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                logger.info(f"Available columns: {list(df.columns)}")
                
                # Check if timestamp is in index
                if df.index.name in ['timestamp', 'Timestamp']:
                    df.reset_index(inplace=True)
                    df.rename(columns={df.index.name: 'Date'}, inplace=True)
                    logger.info("Timestamp found in index, moved to column")
                
                # Final check
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.error(f"Still missing required columns: {missing_cols}")
                    return False
            
            # Clean and validate data
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remove timezone
            
            # Add Adj_Close if missing
            if 'Adj_Close' not in df.columns:
                df['Adj_Close'] = df['Close']
            
            # Remove any rows with NaN values
            initial_rows = len(df)
            df = df.dropna()
            if len(df) < initial_rows:
                logger.info(f"Removed {initial_rows - len(df)} rows with NaN values")
            
            # Validate price data
            df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]
            df = df[df['High'] >= df['Low']]
            df = df[df['Volume'] >= 0]
            
            logger.info(f"After validation: {len(df)} valid rows")
            
            # Log unique symbols and their data type for debugging
            if 'symbol' in df.columns:
                logger.info(f"Unique symbols in CSV: {df['symbol'].unique()}")
                logger.info(f"Symbol column dtype: {df['symbol'].dtype}")
            
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            inserted_count = 0
            
            for _, row in df.iterrows():
                try:
                    # Use symbol from CSV if available, otherwise use provided/auto-detected symbol
                    row_symbol = str(row['symbol']).upper() if 'symbol' in df.columns and pd.notna(row['symbol']) else symbol
                    if not row_symbol or not isinstance(row_symbol, str):
                        logger.warning(f"Skipping row with invalid symbol: {row.to_dict()}")
                        continue
                    
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO market_data 
                        (symbol, timestamp, open_price, high_price, low_price, 
                         close_price, volume, adj_close)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row_symbol,  # Use per-row symbol
                        row['Date'],  # Timezone-naive datetime
                        float(row['Open']),
                        float(row['High']),
                        float(row['Low']),
                        float(row['Close']),
                        int(row['Volume']),
                        float(row['Adj_Close'])
                    ))
                    inserted_count += 1
                except Exception as e:
                    logger.warning(f"Error inserting row: {e}, Row data: {row.to_dict()}")
                    continue
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully loaded {inserted_count} records")
            return True
            
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file_path}: {e}")
            return False
    
    def load_all_csv_files(self, directory: str = ".", file_pattern: str = "*.csv"):
        """Load all CSV files from a directory"""
        csv_files = glob.glob(os.path.join(directory, file_pattern))
        
        if not csv_files:
            logger.warning(f"No CSV files found matching pattern '{file_pattern}' in '{directory}'")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        successful_loads = 0
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            logger.info(f"\nProcessing: {filename}")
            
            # Try to extract symbol from filename
            symbol = self.extract_symbol_from_filename(filename)
            
            if self.load_csv_file(csv_file, symbol):
                successful_loads += 1
        
        logger.info(f"\nCompleted loading {successful_loads}/{len(csv_files)} files successfully")
    
    def extract_symbol_from_filename(self, filename: str) -> str:
        """Extract stock symbol from filename"""
        # Remove .csv extension
        name = filename.replace('.csv', '')
        
        # Common patterns: AAPL.csv, AAPL_data.csv, stock_AAPL.csv
        parts = name.split('_')
        
        # Look for a part that looks like a stock symbol (2-5 uppercase letters)
        for part in parts:
            if 2 <= len(part) <= 5 and part.isalpha():
                return part.upper()
        
        # If no clear symbol found, use the first part
        return parts[0].upper()
    
    def verify_loaded_data(self):
        """Verify the loaded data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get summary statistics
        cursor.execute("SELECT COUNT(*) FROM market_data")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT symbol, COUNT(*) FROM market_data GROUP BY symbol ORDER BY symbol")
        symbol_counts = cursor.fetchall()
        
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM market_data")
        date_range = cursor.fetchone()
        
        print(f"\n{'='*50}")
        print("DATABASE VERIFICATION")
        print(f"{'='*50}")
        print(f"Total records: {total_records}")
        print(f"Date range: {date_range[0]} to {date_range[1]}")
        print(f"\nRecords by symbol:")
        
        for symbol, count in symbol_counts:
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp), 
                       MIN(close_price), MAX(close_price)
                FROM market_data WHERE symbol = ?
            """, (symbol,))
            min_date, max_date, min_price, max_price = cursor.fetchone()
            
            print(f"  {symbol}: {count} records")
            print(f"    Date range: {min_date} to {max_date}")
            print(f"    Price range: ${min_price:.2f} - ${max_price:.2f}")
        
        # Show sample data
        cursor.execute("SELECT symbol, timestamp, close_price FROM market_data ORDER BY timestamp DESC LIMIT 5")
        recent_data = cursor.fetchall()
        
        print(f"\nMost recent data:")
        for symbol, timestamp, price in recent_data:
            print(f"  {symbol} ({timestamp}): ${price:.2f}")
        
        conn.close()

def main():
    """Main function to load CSV files"""
    loader = CSVLoader()
    
    print("CSV to Database Loader")
    print("="*30)
    
    choice = input("\nChoose loading method:\n1. Load single CSV file\n2. Load all CSV files in current directory\n3. Load all CSV files in specific directory\n\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        csv_file = input("Enter CSV file path: ").strip()
        if not os.path.exists(csv_file):
            print(f"File not found: {csv_file}")
            return
        
        symbol = input("Enter symbol (press Enter to auto-detect or use CSV symbol column): ").strip().upper()
        if not symbol:
            symbol = None
        
        loader.load_csv_file(csv_file, symbol)
    
    elif choice == '2':
        loader.load_all_csv_files()
    
    elif choice == '3':
        directory = input("Enter directory path: ").strip()
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return
        
        pattern = input("Enter file pattern (default: *.csv): ").strip()
        if not pattern:
            pattern = "*.csv"
        
        loader.load_all_csv_files(directory, pattern)
    
    else:
        print("Invalid choice")
        return
    
    # Verify the loaded data
    loader.verify_loaded_data()
    
    print(f"\n✓ Data loading complete!")
    print("You can now run technical analysis scripts.")

if __name__ == "__main__":
    main()