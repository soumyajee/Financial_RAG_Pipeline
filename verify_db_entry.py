import sqlite3
import pandas as pd

def fetch_all_entries(db_path):
    """
    Fetches all rows from the market_data table and returns them as a pandas DataFrame.
    Args:
        db_path (str): Path to your SQLite database file.
    Returns:
        pd.DataFrame: DataFrame containing all rows from market_data.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM market_data", conn)
    finally:
        conn.close()
    return df

if __name__ == "__main__":
    db_path = "market_data.db"  # Change this to your actual DB path

    df = fetch_all_entries(db_path)
    if df.empty:
        print("No entries found in the market_data table.")
    else:
        print(f"Total entries: {len(df)}")
        print(df.head(20))  # Show first 10 rows; adjust as needed
        # Optionally, to save to CSV:
        # df.to_csv("all_market_data.csv", index=False)
