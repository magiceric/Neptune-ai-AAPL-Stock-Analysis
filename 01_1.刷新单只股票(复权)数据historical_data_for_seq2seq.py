from config import tushare_api_key
import tushare as ts
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.sql import select, delete
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

print("Setting Tushare token...")
# Set Tushare token
ts.set_token(tushare_api_key)
pro = ts.pro_api()

print("Setting up database connection...")
# Database connection setup
database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock'
engine = create_engine(database_connection_string)
Session = sessionmaker(bind=engine)
session = Session()

def update_stock_data(ts_code='000001.SZ'):
    print(f"Updating stock data for {ts_code}...")
    # Define the table structure
    metadata = MetaData()
    metadata.reflect(bind=engine)
    historical_data_for_seq2seq = metadata.tables['historical_data_for_seq2seq']
    
    print("Fetching historical data from Tushare...")
    # Fetch historical data from Tushare
    # Initialize an empty DataFrame to hold the concatenated results
    df_full = pd.DataFrame()

    # Define the start date for fetching data
    start_date = '19910101'

    # Define the end date as today
    end_date = pd.Timestamp.today().strftime('%Y%m%d')

    # Convert start_date and end_date to datetime objects
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Calculate the number of years between start and end date
    years = end_date_dt.year - start_date_dt.year + 1

    # Fetch the adjusted historical data in yearly batches
    for year in range(start_date_dt.year, start_date_dt.year + years):
        # Define yearly start and end dates
        yearly_start_date = f"{year}0101"
        yearly_end_date = f"{year}1231"
        # Ensure the end date does not exceed the current date
        if pd.to_datetime(yearly_end_date) > end_date_dt:
            yearly_end_date = end_date

        df_batch = pro.daily(ts_code=ts_code, adj='hfq', start_date=yearly_start_date, end_date=yearly_end_date)
        # Adjust the order of the fetched data before concatenation
        df_batch = df_batch.sort_values(by='trade_date', ascending=True)
        df_full = pd.concat([df_full, df_batch], ignore_index=True)

    # Selecting required columns: 'trade_date', 'vol', 'open', 'high', 'low', 'close'
    df = df_full[['trade_date', 'ts_code', 'vol', 'open', 'high', 'low', 'close']]

    print(df)  # Print fetched data
    
    print("Deleting existing data for the stock...")
    # Delete existing data for the stock
    delete_stmt = delete(historical_data_for_seq2seq).where(historical_data_for_seq2seq.c.ts_code == ts_code)
    session.execute(delete_stmt)
    try:
        session.commit()  # Commit immediately after delete
    except OperationalError as e:
        print(f"Error committing delete operation: {e}")
        session.rollback()  # Rollback in case of error
    
    print("Preparing data for insertion...")
    # Prepare data for insertion
    df = df[['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol']]
    df = df.sort_values(by='trade_date')  # Ensure data is in ascending order by date
    
    print("Inserting new data...")
    # Insert new data in batches to avoid lock wait timeout
    batch_size = 500
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]
        try:
            batch.to_sql('historical_data_for_seq2seq', con=engine, if_exists='append', index=False)
            session.commit()  # Commit after each batch
        except OperationalError as e:
            print(f"Error inserting data: {e}")
            session.rollback()  # Rollback in case of error
    print("Data update complete.")

if __name__ == '__main__':
    # Example usage
    update_stock_data()  # Updates data for the default stock code 000001.SZ
    # update_stock_data('600000.SH')  # Updates data for the specified stock code 600000.SH
