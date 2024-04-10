from config import tushare_api_key
import tushare as ts
ts.set_token(tushare_api_key)
pro = ts.pro_api()

import pandas as pd
import time
from sqlalchemy import Table, Column, Integer, Float, String, MetaData, Date, select
from sqlalchemy import inspect

import sqlalchemy
database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock'
engine = sqlalchemy.create_engine(database_connection_string)
connection = engine.connect()

metadata = MetaData()
historical_data = Table('historical_data', metadata,
                       Column('trade_date', Date, primary_key=True),
                       Column('ts_code', String(10), primary_key=True),
                       Column('vol', Float),
                       Column('high', Float),
                       Column('low', Float),
                       Column('open', Float),
                       Column('close', Float))

# Check if the table exists, if not create it
inspector = inspect(engine)
if not inspector.has_table('historical_data'):
    metadata.create_all(engine)

    # Define global variable for API call limit
API_CALL_LIMIT = 190

# Establishing database connection

# Define or create the historical_data table# Fetching stock basic information, only for Shanghai (SH) and Shenzhen (SZ) exchanges
df_stocks = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code').query("ts_code.str.endswith('.SH') or ts_code.str.endswith('.SZ')", engine='python')

total_stocks = len(df_stocks)
print(f"Total stocks to process: {total_stocks}")

# Initialize total time counter for fetching data
total_fetch_time = 0

for index, ts_code in enumerate(df_stocks['ts_code']):
    # Control the access frequency to avoid exceeding the API limit
    if index % API_CALL_LIMIT == 0 and index != 0:
        sleep_time = max(0, 60 - (end_time - start_time))
        print(f"Processed {index}/{total_stocks}, sleeping for {sleep_time} seconds to avoid API limit")
        time.sleep(sleep_time)
    
    # Define the date range for the query
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y%m%d')
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    
    # Check if data for the current stock and date range already exists in the database
    existing_dates_query = select(historical_data.c.trade_date).where(historical_data.c.ts_code == ts_code)
    existing_dates = pd.read_sql(existing_dates_query, con=engine)
    existing_dates_list = existing_dates['trade_date'].tolist()
    
    # Start timing the data fetching process
    start_time = time.time()
    
    # Fetching daily trade data for the past 30 days
    df_daily = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    
    # Filter out existing dates from the fetched data
    df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'], format='%Y%m%d').dt.date
    df_daily = df_daily[~df_daily['trade_date'].isin(existing_dates_list)]
    
    # Calculate and print the time taken to fetch the data
    end_time = time.time()
    fetch_time = end_time - start_time
    total_fetch_time += fetch_time
    print(f"Time taken to fetch data for {ts_code}: {fetch_time:.2f} seconds")
    
    if not df_daily.empty:
        # Preparing data for insertion
        df_daily = df_daily.rename(columns={'vol': 'vol', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'})
        df_daily = df_daily[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']]  # Ensure the order matches the table schema
        df_daily['ts_code'] = ts_code
        
        # Insert fetched data into the database
        df_daily.to_sql('historical_data', con=engine, if_exists='append', index=False)
    
    print(f"Processed {index + 1}/{total_stocks}")

# Print total time taken to fetch data for all stocks
print(f"Total time taken to fetch data for all stocks: {total_fetch_time:.2f} seconds")
