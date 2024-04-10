import pandas as pd
import time
from sqlalchemy import Table, Column, Integer, Float, String, MetaData, Date, select, text
from sqlalchemy import inspect
import sqlalchemy
from tqdm import tqdm
import os

database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock'
engine = sqlalchemy.create_engine(database_connection_string)
connection = engine.connect()

metadata = MetaData()
historical_data = Table('historical_data', metadata,
                       Column('date', Date, primary_key=True),
                       Column('ts_code', String(10), primary_key=True),
                       Column('vol', Float),
                       Column('high', Float),
                       Column('low', Float),
                       Column('open', Float),
                       Column('close', Float),
                       Column('activity_score', Float))  # Ensure activity_score column exists

# Check if the table exists, if not create it
inspector = inspect(engine)
if not inspector.has_table('historical_data'):
    metadata.create_all(engine)

# Fetch the last 20 days of data for volume and volatility calculations
query = """
SELECT ts_code, trade_date, vol, open, high, low, close
FROM historical_data
WHERE trade_date >= (SELECT MAX(trade_date) FROM historical_data) - INTERVAL 20 DAY
"""
df = pd.read_sql(query, con=engine)

# Calculate average volume and volatility for the past 20 days
avg_volume_20d = df.groupby('ts_code')['vol'].mean().reset_index()
avg_volume_20d.columns = ['ts_code', 'avg_vol_20d']

avg_volatility_20d = df.groupby('ts_code').apply(lambda x: ((x['high'] - x['low']).mean())).reset_index(name='avg_volatility_20d')

# Fetch today's data
today_df = df[df['trade_date'] == df['trade_date'].max()]

# Calculate today's volume and volatility ratios
today_df = today_df.merge(avg_volume_20d, on='ts_code')
today_df = today_df.merge(avg_volatility_20d, on='ts_code')

today_df['volume_ratio'] = today_df['vol'] / today_df['avg_vol_20d']
today_df['volatility_ratio'] = (today_df['high'] - today_df['low']) / today_df['avg_volatility_20d']

# Calculate activity score according to the new logic
today_df['activity_score'] = (today_df['volume_ratio'] + today_df['volatility_ratio']) / 2

# Update the activity_score in the database
update_query = """
UPDATE historical_data
SET activity_score = :activity_score
WHERE ts_code = :ts_code AND trade_date = :trade_date
"""
with engine.begin() as conn:
    for index, row in tqdm(today_df.iterrows(), total=today_df.shape[0], desc="Updating activity scores"):
        conn.execute(text(update_query), {'activity_score': row['activity_score'], 'ts_code': row['ts_code'], 'trade_date': row['trade_date']})

# Fetch and print the last trading date and top 50 active stocks based on activity score
last_trading_date_query = "SELECT MAX(trade_date) FROM historical_data"
last_trading_date = pd.read_sql(last_trading_date_query, con=engine).iloc[0, 0]

top_active_stocks_query = """
SELECT ts_code
FROM historical_data
WHERE trade_date = %(last_trading_date)s
ORDER BY activity_score DESC
LIMIT 50
"""
top_active_stocks = pd.read_sql(top_active_stocks_query, con=engine, params={'last_trading_date': last_trading_date})

# Clearing the existing stock.list file before writing the top 50 active stocks
stock_list_path = 'stock.list'
if os.path.exists(stock_list_path):
    os.remove(stock_list_path)

with open(stock_list_path, 'w') as file:
    for index, row in top_active_stocks.iterrows():
        file.write(row['ts_code'] + '\n')

print(f"Last trading date: {last_trading_date}")
print("Top 50 active stocks based on activity score have been written to stock.list")

