import akshare as ak
from tqdm import tqdm
from sqlalchemy import create_engine, text
import pandas as pd
import ta
import numpy as np
from datetime import datetime, timedelta

# 创建数据库连接
engine = create_engine('mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock')

# 获取所有3、6开头的证券代码（剔除688和689）
stock_df = ak.stock_info_a_code_name()
filtered_stock_df = stock_df[(stock_df['code'].str.startswith('0')) | 
                             (stock_df['code'].str.startswith('6'))]
filtered_stock_df = filtered_stock_df[~filtered_stock_df['code'].str.startswith('688')]
filtered_stock_df = filtered_stock_df[~filtered_stock_df['code'].str.startswith('689')]

# 设定测算日期
calculation_date = '2024-03-18'

# 计算起始日期
start_date = (datetime.strptime(calculation_date, '%Y-%m-%d') - timedelta(days=365*2)).strftime('%Y-%m-%d')

# 获取股票的历史数据
def get_stock_history(stock_code):
    # 获取更长时间的数据
    query = f"""
    SELECT trade_date, open, high, low, close, volume, turnover, amplitude, change_rate, change_amount, turnover_rate
    FROM historicaldata
    WHERE symbol = '{stock_code}' AND trade_date BETWEEN '{start_date}' AND '{calculation_date}'
    """
    return pd.read_sql(query, engine)

# 获取calculation_date及其后两天的收盘价
def get_close_prices(stock_code, calculation_date):
    query = f"""
    SELECT trade_date, close
    FROM historicaldata
    WHERE symbol = '{stock_code}' AND trade_date >= '{calculation_date}'
    ORDER BY trade_date ASC
    LIMIT 3
    """
    df = pd.read_sql(query, engine)
    close_on_calculation_date = df['close'].iloc[0] if len(df) > 0 else None
    close_on_next_day = df['close'].iloc[1] if len(df) > 1 else None
    close_on_next_next_day = df['close'].iloc[2] if len(df) > 2 else None
    return close_on_calculation_date, close_on_next_day, close_on_next_next_day

# 计算并存储指标
def calculate_and_store_indicators(stock_code):
    df = get_stock_history(stock_code)
    if df.empty:
        return

    # 计算均线
    df['MA5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['MA10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['MA30'] = ta.trend.sma_indicator(df['close'], window=30)
    df['MA60'] = ta.trend.sma_indicator(df['close'], window=60)

    # 计算MACD
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACDsignal'] = macd.macd_signal()
    df['MACDhist'] = macd.macd_diff()

    # 计算布林带
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['upper'] = bb.bollinger_hband()
    df['middle'] = bb.bollinger_mavg()
    df['lower'] = bb.bollinger_lband()

    # 计算KDJ
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['slowk'] = stoch.stoch()
    df['slowd'] = stoch.stoch_signal()
    df['slowj'] = 3 * df['slowk'] - 2 * df['slowd']

    # 取过去30天的数据
    df_last_30_days = df.tail(30)

    # 获取calculation_date及其后两天的收盘价
    close_on_calculation_date, close_on_next_day, close_on_next_next_day = get_close_prices(stock_code, calculation_date)

    # 将30天的数据转换为逗号分隔的字符串
    def to_comma_separated_string(series):
        return ','.join(series.astype(str).values)

    # 创建表格（如果不存在）
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS historicalIndicator_forTraining (
            id INT AUTO_INCREMENT PRIMARY KEY,
            calculation_date DATE,
            stock_code VARCHAR(10),
            close TEXT,
            MA5 TEXT,
            MA10 TEXT,
            MA30 TEXT,
            MA60 TEXT,
            MACD TEXT,
            MACDsignal TEXT,
            MACDhist TEXT,
            upper TEXT,
            middle TEXT,
            lower TEXT,
            slowk TEXT,
            slowd TEXT,
            slowj TEXT,
            close_on_calculation_date FLOAT,
            close_on_next_day FLOAT,
            close_on_next_next_day FLOAT
        )
        """))

    # 插入数据
    data = {
        'stock_code': stock_code,
        'close': to_comma_separated_string(df_last_30_days['close']),
        'MA5': to_comma_separated_string(df_last_30_days['MA5']),
        'MA10': to_comma_separated_string(df_last_30_days['MA10']),
        'MA30': to_comma_separated_string(df_last_30_days['MA30']),
        'MA60': to_comma_separated_string(df_last_30_days['MA60']),
        'MACD': to_comma_separated_string(df_last_30_days['MACD']),
        'MACDsignal': to_comma_separated_string(df_last_30_days['MACDsignal']),
        'MACDhist': to_comma_separated_string(df_last_30_days['MACDhist']),
        'upper': to_comma_separated_string(df_last_30_days['upper']),
        'middle': to_comma_separated_string(df_last_30_days['middle']),
        'lower': to_comma_separated_string(df_last_30_days['lower']),
        'slowk': to_comma_separated_string(df_last_30_days['slowk']),
        'slowd': to_comma_separated_string(df_last_30_days['slowd']),
        'slowj': to_comma_separated_string(df_last_30_days['slowj']),
        'calculation_date': calculation_date,
        'close_on_calculation_date': close_on_calculation_date,
        'close_on_next_day': close_on_next_day,
        'close_on_next_next_day': close_on_next_next_day
    }
    df_to_insert = pd.DataFrame([data])
    df_to_insert.to_sql('historicalIndicator_forTraining', engine, if_exists='append', index=False)

# 计算并存储所有股票的指标
with tqdm(total=filtered_stock_df.shape[0], desc="Processing Stock Codes") as pbar:
    for index, row in filtered_stock_df.iterrows():
        pbar.set_description(f"Processing stock code: {row['code']}")
        pbar.update(1)
        stock_code = row['code']
        calculate_and_store_indicators(stock_code)

print("Indicators calculated and stored successfully.")
