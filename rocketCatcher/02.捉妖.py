import akshare as ak
from tqdm import tqdm
from sqlalchemy import create_engine
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

# 检查股票是否符合条件
def check_stock_conditions(stock_code):
    df = get_stock_history(stock_code)
    if df.empty:
        return False

    # 计算均线
    df['MA5'] = ta.trend.sma_indicator(df['close'], window=5)
    df['MA10'] = ta.trend.sma_indicator(df['close'], window=10)

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

    # 趋势确认
    if not (df['MA5'].iloc[-1] > df['MA10'].iloc[-1] and df['close'].iloc[-1] > df['MA5'].iloc[-1] and df['close'].iloc[-1] > df['MA10'].iloc[-1]):
        return False

    # MACD金叉
    if not (df['MACD'].iloc[-1] > df['MACDsignal'].iloc[-1] and df['MACD'].iloc[-2] <= df['MACDsignal'].iloc[-2] and df['MACD'].iloc[-1] > 0):
        return False

    # 布林带中轨支撑
    if not (df['close'].iloc[-1] > df['middle'].iloc[-1]):
        return False

    # KDJ金叉
    if not (df['slowk'].iloc[-1] > df['slowd'].iloc[-1] and df['slowk'].iloc[-2] <= df['slowd'].iloc[-2]):
        return False

    return True

# 筛选符合条件的股票
selected_stocks = []
with tqdm(total=filtered_stock_df.shape[0], desc="Printing Stock Codes") as pbar:
    for index, row in filtered_stock_df.iterrows():
        pbar.set_description(f"Processing stock code: {row['code']}")
        pbar.update(1)
        stock_code = row['code']
        if check_stock_conditions(stock_code):
            selected_stocks.append(stock_code)

print("Selected Stocks:", selected_stocks)
