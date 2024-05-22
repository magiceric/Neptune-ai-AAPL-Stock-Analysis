from TushareCaller import TushareCaller
import time
import sqlalchemy as db
import pandas as pd

class StockAnalysisTool:
    def __init__(self, stock_code, global_timestamp=None):
        self._stock_code = stock_code
        self.data = "This is a stock analysis tool."
        if global_timestamp is None:
            global_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.tushare_caller = TushareCaller(global_timestamp)
        # stock_info = self.tushare_caller.call_api('stock_basic', exchange='', list_status='L', fields='name', ts_code=self._stock_code)
        # self._stock_name = stock_info['name'].iloc[0] if not stock_info.empty else 'Unknown'
        self._database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock'
        self._historical_data = self._fetch_historical_data()

    def analyze(self):
        print(f"Analyzing stock data for {self._stock_code} - {self._stock_name}...")
        stock_info = self.tushare_caller.call_api('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date', ts_code=self._stock_code)
        print(stock_info)

    def get_basic_info(self):
        return {"Stock Code": self._stock_code, "Stock Name": self._stock_name}

    def _fetch_historical_data(self):
        engine = db.create_engine(self._database_connection_string)
        connection = engine.connect()
        query = db.text(f"""
        SELECT * FROM historicaldata
        WHERE symbol = :stock_code
        """)
        result = connection.execute(query, {'stock_code': self._stock_code}).fetchall()
        connection.close()
        engine.dispose()
        return result

    def check_macd_crossover(self, date):
        df = self._prepare_macd_dataframe()
        if date in df.index:
            current_idx = df.index.get_loc(date)
            if current_idx > 0:
                previous_macd = df.iloc[current_idx - 1]['MACD_Histogram']
                current_macd = df.iloc[current_idx]['MACD_Histogram']
                if previous_macd < 0 and current_macd > 0:
                    return True
        return False

    def find_recent_macd_crossover(self, date, days):
        df = self._prepare_macd_dataframe()
        end_date = pd.to_datetime(date)
        if end_date not in df.index:
            return None
        end_idx = df.index.get_loc(end_date)
        start_idx = max(end_idx - days, 0)  # Ensure start_idx is not less than 0
        start_date = df.index[start_idx]
        df = df.loc[start_date:end_date]
        last_crossover_date = None
        for idx in range(1, len(df)):
            if df.iloc[idx - 1]['MACD_Histogram'] < 0 and df.iloc[idx]['MACD_Histogram'] > 0:
                last_crossover_date = df.index[idx].date()  # Changed to return only the date part
        return last_crossover_date
    
    def _prepare_macd_dataframe(self):
        expected_columns = ['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']
        df = pd.DataFrame(self._historical_data, columns=expected_columns)
        df = df[['trade_date', 'open', 'high', 'low', 'close', 'volume']]  # Select only the necessary columns
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')  # Ensure correct datetime conversion
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['DIFF'] = df['EMA12'] - df['EMA26']  # DIFF is the same as MACD
        df['DEA'] = df['DIFF'].ewm(span=9, adjust=False).mean()  # DEA is the same as Signal
        df['MACD_Histogram'] = df['DIFF'] - df['DEA']
        return df

    def check_volume_crossover_5_over_10(self, date):
        df = self._prepare_volume_dataframe_with_10_day_ma()
        if date in df.index:
            current_idx = df.index.get_loc(date)
            if current_idx > 0:
                previous_volume_ma5 = df.iloc[current_idx - 1]['Volume_MA5']
                previous_volume_ma10 = df.iloc[current_idx - 1]['Volume_MA10']
                current_volume_ma5 = df.iloc[current_idx]['Volume_MA5']
                current_volume_ma10 = df.iloc[current_idx]['Volume_MA10']
                if previous_volume_ma5 < previous_volume_ma10 and current_volume_ma5 > current_volume_ma10:
                    return True
        return False

    def find_recent_volume_crossover_5_over_10(self, date, days):
        df = self._prepare_volume_dataframe_with_10_day_ma()
        end_date = pd.to_datetime(date)
        if end_date not in df.index:
            return None
        end_idx = df.index.get_loc(end_date)
        start_idx = max(end_idx - days, 0)  # Ensure start_idx is not less than 0
        start_date = df.index[start_idx]

        df = df.loc[start_date:end_date]
        last_crossover_date = None
        for idx in range(1, len(df)):
            if df.iloc[idx - 1]['Volume_MA5'] < df.iloc[idx - 1]['Volume_MA10'] and df.iloc[idx]['Volume_MA5'] > df.iloc[idx]['Volume_MA10']:
                last_crossover_date = df.index[idx].date()  # Changed to return only the date part
        return last_crossover_date
    
    def _prepare_volume_dataframe_with_10_day_ma(self):
        expected_columns = ['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']
        df = pd.DataFrame(self._historical_data, columns=expected_columns)
        df = df[['trade_date', 'volume']]  # Select only the necessary columns
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')  # Ensure correct datetime conversion
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
        df['Volume_MA10'] = df['volume'].rolling(window=10).mean()
        return df
    
    def check_price_above_middle_bollinger_band(self, date):
        df = self._prepare_bollinger_bands_dataframe()
        if date in df.index:
            current_idx = df.index.get_loc(date)
            current_close = df.iloc[current_idx]['close']
            current_middle_band = df.iloc[current_idx]['Middle_Band']
            if current_close > current_middle_band:
                return True
        return False

    def find_recent_price_above_middle_bollinger_band(self, date, days):
        df = self._prepare_bollinger_bands_dataframe()
        end_date = pd.to_datetime(date)
        if end_date not in df.index:
            return None
        end_idx = df.index.get_loc(end_date)
        start_idx = max(end_idx - days, 0)  # Ensure start_idx is not less than 0
        start_date = df.index[start_idx]

        df = df.loc[start_date:end_date]
        for idx in range(len(df)):
            if df.iloc[idx]['close'] > df.iloc[idx]['Middle_Band']:
                return df.index[idx].date()  # Return the date when the price was above the middle band
        return None
    
    def _prepare_bollinger_bands_dataframe(self):
        expected_columns = ['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']
        df = pd.DataFrame(self._historical_data, columns=expected_columns)
        df = df[['trade_date', 'close']]  # Select only the necessary columns
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')  # Ensure correct datetime conversion
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        df['Middle_Band'] = df['close'].rolling(window=20).mean()
        df['Upper_Band'] = df['Middle_Band'] + 2 * df['close'].rolling(window=20).std()
        return df

    def check_kdj_crossover(self, date):
        df = self._prepare_kdj_dataframe()
        if date in df.index:
            current_idx = df.index.get_loc(date)
            if df.iloc[current_idx]['K'] > df.iloc[current_idx]['D'] and df.iloc[current_idx - 1]['K'] <= df.iloc[current_idx - 1]['D']:
                return True
        return False

    def find_recent_kdj_crossover(self, date, days):
        df = self._prepare_kdj_dataframe()
        end_date = pd.to_datetime(date)
        if end_date not in df.index:
            return None
        end_idx = df.index.get_loc(end_date)
        start_idx = max(end_idx - days, 0)  # Ensure start_idx is not less than 0
        start_date = df.index[start_idx]

        df = df.loc[start_date:end_date]
        for idx in range(1, len(df)):
            if df.iloc[idx]['K'] > df.iloc[idx]['D'] and df.iloc[idx - 1]['K'] <= df.iloc[idx - 1]['D']:
                return (df.index[idx].date(), df.iloc[idx]['J'])  # Return the date when the KDJ crossover occurred and the J value
        return None
    
    def _prepare_kdj_dataframe(self):
        expected_columns = ['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']
        df = pd.DataFrame(self._historical_data, columns=expected_columns)
        df = df[['trade_date', 'high', 'low', 'close']]  # Select only the necessary columns
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')  # Ensure correct datetime conversion
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(alpha=1/3).mean()
        df['D'] = df['K'].ewm(alpha=1/3).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        return df

    def check_ma5_ma10_support(self, date):
        df = self._prepare_ma_dataframe()
        if date in df.index:
            current_idx = df.index.get_loc(date)
            if current_idx >= 1:  # Ensure there is a previous day to compare
                ma5_today = df.iloc[current_idx]['MA5']
                ma10_today = df.iloc[current_idx]['MA10']
                ma5_yesterday = df.iloc[current_idx - 1]['MA5']
                ma10_yesterday = df.iloc[current_idx - 1]['MA10']
                # Check if MA5 was above MA10 and has fallen below but not too far
                if ma5_yesterday > ma10_yesterday and ma5_today <= ma10_today and ma5_today >= 0.97 * ma10_today:
                    return True
        return False

    def find_recent_ma5_ma10_support(self, date, days):
        df = self._prepare_ma_dataframe()
        end_date = pd.to_datetime(date)
        if end_date not in df.index:
            return None
        end_idx = df.index.get_loc(end_date)
        start_idx = max(end_idx - days, 0)  # Ensure start_idx is not less than 0
        start_date = df.index[start_idx]

        df = df.loc[start_date:end_date]
        for idx in range(1, len(df)):
            ma5_today = df.iloc[idx]['MA5']
            ma10_today = df.iloc[idx]['MA10']
            ma5_yesterday = df.iloc[idx - 1]['MA5']
            ma10_yesterday = df.iloc[idx - 1]['MA10']
            if ma5_yesterday > ma10_yesterday and ma5_today <= ma10_today and ma5_today >= 0.97 * ma10_today:
                return df.index[idx].date()  # Return the date when the MA5 fell below MA10 but found support
        return None

    def _prepare_ma_dataframe(self):
        expected_columns = ['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']
        df = pd.DataFrame(self._historical_data, columns=expected_columns)
        df = df[['trade_date', 'close']]  # Select only the necessary columns
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')  # Ensure correct datetime conversion
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA10'] = df['close'].rolling(window=10).mean()
        return df

    def check_price_cross_ma5(self, date):
        df = self._prepare_ma_dataframe()
        if date in df.index:
            current_idx = df.index.get_loc(date)
            if current_idx >= 1:  # Ensure there is a previous day to compare
                close_today = df.iloc[current_idx]['close']
                ma5_today = df.iloc[current_idx]['MA5']
                close_yesterday = df.iloc[current_idx - 1]['close']
                ma5_yesterday = df.iloc[current_idx - 1]['MA5']
                # Check if the close price was below MA5 yesterday and above MA5 today
                if close_yesterday < ma5_yesterday and close_today > ma5_today:
                    return True
        return False

    def find_recent_price_cross_ma5(self, date, days):
        df = self._prepare_ma_dataframe()
        end_date = pd.to_datetime(date)
        if end_date not in df.index:
            return None
        end_idx = df.index.get_loc(end_date)
        start_idx = max(end_idx - days, 0)  # Ensure start_idx is not less than 0
        start_date = df.index[start_idx]

        df = df.loc[start_date:end_date]
        for idx in range(1, len(df)):
            close_today = df.iloc[idx]['close']
            ma5_today = df.iloc[idx]['MA5']
            close_yesterday = df.iloc[idx - 1]['close']
            ma5_yesterday = df.iloc[idx - 1]['MA5']
            if close_yesterday < ma5_yesterday and close_today > ma5_today:
                return df.index[idx].date()  # Return the date when the close price crossed above MA5
        return None

    def check_kdj_cross(self, date):
        expected_columns = ['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']
        df = pd.DataFrame(self._historical_data, columns=expected_columns)
        df = df[['trade_date', 'high', 'low', 'close']]  # Select only the necessary columns
        df['K'], df['D'], df['J'] = self._calculate_kdj(df['high'], df['low'], df['close'])
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        if date in df.index:
            current_idx = df.index.get_loc(date)
            if current_idx >= 1:  # Ensure there is a previous day to compare
                k_today = df.iloc[current_idx]['K']
                d_today = df.iloc[current_idx]['D']
                j_today = df.iloc[current_idx]['J']
                k_yesterday = df.iloc[current_idx - 1]['K']
                d_yesterday = df.iloc[current_idx - 1]['D']
                j_yesterday = df.iloc[current_idx - 1]['J']
                # Check if K and J both crossed above D today and J crossed above K
                if k_yesterday <= d_yesterday and j_yesterday <= d_yesterday and k_today > d_today and j_today > d_today and j_today > k_today:
                    return True
        return False

    def find_recent_kdj_cross(self, date, days):
        expected_columns = ['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']
        df = pd.DataFrame(self._historical_data, columns=expected_columns)
        df = df[['trade_date', 'symbol', 'high', 'low', 'close']]  # Select only the necessary columns
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')  # Ensure correct datetime conversion
        df.set_index('trade_date', inplace=True)
        df.sort_index(inplace=True)
        df['K'], df['D'], df['J'] = self._calculate_kdj(df['high'], df['low'], df['close'])
        end_date = pd.to_datetime(date)
        if end_date not in df.index:
            return None
        end_idx = df.index.get_loc(end_date)
        start_idx = max(end_idx - days, 0)  # Ensure start_idx is not less than 0
        start_date = df.index[start_idx]

        df = df.loc[start_date:end_date]
        for idx in range(1, len(df)):
            k_today = df.iloc[idx]['K']
            d_today = df.iloc[idx]['D']
            j_today = df.iloc[idx]['J']
            k_yesterday = df.iloc[idx - 1]['K']
            d_yesterday = df.iloc[idx - 1]['D']
            j_yesterday = df.iloc[idx - 1]['J']
            # if df.iloc[idx]['symbol'] == '600066':
            #     print(df)
            #     print(f"Date: {df.index[idx].date()}, K_today: {k_today}, D_today: {d_today}, J_today: {j_today}, K_yesterday: {k_yesterday}, D_yesterday: {d_yesterday}, J_yesterday: {j_yesterday}")
            #     exit()
            if k_yesterday <= d_yesterday and j_yesterday <= d_yesterday and k_today > d_today and j_today > d_today and j_today > k_today:
                return df.index[idx].date()  # Return the date when K and J both crossed above D and J crossed above K
        return None

    def _calculate_kdj(self, high, low, close, period=9):
        low_min = low.rolling(window=period).min()
        high_max = high.rolling(window=period).max()

        rsv = (close - low_min) / (high_max - low_min) * 100
        K = rsv.ewm(alpha=1/3, adjust=False).mean()
        D = K.ewm(alpha=1/3, adjust=False).mean()
        J = 3 * K - 2 * D

        return K, D, J

def check_moving_average_conditions(self, date, days):
    expected_columns = ['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']
    df = pd.DataFrame(self._historical_data, columns=expected_columns)
    df = df[['trade_date', 'symbol', 'close']]  # Select only the necessary columns
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')  # Ensure correct datetime conversion
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)

    # Calculate moving averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA30'] = df['close'].rolling(window=30).mean()

    end_date = pd.to_datetime(date)
    if end_date not in df.index:
        return False
    end_idx = df.index.get_loc(end_date)
    start_idx = max(end_idx - days, 0)  # Ensure start_idx is not less than 0
    start_date = df.index[start_idx]

    df = df.loc[start_date:end_date]

    for idx in range(1, len(df)):
        if df.iloc[idx]['MA5'] <= df.iloc[idx]['MA10'] or df.iloc[idx]['MA10'] <= df.iloc[idx]['MA30']:
            return False

    return True

def check_diff_above_dea(self, date, days):
    expected_columns = ['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']
    df = pd.DataFrame(self._historical_data, columns=expected_columns)
    df = df[['trade_date', 'symbol', 'close']]  # Select only the necessary columns
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')  # Ensure correct datetime conversion
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)

    # Calculate MACD
    short_window = 12
    long_window = 26
    signal_window = 9

    df['EMA12'] = df['close'].ewm(span=short_window, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=long_window, adjust=False).mean()
    df['DIFF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIFF'].ewm(span=signal_window, adjust=False).mean()

    end_date = pd.to_datetime(date)
    if end_date not in df.index:
        return False
    end_idx = df.index.get_loc(end_date)
    start_idx = max(end_idx - days, 0)  # Ensure start_idx is not less than 0
    start_date = df.index[start_idx]

    df = df.loc[start_date:end_date]

    for idx in range(1, len(df)):
        if df.iloc[idx]['DIFF'] <= df.iloc[idx]['DEA']:
            return False

    return True

def check_price_above_bollinger_middle(self, date, days):
    expected_columns = ['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']
    df = pd.DataFrame(self._historical_data, columns=expected_columns)
    df = df[['trade_date', 'symbol', 'close']]  # Select only the necessary columns
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')  # Ensure correct datetime conversion
    df.set_index('trade_date', inplace=True)
    df.sort_index(inplace=True)

    # Calculate Bollinger Bands
    window = 20
    df['MA20'] = df['close'].rolling(window=window).mean()
    df['STD20'] = df['close'].rolling(window=window).std()
    df['UpperBand'] = df['MA20'] + (df['STD20'] * 2)
    df['LowerBand'] = df['MA20'] - (df['STD20'] * 2)

    end_date = pd.to_datetime(date)
    if end_date not in df.index:
        return False
    end_idx = df.index.get_loc(end_date)
    start_idx = max(end_idx - days, 0)  # Ensure start_idx is not less than 0
    start_date = df.index[start_idx]

    df = df.loc[start_date:end_date]

    for idx in range(len(df)):
        if df.iloc[idx]['close'] < df.iloc[idx]['MA20']:
            return False

    return True



# Example usage of the class
if __name__ == "__main__":
    tool = StockAnalysisTool("000001.SZ")
    tool.analyze()
    basic_info = tool.get_basic_info()
    print(basic_info)
    print("MACD Crossover on 2021-01-01:", tool.check_macd_crossover(pd.Timestamp('2021-01-01')))
    print("Recent MACD Crossover within 30 days before 2021-01-01:", tool.find_recent_macd_crossover('2021-01-01', 30))


