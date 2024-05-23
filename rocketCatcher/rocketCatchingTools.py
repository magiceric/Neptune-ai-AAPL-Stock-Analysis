import pandas as pd
from sqlalchemy import create_engine
import datetime
import akshare as ak
import ta
import os
import logging
from logging.handlers import TimedRotatingFileHandler

class RocketCatchingTools:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.setup_logging()
        self.engine = create_engine('mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock')
        self.historical_data = self.fetch_historical_data()
        self.update_with_realtime_data_if_trading()

    def __del__(self):
        """确保日志处理器和数据库连接关闭"""
        if hasattr(self, 'logger'):
            for handler in self.logger.handlers:
                handler.close()
                self.logger.removeHandler(handler)
        if hasattr(self, 'engine'):
            self.engine.dispose()  # 关闭数据库连接
            
    def setup_logging(self):
        """设置日志记录"""
        current_dir = os.path.dirname(__file__)
        logs_dir = os.path.join(current_dir, "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        log_filename = datetime.datetime.now().strftime("rocket_catching_tools_%Y%m%d.log")
        log_path = os.path.join(logs_dir, log_filename)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[TimedRotatingFileHandler(log_path, when='midnight', encoding='utf-8')])
        self.logger = logging.getLogger()
        self.logger.propagate = False  # Prevent logging from propagating to the root logger

    def fetch_historical_data(self):
        query = f"""
        SELECT trade_date, open, high, low, close, volume, turnover, amplitude, change_rate, change_amount, turnover_rate
        FROM historicaldata
        WHERE symbol = '{self.stock_code}'
        """
        data = pd.read_sql(query, self.engine)
        data['trade_date'] = data['trade_date'].astype(str)  # 将trade_date字段转换为字符串类型
        return data

    def update_with_realtime_data_if_trading(self):
        now = datetime.datetime.now()
        if now.weekday() < 5 and 9 <= now.hour < 15:  # 检查是否为工作日且在交易时间内
            try:
                real_time_data = ak.stock_zh_a_spot_em()  # 获取实时数据
                real_time_data = real_time_data[real_time_data['代码'] == self.stock_code]
                if not real_time_data.empty:
                    latest_data = pd.DataFrame({
                        'trade_date': [now.date().isoformat()],  # 将trade_date转换为ISO格式的字符串
                        'open': [real_time_data['今开'].values[0]],
                        'high': [real_time_data['最高'].values[0]],
                        'low': [real_time_data['最低'].values[0]],
                        'close': [real_time_data['最新价'].values[0]],
                        'volume': [real_time_data['成交量'].values[0]],
                        'turnover': [real_time_data['成交额'].values[0]],
                        'amplitude': [real_time_data['振幅'].values[0]],
                        'change_rate': [real_time_data['涨跌幅'].values[0]],
                        'change_amount': [real_time_data['涨跌额'].values[0]],
                        'turnover_rate': [real_time_data['换手率'].values[0]]
                    })
                    self.historical_data = pd.concat([self.historical_data, latest_data], ignore_index=True)
                    self.logger.info(self.historical_data)
            except Exception as e:
                self.logger.error(f"获取或更新实时数据失败: {e}")

    def is_stock_meeting_selection_criteria(self, reference_date, n_days):
        """
        判断股票是否符合选股特征，基于以下四个条件：
        1. 在指定日期之前n日内，5天均线从下方与10天均线重合或上穿，并在上穿后保持在10天均线上方或保持重合。
        2. 在指定日期之前n日内，出现0线上方的MACD金叉。
        3. 在指定日期之前n日内，股价从下方上穿（或重合）布林线中轨线。
        4. 在指定日期之前n日内，出现KDJ低位上穿（或重合），即在上穿（或重合）点之前，J<K<D，在上穿（或重合）点之后，J>=K>=D，并保持，且J<=50。
        
        参数:
        reference_date: 指定的日期，格式为 'YYYY-MM-DD'
        n_days: 指定的天数范围
        
        返回:
        bool: 如果所有条件都满足，则返回True，否则返回False
        """
        ma_crossover = self.check_ma_crossover_within_days(reference_date, n_days)
        self.logger.info(f"5天均线与10天均线的交叉验证结果: {ma_crossover}\n")
        
        macd_crossover = self.check_macd_crossover_above_zero_within_days(reference_date, n_days)
        self.logger.info(f"MACD金叉在0线上方的验证结果: {macd_crossover}\n")
        
        bollinger_crossover = self.check_price_cross_bollinger_mid_within_days(reference_date, n_days)
        self.logger.info(f"股价与布林线中轨线的交叉验证结果: {bollinger_crossover}\n")
        
        kdj_crossover = self.check_kdj_cross_within_days(reference_date, n_days)
        self.logger.info(f"KDJ低位交叉的验证结果: {kdj_crossover}\n")

        all_criteria_met = ma_crossover and macd_crossover and bollinger_crossover and kdj_crossover
        self.logger.info(f"\n所有选股条件是否满足: {all_criteria_met}")
        return all_criteria_met

    def check_ma_crossover_within_days(self, reference_date, n_days):
        """
        判断在指定日期之前n日内，是否存在5天均线从下方与10天均线重合或上穿的情况，并在上穿后保持在10天均线上方或保持重合。
        参数:
        reference_date: 指定的日期，格式为 'YYYY-MM-DD'
        n_days: 指定的天数范围
        
        返回:
        bool: 如果在指定日期之前n日内满足条件，则返回True，否则返回False
        """
        # 确保historical_data已经包含了均线数据，如果没有则计算
        if 'MA5' not in self.historical_data.columns or 'MA10' not in self.historical_data.columns:
            self.historical_data['MA5'] = self.historical_data['close'].rolling(window=5).mean().round(2)
            self.historical_data['MA10'] = self.historical_data['close'].rolling(window=10).mean().round(2)

        # 找到参考日期在数据集中的位置
        try:
            reference_idx = self.historical_data[self.historical_data['trade_date'] == reference_date].index[0]
        except IndexError:
            self.logger.error(f"指定参考日期的数据不存在: {reference_date}")
            return False

        # 计算从参考日期向前n天的数据范围
        start_idx = max(0, reference_idx - n_days + 1)
        relevant_data = self.historical_data.iloc[start_idx:reference_idx + 1]

        # 检查5日均线是否在10日均线之上或重合，并且之前是在10日均线之下
        crossed = False
        for i in range(1, len(relevant_data)):
            if (relevant_data.iloc[i]['MA5'] >= relevant_data.iloc[i]['MA10'] and
                relevant_data.iloc[i-1]['MA5'] < relevant_data.iloc[i-1]['MA10']):
                crossed = True
            # 一旦上穿，检查之后是否一直在10日均线之上或重合
            if crossed:
                if relevant_data.iloc[i]['MA5'] < relevant_data.iloc[i]['MA10']:
                    crossed = False
                    break

        self.logger.info(f"参与MA交叉判断的数据: \n{relevant_data[['trade_date', 'MA5', 'MA10']]}")
        return crossed
    
    def check_macd_crossover_above_zero_within_days(self, reference_date, n_days):
        """
        判断在指定日期之前n日内，是否存在MACD金叉且金叉点MACD值在0线上方的情况。
        参数:
        reference_date: 指定的日期，格式为 'YYYY-MM-DD'
        n_days: 指定的天数范围
        
        返回:
        bool: 如果在指定日期之前n日内满足条件，则返回True，否则返回False
        """
        # 确保historical_data已经包含了MACD数据，如果没有则计算
        if 'DIFF' not in self.historical_data.columns or 'DEA' not in self.historical_data.columns:
            macd = ta.trend.MACD(self.historical_data['close'], window_slow=26, window_fast=12, window_sign=9)
            self.historical_data['DIFF'] = macd.macd().round(3)
            self.historical_data['DEA'] = macd.macd_signal().round(3)
            self.historical_data['MACD'] = (self.historical_data['DIFF'] - self.historical_data['DEA']).round(3)

        # 找到参考日期在数据集中的位置
        try:
            reference_idx = self.historical_data[self.historical_data['trade_date'] == reference_date].index[0]
        except IndexError:
            self.logger.error(f"指定参考日期的数据不存在: {reference_date}")
            return False

        # 计算从参考日期向前n天的数据范围
        start_idx = max(0, reference_idx - n_days + 1)
        relevant_data = self.historical_data.iloc[start_idx:reference_idx + 1]

        # 记录相关数据
        self.logger.info(f"参与MACD金叉判断的数据: \n{relevant_data[['trade_date', 'DIFF', 'DEA', 'MACD']]}")

        # 检查是否存在MACD金叉且金叉点MACD值在0线上方
        for i in range(1, len(relevant_data)):
            if (relevant_data.iloc[i]['DIFF'] > relevant_data.iloc[i]['DEA'] and
                relevant_data.iloc[i-1]['DIFF'] <= relevant_data.iloc[i-1]['DEA'] and
                relevant_data.iloc[i]['MACD'] > 0):
                return True

        return False

    def check_price_cross_bollinger_mid_within_days(self, reference_date, n_days):
        """
        判断在指定日期之前n日内，股价是否从下方上穿（或重合）布林线中轨线的情况。
        参数:
        reference_date: 指定的日期，格式为 'YYYY-MM-DD'
        n_days: 指定的天数范围
        
        返回:
        bool: 如果在指定日期之前n日内股价从下方上穿（或重合）布林线中轨线，则返回True，否则返回False
        """
        # 确保historical_data已经包含了布林线中轨数据，如果没有则计算
        if 'bollinger_mid' not in self.historical_data.columns:
            bollinger = ta.volatility.BollingerBands(self.historical_data['close'], window=20, window_dev=2)
            self.historical_data['bollinger_mid'] = bollinger.bollinger_mavg().round(2)

        # 找到参考日期在数据集中的位置
        try:
            reference_idx = self.historical_data[self.historical_data['trade_date'] == reference_date].index[0]
        except IndexError:
            self.logger.error(f"指定参考日期的数据不存在: {reference_date}")
            return False

        # 计算从参考日期向前n天的数据范围
        start_idx = max(0, reference_idx - n_days + 1)
        relevant_data = self.historical_data.iloc[start_idx:reference_idx + 1]

        # 记录相关数据
        self.logger.info(f"参与布林线中轨交叉判断的数据: \n{relevant_data[['trade_date', 'close', 'bollinger_mid']]}")

        # 检查是否存在股价从下方上穿（或重合）布林线中轨线的情况
        for i in range(1, len(relevant_data)):
            if (relevant_data.iloc[i]['close'] >= relevant_data.iloc[i]['bollinger_mid'] and
                relevant_data.iloc[i-1]['close'] < relevant_data.iloc[i-1]['bollinger_mid']):
                return True

        return False
    
    def check_kdj_cross_within_days(self, reference_date, n_days):
        """
        判断在指定日期之前n日内，是否出现KDJ低位上穿（或重合）的情况。
        低位定义为J值<=50。
        上穿（或重合）点之前，J<K<D；上穿（或重合）点之后，J>=K>=D，并保持。
        参数:
        reference_date: 指定的日期，格式为 'YYYY-MM-DD'
        n_days: 指定的天数范围
        
        返回:
        bool: 如果在指定日期之前n日内出现KDJ低位上穿（或重合），则返回True，否则返回False
        """
        # 自定义KDJ计算（同花顺公式）
        def calculate_kdj(data, n1=9, n2=3, n3=3):
            low_list = data['low'].rolling(window=n1, min_periods=1).min()
            high_list = data['high'].rolling(window=n1, min_periods=1).max()
            rsv = (data['close'] - low_list) / (high_list - low_list) * 100
            K = rsv.ewm(com=n2-1, adjust=False).mean()
            D = K.ewm(com=n3-1, adjust=False).mean()
            J = 3 * K - 2 * D
            return K, D, J

        # 确保historical_data已经包含了KDJ指标数据，如果没有则计算
        if 'K' not in self.historical_data.columns or 'D' not in self.historical_data.columns or 'J' not in self.historical_data.columns:
            self.historical_data['K'], self.historical_data['D'], self.historical_data['J'] = calculate_kdj(self.historical_data)

        # 找到参考日期在数据集中的位置
        try:
            reference_idx = self.historical_data[self.historical_data['trade_date'] == reference_date].index[0]
        except IndexError:
            self.logger.error(f"指定参考日期的数据不存在: {reference_date}")
            return False

        # 计算从参考日期向前n天的数据范围
        start_idx = max(0, reference_idx - n_days + 1)
        relevant_data = self.historical_data.iloc[start_idx:reference_idx + 1]

        # 记录相关数据
        self.logger.info(f"参与KDJ低位交叉判断的数据: \n{relevant_data[['trade_date', 'J', 'K', 'D']]}")

        # 检查是否存在KDJ低位上穿（或重合）的情况
        for i in range(1, len(relevant_data)):
            if (relevant_data.iloc[i]['J'] <= 50 and
                relevant_data.iloc[i-1]['J'] < relevant_data.iloc[i-1]['K'] < relevant_data.iloc[i-1]['D'] and
                relevant_data.iloc[i]['J'] >= relevant_data.iloc[i]['K'] >= relevant_data.iloc[i]['D']):
                return True

        return False

if __name__ == "__main__":
    stock_code = '000620'  # 测试股票代码
    rc_tools = RocketCatchingTools(stock_code)
    result = rc_tools.is_stock_meeting_selection_criteria('2024-05-17', 5)
    print(f"选股标准是否满足: {result}")


