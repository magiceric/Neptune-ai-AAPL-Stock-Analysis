import akshare as ak
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.exc import ProgrammingError
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config import tushare_api_key

class StockHistoryFetcher:

    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock'
        self.engine = create_engine(self.database_connection_string)
        self.ensure_tables_exist()

    def __del__(self):
        """
        释放数据库连接等资源。
        """
        if self.engine:
            self.engine.dispose()
            
    def ensure_tables_exist(self):
        """
        确保数据库中存在所需的表，如果不存在则创建。
        """
        historicaldata_table_creation_query = """
        CREATE TABLE IF NOT EXISTS historicaldata (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            trade_date DATE NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume BIGINT,
            turnover FLOAT,
            amplitude FLOAT,
            change_rate FLOAT,
            change_amount FLOAT,
            turnover_rate FLOAT,
            UNIQUE KEY `unique_index` (`symbol`, `trade_date`)
        );
        """
        historicaldata_index_table_creation_query = """
        CREATE TABLE IF NOT EXISTS historicaldata_index (
            id INT AUTO_INCREMENT PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            trade_date DATE NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume BIGINT,
            turnover FLOAT,
            amplitude FLOAT,
            change_rate FLOAT,
            change_amount FLOAT,
            turnover_rate FLOAT,
            UNIQUE KEY `unique_index` (`symbol`, `trade_date`)
        );
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(text(historicaldata_table_creation_query))
                connection.execute(text(historicaldata_index_table_creation_query))
                connection.close()
        except Exception as e:
            print(f"创建表失败: {e}")

    def fetch_history(self):
        """
        使用akshare获取历史股票和ETF数据并写入数据库。
        """
        try:
            # 检查数据库中该股票的最新日期
            with self.engine.connect() as connection:
                latest_date_query = "SELECT MAX(trade_date) FROM historicaldata WHERE symbol = :stock_code"
                result = connection.execute(text(latest_date_query), {'stock_code': self.stock_code})
                latest_date = result.fetchone()[0]
                print("最新日期 : ", latest_date)
                if latest_date is not None:
                    latest_date = pd.to_datetime(latest_date).strftime('%Y%m%d')
                else:
                    # 如果latest_date为None，设置start_date为一个早期日期以获取全部历史数据
                    latest_date = '19900101'
            
                # 根据股票代码前缀判断是否为ETF，并从akshare获取相应的历史数据
                if self.stock_code.startswith('5') or self.stock_code.startswith('1'):
                    # ETF股票代码通常以5或1开头
                    market_prefix = "sh" if self.stock_code[0] in ["5"] else "sz"
                    stock_zh_a_hist_df = ak.fund_etf_hist_sina(symbol=market_prefix + self.stock_code, adjust="qfq")
                    # 筛选出最新日期之后的数据，避免数据库中的重复数据
                    # 确保日期字段是日期格式以进行比较
                    stock_zh_a_hist_df.rename(columns={'date': 'trade_date'}, inplace=True)
                    stock_zh_a_hist_df['trade_date'] = pd.to_datetime(stock_zh_a_hist_df['trade_date'])
                    stock_zh_a_hist_df = stock_zh_a_hist_df[stock_zh_a_hist_df['trade_date'] >= pd.to_datetime(latest_date)]
                    # 补全缺失的字段
                    stock_zh_a_hist_df['turnover'] = None
                    stock_zh_a_hist_df['amplitude'] = None
                    stock_zh_a_hist_df['change_rate'] = None
                    stock_zh_a_hist_df['change_amount'] = None
                    stock_zh_a_hist_df['turnover_rate'] = None
                else:
                    # 获取前复权数据
                    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=self.stock_code, adjust="qfq", start_date=latest_date)

                if not stock_zh_a_hist_df.empty:
                    print(f"成功获取{self.stock_code}的历史数据")
                    self.write_to_db(stock_zh_a_hist_df)
                else:
                    print(f"{self.stock_code}没有新数据可获取")
                connection.close()
        except Exception as e:
            print(f"获取{self.stock_code}的历史数据失败: {e}")

    def load_history(self, start_date=None, end_date=None):
        """
        从数据库表获取指定日期范围内的历史股票数据，并返回获取到的数据。
        """
        historical_data_df = pd.DataFrame()  # 初始化空的DataFrame
        try:
            # 如果未指定开始日期或结束日期，则查询数据库中该股票的最新日期
            if start_date is None or end_date is None:
                with self.engine.connect() as connection:
                    latest_date_query = "SELECT MAX(trade_date) FROM historicaldata WHERE symbol = :stock_code"
                    result = connection.execute(text(latest_date_query), {'stock_code': self.stock_code})
                    latest_date = result.fetchone()[0]
                    print("最新日期 : ", latest_date)
                    if latest_date is not None:
                        latest_date = pd.to_datetime(latest_date).strftime('%Y%m%d')
                    else:
                        # 如果latest_date为None，设置start_date为一个早期日期以获取全部历史数据
                        latest_date = '19900101'
                    start_date = latest_date
                    end_date = pd.to_datetime('today').strftime('%Y%m%d')

            # 从数据库获取指定日期范围内的历史数据
            with self.engine.connect() as connection:
                historical_data_query = f"SELECT * FROM historicaldata WHERE symbol = :stock_code AND trade_date BETWEEN :start_date AND :end_date"
                result = connection.execute(text(historical_data_query), {'stock_code': self.stock_code, 'start_date': start_date, 'end_date': end_date})
                historical_data_df = pd.DataFrame(result.fetchall())
                historical_data_df.columns = result.keys()
                connection.close()

            if not historical_data_df.empty:
                print(f"成功从数据库获取{self.stock_code}从{start_date}到{end_date}的历史数据")
            else:
                print(f"{self.stock_code}在指定日期范围内没有新数据可获取")
        except Exception as e:
            print(f"获取{self.stock_code}的历史数据失败: {e}")
        return historical_data_df

    def fetch_all_stock_codes(self):

        """
        先更新
        """
        # Initialize Tushare API
        ts.set_token(tushare_api_key)
        pro = ts.pro_api()

        # Fetch all stock basic information
        stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')

        # Fetch all ETF information
        # etf_basic = pro.fund_basic(market='E', status='L', fields='ts_code,symbol,name')
        etf_basic = pro.fund_basic(market='E', status='L')
        # Remove the suffix from ts_code to get the symbol
        etf_basic['symbol'] = etf_basic['ts_code'].str.split('.').str[0]

        # Filter ETFs that start with 0 or 6 but exclude those starting with 688 or 689
        filtered_etf_data = etf_basic[(etf_basic['symbol'].str.startswith('0')) | 
                                    (etf_basic['symbol'].str.startswith('6'))]
        filtered_etf_data = filtered_etf_data[~filtered_etf_data['symbol'].str.startswith(('688', '689'))]

        filtered_data = pd.concat([filtered_etf_data, etf_basic])

        # Database connection
        engine = create_engine('mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock')

        # Insert data into the database using INSERT ... ON DUPLICATE KEY UPDATE
        with engine.connect() as connection:
            for index, row in filtered_data.iterrows():
                upsert_query = text("""
                    INSERT INTO stockBasic (tsCode, symbol, stockName)
                    VALUES (:tsCode, :symbol, :stockName)
                    ON DUPLICATE KEY UPDATE
                    symbol = VALUES(symbol), stockName = VALUES(stockName)
                """)
                connection.execute(upsert_query, {
                    'tsCode': row['ts_code'],
                    'symbol': row['symbol'],
                    'stockName': row['name']
                })
            connection.commit()

        """
        从数据库表stockBasic获取并返回所有的证券代码。
        """
        try:
            with self.engine.connect() as connection:
                # query = text("SELECT symbol FROM stockBasic WHERE stockName LIKE '%ETF%'")
                query = text("SELECT symbol FROM stockBasic")
                result = connection.execute(query)
                stock_info_df = pd.DataFrame(result.fetchall(), columns=['symbol'])
                print("成功从数据库获取所有证券代码")
                # print(stock_info_df)
                return stock_info_df
        except Exception as e:
            print(f"从数据库获取证券代码失败: {e}")
            return pd.DataFrame()
        
    def fetch_index_history(self):
        """
        获取上海A股指数和深圳A股指数的历史数据并写入数据库。
        """
        index_codes = {'上海A股': '000001', '深圳A股': '399001'}
        for index_name, index_code in index_codes.items():
            try:
                # 检查数据库中该指数的最新日期
                with self.engine.connect() as connection:
                    latest_date_query = f"SELECT MAX(trade_date) FROM historicaldata_index WHERE symbol = :index_code"
                    result = connection.execute(text(latest_date_query), {'index_code': index_code})
                    latest_date = result.fetchone()[0]
                    print(f"{index_name}最新日期 : ", latest_date)
                    if latest_date is not None:
                        latest_date = pd.to_datetime(latest_date).strftime('%Y%m%d')
                    else:
                        # 如果latest_date为None，设置start_date为一个早期日期以获取全部历史数据
                        latest_date = '19900101'
                
                    # 从akshare获取历史数据
                    index_hist_df = ak.index_zh_a_hist(symbol=index_code, start_date=latest_date)
                    if not index_hist_df.empty:
                        print(f"成功获取{index_name}的历史数据")
                        self.write_to_db_index(index_hist_df, index_code)
                    else:
                        print(f"{index_name}没有新数据可获取")
                    connection.close()
            except Exception as e:
                print(f"获取{index_name}的历史数据失败: {e}")

    def write_to_db(self, df):
        """
        将获取的数据写入数据库。
        """
        try:
            # 将DataFrame列名转换为数据库表的英文字段名
            # print(df)
            df = df.rename(columns={
                '日期': 'trade_date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'turnover',
                '振幅': 'amplitude',
                '涨跌幅': 'change_rate',
                '涨跌额': 'change_amount',
                '换手率': 'turnover_rate',
            })
            df.drop(columns=['股票代码'], inplace=True)
            # 添加股票代码字段，并确保它排在第二列
            df.insert(1, 'symbol', self.stock_code)
            with self.engine.connect() as connection:
                # 检查数据库中是否已存在该股票数据
                check_query = "SELECT COUNT(*) FROM historicaldata WHERE symbol = :stock_code"
                result = connection.execute(text(check_query), {'stock_code': self.stock_code}).fetchone()
                if result[0] == 0:
                    # 如果数据库中还不存在该股票数据，使用批量插入
                    df.to_sql('historicaldata', con=self.engine, if_exists='append', index=False, method='multi')
                    print(f"成功将{self.stock_code}的历史数据批量写入数据库，新增数据条数：{len(df)}")
                else:
                    # 如果数据库中已存在该股票数据，逐条更新或插入
                    for index, row in df.iterrows():
                        # 尝试更新
                        update_result = connection.execute(text("""
                        UPDATE historicaldata
                        SET open = :open, high = :high, low = :low, close = :close, volume = :volume, turnover = :turnover, amplitude = :amplitude, change_rate = :change_rate, change_amount = :change_amount, turnover_rate = :turnover_rate
                        WHERE trade_date = :trade_date AND symbol = :symbol;
                        """), row.to_dict())
                        
                        # 如果没有更新到数据，则尝试插入
                        if update_result.rowcount == 0:
                            connection.execute(text("""
                            INSERT INTO historicaldata (trade_date, symbol, open, high, low, close, volume, turnover, amplitude, change_rate, change_amount, turnover_rate)
                            VALUES (:trade_date, :symbol, :open, :high, :low, :close, :volume, :turnover, :amplitude, :change_rate, :change_amount, :turnover_rate)
                            """), row.to_dict())
                        # 提交更新或插入操作
                        connection.commit()
                        # print(f"处理数据：{row.to_dict()}")
                    print(f"成功将{self.stock_code}的历史数据逐条更新或插入到数据库，处理数据条数：{len(df)}")
                connection.close()
        except Exception as e:
            print(f"将{self.stock_code}的历史数据写入数据库失败: {e}")

    def write_to_db_index(self, df, index_code):
        """
        将指数数据写入数据库的historicaldata_index表中。
        """
        try:
            df = df.rename(columns={
                '日期': 'trade_date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'turnover',
                '振幅': 'amplitude',
                '涨跌幅': 'change_rate',
                '涨跌额': 'change_amount',
                '换手率': 'turnover_rate',
            })
            # 添加指数代码字段，并确保它排在第二列
            df.insert(1, 'symbol', index_code)
            with self.engine.connect() as connection:
                # 检查数据库中是否已存在该指数数据
                check_query = "SELECT COUNT(*) FROM historicaldata_index WHERE symbol = :index_code"
                result = connection.execute(text(check_query), {'index_code': index_code}).fetchone()
                if result[0] == 0:
                    # 如果数据库中还不存在该指数数据，使用批量插入
                    df.to_sql('historicaldata_index', con=self.engine, if_exists='append', index=False, method='multi')
                    print(f"成功将{index_code}的指数数据批量写入数据库，新增数据条数：{len(df)}")
                else:
                    # 如果数据库中已存在该指数数据，逐条更新或插入
                    for index, row in df.iterrows():
                        # 尝试更新
                        update_result = connection.execute(text("""
                        UPDATE historicaldata_index
                        SET open = :open, high = :high, low = :low, close = :close, volume = :volume, turnover = :turnover, amplitude = :amplitude, change_rate = :change_rate, change_amount = :change_amount, turnover_rate = :turnover_rate
                        WHERE trade_date = :trade_date AND symbol = :symbol;
                        """), row.to_dict())
                        
                        # 如果没有更新到数据，则尝试插入
                        if update_result.rowcount == 0:
                            connection.execute(text("""
                            INSERT INTO historicaldata_index (trade_date, symbol, open, high, low, close, volume, turnover, amplitude, change_rate, change_amount, turnover_rate)
                            VALUES (:trade_date, :symbol, :open, :high, :low, :close, :volume, :turnover, :amplitude, :change_rate, :change_amount, :turnover_rate)
                            """), row.to_dict())
                        # 提交更新或插入操作
                        connection.commit()
                        print(f"处理指数数据：{row.to_dict()}")
                    print(f"成功将{index_code}的指数数据逐条更新或插入到数据库，处理数据条数：{len(df)}")
                connection.close()
        except Exception as e:
            print(f"将{index_code}的指数数据写入数据库失败: {e}")


    def find_first_limit_up_date(self):
        """
        从当前日期往回找，找到第一个出现了涨停板的日期，返回这个日期值。
        """
        try:
            with self.engine.connect() as connection:
                query = """
                SELECT trade_date FROM historicaldata
                WHERE symbol = :stock_code AND change_rate >= 9.9
                ORDER BY trade_date DESC
                LIMIT 1;
                """
                result = connection.execute(text(query), {'stock_code': self.stock_code}).fetchone()
                if result:
                    print(f"找到涨停板日期: {result[0]}")
                    return result[0]
                else:
                    print(f"{self.stock_code} 没有找到涨停板记录。")
                    return None
                connection.close()
        except Exception as e:
            print(f"查找{self.stock_code}的涨停板日期失败: {e}")
            return None

    def find_limit_up_stocks_on_date(self, target_date):
        """
        获取指定日期所有涨停的股票，包括股票代码和涨幅。
        
        参数:
        target_date (str): 指定的日期，格式应为'YYYY-MM-DD'
        
        返回:
        list of tuples: 每个元组包含股票代码和涨幅，按涨幅降序排列
        """
        try:
            with self.engine.connect() as connection:
                query = """
                SELECT symbol, change_rate
                FROM historicaldata
                WHERE trade_date = :trade_date AND change_rate >= 9.9
                ORDER BY change_rate DESC;
                """
                result = connection.execute(text(query), {'trade_date': target_date}).fetchall()
                if result:
                    # 四舍五入涨幅到小数点后两位
                    formatted_result = [(row.symbol, round(row.change_rate, 2)) for row in result]
                    # print(f"在{target_date}找到涨停股票: {formatted_result}")
                    return formatted_result
                else:
                    print(f"{target_date} 没有找到涨停股票。")
                    return []
        except Exception as e:
            print(f"查询{target_date}的涨停股票失败: {e}")
            return []

if __name__ == "__main__":
    # 示例用法
    fetcher = StockHistoryFetcher("159972")
    # fetcher.fetch_index_history()
    fetcher.fetch_history()
