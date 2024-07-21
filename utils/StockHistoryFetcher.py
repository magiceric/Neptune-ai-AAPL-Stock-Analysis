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
            outstanding_share DOUBLE,
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
            with self.engine.connect() as connection:
                latest_date_query = "SELECT MAX(trade_date) FROM historicaldata WHERE symbol = :stock_code"
                result = connection.execute(text(latest_date_query), {'stock_code': self.stock_code})
                latest_date = result.fetchone()[0]
                print("最新日期 : ", latest_date)
                if latest_date is not None:
                    latest_date = pd.to_datetime(latest_date).strftime('%Y%m%d')
                else:
                    latest_date = '19900101'
                print(latest_date)

                hist_df = None
                if self.stock_code.startswith('5') or self.stock_code.startswith('1'):
                    prefix = 'sh' if self.stock_code.startswith('5') else 'sz'
                    full_symbol = f"{prefix}{self.stock_code}"
                    hist_df = ak.fund_etf_hist_sina(full_symbol)
                    print(full_symbol)
                else:
                    prefix = 'sh' if self.stock_code.startswith('6') else 'sz'
                    full_symbol = f"{prefix}{self.stock_code}"
                    hist_df = ak.stock_zh_a_daily(symbol=full_symbol, start_date=latest_date, adjust="qfq")

                hist_df = hist_df.rename(columns={
                    'date': 'trade_date',
                    'amount': 'turnover',
                    'turnover': 'turnover_rate'
                })
                hist_df['amplitude'] = hist_df['high'] - hist_df['low']
                hist_df['change_rate'] = (hist_df['close'] - hist_df['open']) / hist_df['open']
                hist_df['change_amount'] = hist_df['close'] - hist_df['open']
                hist_df['turnover_rate'] = None
                required_columns = ['turnover', 'outstanding_share', 'turnover_rate']
                for column in required_columns:
                    if column not in hist_df.columns:
                        hist_df[column] = None

                if not hist_df.empty:
                    print(f"成功获取{self.stock_code}的历史数据")
                    self.write_to_db(hist_df)
                else:
                    print(f"{self.stock_code}没有新数据可获取")
                connection.close()

        except Exception as e:
            print(f"{str(e)} 正在处理的证券代码：{full_symbol}")

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

    def fetch_all_stock_basic(self):
        """
        获取所有股票代码及其总股本信息并更新数据库，处理完成后发送邮件通知。
        """
        # 初始化Tushare API
        ts.set_token(tushare_api_key)
        pro = ts.pro_api()

        # 获取所有股票基本信息
        stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')

        # 获取所有ETF信息
        etf_basic = pro.fund_basic(market='E', status='L', fields='ts_code,name')
        etf_basic['totalShare'] = 0
        # 从ts_code中移除后缀获取symbol
        etf_basic['symbol'] = etf_basic['ts_code'].str.split('.').str[0]

        filtered_data = pd.concat([etf_basic, stock_basic])

        # 筛选以0、1、5、6开头的，排除以688或689开头的
        filtered_data = filtered_data[(filtered_data['symbol'].str.startswith('0')) |
                                      (filtered_data['symbol'].str.startswith('1')) |
                                      (filtered_data['symbol'].str.startswith('5')) |
                                      (filtered_data['symbol'].str.startswith('6'))]
        filtered_data = filtered_data[~filtered_data['symbol'].str.startswith(('688', '689'))]
        # 移除名称中包含'LOF'和'ST'的条目
        filtered_data = filtered_data[~filtered_data['name'].str.contains('LOF')]
        filtered_data = filtered_data[~filtered_data['name'].str.contains('ST')]

        # 获取ETF的最新份额信息
        etf_spot_em_df = ak.fund_etf_spot_em()
        print(etf_spot_em_df)

        # 更新filtered_data中的ETF品种的totalShare字段
        for index, row in filtered_data.iterrows():
            if row['symbol'].startswith(('1', '5')):
                latest_shares = etf_spot_em_df.loc[etf_spot_em_df['代码'] == row['symbol'], '最新份额']
                if not latest_shares.empty:
                    row['totalShare'] = latest_shares.values[0]  # 保持字段名为 totalShare
                else:
                    row['totalShare'] = 0  # 保持字段名为 totalShare
            else:
                # 从akshare获取总市值
                stock_individual_info_em_df = ak.stock_individual_info_em(symbol=row['symbol'])
                total_shares = stock_individual_info_em_df[stock_individual_info_em_df["item"] == "总市值"]["value"]
                row['totalShare'] = total_shares.values[0] if not total_shares.empty else None  # 保持字段名为 totalShare
            # print(row['symbol'], row['totalShare'])  # 保持字段名为 totalShare

        # 数据库连接
        engine = create_engine('mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock')

        # 使用INSERT ... ON DUPLICATE KEY UPDATE语句插入数据，并在totalShare为0或空时保持原值
        with engine.connect() as connection:
            for index, row in filtered_data.iterrows():
                if row['symbol'].startswith(('1', '5')):
                    row['totalShare'] = 0
                else:
                    # 从akshare获取总股本
                    stock_individual_info_em_df = ak.stock_individual_info_em(symbol=row['symbol'])
                    total_shares = stock_individual_info_em_df[stock_individual_info_em_df["item"] == "总股本"]["value"]
                    row['totalShare'] = total_shares.values[0] if not total_shares.empty else None

                print(row['symbol'], row['totalShare'])
                if row['totalShare'] is None or row['totalShare'] == 0:
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
                else:
                    upsert_query = text("""
                        INSERT INTO stockBasic (tsCode, symbol, stockName, totalShare)
                        VALUES (:tsCode, :symbol, :stockName, :totalShare)
                        ON DUPLICATE KEY UPDATE
                        symbol = VALUES(symbol), stockName = VALUES(stockName), totalShare = VALUES(totalShare)
                    """)
                    connection.execute(upsert_query, {
                        'tsCode': row['ts_code'],
                        'symbol': row['symbol'],
                        'stockName': row['name'],
                        'totalShare': row['totalShare']
                    })
            connection.commit()

        # 从stockBasic表获取所有证券代码
        try:
            with self.engine.connect() as connection:
                query = text("SELECT symbol FROM stockBasic")
                result = connection.execute(query)
                stock_info_df = pd.DataFrame(result.fetchall(), columns=['symbol'])
                print("成功从数据库获取所有证券代码")

                # 发送邮件通知
                from utils.sendEmail import EmailSender
                email_sender = EmailSender()
                subject = "股票代码更新通知"
                message = "所有股票代码及总股本信息已成功更新到数据库。"
                to_addr = "745339023@qq.com"
                recipient_name = "Mr.Light"
                try:
                    email_sender.send_email(subject, message, to_addr, recipient_name)
                except Exception as e:
                    print(f"邮件发送失败: {e}")
                    subject_failure = "股票代码更新通知失败"
                    message_failure = "尝试更新数据库中的股票代码及总股本信息并发送邮件通知时失败。"
                    email_sender.send_email(subject_failure, message_failure, to_addr, recipient_name)

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

    def write_to_db(self, df, batch_size=1000):
        """
        将获取的数据写入数据库。
        """
        # try:
        # 将DataFrame列名转换为数据库表的英文字段名
        # print(df)
        # 添加股票代码字段，并确保它排在第二列
        df.insert(1, 'symbol', self.stock_code)
        with self.engine.connect() as connection:
            # 检查数据库中是否已存在该股票数据
            check_query = "SELECT COUNT(*) FROM historicaldata WHERE symbol = :stock_code"
            result = connection.execute(text(check_query), {'stock_code': self.stock_code}).fetchone()
            if result[0] == 0:
                # 保留数据库需要的字段
                df = df[['trade_date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'outstanding_share', 'turnover_rate']]
                # print(df)
                # 如果数据库中还不存在该股票数据，使用批量插入
                for start in range(0, len(df), batch_size):
                    end = start + batch_size
                    batch_df = df.iloc[start:end]
                    batch_df.to_sql('historicaldata', con=self.engine, if_exists='append', index=False, method='multi')
                    # print(f"成功将{self.stock_code}的历史数据批量写入数据库，新增数据条数：{len(batch_df)}")
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
        # except Exception as e:
        #     print(f"将{self.stock_code}的历史数据写入数据库失败: {e}")

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

    def delete_today_data(self):
        """
        删除historicaldata表中当前日期的所有数据。
        
        —— 因为当天的数据可能来自实时行情接口，主要用于盘中测算，有些数据项会缺失（比如换手率）
        —— 每次更新整体历史数据前先清一下

        —— 如果在晚上12点前更新整体数据就用这个，如果是凌晨应该用下面的
        """
        import datetime
        today_date = datetime.datetime.now().strftime('%Y-%m-%d')
        try:
            with self.engine.connect() as connection:
                # print(today_date)
                delete_query = """
                DELETE FROM historicaldata
                WHERE trade_date = :trade_date;
                """
                connection.execute(text(delete_query), {'trade_date': today_date})
                print(f"已删除{today_date}的所有数据。")
                connection.commit()
                connection.close()
        except Exception as e:
            print(f"删除{today_date}的数据失败: {e}")

    def delete_previous_trading_day_data(self):
        """
        删除historicaldata表中上一个交易日的所有数据。
        
        —— 使用最大交易日期来确定上一个交易日
        —— 删除这个日期及之后的所有数据
        """
        from utils.sendEmail import EmailSender  # 导入邮件发送模块
        email_sender = EmailSender()  # 创建邮件发送对象
        success_message = "成功删除上一个交易日及之后的所有数据。"
        error_message = "删除上一个交易日的数据时发生错误。"
        try:
            with self.engine.connect() as connection:
                # 获取上一个交易日的日期
                max_date_query = """
                SELECT MAX(trade_date) AS last_trading_day
                FROM historicaldata;
                """
                last_trading_day = connection.execute(text(max_date_query)).fetchone()[0]
                
                if last_trading_day:
                    # 删除上一个交易日及之后的所有数据
                    delete_query = """
                    DELETE FROM historicaldata
                    WHERE trade_date >= :last_trading_day;
                    """
                    connection.execute(text(delete_query), {'last_trading_day': last_trading_day})
                    connection.commit()
                    print(f"已删除{last_trading_day}及之后的所有数据。")
                    email_sender.send_email("删除数据通知", f"{success_message} 删除日期：{last_trading_day}", "745339023@qq.com", "Mr.Light")
                else:
                    print("没有找到上一个交易日的数据。")
                    email_sender.send_email("删除数据通知", "没有找到上一个交易日的数据。", "745339023@qq.com", "Mr.Light")
        except Exception as e:
            print(f"删除上一个交易日的数据失败: {e}")
            email_sender.send_email("删除数据错误", f"{error_message} {str(e)}", "745339023@qq.com", "Mr.Light")

    def update_realtime_etf_data(self):
        """
        一次性更新所有ETF当天实时数据。
        """
        import akshare as ak
        import datetime
        today_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_time = datetime.datetime.now()  # 记录开始时间

        from utils.sendEmail import EmailSender  # 导入邮件发送模块
        email_sender = EmailSender()  # 创建邮件发送对象
        success_message = "成功更新所有ETF基金的实时数据。"
        error_message = "更新ETF基金数据失败。"

        # 获取ETF基金的实时数据
        all_funds_data = ak.fund_etf_category_sina(symbol="ETF基金")
        # 在比较前从 '代码' 列中移除 'sh' 或 'sz' 前缀
        all_funds_data['代码'] = all_funds_data['代码'].str.replace('sh', '').str.replace('sz', '')
        print(all_funds_data)
        
        # 将数据转换为数据库所需格式
        all_funds_data = all_funds_data.rename(columns={
            '代码': 'symbol', '最新价': 'close', '涨跌额': 'change_amount', '涨跌幅': 'change_rate', 
            '今开': 'open', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'turnover'
        })
        all_funds_data['id'] = None
        all_funds_data['trade_date'] = today_date
        all_funds_data['amplitude'] = all_funds_data['high'] - all_funds_data['low']
        all_funds_data['turnover_rate'] = None
        # 移除不需要的列
        all_funds_data = all_funds_data[['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']]
        print(all_funds_data)
        
        # 将数据批量插入数据库
        try:
            with self.engine.connect() as connection:
                for index, row in all_funds_data.iterrows():
                    # 检查是否存在相同的记录
                    check_query = """
                    SELECT 1 FROM historicaldata
                    WHERE symbol = :symbol AND trade_date = :trade_date
                    """
                    exists = connection.execute(text(check_query), {'symbol': row['symbol'], 'trade_date': row['trade_date']}).fetchone()
                    if not exists:
                        # 使用SQLAlchemy的核心表达式语言构建插入语句
                        insert_stmt = text("""
                            INSERT INTO historicaldata (id, symbol, trade_date, open, high, low, close, volume, turnover, amplitude, change_rate, change_amount, turnover_rate)
                            VALUES (:id, :symbol, :trade_date, :open, :high, :low, :close, :volume, :turnover, :amplitude, :change_rate, :change_amount, :turnover_rate)
                        """)
                        connection.execute(insert_stmt, row.to_dict())
                    else:
                        # 更新现有记录
                        update_query = """
                        UPDATE historicaldata
                        SET open = :open, high = :high, low = :low, close = :close, volume = :volume, turnover = :turnover, amplitude = :amplitude, change_rate = :change_rate, change_amount = :change_amount, turnover_rate = :turnover_rate
                        WHERE symbol = :symbol AND trade_date = :trade_date
                        """
                        connection.execute(text(update_query), row.to_dict())
                connection.commit()
                connection.close()
                end_time = datetime.datetime.now()  # 记录结束时间
                duration = end_time - start_time  # 计算持续时间
                formatted_duration = str(duration).split('.')[0]  # 格式化持续时间为易读形式
                email_sender.send_email("ETF数据更新通知", f"{success_message} 处理时间：{formatted_duration}", "745339023@qq.com", "Mr.Light")
                print(f"{success_message} 处理时间：{formatted_duration}")
        except Exception as e:
            email_sender.send_email("ETF数据更新错误", f"{error_message} {str(e)}", "745339023@qq.com", "Mr.Light")
            print(f"{error_message} {str(e)}")

    def update_realtime_stock_data(self):
        """
        一次性更新所有股票当天实时数据，并进行后复权处理。
        """
        import akshare as ak
        import datetime
        today_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_time = datetime.datetime.now()  # 记录开始时间

        from utils.sendEmail import EmailSender  # 导入邮件发送模块
        email_sender = EmailSender()  # 创建邮件发送对象
        success_message = "成功更新所有沪深A股的实时数据。"
        error_message = "更新沪深A股数据失败。"

        # try:
        # 获取实时的股票数据
        all_stocks_data = ak.stock_zh_a_spot_em()
        all_stocks_data = all_stocks_data[all_stocks_data['代码'].str.startswith(('0', '6')) & ~all_stocks_data['代码'].str.startswith(('688', '689'))]
        # 按代码排序
        all_stocks_data = all_stocks_data.sort_values(by='代码')
        # 数据预处理
        all_stocks_data['代码'] = all_stocks_data['代码'].str.replace('sh', '').str.replace('sz', '')
        print(all_stocks_data)
        
        # 将数据转换为数据库所需格式
        all_stocks_data = all_stocks_data.rename(columns={
            '代码': 'symbol', '最新价': 'close', '涨跌额': 'change_amount', '涨跌幅': 'change_rate', 
            '今开': 'open', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'turnover',
            '振幅': 'amplitude', '换手率': 'turnover_rate'
        })
        all_stocks_data['id'] = None
        all_stocks_data['trade_date'] = today_date
        # 移除不需要的列
        all_stocks_data = all_stocks_data[['id', 'symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'amplitude', 'change_rate', 'change_amount', 'turnover_rate']]
        
        with self.engine.connect() as connection:
            from tqdm import tqdm
            for index, row in tqdm(all_stocks_data.iterrows(), total=all_stocks_data.shape[0], desc="Updating Database"):

                # 2024-06-25：去掉后复权，看着太别扭了

                # # 获取复权因子并打印结果
                # query = """
                # SELECT hfq_factor FROM hfqFactors
                # WHERE stockCode = :stockCode
                # ORDER BY tqDate DESC
                # LIMIT 1
                # """
                # result = connection.execute(text(query), {'stockCode': row['symbol']})
                # hfq_factor_row = result.fetchone()
                # if hfq_factor_row:
                #     hfq_factor = hfq_factor_row[0]  # Accessing the first element of the tuple directly
                # else:
                #     hfq_factor = 1  # Default factor in case no record is found
                # # 应用复权因子
                # hfq_factor = float(hfq_factor)  # 确保hfq_factor是浮点数
                # row['close'] = float(row['close']) * hfq_factor
                # row['open'] = float(row['open']) * hfq_factor
                # row['high'] = float(row['high']) * hfq_factor
                # row['low'] = float(row['low']) * hfq_factor
                
                # 检查是否存在相同的记录
                check_query = """
                SELECT 1 FROM historicaldata
                WHERE symbol = :symbol AND trade_date = :trade_date
                """
                exists = connection.execute(text(check_query), {'symbol': row['symbol'], 'trade_date': row['trade_date']}).fetchone()
                if not exists:
                    # 使用SQLAlchemy的核心表达式语言构建插入语句
                    insert_stmt = text("""
                        INSERT INTO historicaldata (id, symbol, trade_date, open, high, low, close, volume, turnover, amplitude, change_rate, change_amount, turnover_rate)
                        VALUES (:id, :symbol, :trade_date, :open, :high, :low, :close, :volume, :turnover, :amplitude, :change_rate, :change_amount, :turnover_rate)
                    """)
                    row_dict = row.to_dict()
                    for key, value in row_dict.items():
                        if pd.isna(value):
                            row_dict[key] = None
                    connection.execute(insert_stmt, row_dict)
                else:
                    # 更新现有记录
                    update_query = """
                    UPDATE historicaldata
                    SET open = :open, high = :high, low = :low, close = :close, volume = :volume, turnover = :turnover, amplitude = :amplitude, change_rate = :change_rate, change_amount = :change_amount, turnover_rate = :turnover_rate
                    WHERE symbol = :symbol AND trade_date = :trade_date
                    """
                    row_dict = row.to_dict()
                    for key, value in row_dict.items():
                        if pd.isna(value):
                            row_dict[key] = None
                    connection.execute(text(update_query), row_dict)  # Corrected from row.to_dict to row_dict
            connection.commit()
            connection.close()
            end_time = datetime.datetime.now()  # 记录结束时间
            duration = end_time - start_time  # 计算持续时间
            formatted_duration = str(duration).split('.')[0]  # 格式化持续时间为易读形式
            email_sender.send_email("沪深A股实时数据更新通知", f"{success_message} 处理时间：{formatted_duration}", "745339023@qq.com", "Mr.Light")
            print(f"{success_message} 处理时间：{formatted_duration}")
        # except Exception as e:
        #     email_sender.send_email("沪深A股数据更新错误", f"{error_message} {str(e)}", "745339023@qq.com", "Mr.Light")
        #     print(f"{error_message} {str(e)}")

    def fetch_and_store_hfq_factors(self):
        """
        从akshare获取后复权因子，并将其存储到数据库中，并通过邮件通知执行结果。
        """
        from tqdm import tqdm  # 导入tqdm模块
        import datetime  # 导入datetime模块以修复AttributeError
        from utils.sendEmail import EmailSender  # 导入邮件发送模块
        email_sender = EmailSender()  # 创建邮件发送对象
        success_message = "成功将所有证券的后复权因子更新到数据库。"
        error_message = "更新后复权因子时发生错误。"
        
        start_time = datetime.datetime.now()  # 记录开始时间
        
        try:
            with self.engine.connect() as connection:
                # 从stockBasic表中获取所有证券代码
                query = text("SELECT symbol FROM stockBasic WHERE (symbol LIKE '0%' OR symbol LIKE '6%') AND symbol NOT LIKE '688%' AND symbol NOT LIKE '689%'")
                result = connection.execute(query)
                stock_codes = result.fetchall()

                # 获取每个证券代码的最新datentime
                last_dates_query = text("SELECT stockCode, MAX(datentime) as maxDate FROM hfqFactors GROUP BY stockCode")
                last_dates_result = connection.execute(last_dates_query)
                last_dates_df = pd.DataFrame(last_dates_result.fetchall(), columns=['stockCode', 'maxDate'])
                last_dates_dict = last_dates_df.set_index('stockCode').to_dict()['maxDate']

                # 遍历所有证券代码，获取每个证券的后复权因子
                progress_bar = tqdm(stock_codes, desc="获取后复权因子")
                for code in progress_bar:
                    stock_code = code[0]
                    prefix = 'sh' if stock_code.startswith(('6')) else 'sz'
                    full_symbol = f"{prefix}{stock_code}"
                    progress_bar.set_description(f"处理 {full_symbol}")  # 正确使用tqdm对象的set_description方法

                    # 检查是否已经更新
                    last_date = last_dates_dict.get(stock_code)
                    if last_date is not None and last_date.date() == datetime.date.today():
                        progress_bar.set_description(f"跳过 {full_symbol}，因为它已经是最新的")
                        continue

                    # 获取后复权因子
                    hfq_factors = ak.stock_zh_a_daily(symbol=full_symbol, adjust='hfq-factor')

                    # 将hfq_factors转换为DataFrame
                    hfq_factors_df = pd.DataFrame(hfq_factors)
                    # 从数据库获取已存在的后复权因子记录
                    existing_records_query = text("""
                        SELECT tqDate FROM hfqFactors WHERE stockCode = :stockCode
                    """)
                    existing_records = connection.execute(existing_records_query, {'stockCode': stock_code}).fetchall()
                    existing_dates = {pd.to_datetime(record[0]) for record in existing_records}

                    # 筛选出不存在于数据库中的记录
                    new_records = hfq_factors_df[~hfq_factors_df['date'].isin(existing_dates)]

                    # 写入新的后复权因子数据到数据库
                    for index, row in new_records.iterrows():
                        insert_query = text("""
                            INSERT INTO hfqFactors (stockCode, tqDate, hfq_factor, datentime)
                            VALUES (:stockCode, :tqDate, :hfq_factor, NOW())
                        """)
                        connection.execute(insert_query, {
                            'stockCode': stock_code,
                            'tqDate': row['date'],
                            'hfq_factor': row['hfq_factor'],
                        })
                        connection.commit()

                    progress_bar.update()
                
                connection.close()
                end_time = datetime.datetime.now()  # 记录结束时间
                duration = end_time - start_time  # 计算持续时间
                formatted_duration = str(duration).split('.')[0]  # 格式化持续时间为易读形式
                email_sender.send_email("后复权因子更新通知", f"{success_message} 处理时间：{formatted_duration}", "745339023@qq.com", "Mr.Light")
                print(f"{success_message} 处理时间：{formatted_duration}")
        except Exception as e:
            email_sender.send_email("后复权因子更新错误", f"{error_message} {str(e)} 正在处理的证券代码：{full_symbol}", "745339023@qq.com", "Mr.Light")
            print(f"{error_message} {str(e)} 正在处理的证券代码：{full_symbol}")

if __name__ == "__main__":
    # 示例用法
    # fetcher = StockHistoryFetcher("562340")
    fetcher = StockHistoryFetcher(None)
    # fetcher.fetch_index_history()
    # fetcher.fetch_history()
    # fetcher.fetch_and_store_hfq_factors()
    # fetcher.update_realtime_stock_data()
    # fetcher.update_realtime_etf_data()
    fetcher.fetch_all_stock_basic()

    