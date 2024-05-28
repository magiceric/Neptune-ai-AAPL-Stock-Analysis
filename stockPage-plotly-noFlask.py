import akshare as ak
from datetime import datetime, timedelta
from click import DateTime
import pandas as pd
from sqlalchemy import BigInteger, Table, Column, Integer, String, MetaData, Float, Date, DateTime, UniqueConstraint, create_engine, select, and_
import numpy as np
import os
import logging

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.orca.config.executable = 'C:\\Users\\magic\\AppData\\Local\\Programs\\orca\\orca.exe'
pio.orca.config.save()

import random

class StockDataVisualizer:
    def __init__(self, stock_code, date_str, backtrack_days):
        self.stock_code = stock_code
        self.date_str = date_str.strftime('%Y-%m-%d')  # Convert Timestamp to string
        self.backtrack_days = backtrack_days
        self.engine = create_engine('mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock')
        self.stock_info = pd.read_sql('SELECT symbol AS code, stockName AS name FROM stockBasic', con=self.engine)
        self.error_message = ""
        self.validate_inputs()

        # Set up logging with UTF-8 encoding
        logging.basicConfig(filename='logs/plots_generating_log.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s', encoding='utf-8')

    def validate_inputs(self):
        try:
            date = datetime.strptime(self.date_str, '%Y-%m-%d')
            if date > datetime.now():
                raise ValueError("日期不能超过当前日期")
        except ValueError:
            self.error_message = "日期格式不正确或超过当前日期"
            self.date_str = datetime.now().strftime('%Y-%m-%d')

        if self.stock_code not in self.stock_info['code'].values:
            self.error_message = "股票代码不存在"

    def generate_plots(self):
        if self.stock_code and self.date_str and not self.error_message:
            stock_name = self.stock_info.loc[self.stock_info['code'] == self.stock_code, 'name'].values[0] if not self.stock_info.loc[self.stock_info['code'] == self.stock_code].empty else "未知股票"
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            base_directory = "plotsForTraining"
            today_directory = datetime.strptime(self.date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
            plot_directory = os.path.join(base_directory, today_directory)
            if not os.path.exists(plot_directory):
                os.makedirs(plot_directory)

            date = datetime.strptime(self.date_str, '%Y-%m-%d')
            trading_days = ak.tool_trade_date_hist_sina()
            trading_days['trade_date'] = pd.to_datetime(trading_days['trade_date'], format='%Y%m%d')
            original_trading_days = trading_days.copy()
            trading_days = trading_days[trading_days['trade_date'] <= date]
            if len(trading_days) >= self.backtrack_days:
                start_date = trading_days.iloc[-self.backtrack_days]['trade_date'].strftime('%Y%m%d')
            else:
                start_date = trading_days.iloc[0]['trade_date'].strftime('%Y%m%d')

            end_date = date

            query = f"""
            SELECT trade_date AS 日期, open AS 开盘, high AS 最高, low AS 最低, close AS 收盘, volume AS 成交量, turnover AS 成交额, change_rate AS 涨跌幅
            FROM historicaldata
            WHERE symbol = '{self.stock_code}' AND trade_date BETWEEN '{start_date}' AND '{end_date.strftime('%Y-%m-%d')}'
            ORDER BY trade_date
            """
            stock_zh_a_hist_df = pd.read_sql(query, self.engine)
            stock_zh_a_hist_df['日期'] = pd.to_datetime(stock_zh_a_hist_df['日期']).dt.date
            stock_zh_a_hist_df.sort_values(by='日期', inplace=True)

            # Plotly: 蜡烛图与移动平均线
            stock_zh_a_hist_df['MA5'] = stock_zh_a_hist_df['收盘'].rolling(window=5, min_periods=1).mean().round(2)
            stock_zh_a_hist_df['MA10'] = stock_zh_a_hist_df['收盘'].rolling(window=10, min_periods=1).mean().round(2)
            stock_zh_a_hist_df['MA20'] = stock_zh_a_hist_df['收盘'].rolling(window=20, min_periods=1).mean().round(2)
            stock_zh_a_hist_df['MA30'] = stock_zh_a_hist_df['收盘'].rolling(window=30, min_periods=1).mean().round(2)
            stock_zh_a_hist_df['MA60'] = stock_zh_a_hist_df['收盘'].rolling(window=60, min_periods=1).mean().round(2)

            # 填充前期缺失的数据
            stock_zh_a_hist_df['MA5'] = stock_zh_a_hist_df['MA5'].bfill()
            stock_zh_a_hist_df['MA10'] = stock_zh_a_hist_df['MA10'].bfill()
            stock_zh_a_hist_df['MA20'] = stock_zh_a_hist_df['MA20'].bfill()
            stock_zh_a_hist_df['MA30'] = stock_zh_a_hist_df['MA30'].bfill()
            stock_zh_a_hist_df['MA60'] = stock_zh_a_hist_df['MA60'].bfill()

            fig_pricevol = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, subplot_titles=('', ''), row_width=[0.2, 0.7])
            fig_pricevol.add_trace(go.Candlestick(x=stock_zh_a_hist_df['日期'],
                                            open=stock_zh_a_hist_df['开盘'],
                                            high=stock_zh_a_hist_df['最高'],
                                            low=stock_zh_a_hist_df['最低'],
                                            close=stock_zh_a_hist_df['收盘'], 
                                            increasing_line_color='red', 
                                            decreasing_line_color='green', 
                                            name=''), row=1, col=1)
            fig_pricevol.add_trace(go.Scatter(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['MA5'], mode='lines', name='5日均线'), row=1, col=1)
            fig_pricevol.add_trace(go.Scatter(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['MA10'], mode='lines', name='10日均线'), row=1, col=1)
            fig_pricevol.add_trace(go.Scatter(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['MA20'], mode='lines', name='20日均线'), row=1, col=1)
            fig_pricevol.add_trace(go.Scatter(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['MA30'], mode='lines', name='30日均线'), row=1, col=1)
            fig_pricevol.add_trace(go.Scatter(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['MA60'], mode='lines', name='60日均线'), row=1, col=1)

            # 成交量的红绿要与K线的红绿保持一致
            colors = [
                'red' if row['收盘'] > row['开盘'] else 
                'green' if row['收盘'] < row['开盘'] else 
                ('red' if row['涨跌幅'] > 0 else 'green') 
                for index, row in stock_zh_a_hist_df.iterrows()
            ]
            fig_pricevol.add_trace(go.Bar(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['成交量'], name='成交量', marker_color=colors), row=2, col=1)

            fig_pricevol.update_layout(title='K线（左侧）', xaxis_rangeslider_visible=False, height=420, margin=dict(l=30, r=30, t=30, b=30), xaxis_type='category', showlegend=False)
            fig_pricevol.update_xaxes(type='category', row=2, col=1, showticklabels=False)  # 更新x轴类型为类别并移除标签

            # Plotly: MACD图
            diff = (stock_zh_a_hist_df['收盘'].ewm(span=12, adjust=False).mean() - stock_zh_a_hist_df['收盘'].ewm(span=26, adjust=False).mean()).round(3)
            dea = diff.ewm(span=9, adjust=False).mean().round(3)
            macd_hist = (diff - dea).round(3)

            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=stock_zh_a_hist_df['日期'], y=diff, mode='lines', name='DIFF'))
            fig_macd.add_trace(go.Scatter(x=stock_zh_a_hist_df['日期'], y=dea, mode='lines', name='DEA'))

            # MACD柱图中，MACD值大于等于0用红色柱，小于0用绿色柱
            macd_colors = ['red' if val >= 0 else 'green' for val in macd_hist]
            fig_macd.add_trace(go.Bar(x=stock_zh_a_hist_df['日期'], y=macd_hist, name='MACD', marker_color=macd_colors, width=0.1, marker_line_width=0))

            fig_macd.update_layout(title='MACD', height=267, margin=dict(l=30, r=30, t=30, b=30), xaxis_type='category', showlegend=False)
            fig_macd.update_xaxes(showticklabels=False)  # 移除x轴标签

            # Plotly: 布林带叠加K线
            window = 20
            stock_zh_a_hist_df['MA20'] = stock_zh_a_hist_df['收盘'].rolling(window=window, min_periods=1).mean().round(2)
            stock_zh_a_hist_df['STD20'] = stock_zh_a_hist_df['收盘'].rolling(window=window, min_periods=1).std().round(2)
            stock_zh_a_hist_df['Upper'] = (stock_zh_a_hist_df['MA20'] + (stock_zh_a_hist_df['STD20'] * 2)).round(2)
            stock_zh_a_hist_df['Lower'] = (stock_zh_a_hist_df['MA20'] - (stock_zh_a_hist_df['STD20'] * 2)).round(2)

            # 填充前期缺失的数据
            stock_zh_a_hist_df['MA20'] = stock_zh_a_hist_df['MA20'].bfill()
            stock_zh_a_hist_df['STD20'] = stock_zh_a_hist_df['STD20'].bfill()
            stock_zh_a_hist_df['Upper'] = stock_zh_a_hist_df['Upper'].bfill()
            stock_zh_a_hist_df['Lower'] = stock_zh_a_hist_df['Lower'].bfill()

            fig_boll = go.Figure(data=[
                go.Scatter(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['Upper'], mode='lines', name='上轨'),
                go.Scatter(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['MA20'], mode='lines', name='中轨'),
                go.Scatter(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['Lower'], mode='lines', name='下轨'),
                go.Candlestick(x=stock_zh_a_hist_df['日期'],
                                open=stock_zh_a_hist_df['开盘'],
                                high=stock_zh_a_hist_df['最高'],
                                low=stock_zh_a_hist_df['最低'],
                                close=stock_zh_a_hist_df['收盘'], 
                                increasing_line_color='red', 
                                decreasing_line_color='green', 
                                name='蜡烛图')
            ])
            fig_boll.update_layout(
                title='布林带叠加K线',
                xaxis_rangeslider_visible=False,
                height=280,
                margin=dict(l=30, r=30, t=30, b=30),
                xaxis_type='category',
                showlegend=False
            )
            fig_boll.update_xaxes(showticklabels=False)  # 移除x轴标签

            # 计算KDJ指标
            low_list = stock_zh_a_hist_df['最低'].rolling(window=9, min_periods=1).min()
            high_list = stock_zh_a_hist_df['最高'].rolling(window=9, min_periods=1).max()
            rsv = (stock_zh_a_hist_df['收盘'] - low_list) / (high_list - low_list) * 100
            stock_zh_a_hist_df['K'] = rsv.ewm(com=2).mean()
            stock_zh_a_hist_df['D'] = stock_zh_a_hist_df['K'].ewm(com=2).mean()
            stock_zh_a_hist_df['J'] = 3 * stock_zh_a_hist_df['K'] - 2 * stock_zh_a_hist_df['D']

            # 填充前期缺失的数据
            stock_zh_a_hist_df['K'] = stock_zh_a_hist_df['K'].bfill()
            stock_zh_a_hist_df['D'] = stock_zh_a_hist_df['D'].bfill()
            stock_zh_a_hist_df['J'] = stock_zh_a_hist_df['J'].bfill()

            # 绘制KDJ图
            fig_kdj = go.Figure()
            fig_kdj.add_trace(go.Scatter(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['K'], mode='lines', name='K'))
            fig_kdj.add_trace(go.Scatter(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['D'], mode='lines', name='D'))
            fig_kdj.add_trace(go.Scatter(x=stock_zh_a_hist_df['日期'], y=stock_zh_a_hist_df['J'], mode='lines', name='J'))

            fig_kdj.update_layout(
                title='KDJ',
                height=267,
                margin=dict(l=30, r=30, t=30, b=30),
                xaxis_type='category',
                showlegend=False  # 去掉图例
            )
            fig_kdj.update_xaxes(showticklabels=False)  # 移除x轴标签

            # 根据指定的日期和天数格式化文件名
            filename_price_vol = f"{self.stock_code}_{self.date_str.replace('-', '')}_price_vol_{self.backtrack_days}.png"
            filename_macd = f"{self.stock_code}_{self.date_str.replace('-', '')}_macd_{self.backtrack_days}.png"
            filename_boll = f"{self.stock_code}_{self.date_str.replace('-', '')}_boll_{self.backtrack_days}.png"
            filename_kdj = f"{self.stock_code}_{self.date_str.replace('-', '')}_kdj_{self.backtrack_days}.png"
            # 指定保存路径
            file_path_price_vol = os.path.join(plot_directory, filename_price_vol)
            file_path_macd = os.path.join(plot_directory, filename_macd)
            file_path_boll = os.path.join(plot_directory, filename_boll)
            file_path_kdj = os.path.join(plot_directory, filename_kdj)
            
            # 保存fig_pricevol图像，如果文件已存在则直接覆盖
            if os.path.exists(file_path_price_vol):
                os.remove(file_path_price_vol)
            fig_pricevol.write_image(file_path_price_vol, engine="orca")
            logging.info(f"图像已保存至: {file_path_price_vol}")
            
            # 保存fig_macd图像，如果文件已存在则直接覆盖
            if os.path.exists(file_path_macd):
                os.remove(file_path_macd)
            fig_macd.write_image(file_path_macd, engine="orca")
            logging.info(f"图像已保存至: {file_path_macd}")
            
            # 保存fig_boll图像，如果文件已存在则直接覆盖
            if os.path.exists(file_path_boll):
                os.remove(file_path_boll)
            fig_boll.write_image(file_path_boll, engine="orca")
            logging.info(f"图像已保存至: {file_path_boll}")
            
            # 保存fig_kdj图像，如果文件已存在则直接覆盖
            if os.path.exists(file_path_kdj):
                os.remove(file_path_kdj)
            fig_kdj.write_image(file_path_kdj, engine="orca")
            logging.info(f"图像已保存至: {file_path_kdj}")

            # 将数据写入数据库表train_data_v2
            metadata = MetaData()
            train_data_v2 = Table('train_data_v2', metadata,
                                    Column('id', Integer, primary_key=True, autoincrement=True, comment='数据id'),
                                    Column('stockCode', String(6), nullable=True, comment='股票代码'),
                                    Column('stockName', String(20), nullable=True, comment='股票名称'),
                                    Column('tDate', Date, nullable=True, comment='观察日期（T日）'),
                                    Column('backtrackDays', Integer, nullable=True, comment='回溯天数'),
                                    Column('price_vol_plot_path', String(200), nullable=True, comment='价格和成交量图像路径'),
                                    Column('macd_plot_path', String(200), nullable=True, comment='MACD图像路径'),
                                    Column('boll_plot_path', String(200), nullable=True, comment='布林线图像路径'),
                                    Column('kdj_plot_path', String(200), nullable=True, comment='KDJ图像路径'),
                                    Column('t1Date', Date, nullable=True, comment='T+1日'),
                                    Column('t1ChangeRate', Float, nullable=True, comment='T+1涨跌幅'),
                                    Column('datentime', DateTime(), nullable=False, comment='该数据行生成或更新的日期时间')
                                    )

            # 定义historicaldata表
            historicaldata = Table('historicaldata', metadata,
                                    Column('id', Integer, primary_key=True, autoincrement=True, comment='数据id'),
                                    Column('symbol', String(10), nullable=False, comment='证券代码'),
                                    Column('trade_date', Date, nullable=False, comment='交易日期'),
                                    Column('open', Float, nullable=True, comment='开盘价'),
                                    Column('high', Float, nullable=True, comment='最高价'),
                                    Column('low', Float, nullable=True, comment='最低价'),
                                    Column('close', Float, nullable=True, comment='收盘价'),
                                    Column('volume', BigInteger, nullable=True, comment='成交量(在交易日内买卖的股票数量)'),
                                    Column('turnover', Float, nullable=True, comment='成交金额(在交易日内的总交易金额)'),
                                    Column('amplitude', Float, nullable=True, comment='振幅(当日股价的最高价和最低价之间的差值)'),
                                    Column('change_rate', Float, nullable=True, comment='涨跌幅(当日收盘价相对于前一交易日收盘价的涨跌幅度)'),
                                    Column('change_amount', Float, nullable=True, comment='涨跌额(当日收盘价相对于前一交易日收盘价的涨跌金额)'),
                                    Column('turnover_rate', Float, nullable=True, comment='换手率(当日的成交量与流通股本的比率)'),
                                    UniqueConstraint('symbol', 'trade_date', name='unique_index')
                                    )
                                    
            # 检查是否存在相同记录
            check_query = train_data_v2.select().where(
                and_(
                    train_data_v2.c.stockCode == self.stock_code,
                    train_data_v2.c.tDate == datetime.strptime(self.date_str, '%Y-%m-%d'),
                    train_data_v2.c.backtrackDays == self.backtrack_days
                )
            )
            conn = self.engine.connect()
            existing_record = conn.execute(check_query).fetchone()

            # 如果存在相同记录，则更新
            if existing_record:
                next_trading_day = original_trading_days.loc[original_trading_days['trade_date'] > self.date_str].iloc[0]['trade_date']
                # 从数据库获取t1Date的涨跌幅
                t1_change_rate_query = historicaldata.select().where(
                    historicaldata.c.trade_date == next_trading_day
                )
                t1_change_rate_record = conn.execute(t1_change_rate_query).fetchone()
                t1_change_rate = t1_change_rate_record.change_rate if t1_change_rate_record else None

                update_stmt = train_data_v2.update().where(
                    train_data_v2.c.id == existing_record.id
                ).values(
                    price_vol_plot_path=plot_directory + f'\{filename_price_vol}',
                    macd_plot_path=plot_directory + f'\{filename_macd}',
                    boll_plot_path=plot_directory + f'\{filename_boll}',
                    kdj_plot_path=plot_directory + f'\{filename_kdj}',
                    t1Date=next_trading_day,
                    t1ChangeRate=t1_change_rate,
                    datentime=datetime.now()
                )
                result = conn.execute(update_stmt)
                logging.info("数据库表train_data_v2中的记录已更新")
            else:
                # 插入新记录
                next_trading_day = original_trading_days.loc[original_trading_days['trade_date'] > self.date_str].iloc[0]['trade_date']
                # 从数据库获取t1Date的涨跌幅，同时确保股票代码匹配
                t1_change_rate_query = historicaldata.select().where(
                    and_(
                        historicaldata.c.trade_date == next_trading_day,
                        historicaldata.c.symbol == self.stock_code
                    )
                )
                t1_change_rate_record = conn.execute(t1_change_rate_query).fetchone()
                t1_change_rate = t1_change_rate_record.change_rate if t1_change_rate_record else None

                insert_stmt = train_data_v2.insert().values(
                    stockCode=self.stock_code,
                    stockName=stock_name,
                    tDate=datetime.strptime(self.date_str, '%Y-%m-%d'),
                    backtrackDays=self.backtrack_days,
                    price_vol_plot_path=plot_directory + f'\{filename_price_vol}',
                    macd_plot_path=plot_directory + f'\{filename_macd}',
                    boll_plot_path=plot_directory + f'\{filename_boll}',
                    kdj_plot_path=plot_directory + f'\{filename_kdj}',
                    t1Date=next_trading_day,
                    t1ChangeRate=t1_change_rate,
                    datentime=datetime.now()
                )
                result = conn.execute(insert_stmt)
                logging.info("新记录已添加到数据库表train_data_v2")

            conn.commit()
            conn.close()

import random
import akshare as ak

# 获取有效交易日期
trading_days = ak.tool_trade_date_hist_sina()
# Ensure the 'trade_date' column is of type datetime for comparison
trading_days['trade_date'] = pd.to_datetime(trading_days['trade_date'])
valid_days = trading_days[(trading_days['trade_date'] >= pd.Timestamp('2024-01-01')) & (trading_days['trade_date'] <= pd.Timestamp('2024-05-25'))]

# 连接数据库并获取股票信息
engine = create_engine('mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock')
stock_info = pd.read_sql('SELECT symbol AS code, stockName AS name FROM stockBasic', con=engine)

# 初始化进度条
from tqdm import tqdm
progress_bar = tqdm(total=1000, desc="初始化")

for _ in range(progress_bar.total):
    # 随机选择一个有效交易日
    random_date = random.choice(valid_days['trade_date'].tolist())
    
    # 随机选择一个股票代码
    random_stock = stock_info.sample(n=1).iloc[0]
    stock_code = random_stock['code']

    # Check if there is an existing record in train_data_v2 for the selected stock, date, and backtrack days
    backtrack_days = 60  # Assuming backtrack days is a constant for this context
    check_stmt = f"SELECT * FROM train_data_v2 WHERE stockCode = '{stock_code}' AND tDate = '{random_date}' AND backtrackDays = {backtrack_days}"
    existing_record = pd.read_sql(check_stmt, con=engine)
    
    # If there is an existing record, reselect random values
    while not existing_record.empty:
        random_date = random.choice(valid_days['trade_date'].tolist())
        random_stock = stock_info.sample(n=1).iloc[0]
        stock_code = random_stock['code']
        check_stmt = f"SELECT * FROM train_data_v2 WHERE stockCode = '{stock_code}' AND tDate = '{random_date}' AND backtrackDays = {backtrack_days}"
        existing_record = pd.read_sql(check_stmt, con=engine)

        if (existing_record):
            logging.info("随机结果已存在，重算。")

    # 更新进度条描述
    progress_bar.set_description(f"处理 {stock_code} 于 {random_date}")
    
    # 创建实例并生成图表
    instance = StockDataVisualizer(stock_code, random_date, 60)
    instance.generate_plots()

    # 更新进度条
    progress_bar.update(1)


