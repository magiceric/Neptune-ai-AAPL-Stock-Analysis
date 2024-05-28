from flask import Flask, render_template_string, request
import akshare as ak
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import Table, Column, Integer, String, MetaData, Float, Date, create_engine, select, and_
import numpy as np
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.orca.config.executable = 'C:\\Users\\magic\\AppData\\Local\\Programs\\orca\\orca.exe'
pio.orca.config.save()

import random

# 创建数据库连接
engine = create_engine('mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock')
# 从数据库获取股票信息
stock_info = pd.read_sql('SELECT symbol AS code, stockName AS name FROM stockBasic', con=engine)
# print(stock_info)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        print("Received POST request with the following parameters:")
        for key, value in request.form.items():
            print(f"{key}: {value}")

    today_date = datetime.now().strftime('%Y-%m-%d')
    stock_data = None
    ma_graph_url = None
    macd_graph_url = None
    boll_graph_url = None
    kdj_graph_url = None
    ma_graph_url_right = None
    stock_code = request.form.get('stock_code', '000001')
    date_str = request.form.get('date', today_date)
    backtrack_days = request.form.get('backtrack_days', '60')  # 默认回溯60天
    stock_name = ""
    error_message = ""

    try:
        backtrack_days = int(backtrack_days)
        if backtrack_days <= 0:
            raise ValueError("回溯天数必须是正整数")
    except ValueError:
        error_message = "回溯天数必须是正整数"
        backtrack_days = 60

    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        if date > datetime.now():
            raise ValueError("日期不能超过当前日期")
    except ValueError:
        error_message = "日期格式不正确或超过当前日期"
        date_str = today_date

    if stock_code not in stock_info['code'].values:
        error_message = "股票代码不存在"

    if request.method == 'POST':
        if stock_code and date_str and not error_message:
            try:
                stock_name = stock_info.loc[stock_info['code'] == stock_code, 'name'].values[0] if not stock_info.loc[stock_info['code'] == stock_code].empty else "未知股票"

                # 将图表保存到指定目录，按照给定的命名规则
                # 确保目录存在
                # 确保当前目录是代码所在目录
                os.chdir(os.path.dirname(os.path.abspath(__file__)))

                # 创建plotsForTraining目录及其子目录
                base_directory = "plotsForTraining"
                today_directory = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
                plot_directory = os.path.join(base_directory, today_directory)
                if not os.path.exists(plot_directory):
                    os.makedirs(plot_directory)

                date = datetime.strptime(date_str, '%Y-%m-%d')
                trading_days = ak.tool_trade_date_hist_sina()
                trading_days['trade_date'] = pd.to_datetime(trading_days['trade_date'], format='%Y%m%d')
                trading_days = trading_days[trading_days['trade_date'] <= date]
                if len(trading_days) >= backtrack_days:
                    start_date = trading_days.iloc[-backtrack_days]['trade_date'].strftime('%Y%m%d')
                else:
                    start_date = trading_days.iloc[0]['trade_date'].strftime('%Y%m%d')
                
                end_date = date
                
                # 创建数据库连接
                engine = create_engine('mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock')

                # 从数据库查询历史数据
                query = f"""
                SELECT trade_date AS 日期, open AS 开盘, high AS 最高, low AS 最低, close AS 收盘, volume AS 成交量, turnover AS 成交额, change_rate AS 涨跌幅
                FROM historicaldata
                WHERE symbol = '{stock_code}' AND trade_date BETWEEN '{start_date}' AND '{end_date.strftime('%Y-%m-%d')}'
                ORDER BY trade_date
                """
                stock_zh_a_hist_df = pd.read_sql(query, engine)

                # 确保数据按日期排序
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

                # 找到5日线最近一个上穿10日均线的位置，从后往前找
                golden_cross = None
                for i in range(len(stock_zh_a_hist_df) - 1, 0, -1):
                    if stock_zh_a_hist_df['MA5'].iloc[i] > stock_zh_a_hist_df['MA10'].iloc[i] and stock_zh_a_hist_df['MA5'].iloc[i - 1] <= stock_zh_a_hist_df['MA10'].iloc[i - 1]:
                        golden_cross = stock_zh_a_hist_df['日期'].iloc[i]
                        break

                # 定义支撑区域参数
                epsilon = 0.01

                # 判断股价是否在5日均线附近获得支撑
                stock_zh_a_hist_df['支撑'] = np.where(
                    (stock_zh_a_hist_df['收盘'] >= (1 - epsilon) * stock_zh_a_hist_df['MA5']) & 
                    (stock_zh_a_hist_df['收盘'] <= (1 + epsilon) * stock_zh_a_hist_df['MA5']) & 
                    (stock_zh_a_hist_df['收盘'].shift(-1) > stock_zh_a_hist_df['收盘']),
                    True, False
                )

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
                
                # 如果找到金叉位置，添加标记
                if golden_cross:
                    fig_pricevol.add_trace(go.Scatter(x=[golden_cross], y=[stock_zh_a_hist_df.loc[stock_zh_a_hist_df['日期'] == golden_cross, 'MA5'].values[0]], mode='markers', marker=dict(color='blue', size=10), name='金叉'))

                # 如果找到支撑位置，添加标记
                support_points = stock_zh_a_hist_df[stock_zh_a_hist_df['支撑']]
                fig_pricevol.add_trace(go.Scatter(x=support_points['日期'], y=support_points['收盘'], mode='markers', marker=dict(color='orange', size=8), name='支撑'))

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

                # 找到最后一个金叉 (DIFF 上穿 DEA)
                golden_cross = None
                for i in range(1, len(stock_zh_a_hist_df)):
                    if diff[i] > dea[i] and diff[i-1] <= dea[i-1]:
                        golden_cross = stock_zh_a_hist_df['日期'][i]

                if golden_cross:
                    fig_macd.add_trace(go.Scatter(x=[golden_cross], y=[diff[stock_zh_a_hist_df['日期'] == golden_cross].values[0]], mode='markers', marker=dict(color='red', size=10), name='金叉'))

                fig_macd.update_layout(title='MACD', height=267, margin=dict(l=30, r=30, t=30, b=30), xaxis_type='category', showlegend=False)
                fig_macd.update_xaxes(showticklabels=False)  # 移除x轴标签

                macd_graph_url = fig_macd.to_html(full_html=False)

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

                boll_graph_url = fig_boll.to_html(full_html=False)

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

                kdj_graph_url = fig_kdj.to_html(full_html=False)
                # 复制一个maGraph到rightPanel下面，数据范围从左侧k线图最右边一条数据的下一个交易日开始，取backtrack_days条数据
                last_date_left = stock_zh_a_hist_df['日期'].iloc[-1]
                
                # 重新获取交易日数据，并换个名字
                trading_days_new = ak.tool_trade_date_hist_sina()
                next_day = trading_days_new[trading_days_new['trade_date'].apply(lambda x: pd.Timestamp(x)) > pd.Timestamp(last_date_left)].iloc[0]['trade_date']
                end_date_new = trading_days_new[trading_days_new['trade_date'].apply(lambda x: pd.Timestamp(x)) > pd.Timestamp(next_day)].iloc[backtrack_days - 1]['trade_date']
                
                stock_zh_a_hist_df_right = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=next_day.strftime('%Y%m%d'), end_date=end_date_new.strftime('%Y%m%d'), adjust="qfq")
                stock_zh_a_hist_df_right = stock_zh_a_hist_df_right.tail(backtrack_days)
                
                if not stock_zh_a_hist_df_right.empty:
                    stock_zh_a_hist_df_right['MA5'] = stock_zh_a_hist_df_right['收盘'].rolling(window=5, min_periods=1).mean().round(2)
                    stock_zh_a_hist_df_right['MA10'] = stock_zh_a_hist_df_right['收盘'].rolling(window=10, min_periods=1).mean().round(2)
                    stock_zh_a_hist_df_right['MA20'] = stock_zh_a_hist_df_right['收盘'].rolling(window=20, min_periods=1).mean().round(2)
                    stock_zh_a_hist_df_right['MA30'] = stock_zh_a_hist_df_right['收盘'].rolling(window=30, min_periods=1).mean().round(2)
                    stock_zh_a_hist_df_right['MA60'] = stock_zh_a_hist_df_right['收盘'].rolling(window=60, min_periods=1).mean().round(2)
                    stock_zh_a_hist_df_right['MA5'] = stock_zh_a_hist_df_right['MA5'].bfill()
                    stock_zh_a_hist_df_right['MA10'] = stock_zh_a_hist_df_right['MA10'].bfill()
                    stock_zh_a_hist_df_right['MA20'] = stock_zh_a_hist_df_right['MA20'].bfill()
                    stock_zh_a_hist_df_right['MA30'] = stock_zh_a_hist_df_right['MA30'].bfill()
                    stock_zh_a_hist_df_right['MA60'] = stock_zh_a_hist_df_right['MA60'].bfill()

                    fig_right = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, subplot_titles=('', ''), row_width=[0.2, 0.7])
                    fig_right.add_trace(go.Candlestick(x=stock_zh_a_hist_df_right['日期'],
                                                       open=stock_zh_a_hist_df_right['开盘'],
                                                       high=stock_zh_a_hist_df_right['最高'],
                                                       low=stock_zh_a_hist_df_right['最低'],
                                                       close=stock_zh_a_hist_df_right['收盘'], 
                                                       increasing_line_color='red', 
                                                       decreasing_line_color='green', 
                                                       name=''), row=1, col=1)
                    fig_right.add_trace(go.Scatter(x=stock_zh_a_hist_df_right['日期'], y=stock_zh_a_hist_df_right['MA5'], mode='lines', name='5日均线'), row=1, col=1)
                    fig_right.add_trace(go.Scatter(x=stock_zh_a_hist_df_right['日期'], y=stock_zh_a_hist_df_right['MA10'], mode='lines', name='10日均线'), row=1, col=1)
                    fig_right.add_trace(go.Scatter(x=stock_zh_a_hist_df_right['日期'], y=stock_zh_a_hist_df_right['MA20'], mode='lines', name='20日均线'), row=1, col=1)
                    fig_right.add_trace(go.Scatter(x=stock_zh_a_hist_df_right['日期'], y=stock_zh_a_hist_df_right['MA30'], mode='lines', name='30日均线'), row=1, col=1)
                    fig_right.add_trace(go.Scatter(x=stock_zh_a_hist_df_right['日期'], y=stock_zh_a_hist_df_right['MA60'], mode='lines', name='60日均线'), row=1, col=1)
                    colors_right = [
                        'red' if row['收盘'] > row['开盘'] else 
                        'green' if row['收盘'] < row['开盘'] else 
                        ('red' if row['涨跌幅'] > 0 else 'green') 
                        for index, row in stock_zh_a_hist_df_right.iterrows()
                    ]
                    fig_right.add_trace(go.Bar(x=stock_zh_a_hist_df_right['日期'], y=stock_zh_a_hist_df_right['成交量'], name='成交量', marker_color=colors_right), row=2, col=1)
                    fig_right.update_layout(title='K线 (右侧)', xaxis_rangeslider_visible=False, height=420, margin=dict(l=30, r=30, t=30, b=30), xaxis_type='category', showlegend=False)
                    fig_right.update_xaxes(type='category', row=2, col=1, showticklabels=False, range=[0, backtrack_days])  # 设置x轴数据宽度为backtrack_days
                    
                    # 确保y轴一致，通过找到两边的极值
                    min_low = min(stock_zh_a_hist_df['最低'].min(), stock_zh_a_hist_df_right['最低'].min())
                    max_high = max(stock_zh_a_hist_df['最高'].max(), stock_zh_a_hist_df_right['最高'].max())
                    
                    # 为y轴范围添加一些填充
                    padding = (max_high - min_low) * 0.05  # 5%填充
                    yaxis_range = [min_low - padding, max_high + padding]
                    
                    fig_pricevol.update_yaxes(range=yaxis_range, row=1, col=1)
                    fig_right.update_yaxes(range=yaxis_range, row=1, col=1)
                    
                    # 确保成交量的y轴一致，通过找到两边的极值
                    min_volume = min(stock_zh_a_hist_df['成交量'].min(), stock_zh_a_hist_df_right['成交量'].min())
                    max_volume = max(stock_zh_a_hist_df['成交量'].max(), stock_zh_a_hist_df_right['成交量'].max())
                    
                    # 为成交量的y轴范围添加一些填充
                    volume_padding = (max_volume - min_volume) * 0.05  # 5%填充
                    volume_yaxis_range = [min_volume - volume_padding, max_volume + volume_padding]
                    
                    fig_pricevol.update_yaxes(range=volume_yaxis_range, row=2, col=1)
                    fig_right.update_yaxes(range=volume_yaxis_range, row=2, col=1)
                    
                    ma_graph_url_right = fig_right.to_html(full_html=False)
                else:
                    ma_graph_url_right = "<div>（右侧）无可用数据。</div>"

                ma_graph_url = fig_pricevol.to_html(full_html=False)

                # 准备显示的股票数据
                # 合并stock_zh_a_hist_df和stock_zh_a_hist_df_right
                combined_df = pd.concat([stock_zh_a_hist_df, stock_zh_a_hist_df_right]).drop_duplicates().reset_index(drop=True)
                display_df = combined_df.copy()
                # display_df = display_df.sort_values(by='日期', ascending=False)
                display_df['成交量'] = (display_df['成交量'] / 10000).apply(lambda x: "{:,.0f}".format(x))
                display_df['成交额'] = (display_df['成交额'] / 10000).apply(lambda x: "{:,.0f}".format(x))
                stock_data = display_df.to_html(classes='stock-table', border=0, index=False, escape=False)
                stock_data = stock_data.replace('<table', '<table style="width: 100%; overflow-y: auto;"')
                stock_data = stock_data.replace('<thead>', '<thead style="width: 100%; position: sticky; top: 0; background-color: white;">')
                stock_data = stock_data.replace('<tbody>', '<tbody style="width: 100%; height: 300px; overflow-y: scroll;">')

                # 根据指定的日期和天数格式化文件名
                filename_price_vol = f"{stock_code}_{date_str.replace('-', '')}_price_vol_{backtrack_days}.png"
                filename_macd = f"{stock_code}_{date_str.replace('-', '')}_macd_{backtrack_days}.png"
                filename_boll = f"{stock_code}_{date_str.replace('-', '')}_boll_{backtrack_days}.png"
                filename_kdj = f"{stock_code}_{date_str.replace('-', '')}_kdj_{backtrack_days}.png"
                # 指定保存路径
                file_path_price_vol = os.path.join(plot_directory, filename_price_vol)
                file_path_macd = os.path.join(plot_directory, filename_macd)
                file_path_boll = os.path.join(plot_directory, filename_boll)
                file_path_kdj = os.path.join(plot_directory, filename_kdj)
                
                # 保存fig_pricevol图像，如果文件已存在则直接覆盖
                if os.path.exists(file_path_price_vol):
                    os.remove(file_path_price_vol)
                fig_pricevol.write_image(file_path_price_vol, engine="orca")
                print(f"图像已保存至: {file_path_price_vol}")
                
                # 保存fig_macd图像，如果文件已存在则直接覆盖
                if os.path.exists(file_path_macd):
                    os.remove(file_path_macd)
                fig_macd.write_image(file_path_macd, engine="orca")
                print(f"图像已保存至: {file_path_macd}")
                
                # 保存fig_boll图像，如果文件已存在则直接覆盖
                if os.path.exists(file_path_boll):
                    os.remove(file_path_boll)
                fig_boll.write_image(file_path_boll, engine="orca")
                print(f"图像已保存至: {file_path_boll}")
                
                # 保存fig_kdj图像，如果文件已存在则直接覆盖
                if os.path.exists(file_path_kdj):
                    os.remove(file_path_kdj)
                fig_kdj.write_image(file_path_kdj, engine="orca")
                print(f"图像已保存至: {file_path_kdj}")    

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
                                      Column('t1ChangeRate', Float, nullable=True, comment='T+1涨跌幅')
                                     )

                # 检查是否存在相同记录
                check_query = train_data_v2.select().where(
                    and_(
                        train_data_v2.c.stockCode == stock_code,
                        train_data_v2.c.tDate == datetime.strptime(date_str, '%Y-%m-%d'),
                        train_data_v2.c.backtrackDays == backtrack_days
                    )
                )
                conn = engine.connect()
                existing_record = conn.execute(check_query).fetchone()

                # 如果存在相同记录，则更新
                if existing_record:
                    update_stmt = train_data_v2.update().where(
                        train_data_v2.c.id == existing_record.id
                    ).values(
                        price_vol_plot_path=file_path_price_vol,
                        macd_plot_path=file_path_macd,
                        boll_plot_path=file_path_boll,
                        kdj_plot_path=file_path_kdj,
                        t1Date=stock_zh_a_hist_df_right.iloc[0]['日期'] if not stock_zh_a_hist_df_right.empty else None,
                        t1ChangeRate=stock_zh_a_hist_df_right.iloc[0]['涨跌幅'] if not stock_zh_a_hist_df_right.empty else None
                    )
                    result = conn.execute(update_stmt)
                    print("数据库表train_data_v2中的记录已更新")
                else:
                    # 插入新记录
                    insert_stmt = train_data_v2.insert().values(
                        stockCode=stock_code,
                        stockName=stock_name,
                        tDate=datetime.strptime(date_str, '%Y-%m-%d'),
                        backtrackDays=backtrack_days,
                        price_vol_plot_path=file_path_price_vol,
                        macd_plot_path=file_path_macd,
                        boll_plot_path=file_path_boll,
                        kdj_plot_path=file_path_kdj,
                        t1Date=stock_zh_a_hist_df_right.iloc[0]['日期'] if not stock_zh_a_hist_df_right.empty else None,
                        t1ChangeRate=stock_zh_a_hist_df_right.iloc[0]['涨跌幅'] if not stock_zh_a_hist_df_right.empty else None
                    )
                    result = conn.execute(insert_stmt)
                    print("新记录已添加到数据库表train_data_v2")

                conn.commit()
                conn.close()

                # 随机生成日期和选择股票，确保数据库中不存在相同的记录
                existing = True
                while existing:
                    start_date = datetime.strptime('2024-01-01', '%Y-%m-%d')
                    end_date = datetime.strptime('2024-05-24', '%Y-%m-%d')
                    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
                    random_date_str = random_date.strftime('%Y-%m-%d')

                    # 从stock_info随机选择一个股票
                    random_stock = random.choice(stock_info['code'])

                    # 检查数据库中是否存在相同的记录
                    check_query = train_data_v2.select().where(
                        and_(
                            train_data_v2.c.stockCode == random_stock,
                            train_data_v2.c.tDate == random_date,
                            train_data_v2.c.backtrackDays == backtrack_days
                        )
                    )
                    conn = engine.connect()
                    result = conn.execute(check_query)
                    existing = result.fetchone() is not None
                    conn.close()
                    if existing:
                        print("数据库中已存在相同记录，重新生成日期和股票代码。")

                print(f"随机选定的日期: {random_date_str}")
                print(f"随机选定的股票代码: {random_stock}")

                # 用javascript找到date_str所在的行，然后全行设为粗体字，并滚动数据表，使这行成为最上面一行
                stock_data += f"""
                <script>

                    function startRandom() {{
                        isIntervalActive = true;
                        randomInterval = setInterval(function() {{
                            console.log('111111111111111111111111111111111111111111111111111111111');
                            var stockCodeInput = document.getElementById('stock_code');
                            var dateInput = document.getElementById('date');
                            dateInput.value = '{random_date_str}';
                            stockCodeInput.value = '{random_stock}';
                            var submitButton = document.querySelector('input[type="submit"]');
                            var isIntervalActiveInput = document.createElement('input');
                            isIntervalActiveInput.type = 'hidden';
                            isIntervalActiveInput.name = 'isIntervalActive';
                            isIntervalActiveInput.value = isIntervalActive.toString();
                            document.forms[0].appendChild(isIntervalActiveInput);
                            submitButton.click();
                        }}, 3000);
                    }}

                    function stopRandom() {{
                        if (isIntervalActive) {{
                            clearInterval(randomInterval);
                            isIntervalActive = false;
                        }}
                    }}

                    document.addEventListener('DOMContentLoaded', function() {{
                        
                        var date_str = '{last_date_left}';
                        var rows = document.querySelectorAll('.stock-table tbody tr');
                        var rightPanel = document.getElementById('rightPanel'); // 假设有一个id为'rightPanel'的元素
                        rows.forEach(function(row) {{
                            if (row.innerText.includes(date_str)) {{
                                row.style.fontWeight = 'bold';
                                row.scrollIntoView({{block: 'center'}});
                                var nextRow1 = row.nextElementSibling;
                                var nextRow2 = nextRow1 ? nextRow1.nextElementSibling : null;
                                var targetRow = nextRow1 ? nextRow1 : null;
                                if (targetRow) {{

                                    var indicatorLabel = document.createElement('label');
                                    if (targetRow === nextRow1) {{
                                        indicatorLabel.textContent = 'T+1涨跌幅（实际值） : ';
                                    }} else if (targetRow === nextRow2) {{
                                        indicatorLabel.textContent = 'T+2涨跌幅（实际值） : ';
                                    }}
                                    indicatorLabel.style.fontWeight = 'bold'; // 使标签文本加粗
                                    rightPanel.appendChild(indicatorLabel);
                                    var eighthColumnValue = targetRow.cells[7].innerText;
                                    targetRow.cells[7].style.fontWeight = 'bold'; // 使文本加粗
                                    var valueFloat = parseFloat(eighthColumnValue.replace(',', ''));
                                    if (valueFloat > 0) {{
                                        targetRow.cells[7].style.color = 'red'; // 将文本颜色设置为红色表示正值
                                    }} else {{
                                        targetRow.cells[7].style.color = 'green'; // 将文本颜色设置为绿色表示负值
                                    }}
                                    console.log('第二行第八列的值:', eighthColumnValue);
                                    var valuePercentage = parseFloat(eighthColumnValue.replace(',', ''));
                                    var indicatorDiv = document.createElement('div');
                                    var indicatorInnerDiv = document.createElement('div');

                                    // 设置外部div的样式
                                    indicatorDiv.style.display = 'inline-block'; // 设置显示为内联块以便与标签并排
                                    indicatorDiv.style.width = 'calc(90% - 120px)'; // 调整宽度以适应标签的同一行
                                    indicatorDiv.style.height = '20px';
                                    indicatorDiv.style.backgroundColor = '#ddd';
                                    indicatorDiv.style.position = 'relative';
                                    indicatorDiv.style.verticalAlign = 'middle'; // 与标签垂直对齐
                                    indicatorDiv.style.marginTop = '50px'; // 增加边距以与上方元素分开
                                    indicatorDiv.style.marginLeft = '10px'; // 小边距以与标签分开

                                    // 设置内部div的样式
                                    indicatorInnerDiv.style.height = '100%';
                                    indicatorInnerDiv.style.width = '4px'; // 稍微加粗的垂直线
                                    indicatorInnerDiv.style.position = 'absolute';
                                    // 根据值的正负设置线的颜色
                                    if (valuePercentage > 0) {{
                                        indicatorInnerDiv.style.backgroundColor = 'red'; // 正值用红线
                                    }} else {{
                                        indicatorInnerDiv.style.backgroundColor = 'green'; // 负值用绿线
                                    }}
                                    // 根据valuePercentage计算位置
                                    var position = (valuePercentage + 10) / 20 * 100; // 转换为外部div的百分比
                                    indicatorInnerDiv.style.left = position + '%';

                                    // 将内部div添加到外部div
                                    indicatorDiv.appendChild(indicatorInnerDiv);

                                    // 创建虚线和标签用于-10%，0, 10%
                                    ['0%', '50%', '100%'].forEach(function(pos, index) {{
                                        var dashLine = document.createElement('div');
                                        dashLine.style.position = 'absolute';
                                        dashLine.style.left = pos;
                                        dashLine.style.width = '1px';
                                        dashLine.style.height = '100%';
                                        dashLine.style.backgroundColor = '#888';
                                        dashLine.style.borderLeft = 'dashed 1px black';
                                        indicatorDiv.appendChild(dashLine);

                                        var label = document.createElement('div');
                                        label.style.position = 'absolute';
                                        label.style.left = pos;
                                        label.style.top = '20px'; // 标签位置在条形图上方
                                        label.style.transform = 'translateX(-50%)';
                                        label.textContent = (index - 1) * 10 + '%';
                                        indicatorDiv.appendChild(label);
                                    }});

                                    // 标签用于显示值
                                    var valueLabel = document.createElement('div');
                                    valueLabel.style.position = 'absolute';
                                    valueLabel.style.left = position + '%';
                                    valueLabel.style.top = '-20px'; // 标签位置在条形图上方
                                    valueLabel.style.transform = 'translateX(-50%)';
                                    valueLabel.textContent = eighthColumnValue;
                                    // 根据值的正负设置标签颜色：正值红色，负值绿色
                                    if (valuePercentage > 0) {{
                                        valueLabel.style.color = 'red';
                                    }} else {{
                                        valueLabel.style.color = 'green';
                                    }}
                                    indicatorDiv.appendChild(valueLabel);
                                    // 将外部div添加到rightPanel
                                    rightPanel.appendChild(indicatorDiv);
                                }}
                            }}
                        }});

                        console.log('isIntervalActive : ', isIntervalActive);

                        if (isIntervalActive) {{
                            console.log('2222222222222222222222222222222222222222222222222');
                            startRandom();
                        }}

                    }});

                </script>
                """

            except Exception as e:
                stock_data = f"<div class='error'>获取数据错误: {e}</div>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>股票数据</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                color: #333;
                margin: 40px;
                height: 100vh; /* 设置body高度为视口高度 */
                overflow: hidden; /* 隐藏溢出以防止垂直滚动 */
            }}
            form {{
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                justify-content: flex-end;
                margin-bottom: 20px;
            }}
            input[type="text"], input[type="submit"] {{
                padding: 10px;
                margin: 0 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }}
            input[type="text"]:focus {{
                outline: none;
                border-color: #026e99;
                box-shadow: 0 0 5px rgba(2, 110, 153, 0.5);
            }}
            input[type="submit"] {{
                background-color: #026e99;
                color: white;
                cursor: pointer;
            }}
            input[type="submit"]:hover {{
                background-color: #024e76;
            }}
            .stock-table {{
                width: 100%;
                height: auto;
                overflow-y: auto;
                border-collapse: collapse;
                table-layout: auto;
                font-size: smaller;
            }}
            .stock-table th, .stock-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: right;
                white-space: nowrap;
            }}
            .stock-table th:first-child, .stock-table td:first-child {{
                text-align: center;
            }}
            .stock-table th {{
                background-color: #026e99;
                color: white;
                text-align: center;
            }}
            .error {{
                color: red;
            }}
            #mainBody {{
                display: flex;
                margin-top: 0;
            }}
            #leftPanel {{
                width: 49%;  /* 设置宽度为49% */
                height: 100vh;  /* 设置高度为视口高度以防止溢出 */
                overflow-y: auto;  /* 启用滚动如果内容超过视口高度 */
                margin-right: 2%;  /* 添加右边距以在leftPanel和rightPanel之间创建缝隙 */
            }}
            #rightPanel {{
                width: 49%;  /* 设置宽度为49% */
                height: 100vh;  /* 设置高度为视口高度以防止溢出 */
                overflow-y: auto;  /* 启用滚动如果内容超过视口高度 */
            }}
            #stockDataContainer {{
                height: 420px;  /* 设置高度与maGraph一致 */
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }}
            .plotly-graph-div {{
                width: 100%;  /* 使图表div占满其容器的宽度 */
                margin-bottom: 20px; /* 添加边距以在图表之间创建空间 */
            }}
            .gradient-bar {{
                height: 20px;
                background: linear-gradient(to right, #ff0000 0%, #ffff00 50%, #00ff00 100%);
                border: 1px solid #ccc;
                border-radius: 4px;
                margin: 10px 0;
            }}
            .gradient-bar-indicator {{
                height: 100%;
                width: 10px;
                background-color: black;
                position: relative;
                left: calc(50% - 5px); /* Center the indicator */
            }}
        </style>
        <link rel="stylesheet" href="https://code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.min.js"></script>
        <script>

            var randomInterval;
            var isIntervalActive = { request.form.get('isIntervalActive', 'false').lower() };
            console.log('isIntervalActive : ', isIntervalActive);

            $(function() {{
                $("#date").datepicker({{
                    dateFormat: "yy-mm-dd",
                    dayNamesMin: ["日", "一", "二", "三", "四", "五", "六"],
                    monthNames: ["一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月"],
                    showMonthAfterYear: true,
                    yearSuffix: "年"
                }});
                $("#stock_code").focus().select();
            }});

            $(document).ready(function() {{
                $("form").on("submit", function() {{
                    setTimeout(function() {{
                        $("#stock_code").focus().select();
                    }}, 10);
                }});
            }});
        </script>
    </head>
    <body>
        <form method="post" style="display: flex; justify-content: space-between;">
            <div style="flex: 1; text-align: left;">
                观察日期: <input type="text" id="date" name="date" value="{date_str}" style="width: 120px; text-align: center;" placeholder="YYYY-MM-DD">
                回溯天数: <input type="text" name="backtrack_days" value="{backtrack_days}" style="width: 80px; text-align: center;">
            </div>
            <div style="flex: 1; text-align: right;">
                <label><b>{stock_name}</b></label>&nbsp&nbsp
                股票代码: <input type="text" id="stock_code" name="stock_code" value="{stock_code}" style="text-align: center;">
                <input type="submit" value="提交">
                <input type="button" value="开始随机" onclick="startRandom();">
                <input type="button" value="停止随机" onclick="stopRandom();">
                <script>
                </script>
            </div>
        </form>
        <div id="mainBody">
            <div id="leftPanel">
                <div id="maGraph" class="plotly-graph-div">
                    {ma_graph_url}
                </div>
                <div id="macdGraph" class="plotly-graph-div">
                    {macd_graph_url}
                </div>
                <div id="bollGraph" class="plotly-graph-div">
                    {boll_graph_url}
                </div>
                <div id="kdjGraph" class="plotly-graph-div">
                    {kdj_graph_url}
                </div>
            </div>
            <div id="rightPanel">
                <div id="maGraphRight" class="plotly-graph-div">
                    {ma_graph_url_right}
                </div>
                <div id="stockDataContainer">
                    {stock_data if stock_data else (f'<div>{error_message}</div>' if error_message else '<div>无可用数据。</div>')}
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(debug=True)

