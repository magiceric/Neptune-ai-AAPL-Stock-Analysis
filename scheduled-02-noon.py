from utils.StockHistoryFetcher import StockHistoryFetcher
import logging
import time
import tushare as ts
from config import tushare_api_key
from utils.sendEmail import EmailSender  # 导入 EmailSender 类
import datetime

# 配置日志
log_filename = f"logs/intraday_analysis_{time.strftime('%Y%m%d')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# 初始化 tushare
ts.set_token(tushare_api_key)
pro = ts.pro_api()

# 使用今天的日期作为搜索日期
search_date = datetime.datetime.now().strftime('%Y-%m-%d')
# search_date = '2024-05-22'

# 获取交易日历并检查今天是否是交易日，同时找出search_date前的倒数第10个交易日的日期，即T-10
search_date_no_dash = search_date.replace('-', '')
trade_cal = pro.trade_cal(exchange='', start_date='20200101', end_date=search_date_no_dash, is_open='1')
trade_cal = trade_cal.sort_values(by='cal_date', ascending=False)

# 输出trade_cal
print(trade_cal)

# 检查今天是否是交易日
today = time.strftime('%Y%m%d')
# search_date = today
if search_date_no_dash not in trade_cal['cal_date'].values:
    logging.info(f"search_date不是交易日，程序退出。search_date: {search_date}")
    exit()

# 找出T-10日期
if len(trade_cal) >= 10:
    t_minus_10_date = trade_cal.iloc[9]['cal_date']
    logging.info(f"T-10 日期为: {t_minus_10_date}")
else:
    logging.error("交易日历数据不足以计算T-10日期")
    t_minus_10_date = None

# 给t_minus_10_date加上年月日之间的破折号
if t_minus_10_date:
    t_minus_10_date = f"{t_minus_10_date[:4]}-{t_minus_10_date[4:6]}-{t_minus_10_date[6:]}"

# 打印输出T-10日期
print(f"T-10 日期为: {t_minus_10_date}")
# 创建 StockHistoryFetcher 实例
fetcher = StockHistoryFetcher(None)

# 更新实时数据
logging.info("开始更新实时数据。")
fetcher.update_realtime_stock_data()
# fetcher.update_realtime_etf_data()
logging.info("实时数据更新完成。")

import pandas as pd
from StockAnalysisTool import StockAnalysisTool
import sqlalchemy as db

database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock'
engine = db.create_engine(database_connection_string)
connection = engine.connect()
query = db.text("""
SELECT symbol, stockName FROM stockBasic
""")
allStocks = connection.execute(query).fetchall()
connection.close()

# # 配置盘中分析日志文件
# analysis_log_filename = f"logs/intraday_analysis_{time.strftime('%Y%m%d')}.log"
# logging.basicConfig(filename=analysis_log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# 记录开始盘中分析
logging.info("开始进行盘中分析。")

# 在进一步分析之前清空数据库中的 'goldencross' 表   
try:
    connection = engine.connect()
    connection.execute(db.text("TRUNCATE TABLE goldencross"))
    logging.info("成功清空了 'goldencross' 表。")
except Exception as e:
    logging.error(f"清空 'goldencross' 表失败: {e}")
finally:
    connection.close()

etf_results = pd.DataFrame(columns=['Symbol', 'Name', 'CrossDate', 'Depth', 'T+1 Profit'])

start_time = datetime.datetime.now()  # 记录处理开始时间
email_sender = EmailSender()  # 创建 EmailSender 对象
success_message = "成功完成盘中扫描。"
error_message = "盘中扫描失败。"
stock_count = 0  # 初始化股票计数

# 从allStocks中去掉ETF，也就是1和5开头的
filtered_stocks = [stock for stock in allStocks if not (stock.symbol.startswith('1') or stock.symbol.startswith('5'))]
allStocks = filtered_stocks

# try:
total_stocks = len(allStocks)
# allStocks = []
for stock in allStocks:
    stock_count += 1
    logging.info(f"处理股票 {stock_count}/{total_stocks}: {stock.symbol}")
    analysis_tool = StockAnalysisTool(stock.symbol)
    # last_negative_macd_cross = analysis_tool.find_last_negative_macd_crossover(search_date)
    last_negative_macd_cross = analysis_tool.find_last_negative_macd_crossover(search_date)
    if last_negative_macd_cross and last_negative_macd_cross[0].strftime('%Y-%m-%d') >= t_minus_10_date:
        profit_after_one_day = analysis_tool.calculate_profit_after_n_days(search_date, 1)
        result = analysis_tool.calculate_diff_slope_sum(search_date, 3)
        if result is not None:
            diff_slope_sum, volume_percentage, date_volume, previous_day_volume = result
        else:
            diff_slope_sum, volume_percentage, date_volume, previous_day_volume = None, None, None, None
        powerX = None
        vwap, previous_close_price, closing_price = analysis_tool.calculate_vwap(search_date)
        
        # 判断search_date当天成交量是否为之前一个交易日成交量的一倍以上
        if date_volume is not None and previous_day_volume is not None and date_volume > 2 * previous_day_volume:
            insert_data = {
                'miningDate': today,
                'targetDate': search_date,
                'stockCode': stock.symbol,
                'stockName': stock.stockName,
                'lastGCbelow0date': last_negative_macd_cross[0],
                'diffDepth': last_negative_macd_cross[3],
                't1Profit': profit_after_one_day,
                'diffSlopeSum': diff_slope_sum,
                'powerX': powerX,
                'vwap': vwap,
                'tM1Close': previous_close_price,
                'tClose': closing_price,
                'volumePercentage': volume_percentage
            }
            connection = engine.connect()
            query = db.text(
                "INSERT INTO goldencross (miningDate, targetDate, stockCode, stockName, lastGCbelow0date, diffDepth, t1Profit, diffSlopeSum, powerX, vwap, tM1Close, tClose, volumePercentage) "
                "VALUES (:miningDate, :targetDate, :stockCode, :stockName, :lastGCbelow0date, :diffDepth, :t1Profit, :diffSlopeSum, :powerX, :vwap, :tM1Close, :tClose, :volumePercentage)"
            ).bindparams(**insert_data)
            connection.execute(query)
            connection.commit()
            connection.close()
        del analysis_tool

end_time = datetime.datetime.now()  # 记录处理结束时间
processing_time = (end_time - start_time).total_seconds()  # 计算处理时间

# 查询数据库并格式化结果
# query = """
# SELECT stockCode,
#         stockName,
#         lastGCbelow0date       AS gcbzDate,
#         ROUND(diffDepth, 3)    AS diffDepth,
#         ROUND(diffSlopeSum, 3) AS diffSlopeSum,
#         ROUND(tClose, 3)       AS tClose,
#         ROUND(vwap, 3)         AS vwap,
#         CASE 
#             WHEN tOpen IS NULL OR tOpen = 0 THEN 0
#             ELSE ROUND((tClose - tOpen) / tOpen * 100, 2)
#         END AS priceChangePercentage,
#         ROUND(volumePercentage, 3)       AS volumePercentage
# FROM goldencross
# WHERE (diffDepth > 30
#     AND volumePercentage > 2)
#     OR volumePercentage > 5
# ORDER BY volumePercentage DESC;
# """
# query = """
# SELECT stockCode,
#         stockName,
#         lastGCbelow0date       AS gcbzDate,
#         ROUND(diffDepth, 3)    AS diffDepth,
#         ROUND(diffSlopeSum, 3) AS diffSlopeSum,
#         ROUND(tClose, 3)       AS tClose,
#         ROUND(vwap, 3)         AS vwap,
#         CASE 
#             WHEN tM1Close IS NULL OR tM1Close = 0 THEN 0
#             ELSE ROUND((tClose - tM1Close) / tM1Close * 100, 2)
#         END AS priceChangePercentage,
#         ROUND(volumePercentage, 3)       AS volumePercentage
# FROM goldencross
# WHERE volumePercentage > 8
# ORDER BY diffDepth DESC;
# """

# 2024-07-15，成交量占比放宽到大于6
query = """
SELECT stockCode,
        stockName,
        lastGCbelow0date       AS gcbzDate,
        ROUND(diffDepth, 3)    AS diffDepth,
        ROUND(diffSlopeSum, 3) AS diffSlopeSum,
        ROUND(tClose, 3)       AS tClose,
        ROUND(vwap, 3)         AS vwap,
        CASE 
            WHEN tM1Close IS NULL OR tM1Close = 0 THEN 0
            ELSE ROUND((tClose - tM1Close) / tM1Close * 100, 2)
        END AS priceChangePercentage,
        ROUND(volumePercentage, 3)       AS volumePercentage
FROM goldencross
WHERE volumePercentage > 6
ORDER BY diffDepth DESC;
"""
connection = engine.connect()
result = connection.execute(db.text(query))
connection.close()

# 将查询结果转换为HTML表格，并修改表头
df = pd.DataFrame(result.fetchall(), columns=result.keys())
df.columns = ['代码', '名称', '0下金叉日期', '下沉深度', '斜率总和', '最近价', '市场持仓成本', '当前涨跌幅', '成交量占比']

# 分离已经涨停的股票
limit_up_stocks = df[df['当前涨跌幅'] > 9.9]
other_stocks = df[df['当前涨跌幅'] <= 9.9]

# 设置文本类字段居中，数字类字段靠右对齐
df_style = other_stocks.style.set_properties(subset=['代码', '名称', '0下金叉日期', '成交量占比'], **{'text-align': 'center'})\
                    .set_properties(subset=['下沉深度', '斜率总和', '最近价', '市场持仓成本', '当前涨跌幅'], **{'text-align': 'right'})\
                    .set_table_styles([dict(selector='th', props=[('text-align', 'center')]),
                                        dict(selector='td style="border: 1px solid #999; border-collapse: collapse;"', props=[('padding', '2px')]),
                                        dict(selector='table', props=[('border-collapse', 'collapse'), ('border', '1px solid #999'), ('border-spacing', '0')])])  # 所有表头标题居中
html_table = df_style.to_html(classes='table table-striped table-hover', border=1)

# 设置已经涨停股票的表格样式
limit_up_style = limit_up_stocks.style.set_properties(subset=['代码', '名称', '0下金叉日期', '成交量占比'], **{'text-align': 'center'})\
                    .set_properties(subset=['下沉深度', '斜率总和', '最近价', '市场持仓成本'], **{'text-align': 'right'})\
                    .set_table_styles([dict(selector='th', props=[('text-align', 'center')]),
                                        dict(selector='td style="border: 1px solid #999; border-collapse: collapse;"', props=[('padding', '2px')]),
                                        dict(selector='table', props=[('border-collapse', 'collapse'), ('border', '1px solid #999'), ('border-spacing', '0')])])  # 所有表头标题居中
html_limit_up_table = limit_up_style.to_html(classes='table table-striped table-hover', border=1)

# 添加内联样式
html_table = html_table.replace('<table ', '<table style="border: 1px solid #999; border-collapse: collapse; padding: 2px;" ')
html_limit_up_table = html_limit_up_table.replace('<table ', '<table style="border: 1px solid #999; border-collapse: collapse; padding: 2px;" ')

# 发送邮件
email_body = f"{success_message} 处理时间: {processing_time} 秒<br>观察日期: {search_date}<br><br>值得关注的股票:<br>{html_table}<br><br>已经涨停的股票:<br>{html_limit_up_table}"
email_sender.send_email("盘中扫描完成", email_body, "745339023@qq.com", "Mr.Light", html=True)
# except Exception as e:
#     email_sender.send_email("盘中扫描错误", f"{error_message} {str(e)} 正在处理的证券代码：{stock.symbol} 处理时间：{processing_time} 秒", "745339023@qq.com", "Mr.Light")
#     logging.error(f"{error_message} {str(e)} 正在处理的证券代码：{stock.symbol}")

engine.dispose()

logging.shutdown()

