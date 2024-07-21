# 创建 StockHistoryFetcher 实例
from contextlib import redirect_stdout
import datetime
import time
from utils.StockHistoryFetcher import StockHistoryFetcher
import io
import logging
import tushare as ts
from config import tushare_api_key
import sys
import os

# 设置日志
log_filename = f"logs/history_fetcher_{time.strftime('%Y%m%d')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# 使用 tushare 判断今天是否为交易日
ts.set_token(tushare_api_key)
pro = ts.pro_api()

today = time.strftime('%Y%m%d')
df = pro.trade_cal(exchange='', start_date=today, end_date=today)
if df.at[0, 'is_open'] != 1:
    logging.info(f"今天 ({today}) 不是交易日，程序退出。")
    exit()

logging.info("今天是交易日，开始执行股票历史数据抓取任务。")

fetcher = StockHistoryFetcher(None)

# 1. 更新股票基本信息
all_stock_codes_df = fetcher.fetch_all_stock_basic()

# 2. 更新后复权因子
# fetcher.fetch_and_store_hfq_factors()

# 3. 删除数据库中前一个交易日及以后的数据
fetcher.delete_previous_trading_day_data()

# 4. 同步历史数据

# 确保 utils 模块可以被找到
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# 设置日志
log_filename = f"logs/history_fetcher_{time.strftime('%Y%m%d')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# 创建 StockHistoryFetcher 实例
fetcher = StockHistoryFetcher(None)

from utils.sendEmail import EmailSender  # 导入邮件发送模块
email_sender = EmailSender()  # 创建邮件发送对象
success_message = "成功获取历史数据并写入数据库。"
error_message = "获取历史数据失败。"
start_time = datetime.datetime.now()  # 记录开始时间

# 初始化股票计数器
stock_count = 0
total_stocks = all_stock_codes_df.shape[0]

try:
    for index, row in all_stock_codes_df.iterrows():
        stock_code = row['symbol']
        fetcher.stock_code = stock_code  # 更新 fetcher 中的股票代码
        logging.info(f"正在抓取股票代码: {stock_code} 的历史数据")
        
        # 重定向 fetch_history 输出到日志文件
        log_stream = io.StringIO()
        with redirect_stdout(log_stream):
            fetcher.fetch_history()
        logging.info(log_stream.getvalue())
        
        # 更新股票计数器
        stock_count += 1
        logging.info(f"已处理 {stock_count}/{total_stocks} 股票")
    
    end_time = datetime.datetime.now()  # 记录结束时间
    duration = end_time - start_time  # 计算持续时间
    formatted_duration = str(duration).split('.')[0]  # 格式化持续时间为易读形式
    email_sender.send_email("历史数据更新通知", f"{success_message} 处理时间：{formatted_duration}", "745339023@qq.com", "Mr.Light")
    logging.info(f"{success_message} 处理时间：{formatted_duration}")
except Exception as e:
    email_sender.send_email("历史数据更新错误", f"{error_message} {str(e)} 正在处理的证券代码：{row['symbol']} 处理时间：{formatted_duration}", "745339023@qq.com", "Mr.Light")
    logging.error(f"{error_message} {str(e)} 正在处理的证券代码：{row['symbol']}")
finally:
    # 确保在抓取完成后释放资源
    logging.info("已完成所有股票历史数据的抓取。")
    del fetcher

logging.shutdown()
