import os, sys

sys.path.insert(
    1, os.path.join(sys.path[0], "..")
)  # Add the parent directory to the path at position 1
from config import tushare_api_key
import tushare as ts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import sqlalchemy
import datetime


# import msvcrt  # Importing module for keypress detection
import time  # Importing module for sleep function
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData  # Import MetaData
import platform  # 导入platform模块以检查操作系统

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号'-'显示为方块的问题

print("设置Tushare令牌...")
ts.set_token(tushare_api_key)
pro = ts.pro_api()

plot_directory = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plot_directory, exist_ok=True)

database_connection_string = (
    "mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock"
)
engine = create_engine(database_connection_string)
Session = sessionmaker(bind=engine)
session = Session()
metadata = MetaData()
metadata.reflect(bind=engine)
goldencross = metadata.tables["goldencross"]

# 使用tushare的trade_cal接口获取T-1, T-2, 和 T-3交易日日期，并打印出来
today_date = pd.Timestamp.today()  # 使用当前日期
trade_cal = pro.trade_cal(
    exchange="",
    start_date=(today_date - pd.Timedelta(days=60)).strftime("%Y%m%d"),
    end_date=today_date.strftime("%Y%m%d"),
)
# print("交易日历:", trade_cal)
trade_days = trade_cal[trade_cal["is_open"] == 1]["cal_date"]  # 筛选出开市的日期
# print("开市的日期:", trade_days)
T_minus_1_date = trade_days.iloc[1]  # 获取T-1的交易日日期
T_minus_2_date = trade_days.iloc[2]  # 获取T-2的交易日日期
T_minus_3_date = trade_days.iloc[3]  # 获取T-3的交易日日期
print("T-1日:", T_minus_1_date)
print("T-2日:", T_minus_2_date)
print("T-3日:", T_minus_3_date)

# 用于判断均线是否上穿的灵敏度参数，取值0-1之间，越大越可能丢失，越小越可能误判
# 目前仅用于成交量均线判断，是否需要在价格均线，需要再观察
# MA_CROSS_PRECISION = 0.9 # 误判太多，风险增大，暂时弃用
MA_CROSS_PRECISION = 1

# 提供一个简单的文本菜单，等待用户输入
input_prompt = "按回车键继续，输入'q'退出程序："
user_input = input(input_prompt)
if user_input.lower() == "q":
    print("程序已退出。")
    exit()


global request_count, last_reset_time
if "request_count" not in globals():
    request_count = 0  # Initialize the request count if it doesn't exist
    last_reset_time = time.time()  # Record the current time as the last reset time


def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    df["DIFF"] = df["close"].ewm(span=short_period, adjust=False).mean() - df["close"].ewm(span=long_period, adjust=False).mean()
    df["DEA"] = df["DIFF"].ewm(span=signal_period, adjust=False).mean()
    df["MACD"] = (df["DIFF"] - df["DEA"]) * 2
    df["Volume"] = df["vol"]  # 添加成交量
    return df


def plot_macd_with_golden_cross(stock_code, stock_name):

    print(f"获取{stock_code}的每日历史报价数据...")
    # 使用今天往回1年的数据
    end_date = pd.Timestamp.today().strftime("%Y%m%d")
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=365)).strftime("%Y%m%d")

    global request_count, last_reset_time
    if request_count >= 190:
        elapsed_time = (
            time.time() - last_reset_time
        )  # Calculate the elapsed time since the last reset
        wait_time = max(
            0, 60 - elapsed_time
        )  # Calculate the required wait time, ensuring it's not negative
        if wait_time > 0:
            print(f"已达到API请求限制，等待{wait_time}秒以继续...")
        time.sleep(
            wait_time
        )  # Wait for the calculated time if the request count reaches 190
        request_count = 0  # Reset the request count after waiting
        last_reset_time = time.time()  # Update the last reset time after waiting
    df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
    request_count += 1  # Increment the request count after a successful API call

    try:
        print("按日期排序数据框...")
        df_sorted = df.sort_values(by="trade_date")
    except KeyError as e:
        print(f"警告: 数据框中缺少期望的列 {e}. 继续处理其他数据.")
        return None

    print("将trade_date转换为datetime格式...")
    df_sorted["trade_date"] = pd.to_datetime(df_sorted["trade_date"])

    print("计算MACD值和成交量...")
    df_macd = calculate_macd(df_sorted)
    # print("df_macd", df_macd)
    min_diff_of_last_365 = df_macd["DIFF"].min()
    # print(f"过去一年最小的DIFF值: {min_diff_of_last_365}")

    print("绘制MACD图和成交量...")
    fig, ax1 = plt.subplots(figsize=(24, 3))  # 修改图的大小为1600x200
    ax2 = ax1.twinx()  # 创建第二个坐标轴
    ax1.plot(df_macd["trade_date"], df_macd["DIFF"], label="DIFF")
    ax1.plot(df_macd["trade_date"], df_macd["DEA"], label="DEA")

    # 根据DIFF和DEA的位置调整成交量的颜色和位置
    for index, row in df_macd.iterrows():
        if row["DIFF"] > row["DEA"]:
            color = "red"
            ax2.bar(row["trade_date"], row["Volume"], color=color, alpha=0.3)
        else:
            color = "green"
            ax2.bar(row["trade_date"], -row["Volume"], color=color, alpha=0.3)

    # 确保x轴上下方空间相等
    max_volume = df_macd["Volume"].max()
    ax2.set_ylim(
        -max_volume * 1.1, max_volume * 1.1
    )  # 设置y轴的最大值和最小值，使其绝对值相等并适当放大

    max_macd = max(df_macd["DIFF"].max(), abs(df_macd["DIFF"].min()))
    ax1.set_ylim(
        -max_macd * 1.1, max_macd * 1.1
    )  # 设置MACD y轴的最大值和最小值，使其绝对值相等并适当放大

    print("突出显示金叉点和死叉点...")
    # 修改金叉的计算逻辑，确保只有当MACD从下方穿越Signal Line时才被视为金叉
    golden_crosses = df_macd[
        (df_macd["DIFF"] > df_macd["DEA"])
        & (df_macd["DIFF"].shift(1) < df_macd["DEA"].shift(1))
    ]
    # 新增死叉的计算逻辑，确保只有当MACD从上方穿越Signal Line时才被视为死叉
    death_crosses = df_macd[
        (df_macd["DIFF"] < df_macd["DEA"])
        & (df_macd["DIFF"].shift(1) > df_macd["DEA"].shift(1))
    ]

    # 初始化变量以存储最近一次0线下方金叉的日期和偏移值，且这个金叉之后没有任何其他叉
    last_gc_below_0_date = None
    gcbz_offset = None
    last_gc_date = None
    last_gc_offset = None

    # 合并金叉和死叉，按照trade_date排序
    crosses = pd.concat([golden_crosses, death_crosses]).sort_values(
        by="trade_date"
    )  # 修改为按照trade_date排序
    # print("合并后的金叉和死叉数据：\n", crosses)
    # 从所有金叉中找到最后一个0线下方的金叉，且这个金叉之后没有任何其他叉
    for index, row in crosses[::-1].iterrows():  # 修改为反向遍历以正确处理时间顺序
        if (
            index in golden_crosses.index
        ):  # 修改检查逻辑，确保只有当行索引在golden_crosses的索引中时才被视为金叉
            if last_gc_date is None:
                subsequent_crosses = crosses[
                    crosses["trade_date"] > row["trade_date"]
                ]  # 使用trade_date来检查这个金叉之后是否有其他叉
                if subsequent_crosses.empty:  # 如果这个金叉之后没有其他叉
                    last_gc_date = row["trade_date"]
                    last_gc_offset = row["DIFF"]

            if row["DIFF"] < 0:  # 忽略0线上方的金叉和死叉
                subsequent_crosses = crosses[
                    crosses["trade_date"] > row["trade_date"]
                ]  # 使用trade_date来检查这个金叉之后是否有其他叉
                if subsequent_crosses.empty:  # 如果这个金叉之后没有其他叉
                    # print(
                    #     f"找到最后一个0线下方的金叉: 日期={row['trade_date']}, MACD偏移={row['MACD']}"
                    # )
                    last_gc_below_0_date = row["trade_date"]
                    gcbz_offset = row["DIFF"]
                    break  # 找到最后一个0线下方的金叉后退出循环

    # 统计last_gc_date之前连续的“绿柱”成交额
    continuous_green_turnover = None
    ma_cross_date = None
    if last_gc_date:
        reversed_df_macd = df_macd.iloc[::-1]  # 颠倒df_macd的顺序并保存到新的数组
        last_gc_date_index = reversed_df_macd[reversed_df_macd["trade_date"] == last_gc_date].index[0]
        # print('last_gc_date_index', last_gc_date_index)
        continuous_green_turnover = 0
        for i in range(last_gc_date_index + 1, len(reversed_df_macd)):  # 修正为向后统计
            # print('i, DIFF, DEA : ', i, reversed_df_macd.iloc[i]["DIFF"], reversed_df_macd.iloc[i]["DEA"])
            if reversed_df_macd.iloc[i]["DIFF"] < reversed_df_macd.iloc[i]["DEA"]:
                continuous_green_turnover += reversed_df_macd.iloc[i]["amount"]  # 使用成交额代替成交量
            else:
                break
        # print(f"在{last_gc_date}之前连续的绿柱成交额: {continuous_green_turnover}")

        # 如果last_gc_date存在，找出T+1之后(T+1不算)出现5天均线上穿10天均线的日期
        if last_gc_date:
            # 通过tushare接口确定last_gc_date的下一个交易日（即有效的T+1）

            if request_count >= 190:
                elapsed_time = (
                    time.time() - last_reset_time
                )  # Calculate the elapsed time since the last reset
                wait_time = max(
                    0, 60 - elapsed_time
                )  # Calculate the required wait time, ensuring it's not negative
                if wait_time > 0:
                    print(f"已达到API请求限制，等待{wait_time}秒以继续...")
                time.sleep(
                    wait_time
                )  # Wait for the calculated time if the request count reaches 190
                request_count = 0  # Reset the request count after waiting
                last_reset_time = time.time()  # Update the last reset time after waiting
            next_trade_day = pro.trade_cal(
                exchange="",
                start_date=(pd.to_datetime(last_gc_date) + pd.Timedelta(days=1)).strftime("%Y%m%d"),
                end_date=(pd.to_datetime(last_gc_date) + pd.Timedelta(days=10)).strftime("%Y%m%d"),
            )
            request_count += 1
            next_trade_day = next_trade_day[next_trade_day["is_open"] == 1]["cal_date"].iloc[0]
            print('next_trade_day after last_gc_date : ', next_trade_day)

            df_ma = df_macd[['trade_date', 'close', 'amount']].copy()
            df_ma['MA5_close'] = df_ma['close'].rolling(window=5).mean().round(2)
            df_ma['MA10_close'] = df_ma['close'].rolling(window=10).mean().round(2)
            ma_crosses_close = df_ma[
                (df_ma['MA5_close'] > df_ma['MA10_close']) & (df_ma['MA5_close'].shift(1) <= df_ma['MA10_close'].shift(1))
            ]
            ma_cross_after_last_gc = ma_crosses_close[ma_crosses_close['trade_date'] > pd.to_datetime(next_trade_day)]
            if not ma_cross_after_last_gc.empty:
                ma_cross_date = ma_cross_after_last_gc.iloc[0]['trade_date']
            else:
                # 如果价格的移动平均线没有交叉，检查成交量的移动平均线
                df_ma['MA5_amount'] = df_ma['amount'].rolling(window=5).mean().round(2)
                df_ma['MA10_amount'] = df_ma['amount'].rolling(window=10).mean().round(2)
                # with pd.option_context('display.max_rows', None):  # Ensure all rows are displayed
                #     print("Last 30 rows of MA5 and MA10 for Amount:")
                #     print(df_ma[['MA5_amount', 'MA10_amount']].tail(30))
                ma_crosses_amount = df_ma[
                    (df_ma['MA5_amount'] > df_ma['MA10_amount']) & (df_ma['MA5_amount'].shift(1) * MA_CROSS_PRECISION <= df_ma['MA10_amount'].shift(1))
                ]
                ma_cross_after_last_gc = ma_crosses_amount[ma_crosses_amount['trade_date'] > pd.to_datetime(next_trade_day)]
                if not ma_cross_after_last_gc.empty:
                    ma_cross_date = ma_cross_after_last_gc.iloc[0]['trade_date']
                else:
                    ma_cross_date = None
            print(f"在{last_gc_date}之后5天均线上穿10天均线的日期: {ma_cross_date}")

    # 对找到的最后一个0线下方金叉进行标注
    for index, row in golden_crosses.iterrows():
        if (
            row["trade_date"] == last_gc_below_0_date
        ):  # 只为last_gc_below_0_date对应的金叉加上“B”字符
            ax1.annotate(
                "B",
                xy=(
                    mdates.date2num(row["trade_date"]),
                    min(row["DIFF"], row["DEA"]),
                ),
                xytext=(
                    mdates.date2num(row["trade_date"]),
                    min(row["DIFF"], row["DEA"]) - 0.1 * 1.1,
                ),  # 尾巴长度调整为不超过10个像素并适当放大
                arrowprops=dict(
                    facecolor="red", shrink=0.05, headlength=5, headwidth=5, width=1
                ),
            )
        else:

            ax1.annotate(
                "",
                xy=(
                    mdates.date2num(row["trade_date"]),
                    min(row["DIFF"], row["DEA"]),
                ),
                xytext=(
                    mdates.date2num(row["trade_date"]),
                    min(row["DIFF"], row["DEA"]) - 0.1 * 1.1,
                ),  # 尾巴长度调整为不超过10个像素并适当放大
                arrowprops=dict(
                    facecolor="red", shrink=0.05, headlength=5, headwidth=5, width=1
                ),
            )
    for index, row in death_crosses.iterrows():
        ax1.annotate(
            "",
            xy=(
                mdates.date2num(row["trade_date"]),
                max(row["DIFF"], row["DEA"]),
            ),
            xytext=(
                mdates.date2num(row["trade_date"]),
                max(row["DIFF"], row["DEA"]) + 0.1 * 1.1,
            ),  # 尾巴长度调整为不超过10个像素并适当放大
            arrowprops=dict(
                facecolor="green", shrink=0.05, headlength=5, headwidth=5, width=1
            ),
        )

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title(
        f"{stock_code} {stock_name} - MACD图与金叉、死叉及成交量"
    )  # 在标题中加入证券名称
    ax1.set_xlabel("日期")
    ax1.set_ylabel("DIFF值")
    ax2.set_ylabel("成交量")
    if (
        last_gc_below_0_date is not None
        and pd.to_datetime(last_gc_below_0_date).strftime("%Y%m%d")
        in [T_minus_1_date, T_minus_2_date, T_minus_3_date]
        and abs(gcbz_offset) > abs(min_diff_of_last_365) / 2
    ):
        current_date_subdir = datetime.date.today().strftime("%Y%m%d")
        full_path = os.path.join(
            plot_directory,
            current_date_subdir,
            f"{stock_code}_MACD_{current_date_subdir}.png",
        )
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        plt.savefig(
            full_path
        )  # 仅当last_gc_below_0_date为T-1或T-2或T-3交易日时，且gcbz_offset的绝对值大于min_diff_of_last_365绝对值的二分之一时，保存图表到指定的子目录中
        plt.close()  # 关闭图表，不显示
    return (
        last_gc_below_0_date,
        gcbz_offset,
        min_diff_of_last_365,
        last_gc_date,
        last_gc_offset,
        continuous_green_turnover,
        ma_cross_date
    )


# 遍历深沪两市所有“0”和“6”开头的股票
# 使用tushare获取股票列表
stock_list = pro.query(
    "stock_basic", exchange="", list_status="L", fields="ts_code"
).query("ts_code.str.startswith('0') or ts_code.str.startswith('6')", engine="python")
# 剔除3开头、01开头和688开头的品种
stock_list = stock_list[
    ~stock_list["ts_code"].str.startswith(("3", "01", "688", "689"))
]

import time  # 导入time模块以便记录时间

stock_processed_count = 0  # 初始化处理过的股票计数器
total_start_time = time.time()  # 记录处理开始的总时间

for index, row in stock_list.iterrows():
    loop_start_time = time.time()  # 记录当前循环开始的时间

    stock_code = row["ts_code"]

    # 检查今天是否已经测算过
    today = datetime.date.today().strftime("%Y-%m-%d")  # 修正日期格式以匹配数据库格式
    result = (
        session.query(goldencross)
        .filter(
            goldencross.c.stockCode == stock_code, goldencross.c.miningDate == today
        )
        .first()
    )
    if result:
        print(f"{stock_code} 今天已经测算过了")
        continue

    # 从tushare获取股票的基本信息以解决stock_name未定义的问题
    current_time = datetime.datetime.now()
    if (
        current_time - datetime.datetime.fromtimestamp(last_reset_time)
    ).total_seconds() < 60 and request_count >= 190:
        wait_time = (
            60
            - (
                current_time - datetime.datetime.fromtimestamp(last_reset_time)
            ).total_seconds()
        )
        print(f"达到请求限制，等待{wait_time:.2f}秒后继续...")
        time.sleep(wait_time)  # 等待直到1分钟结束
        request_count = 0  # 重置计数器
        last_reset_time = time.time()  # 更新清零时间为当前时间戳
    elif (
        current_time - datetime.datetime.fromtimestamp(last_reset_time)
    ).total_seconds() >= 60:
        request_count = 0  # 重置计数器
        last_reset_time = time.time()  # 更新清零时间为当前时间戳
    stock_info = pro.stock_basic(ts_code=stock_code, fields="name")
    request_count += 1  # 更新请求计数

    if stock_info.empty:
        print(f"{stock_code} 的基本信息未找到，跳过此股票。")
        continue
    stock_name = stock_info["name"].iloc[0]

    # 调用类似plot_macd_with_golden_cross的函数计算金叉，如果所有结果都是None，尝试最多3次
    attempt_count = 0
    last_gc_below_0_date = None
    gcbz_offset = None
    min_diff_of_last_365 = None
    last_gc_date = None
    gc_offset = None
    continuous_green_turnover = None
    ma_cross_date = None

    while attempt_count < 3:
        (
            last_gc_below_0_date,
            gcbz_offset,
            min_diff_of_last_365,
            last_gc_date,
            gc_offset,
            continuous_green_turnover,
            ma_cross_date
        ) = plot_macd_with_golden_cross(stock_code, stock_name)
        
        if all(value is None for value in [
            last_gc_below_0_date,
            gcbz_offset,
            min_diff_of_last_365,
            last_gc_date,
            gc_offset,
            continuous_green_turnover,
            ma_cross_date
        ]):
            print(f"尝试 {attempt_count + 1}: 未找到有效数据，重试中............................................................")
            attempt_count += 1
        else:

            break

    print(
        last_gc_below_0_date,
        gcbz_offset,
        min_diff_of_last_365,
        last_gc_date,
        gc_offset,
        continuous_green_turnover,
        ma_cross_date
    )

    if last_gc_date is None and gc_offset is None:
        insert_query = goldencross.insert().values(
            miningDate=today,
            stockCode=stock_code,
            stockName=stock_name,
            lastGCbelow0date=None,
            gcbzOffset=None,
            minDIFFofLast365=min_diff_of_last_365,
            lastGCdate=None,
            gcOffset=None,
            continuousGreenTurnover=None,
            maCrossDate=None
        )
    else:
        formatted_last_gc_below_0_date = (
            last_gc_below_0_date.strftime("%Y-%m-%d")
            if last_gc_below_0_date is not None
            else None
        )
        insert_query = goldencross.insert().values(
            miningDate=today,
            stockCode=stock_code,
            stockName=stock_name,
            lastGCbelow0date=formatted_last_gc_below_0_date,
            gcbzOffset=gcbz_offset,
            minDIFFofLast365=min_diff_of_last_365,
            lastGCdate=last_gc_date,
            gcOffset=gc_offset,
            continuousGreenTurnover=continuous_green_turnover,
            maCrossDate=ma_cross_date
        )
    print("INSERT statement : ", insert_query)

    conn = engine.connect()  # 创建连接
    conn.execute(insert_query)  # 使用连接对象直接执行SQL语句
    conn.commit()
    conn.close()  # 关闭连接
    print(f"{stock_code} 的金叉数据已成功写入数据库")

    stock_processed_count += 1  # 更新处理过的股票计数器

    loop_end_time = time.time()  # 记录当前循环结束的时间
    print(f"处理{stock_code}耗时：{loop_end_time - loop_start_time:.2f}秒")

    # # Check for keypress after each stock calculation
    # if msvcrt.kbhit():
    #     if msvcrt.getch() == b'q':  # If 'q' is pressed
    #         print("用户已请求中断运行。")
    #         break  # Exit the loop

    print("\n\n")

total_end_time = time.time()  # 记录处理结束的总时间
print(f"总耗时：{total_end_time - total_start_time:.2f}秒")
# 关闭数据库会话和连接
session.close()
engine.dispose()
