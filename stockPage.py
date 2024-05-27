from flask import Flask, render_template_string, request
import akshare as ak
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import mplfinance as mpf

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    today_date = datetime.now().strftime('%Y-%m-%d')
    stock_data = None
    ma_graph_url = None
    macd_graph_url = None
    boll_graph_url = None  # Added Bollinger Bands graph URL
    stock_code = request.form.get('stock_code', '000001')
    date_str = request.form.get('date', today_date)
    stock_name = ""  # Initialize stock name variable
    if request.method == 'POST':
        if stock_code and date_str:
            try:
                # Fetch stock name using akshare
                stock_info = ak.stock_info_a_code_name()
                stock_name = stock_info.loc[stock_info['code'] == stock_code, 'name'].values[0] if not stock_info.loc[stock_info['code'] == stock_code].empty else "Unknown Stock"

                date = datetime.strptime(date_str, '%Y-%m-%d')
                trading_days = ak.tool_trade_date_hist_sina()
                trading_days['trade_date'] = pd.to_datetime(trading_days['trade_date'], format='%Y%m%d')
                trading_days = trading_days[trading_days['trade_date'] <= date]
                if len(trading_days) >= 60:
                    start_date = trading_days.iloc[-60]['trade_date'].strftime('%Y%m%d')
                else:
                    start_date = trading_days.iloc[0]['trade_date'].strftime('%Y%m%d')
                
                end_date = date
                stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_date, end_date=end_date.strftime('%Y%m%d'), adjust="qfq")
                stock_zh_a_hist_df['成交量'] = stock_zh_a_hist_df['成交量'].astype(float)
                stock_zh_a_hist_df['成交额'] = stock_zh_a_hist_df['成交额'].astype(float)
                
                display_df = stock_zh_a_hist_df.copy()
                display_df.sort_values(by='日期', ascending=False, inplace=True)
                # Format '成交量' and '成交额' for display in millions with comma separators
                display_df['成交量'] = (display_df['成交量'] / 10000).apply(lambda x: "{:,.0f}".format(x))
                display_df['成交额'] = (display_df['成交额'] / 10000).apply(lambda x: "{:,.0f}".format(x))
                
                # Splitting the DataFrame into header and data
                header_df = pd.DataFrame(columns=display_df.columns)
                data_df = display_df.copy()
                
                # Generating HTML without header for data_df
                data_html = data_df.to_html(classes='stock-table', index=False, justify='center', border=0, header=False)
                # Generating HTML with only header for header_df
                header_html = header_df.to_html(classes='stock-table', index=False, justify='center', border=0, header=True)
                
                # Wrapping data_html in a div with a scrollbar and making it fill the remaining height
                data_html_with_scroll = f'<div style="overflow-y: auto; flex-grow: 1;">{data_html}</div>'
                
                # Combining both HTML parts with header fixed
                stock_data = header_html + data_html_with_scroll
                
                stock_zh_a_hist_df['5日均线'] = stock_zh_a_hist_df['收盘'].rolling(window=5).mean()
                stock_zh_a_hist_df['10日均线'] = stock_zh_a_hist_df['收盘'].rolling(window=10).mean()

                # Convert the DataFrame to the format required by mplfinance
                stock_zh_a_hist_df.index = pd.to_datetime(stock_zh_a_hist_df['日期'])
                stock_zh_a_hist_df.rename(columns={'开盘': 'Open', '收盘': 'Close', '最高': 'High', '最低': 'Low', '成交量': 'Volume'}, inplace=True)
                apdict = mpf.make_addplot(stock_zh_a_hist_df[['5日均线', '10日均线']], secondary_y=False)

                # Plotting candlestick chart with moving averages and volume
                fig, axes = mpf.plot(stock_zh_a_hist_df,
                                     type='candle',
                                     style='charles',
                                     figsize=(20, 8),
                                     addplot=apdict,
                                     title=f"Stock Price and Moving Averages for {stock_code}",
                                     ylabel='Price',
                                     volume=True,
                                     returnfig=True)
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                ma_graph_url = base64.b64encode(img.getvalue()).decode()

                # MACD calculation
                exp1 = stock_zh_a_hist_df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = stock_zh_a_hist_df['Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                macd_hist = macd - signal

                # Plotting MACD using mplfinance
                apds = [mpf.make_addplot(macd, panel=0, color='blue', ylabel='MACD'),
                        mpf.make_addplot(signal, panel=0, color='orange'),
                        mpf.make_addplot(macd_hist, panel=0, type='bar', color=['red' if v >= 0 else 'green' for v in macd_hist], secondary_y=False)]
                
                # Create an empty DataFrame for the main panel to hide the stock price line
                empty_data = pd.DataFrame(index=stock_zh_a_hist_df.index)
                empty_data['Open'] = pd.Series([float('nan')] * len(stock_zh_a_hist_df), index=stock_zh_a_hist_df.index)
                empty_data['High'] = pd.Series([float('nan')] * len(stock_zh_a_hist_df), index=stock_zh_a_hist_df.index)
                empty_data['Low'] = pd.Series([float('nan')] * len(stock_zh_a_hist_df), index=stock_zh_a_hist_df.index)
                empty_data['Close'] = pd.Series([float('nan')] * len(stock_zh_a_hist_df), index=stock_zh_a_hist_df.index)

                fig, axes = mpf.plot(empty_data,
                                     type='line',
                                     style='charles',
                                     figsize=(20, 8),
                                     addplot=apds,
                                     title=f"MACD for {stock_code}",
                                     ylabel='MACD',
                                     volume=False,
                                     returnfig=True)
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                macd_graph_url = base64.b64encode(img.getvalue()).decode()

                # Bollinger Bands calculation
                mid_band = stock_zh_a_hist_df['Close'].rolling(window=20).mean()
                std_dev = stock_zh_a_hist_df['Close'].rolling(window=20).std()
                upper_band = mid_band + (std_dev * 2)
                lower_band = mid_band - (std_dev * 2)

                # Plotting Bollinger Bands using mplfinance
                apds_boll = [mpf.make_addplot(mid_band, color='blue', ylabel='Bollinger Bands'),
                             mpf.make_addplot(upper_band, color='green'),
                             mpf.make_addplot(lower_band, color='red')]
                fig, axes = mpf.plot(stock_zh_a_hist_df,
                                     type='line',
                                     style='charles',
                                     figsize=(20, 6),
                                     addplot=apds_boll,
                                     title=f"Bollinger Bands for {stock_code}",
                                     ylabel='Price',
                                     volume=False,
                                     returnfig=True)
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                boll_graph_url = base64.b64encode(img.getvalue()).decode()

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
            }}
            form {{
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                display: flex;
                align-items: center;
                justify-content: flex-end; /* Align form elements to the right */
                margin-bottom: 20px;
            }}
            input[type="text"], input[type="submit"] {{
                padding: 10px;
                margin: 0 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
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
                display: flex;000620
                margin-top: 0; /* Removed top margin */
            }}
            #maGraph img, #macdGraph img, #bollGraph img, #volumeGraph img {{
                width: 100%;
                height: 100%;
                object-fit: contain; /* Ensures the image is scaled correctly within the fixed dimensions */
            }}
            #stockDataContainer {{
                height: 100%; /* Set height to 40% of its parent */
                overflow-y: auto; /* Enable vertical scrolling */
                max-height: 86vh; /* Maximum height to prevent content overflow */
                display: flex;
                flex-direction: column;
                justify-content: space-between; /* Distribute space evenly around the content */
            }}
        </style>
    </head>
    <body>
        <form method="post">
            <label><b>{stock_name}</b></label>&nbsp&nbsp
            股票代码: <input type="text" name="stock_code" value="{stock_code}">
            日期 (YYYY-MM-DD): <input type="text" name="date" value="{date_str}">
            <input type="submit" value="提交">
        </form>
        <div id="mainBody">
            <div id="leftPanel">
                <div id="maGraph">
                    <img src="data:image/png;base64,{ma_graph_url}" alt="Stock Moving Averages">
                </div>
                <div id="macdGraph">
                    <img src="data:image/png;base64,{macd_graph_url}" alt="MACD Chart">
                </div>
                <div id="bollGraph">  
                    <img src="data:image/png;base64,{boll_graph_url}" alt="Bollinger Bands Chart">
                </div>
            </div>
            <div id="rightPanel">
                <div id="stockDataContainer">
                    {stock_data if stock_data else '<div>No data available.</div>'}
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(debug=True)
