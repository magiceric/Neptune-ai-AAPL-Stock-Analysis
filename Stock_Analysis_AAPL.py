#!/usr/bin/env python
# coding: utf-8

# In[55]:


from config import api_key, neptune_key  # 从配置文件导入API密钥
import requests  # 导入requests库用于API请求
import pandas as pd  # 导入pandas库用于数据处理
import numpy as np  # 导入numpy库用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
get_ipython().run_line_magic('matplotlib', 'inline')  # 设置matplotlib图表内嵌显示
import os  # 导入os库用于操作系统功能，如文件路径
from sklearn.preprocessing import StandardScaler  # 从sklearn库导入标准化模块
import tensorflow as tf  # 导入tensorflow库用于深度学习模型
from tensorflow.keras.layers import LSTM, Input, Dense, Dropout, Activation  # 从tensorflow库导入深度学习层
from tensorflow.keras.datasets import mnist  # 从tensorflow库导入MNIST数据集
from tensorflow.keras.models import Sequential  # 从tensorflow库导入序贯模型
from tensorflow.keras.utils import to_categorical as np_utils  # 从tensorflow库导入数据预处理工具


# In[56]:


import neptune  # 导入neptune库用于实验跟踪

run = neptune.init_run(
    project="magiceric/aaa",  # 指定Neptune项目
    api_token=neptune_key,  # 指定Neptune API密钥
    capture_stdout=True,  # 捕获标准输出
    capture_stderr=True,  # 捕获标准错误
    capture_traceback=True,  # 捕获回溯
    capture_hardware_metrics=True  # 捕获硬件指标
)  # 初始化Neptune运行

params = {"learning_rate": 0.001, "optimizer": "Adam"}  # 定义模型参数
run["parameters"] = params  # 将参数记录到Neptune

for epoch in range(10):  # 循环记录训练损失
    run["train/loss"].log(0.9 ** epoch)

run["eval/f1_score"] = 0.66  # 记录评估F1分数

run.stop()  # 停止Neptune运行


# In[57]:


run.stop()  # 再次调用停止Neptune运行（重复调用，应删除）


# In[58]:


# 构造请求URL，获取股票数据
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=000001.SZ&outputsize=full&apikey=' + api_key
r = requests.get(url)  # 发送请求
data = r.json()  # 解析JSON数据

print(data)  # 打印数据


# In[59]:


sz000001_df_json = pd.DataFrame.from_dict(data, orient='index')  # 将JSON数据转换为DataFrame
sz000001_df_json  # 显示DataFrame


# In[60]:


# 构造请求URL，获取CSV格式的股票数据
sz000001_csv = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=000001.SZ&outputsize=full&apikey=' + api_key + '&datatype=csv'
sz000001_csv_df = pd.read_csv(sz000001_csv)  # 读取CSV数据
print(sz000001_csv_df)  # 打印DataFrame


# In[61]:


len(sz000001_csv_df.close)  # 打印收盘价的数量


# In[62]:


df_copy = sz000001_csv_df.copy()  # 复制DataFrame


# In[63]:


date_close_df = df_copy.filter(['timestamp','close'], axis=1)  # 筛选出时间戳和收盘价
date_close_df  # 显示筛选结果


# In[64]:


date_close_df.tail(5)  # 显示最后5条记录


# In[65]:


stockprices = date_close_df  # 将筛选结果赋值给stockprices


# In[66]:


#### 时间序列的训练测试分割 ####
test_ratio = 0.2  # 测试集比例
training_ratio = 1 - test_ratio  # 训练集比例

train_size = int(training_ratio * len(stockprices))  # 计算训练集大小
test_size = int(test_ratio * len(stockprices))  # 计算测试集大小
print("train_size: " + str(train_size))  # 打印训练集大小
print("test_size: " + str(test_size))  # 打印测试集大小

train = stockprices[:train_size][['timestamp', 'close']]  # 划分训练集
test = stockprices[train_size:][['timestamp', 'close']]  # 划分测试集


# In[67]:


## 将时间序列数据分割为训练序列X和输出值Y
def extract_seqX_outcomeY(data, N, offset):
    """
    将时间序列分割为训练序列X和输出值Y
    参数:
        data - 数据集
        N - 窗口大小，例如，50表示50天的历史股价
        offset - 开始分割的位置
    """
    X, y = [], []
    
    for i in range(offset, len(data)):  # 循环分割数据
        X.append(data[i-N:i])
        y.append(data[i])
    
    return np.array(X), np.array(y)  # 返回分割结果


# In[68]:


#### 计算指标RMSE和MAPE ####
def calculate_rmse(y_true, y_pred):
    """
    计算均方根误差(RMSE)
    """
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))  # 计算RMSE                   
    return rmse  # 返回RMSE

def calculate_mape(y_true, y_pred): 
    """
    计算平均绝对百分比误差(MAPE)%
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)  # 转换为numpy数组   
    mape = np.mean(np.abs((y_true-y_pred) / y_true))*100  # 计算MAPE   
    return mape  # 返回MAPE


# In[69]:


def calculate_perf_metrics(var, logNeptune=True, logmodelName='Simple MA'):
    ### RMSE 
    rmse = calculate_rmse(np.array(stockprices[train_size:]['close']), np.array(stockprices[train_size:][var]))  # 计算RMSE
    ### MAPE 
    mape = calculate_mape(np.array(stockprices[train_size:]['close']), np.array(stockprices[train_size:][var]))  # 计算MAPE
    
    ## 如果启用Neptune日志记录
    if logNeptune:        
        npt_exp['RMSE'].log(rmse)  # 记录RMSE到Neptune
        npt_exp['MAPE (%)'].log(mape)  # 记录MAPE到Neptune
    
    return rmse, mape  # 返回性能指标


# In[70]:


def plot_stock_trend(var, cur_title, stockprices=stockprices, logNeptune=True, logmodelName='Simple MA'):
    ax = stockprices[['close', var,'200day']].plot(figsize=(20, 10))  # 绘制股价趋势图
    plt.grid(False)  # 关闭网格线
    plt.title(cur_title)  # 设置标题
    plt.axis('tight')  # 设置坐标轴紧凑
    plt.ylabel('Stock Price ($)')  # 设置y轴标签

    ## 如果启用Neptune日志记录
    if logNeptune:
        npt_exp[f'Plot of Stock Predictions with {logmodelName}'].upload(neptune.types.File.as_image(ax.get_figure()))  # 将图表上传到Neptune


# In[71]:


window_size = 50  # 设置窗口大小

import neptune  # 再次导入neptune库

# 在Neptune中创建实验并记录模型
npt_exp = neptune.init_project(    
        api_token=neptune_key,  # 指定Neptune API密钥
        project="magiceric/aaa",  # 指定Neptune项目
        )

window_var = str(window_size) + 'day'  # 构造窗口变量名
    
stockprices[window_var] = stockprices['close'].rolling(window_size).mean()  # 计算简单移动平均
### 包含200天SMA作为参考
stockprices['200day'] = stockprices['close'].rolling(200).mean()  # 计算200天简单移动平均
    
### 为SMA模型绘制趋势图和计算性能指标
plot_stock_trend(var=window_var, cur_title='Simple Moving Averages', logmodelName='Simple MA')  # 绘制SMA趋势图
rmse_sma, mape_sma = calculate_perf_metrics(var=window_var, logmodelName='Simple MA')  # 计算SMA性能指标

### 记录完成后停止Neptune运行
npt_exp.stop()


# In[72]:


# 在Neptune中创建实验并记录模型（新版本）
npt_exp = neptune.init_project(    
        api_token=neptune_key,  # 指定Neptune API密钥
        project="magiceric/aaa",  # 指定Neptune项目
        )      
    
###### 指数移动平均
window_ema_var = window_var+'_EMA'  # 构造EMA变量名
# 计算50天指数加权移动平均
stockprices[window_ema_var] = stockprices['close'].ewm(span=window_size, adjust=False).mean()  # 计算EMA
stockprices['200day'] = stockprices['close'].rolling(200).mean()  # 计算200天简单移动平均
    
### 为EMA模型绘制趋势图和计算性能指标
plot_stock_trend(var=window_ema_var, cur_title='Exponential Moving Averages', logmodelName='Exp MA')  # 绘制EMA趋势图
rmse_ema, mape_ema = calculate_perf_metrics(var=window_ema_var, logmodelName='Exp MA')  # 计算EMA性能指标
### 记录完成后停止Neptune运行（新版本）
npt_exp.stop()


# In[73]:


layer_units, optimizer = 50, 'adam'  # 设置LSTM层单元数和优化器
cur_epochs = 15  # 设置训练轮次
cur_batch_size = 20  # 设置批次大小
    
cur_LSTM_pars = {'units': layer_units, 
                 'optimizer': optimizer, 
                 'batch_size': cur_batch_size, 
                 'epochs': cur_epochs
                 }  # 构造LSTM参数字典
    
# 在Neptune中创建实验并记录模型（新版本）
npt_exp = neptune.init_project(    
        api_token=neptune_key,  # 指定Neptune API密钥
        project="magiceric/aaa",  # 指定Neptune项目
        )   
npt_exp['LSTMPars'] = cur_LSTM_pars  # 将LSTM参数记录到Neptune


# In[74]:


# 对数据集进行标准化
scaler = StandardScaler()  # 实例化标准化器
scaled_data = scaler.fit_transform(stockprices[['close']])  # 对收盘价进行标准化
scaled_data_train = scaled_data[:train.shape[0]]  # 分割训练集数据
    
# 我们使用过去50天的股价来预测第51天的收盘价。
X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, window_size)  # 提取训练序列X和输出值Y


# In[75]:


### 构建LSTM模型并将模型摘要记录到Neptune ###    
def Run_LSTM(X_train, layer_units=50, logNeptune=True, NeptuneProject=None):     
    inp = Input(shape=(X_train.shape[1], 1))  # 定义输入层
    
    x = LSTM(units=layer_units, return_sequences=True)(inp)  # 定义LSTM层
    x = LSTM(units=layer_units)(x)  # 定义第二个LSTM层
    out = Dense(1, activation='linear')(x)  # 定义输出层
    model = tf.keras.Model(inp, out)  # 构建模型
    
    # 编译LSTM神经网络
    model.compile(loss='mean_squared_error', optimizer='adam')  # 编译模型
    
    ## 如果启用Neptune日志记录
    if logNeptune:
        # 捕获模型摘要并记录到Neptune
        from io import StringIO
        model_summary = StringIO()
        model.summary(print_fn=lambda x, **kwargs: model_summary.write(x + '\n'))  # 获取模型摘要
        model_summary.seek(0)
        NeptuneProject['model_summary'].log(model_summary.read())  # 将模型摘要记录到Neptune
        
    return model  # 返回模型   

model = Run_LSTM(X_train, layer_units=layer_units, logNeptune=True, NeptuneProject=npt_exp)  # 运行LSTM模型

history = model.fit(X_train, y_train, epochs=cur_epochs, batch_size=cur_batch_size, 
                    verbose=1, validation_split=0.1, shuffle=True)  # 训练模型


# In[76]:


# 使用过去window_size天的股价预测股价
def preprocess_testdat(data=stockprices, scaler=scaler, window_size=window_size, test=test):    
    raw = data['close'][len(data) - len(test) - window_size:].values  # 提取测试数据
    raw = raw.reshape(-1,1)  # 调整形状
    raw = scaler.transform(raw)  # 进行标准化
    
    X_test = []  # 初始化测试序列X
    for i in range(window_size, raw.shape[0]):  # 循环构造测试序列X
        X_test.append(raw[i-window_size:i, 0])
        
    X_test = np.array(X_test)  # 转换为numpy数组
    
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # 调整形状
    return X_test  # 返回测试序列X

X_test = preprocess_testdat()  # 预处理测试数据

predicted_price_ = model.predict(X_test)  # 预测股价
predicted_price = scaler.inverse_transform(predicted_price_)  # 反标准化预测股价

# 绘制预测价格与实际收盘价
test['Predictions_lstm'] = predicted_price  # 将预测价格添加到测试集DataFrame


# In[77]:


# 评估性能
rmse_lstm = calculate_rmse(np.array(test['close']), np.array(test['Predictions_lstm']))  # 计算RMSE
mape_lstm = calculate_mape(np.array(test['close']), np.array(test['Predictions_lstm']))  # 计算MAPE

### Neptune新版本
npt_exp['RMSE'].log(rmse_lstm)  # 将RMSE记录到Neptune
npt_exp['MAPE (%)'].log(mape_lstm)  # 将MAPE记录到Neptune

### 绘制预测趋势与真实趋势并记录到Neptune         
def plot_stock_trend_lstm(train, test, logNeptune=True):        
    fig = plt.figure(figsize = (20,10))  # 创建图表
    plt.plot(train['timestamp'], train['close'], label = 'Train Closing Price')  # 绘制训练集收盘价
    plt.plot(test['timestamp'], test['close'], label = 'Test Closing Price')  # 绘制测试集收盘价
    plt.plot(test['timestamp'], test['Predictions_lstm'], label = 'Predicted Closing Price')  # 绘制预测收盘价
    plt.title('LSTM Model')  # 设置标题
    plt.xlabel('Date')  # 设置x轴标签
    plt.ylabel('Stock Price ($)')  # 设置y轴标签
    plt.legend(loc="upper left")  # 设置图例位置
    
## 如果启用Neptune日志记录
    if logNeptune:
        npt_exp['Plot of Stock Predictions with LSTM'].upload(neptune.types.File.as_image(fig))  # 将图表上传到Neptune
        
plot_stock_trend_lstm(train, test)  # 绘制LSTM趋势图

### 记录完成后停止Neptune运行（新版本）
npt_exp.stop()

