import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import sqlalchemy
from config import tushare_api_key
import tushare as ts
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, Date, select, delete
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import os
import time

lastDate = None

# 随机数种子
np.random.seed(0)
 
 
class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
 
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)
 
    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std
 
    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
 

 
def plot_loss_data(data):
    # 使用Matplotlib绘制线图
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')
 
    # 添加标题
    plt.title("loss results Plot")
 
    # 显示图例
    plt.legend(["Loss"])
 
    plt.show(block=False)
    plt.pause(2)
    plt.close()

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
 
    def __len__(self):
        return len(self.sequences)
 
    def __getitem__(self, index):
        sequence, label = self.sequences[index]
        return torch.Tensor(sequence), torch.Tensor(label)
 
 

def create_inout_sequences(input_data, tw, pre_len, config):
    # 创建时间序列数据专用的数据分割器
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + pre_len) > len(input_data):
            break
        if config.feature == 'MS':
            train_label = input_data[:, -1:][i + tw:i + tw + pre_len]
        else:
            train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq
 
 

def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae
 

 
def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock'
    engine = sqlalchemy.create_engine(database_connection_string)
    df = pd.read_sql_query(f"SELECT trade_date, vol, high, low, open, close FROM historical_data_for_seq2seq WHERE ts_code = '{config.tsCode}'", engine)  # 从MySQL数据库读取特定股票代码的数据
    print(df)
    pre_len = config.pre_len  # 预测未来数据的长度
    train_window = config.window_size  # 观测窗口
    # 将特征列移到末尾
    target_data = df[[config.target]]
    df = df.drop(config.target, axis=1)
    df = pd.concat((df, target_data), axis=1)
 
    cols_data = df.columns[1:]
    df_data = df[cols_data]
 
    # 这里加一些数据的预处理, 最后需要的格式是pd.series
    true_data = df_data.values
 
    # 定义标准化优化器
    scaler = StandardScaler()
    scaler.fit(true_data)
 
    train_data = true_data[int(0.3 * len(true_data)):]
    valid_data = true_data[int(0.15 * len(true_data)):int(0.30 * len(true_data))]
    test_data = true_data[:int(0.15 * len(true_data))]
    print("训练集尺寸:", len(train_data), "测试集尺寸:", len(test_data), "验证集尺寸:", len(valid_data))
 
    # 进行标准化处理
    train_data_normalized = scaler.transform(train_data)
    test_data_normalized = scaler.transform(test_data)
    valid_data_normalized = scaler.transform(valid_data)
 
    # 转化为深度学习模型需要的类型Tensor
    train_data_normalized = torch.FloatTensor(train_data_normalized).to(device)
    test_data_normalized = torch.FloatTensor(test_data_normalized).to(device)
    valid_data_normalized = torch.FloatTensor(valid_data_normalized).to(device)
 
    # 定义训练器的的输入
    train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len, config)
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len, config)
    valid_inout_seq = create_inout_sequences(valid_data_normalized, train_window, pre_len, config)
 
    # 创建数据集
    train_dataset = TimeSeriesDataset(train_inout_seq)
    test_dataset = TimeSeriesDataset(test_inout_seq)
    valid_dataset = TimeSeriesDataset(valid_inout_seq)
 
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
 
    print("通过滑动窗口共有训练集数据：", len(train_inout_seq), "转化为批次数据:", len(train_loader))
    print("通过滑动窗口共有测试集数据：", len(test_inout_seq), "转化为批次数据:", len(test_loader))
    print("通过滑动窗口共有验证集数据：", len(valid_inout_seq), "转化为批次数据:", len(valid_loader))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>创建数据加载器完成<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    return train_loader, test_loader, valid_loader, scaler
 

 
class LSTMEncoder(nn.Module):
    def __init__(self, rnn_num_layers=1, input_feature_len=1, sequence_len=168, hidden_size=100, bidirectional=False):
        super().__init__()
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.input_feature_len = input_feature_len
        self.num_layers = rnn_num_layers
        self.rnn_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            num_layers=rnn_num_layers,
            input_size=input_feature_len,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
 
    def forward(self, input_seq):
 
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_seq.size(0), self.hidden_size, device='cuda')
        ct = ht.clone()
        if input_seq.ndim < 3:
            input_seq.unsqueeze_(2)
        lstm_out, (ht, ct) = self.lstm(input_seq, (ht,ct))
        if self.rnn_directions > 1:
            lstm_out = lstm_out.view(input_seq.size(0), self.sequence_len, self.rnn_directions, self.hidden_size)
            lstm_out = torch.sum(lstm_out, axis=2)
        return lstm_out, ht.squeeze(0)
 

class AttentionDecoderCell(nn.Module):
    def __init__(self, input_feature_len, out_put, sequence_len, hidden_size):
        super().__init__()
        # attention - inputs - (decoder_inputs, prev_hidden)
        self.attention_linear = nn.Linear(hidden_size + input_feature_len, sequence_len)
        # attention_combine - inputs - (decoder_inputs, attention * encoder_outputs)
        self.decoder_rnn_cell = nn.LSTMCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.out = nn.Linear(hidden_size, input_feature_len)
 
    def forward(self, encoder_output, prev_hidden, y):
        if prev_hidden.ndimension() == 3:
            prev_hidden = prev_hidden[-1]  # 保留最后一层的信息
        attention_input = torch.cat((prev_hidden, y), axis=1)
        attention_weights = F.softmax(self.attention_linear(attention_input), dim=-1).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        rnn_hidden, rnn_hidden = self.decoder_rnn_cell(attention_combine, (prev_hidden, prev_hidden))
        output = self.out(rnn_hidden)
        return output, rnn_hidden
 
 

class EncoderDecoderWrapper(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, pred_len, window_size, teacher_forcing=0.3):
        super().__init__()
        self.encoder = LSTMEncoder(num_layers, input_size, window_size, hidden_size)
        self.decoder_cell = AttentionDecoderCell(input_size, output_size,  window_size, hidden_size)
        self.output_size = output_size
        self.input_size = input_size
        self.pred_len = pred_len
        self.teacher_forcing = teacher_forcing
        self.linear = nn.Linear(input_size,output_size)
 
 
    def __call__(self, xb, yb=None):
        input_seq = xb
        encoder_output, encoder_hidden = self.encoder(input_seq)
        prev_hidden = encoder_hidden
        if torch.cuda.is_available():
            outputs = torch.zeros(self.pred_len, input_seq.size(0), self.input_size, device='cuda')
        else:
            outputs = torch.zeros(input_seq.size(0), self.output_size)
        y_prev = input_seq[:, -1, :]
        for i in range(self.pred_len):
            if (yb is not None) and (i > 0) and (torch.rand(1) < self.teacher_forcing):
                y_prev = yb[:, i].unsqueeze(1)
            rnn_output, prev_hidden = self.decoder_cell(encoder_output, prev_hidden, y_prev)
            y_prev = rnn_output
            outputs[i, :, :] = rnn_output
        outputs = outputs.permute(1, 0, 2)
        if self.output_size == 1:
            outputs = self.linear(outputs)
        return outputs
 
 
 

def train(model, args, scaler, device):
    start_time = time.time()  # 计算起始时间
    model = model
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    epochs = args.epochs
    model.train()  # 训练模式
    results_loss = []
    print_str = ""
    for i in tqdm(range(epochs), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        losss = []
        for seq, labels in train_loader:
            optimizer.zero_grad()
 
            y_pred = model(seq)
 
            single_loss = loss_function(y_pred, labels)
 
            single_loss.backward()
 
            optimizer.step()
            losss.append(single_loss.detach().cpu().numpy())
        epoch_loss = sum(losss) / len(losss)
        print_str += f"\t Epoch {i + 1} / {epochs}, Loss: {epoch_loss:.6f} "
        results_loss.append(epoch_loss)
 
        # Generate the model save path with timestamp and tsCode
        save_path = f"trainedModels/save_model_{args.tsCode}_{time.strftime('%Y%m%d')}.pth"
        torch.save(model.state_dict(), save_path)
        time.sleep(0.1)
 
    print(print_str)
 
    print(f">>>>>>>>>>>>>>>>>>>>>>模型已保存至{save_path}, 用时:{(time.time() - start_time) / 60:.4f} min<<<<<<<<<<<<<<<<<<")
    plot_loss_data(results_loss)
 

def valid(model, args, scaler, valid_loader):
    lstm_model = model
    # 加载模型进行预测
    model_path = f"trainedModels/save_model_{args.tsCode}_{time.strftime('%Y%m%d')}.pth"
    lstm_model.load_state_dict(torch.load(model_path))
    lstm_model.eval()  # 评估模式
    losss = []
 
    for seq, labels in valid_loader:
        pred = lstm_model(seq)
        mae = calculate_mae(pred.detach().numpy().cpu(), np.array(labels.detach().cpu()))  # MAE误差计算绝对值(预测值  - 真实值)
        losss.append(mae)
 
    print("验证集误差MAE:", losss)
    return sum(losss) / len(losss)
 

def test(model, args, test_loader, scaler):
    # 加载模型进行预测
    losss = []
    model = model
    model_path = f"trainedModels/save_model_{args.tsCode}_{time.strftime('%Y%m%d')}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 评估模式
    results = []
    labels = []
    for seq, label in test_loader:
        pred = model(seq)
        mae = calculate_mae(pred.detach().cpu().numpy(),
                            np.array(label.detach().cpu()))  # MAE误差计算绝对值(预测值  - 真实值)
        losss.append(mae)
        pred = pred[:, 0, :]
        label = label[:, 0, :]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])
    plt.figure(figsize=(10, 5))
    print("测试集误差MAE:", losss)
    # 绘制历史数据
    plt.plot(labels, label='TrueValue')
 
    # 绘制预测数据
    # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
    plt.plot(results, label='Prediction')
 
    # 添加标题和图例
    plt.title("test state")
    plt.legend()
    plt.show(block=False)

    current_date = datetime.now().strftime('%Y%m%d')
    plots_dir = os.path.join('plots', current_date)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plot_filename = f"{lastDate}_{args.tsCode}_test.png"
    full_plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(full_plot_path)    
    print(f"【Test】 plot saved successfully at {full_plot_path}")

    plt.pause(2)
    plt.close()

# 检验模型拟合情况
def inspect_model_fit(model, args, train_loader, scaler):
    model = model
    model_path = f"trainedModels/save_model_{args.tsCode}_{time.strftime('%Y%m%d')}.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 评估模式
    results = []
    labels = []
 
    for seq, label in train_loader:
        pred = model(seq)[:, 0, :]
        label = label[:, 0, :]
        pred = scaler.inverse_transform(pred.detach().cpu().numpy())
        label = scaler.inverse_transform(label.detach().cpu().numpy())
        for i in range(len(pred)):
            results.append(pred[i][-1])
            labels.append(label[i][-1])
    plt.figure(figsize=(10, 5))
    # 绘制历史数据
    plt.plot(labels, label='History')
 
    # 绘制预测数据
    # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
    plt.plot(results, label='Prediction')
 
    # 添加标题和图例
    plt.title("inspect model fit state")
    plt.legend()
    plt.show(block=False)
    plt.pause(2)
    plt.close()
 

def predict(model=None, args=None, device=None, scaler=None, rolling_data=None, show=False):
    # 预测未知数据的功能
    database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock'
    engine = sqlalchemy.create_engine(database_connection_string)
    strSQL = f"SELECT trade_date, vol, high, low, open, close FROM historical_data_for_seq2seq WHERE ts_code = '{args.tsCode}' ORDER BY trade_date DESC"  # 从MySQL数据库读取特定股票代码的数据
    print("strSQL : ", strSQL)
    df = pd.read_sql_query(strSQL, engine)  # 从MySQL数据库读取特定股票代码的数据

    lastDate = str(df.iloc[0]['trade_date']).replace('-', '')
    lastClose = df.iloc[0]['close']
    print("最大交易日期:", lastDate, "对应的收盘价:", lastClose)
    df = pd.concat((df, rolling_data), axis=0).reset_index(drop=True)
    df = df.iloc[:, 1:][-args.window_size:].values  # 转换为nadarry
    pre_data = scaler.transform(df)
    tensor_pred = torch.FloatTensor(pre_data).to(device)
    tensor_pred = tensor_pred.unsqueeze(0)  # 单次预测 , 滚动预测功能暂未开发后期补上
    model = model
    model_save_path = f"trainedModels/save_model_{args.tsCode}_{time.strftime('%Y%m%d')}.pth"
    model.load_state_dict(torch.load(model_save_path))
    model.eval()  # 评估模式
 
    pred = model(tensor_pred)[0]
 
    pred = scaler.inverse_transform(pred.detach().cpu().numpy())
    if show:
        # 计算历史数据的长度
        history_length = len(df[:, -1])
        # 为历史数据生成x轴坐标
        history_x = range(history_length)
        plt.figure(figsize=(10, 5))
        # 为预测数据生成x轴坐标
        # 开始于历史数据的最后一个点的x坐标
        prediction_x = range(history_length - 1, history_length + len(pred[:, -1]) - 1)
 
        # 绘制历史数据
        plt.plot(history_x, df[:, -1], label='History')
 
        print("预测T+1和T+2 : ", pred[:, -1])
        predT1, predT2 = pred[:, -1]
        print(f"预测T+1: {predT1}, 预测T+2: {predT2}")

        # Inserting prediction results into the database
        connection = engine.connect()
        metadata = MetaData()
        predictionHistory = Table('predictionHistory', metadata,
                                Column('id', Integer, primary_key=True),
                                Column('predDate', Date),
                                Column('stockCode', String(10)),
                                Column('lastDate', Date),
                                Column('lastClose', Float),
                                Column('predT1', Float),
                                Column('predT2', Float),
                                )
        metadata.create_all(engine)  # Creates the table if it doesn't exist
        insert_stmt = predictionHistory.insert().values(
                predDate=datetime.now().date(),
                stockCode=args.tsCode,
                lastDate=lastDate,
                lastClose=lastClose,
                predT1=predT1,
                predT2=predT2
            )
        try:
            connection.execute(insert_stmt)
            connection.commit()
            print(f"Successfully inserted prediction results for {args.tsCode} into the database.")
        except Exception as e:
            print(f"Failed to insert prediction results for {args.tsCode} into the database. Error: {e}")

        # 绘制预测数据
        # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
        plt.plot(prediction_x, pred[:, -1], marker='o', label='Prediction')
        plt.axvline(history_length - 1, color='red')  # 在图像的x位置处画一条红色竖线

        # 添加标题和图例
        plt.title("History and Prediction")
        plt.legend()

        plt.draw()

        current_date = datetime.now().strftime('%Y%m%d')
        plots_dir = os.path.join('plots', current_date)
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        plot_filename = f"{lastDate}_{args.tsCode}_predict.png"
        full_plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(full_plot_path)    
        print(f"Predict plot saved successfully at {full_plot_path}")
        
        plt.pause(2)
        plt.close()

    return pred


def rolling_predict(model=None, args=None, device=None, scaler=None):
    # 滚动预测
    database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock'
    engine = sqlalchemy.create_engine(database_connection_string)

    # pre_data = pd.read_csv(args.roolling_data_path)
    # pre_data = history_data
    query = f"""
    SELECT trade_date, vol, high, low, open, close 
    FROM historical_data_for_seq2seq 
    WHERE ts_code = '{args.tsCode}' 
    AND trade_date <= (
        SELECT trade_date 
        FROM historical_data_for_seq2seq 
        WHERE ts_code = '{args.tsCode}' 
        ORDER BY trade_date DESC 
        LIMIT 1 OFFSET 0
    ) 
    ORDER BY trade_date DESC 
    LIMIT 498
    """
    pre_data = pd.read_sql_query(query, engine).iloc[::-1].reset_index(drop=True)  # 从MySQL数据库读取特定股票代码的数据，获取T-2交易日起前500个交易日的数据，并将数据顺序倒转
    # 添加两条新数据
    new_rows = [{'trade_date': '2024-04-02', 'vol': None, 'high': None, 'low': None, 'open': None, 'close': None},
                {'trade_date': '2024-04-03', 'vol': None, 'high': None, 'low': None, 'open': None, 'close': None}]
    pre_data = pd.concat([pre_data, pd.DataFrame(new_rows)], ignore_index=True)
    print(pre_data)

    pre_data_min_date = pre_data['trade_date'].astype(str).min()
    query = f"""
    SELECT {args.target} 
    FROM historical_data_for_seq2seq 
    WHERE ts_code = '{args.tsCode}' 
    AND trade_date < '{pre_data_min_date}' 
    ORDER BY trade_date DESC 
    LIMIT {args.window_size * 4}
    """
    history_data = pd.read_sql_query(query, engine).iloc[::-1].reset_index(drop=True)
    print("history_data : ", history_data)

    columns = pre_data.columns[1:]
    columns = ['forecast' + column for column in columns]
    dict_of_lists = {column: [] for column in columns}
    results = []
    print(args)
    for i in range(int(len(pre_data)/args.pre_len)):
        rolling_data = pre_data.iloc[:args.pre_len * i]  # 转换为nadarry
        pred = predict(model, args, device, scaler, rolling_data)
        if args.feature == 'MS' or args.feature == 'S':
            for i in range(args.pre_len):
                # results.append(pred[i][0].detach().cpu().numpy())
                results.append(pred[i][0])
        else:
            for j in range(args.output_size):
                for i in range(args.pre_len):
                    dict_of_lists[columns[j]].append(pred[i][j])
        print(pred)
    predDate = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    if args.feature == 'MS' or args.feature == 'S':
        df = pd.DataFrame({'predDate': predDate, 'trade_date': pre_data['trade_date'], '{}'.format(args.target): pre_data[args.target],
                           'forecast{}'.format(args.target): results})
    else:
        df = pd.DataFrame(dict_of_lists)
        # print("df before concat : ", df)
        df['predDate'] = predDate
        df = pd.concat((pre_data, df), axis=1)
        # print("df after concat : ", df)
    
    database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock'
    engine = sqlalchemy.create_engine(database_connection_string)
    # Assuming 'engine' is the SQLAlchemy engine object already created for database connection
    df.to_sql('interval_historical_data_for_seq2seq', con=engine, if_exists='append', index=False)
    if args.feature == 'MS' or args.feature == 'S':
        pre_len = len(results)
    else:
        pre_len = len(dict_of_lists['forecast' + args.target])
    # 绘图
    plt.figure()
    if args.feature == 'MS' or args.feature == 'S':
        plt.plot(range(len(history_data)), history_data,
                 label='Past Actual Values')
        # print("pre_data again : ", pre_data)
        # print("pre_data[args.target] : ", pre_data[args.target])0
        # print("pre_data[args.target][:pre_len].tolist() : ", pre_data[args.target][:pre_len].tolist())
        plt.plot(range(len(history_data), len(history_data) + pre_len), pre_data[args.target][:pre_len].tolist(), label='Predicted Actual Values')
        # print("results : ", results)
        plt.plot(range(len(history_data), len(history_data) + pre_len), results, label='Predicted Future Values')
    else:
        plt.plot(range(len(history_data)), history_data,
                 label='Past Actual Values')
        plt.plot(range(len(history_data), len(history_data) + pre_len), pre_data[args.target][:pre_len].tolist(), label='Predicted Actual Values')
        plt.plot(range(len(history_data), len(history_data) + pre_len), dict_of_lists['forecast' + args.target], label='Predicted Future Values')
    # 添加图例
    plt.legend()
    plt.style.use('ggplot')
    # 添加标题和轴标签
    plt.title('Past vs Predicted Future Values')
    plt.xlabel('Time Point')
    plt.ylabel('Value')
    # 在特定索引位置画一条直线
    plt.axvline(x=len(history_data), color='blue', linestyle='--', linewidth=2)
    # 显示图表
    plt.savefig('forcast.png')
    plt.show()
 

def update_stock_data(ts_code='000001.SZ'):

    print("Setting Tushare token...")
    # Set Tushare token
    ts.set_token(tushare_api_key)
    pro = ts.pro_api()

    print("Setting up database connection...")
    # Database connection setup
    database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock'
    engine = create_engine(database_connection_string)
    Session = sessionmaker(bind=engine)
    session = Session()

    print(f"Updating stock data for {ts_code}...")
    # Define the table structure
    metadata = MetaData()
    metadata.reflect(bind=engine)
    historical_data_for_seq2seq = metadata.tables['historical_data_for_seq2seq']
    
    print("Fetching historical data from Tushare...")
    # Fetch historical data from Tushare
    # Initialize an empty DataFrame to hold the concatenated results
    df_full = pd.DataFrame()

    # Define the start date for fetching data
    start_date = '19910101'

    # Define the end date as today
    end_date = pd.Timestamp.today().strftime('%Y%m%d')

    # Convert start_date and end_date to datetime objects
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Calculate the number of years between start and end date
    years = end_date_dt.year - start_date_dt.year + 1

    # Fetch the adjusted historical data in yearly batches
    for year in range(start_date_dt.year, start_date_dt.year + years):
        # Define yearly start and end dates
        yearly_start_date = f"{year}0101"
        yearly_end_date = f"{year}1231"
        # Ensure the end date does not exceed the current date
        if pd.to_datetime(yearly_end_date) > end_date_dt:
            yearly_end_date = end_date

        df_batch = pro.daily(ts_code=ts_code, adj='hfq', start_date=yearly_start_date, end_date=yearly_end_date)
        # Adjust the order of the fetched data before concatenation
        df_batch = df_batch.sort_values(by='trade_date', ascending=True)
        if not df_batch.empty:
            df_full = pd.concat([df_full, df_batch], ignore_index=True)
    # Selecting required columns: 'trade_date', 'vol', 'open', 'high', 'low', 'close'
    df = df_full[['trade_date', 'ts_code', 'vol', 'open', 'high', 'low', 'close']]
    # Remove data for the date 2024-04-03

    # 测试需要，暂时去掉2-24-04-03的数据
    # TODO：完成测试记得去掉！！！！！！！！！！！！
    df = df[df['trade_date'] != '20240403']

    global lastDate
    lastDate = df['trade_date'].max()
    print("global lastDate set : ", lastDate)

    print(df)  # Print fetched data
    print("Deleting existing data for the stock...")
    # Delete existing data for the stock
    delete_stmt = delete(historical_data_for_seq2seq).where(historical_data_for_seq2seq.c.ts_code == ts_code)
    session.execute(delete_stmt)
    try:
        session.commit()  # Commit immediately after delete
    except OperationalError as e:
        print(f"Error committing delete operation: {e}")
        session.rollback()  # Rollback in case of error
        

    print("Preparing data for insertion...")
    # Prepare data for insertion
    df = df[['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol']]
    df = df.sort_values(by='trade_date')  # Ensure data is in ascending order by date
    
    print("Inserting new data...")
    # Insert new data in batches to avoid lock wait timeout
    batch_size = 500
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]
        try:
            batch.to_sql('historical_data_for_seq2seq', con=engine, if_exists='append', index=False)
            session.commit()  # Commit after each batch
        except OperationalError as e:
            print(f"Error inserting data: {e}")
            session.rollback()  # Rollback in case of error
    print("Data update complete.")
 
parser = argparse.ArgumentParser(description='Time Series forecast')


parser.add_argument('-model', type=str, default='LSTM2LSTM', help="模型持续更新")
parser.add_argument('-window_size', type=int, default=512, help="时间窗口大小, window_size > pre_len")
parser.add_argument('-pre_len', type=int, default=2, help="预测未来数据长度")


# data
parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")
# parser.add_argument('-data_path', type=str, default='ETTh1.csv', help="你的数据数据地址")
parser.add_argument('-target', type=str, default='close', help='你需要预测的特征列，这个值会最后保存在csv文件里')
parser.add_argument('-input_size', type=int, default=5, help='你的特征个数不算时间那一列')
# parser.add_argument('-feature', type=str, default='M', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
parser.add_argument('-feature', type=str, default='MS', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
parser.add_argument('-tsCode', type=str, default='000001.SZ', help="指定股票代码")


# learning
parser.add_argument('-lr', type=float, default=0.001, help="学习率")
parser.add_argument('-drop_out', type=float, default=0.05, help="随机丢弃概率,防止过拟合")
parser.add_argument('-epochs', type=int, default=50, help="训练轮次")
parser.add_argument('-batch_size', type=int, default=16, help="批次大小")
parser.add_argument('-save_path', type=str, default='models')


# model
parser.add_argument('-hidden_size', type=int, default=128, help="隐藏层单元数")
parser.add_argument('-laryer_num', type=int, default=2)


# device
parser.add_argument('-use_gpu', type=bool, default=True)
parser.add_argument('-device', type=int, default=0, help="只设置最多支持单个gpu训练")


# option
parser.add_argument('-train', type=bool, default=True)
parser.add_argument('-test', type=bool, default=True)
parser.add_argument('-predict', type=bool, default=True)
parser.add_argument('-inspect_fit', type=bool, default=True)
parser.add_argument('-lr-scheduler', type=bool, default=True)


# 可选部分，滚动预测如果想要进行这个需要你有一个额外的文件和你的训练数据集完全相同但是数据时间点不同。
parser.add_argument('-rolling_predict', type=bool, default=True)

import sys

# Modify the condition to check for any argument that starts with '--f=' and remove it
sys.argv = [arg for arg in sys.argv if not arg.startswith('--f=')]
args = parser.parse_args()


with open('stock.list', 'r') as file:
    stock_codes = [line.strip() for line in file if line.strip()]

total_start_time = time.time()  # Start timing for the entire loop

for index, stockCode in enumerate(stock_codes):
    loop_start_time = time.time()  # Start timing for this iteration of the loop
    print(f"正在处理第 {index + 1}/{len(stock_codes)} 支股票: {stockCode}")

    # 首先，更新最新的交易数据
    step_start_time = time.time()  # Start timing for updating stock data
    update_stock_data(stockCode)
    print(f"更新股票数据耗时: {time.time() - step_start_time}秒")

    # 修改args参数项，为后面处理做准备
    args.tsCode = stockCode

    step_start_time = time.time()  # Start timing for device setup
    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")
    print("使用设备:", device, f"耗时: {time.time() - step_start_time}秒")
    train_loader, test_loader, valid_loader, scaler = create_dataloader(args, device)

    step_start_time = time.time()  # Start timing for output size setup
    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size
    print(f"设置输出大小耗时: {time.time() - step_start_time}秒")

    print(args)

    # 实例化模型
    step_start_time = time.time()  # Start timing for model instantiation
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = EncoderDecoderWrapper(args.input_size, args.output_size, args.hidden_size, args.laryer_num, args.pre_len, args.window_size).to(device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"模型初始化耗时: {time.time() - step_start_time}秒")

    # 训练模型
    if args.train:
        step_start_time = time.time()  # Start timing for model training
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, scaler, device)
        print(f"模型训练耗时: {time.time() - step_start_time}秒")

    if args.test:
        step_start_time = time.time()  # Start timing for model testing
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        test(model, args, test_loader, scaler)
        print(f"模型测试耗时: {time.time() - step_start_time}秒")

    if args.inspect_fit:
        step_start_time = time.time()  # Start timing for model fit inspection
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始检验{args.model}模型拟合情况<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        inspect_model_fit(model, args, train_loader, scaler)
        print(f"模型拟合检验耗时: {time.time() - step_start_time}秒")

    if args.predict:
        step_start_time = time.time()  # Start timing for prediction
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        predict(model, args, device, scaler, show=True)
        print(f"预测耗时: {time.time() - step_start_time}秒")

    print(f"处理第 {index + 1} 支股票总耗时: {time.time() - loop_start_time}秒")
    # if args.predict:
    #     print(f">>>>>>>>>>>>>>>>>>>>>>>>>滚动预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #     import time
    #     start_time = time.time()
    #     rolling_predict(model, args, device, scaler)
    #     end_time = time.time()
    #     print(f"滚动预测执行消耗时间: {end_time - start_time}秒")

print(f"全部 {len(stock_codes)} 支股票共耗时: {time.time() - total_start_time}秒")
 