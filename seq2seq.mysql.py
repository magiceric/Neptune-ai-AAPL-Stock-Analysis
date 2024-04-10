import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import sqlalchemy
import matplotlib.dates as mdates  # Importing mdates for date formatting in matplotlib

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
    plt.figure()
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o')
 
    # 添加标题
    plt.title("loss results Plot")
 
    # 显示图例
    plt.legend(["Loss"])
 
    plt.show()
 
 
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
    for i in tqdm(range(epochs)):
        losss = []
        for seq, labels in train_loader:
            optimizer.zero_grad()
 
            y_pred = model(seq)
 
            single_loss = loss_function(y_pred, labels)
 
            single_loss.backward()
 
            optimizer.step()
            losss.append(single_loss.detach().cpu().numpy())
        tqdm.write(f"\t Epoch {i + 1} / {epochs}, Loss: {sum(losss) / len(losss)}")
        results_loss.append(sum(losss) / len(losss))
 
        # Generate the model save path with timestamp and tsCode
        save_path = f"trainedModels/save_model_{args.tsCode}_{time.strftime('%Y%m%d')}.pth"
        torch.save(model.state_dict(), save_path)
        time.sleep(0.1)
 
    # valid_loss = valid(model, args, scaler, valid_loader)
    # 尚未引入学习率计划后期补上
    # 保存模型
 
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
    plt.show()
 
# 检验模型拟合情况
def inspect_model_fit(model, args, train_loader, scaler):
    model = model
    model_save_path = f"trainedModels/save_model_{args.tsCode}_{time.strftime('%Y%m%d')}.pth"
    model.load_state_dict(torch.load(model_save_path))
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
    plt.show()
 
 
def predict(model=None, args=None, device=None, scaler=None, rolling_data=None, show=False):
    # 预测未知数据的功能
    database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock'
    engine = sqlalchemy.create_engine(database_connection_string)
    df = pd.read_sql_query(f"SELECT trade_date, vol, high, low, open, close FROM historical_data_for_seq2seq WHERE ts_code = '{args.tsCode}'", engine)  # 从MySQL数据库读取特定股票代码的数据
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

        print("history data : ", df[:, -1])
        
        # 绘制蜡烛图
        ohlc = zip(range(len(df)), df[:, 1], df[:, 2], df[:, 3], df[:, 4])  # Adjusted for numpy array indexing
        candlestick_ohlc(plt.gca(), ohlc, width=0.6, colorup='g', colordown='r')
        plt.gca().xaxis_date()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()  # 自动旋转日期标记

        print("预测T+1和T+2 : ", pred[:, -1])
        predT1, predT2 = pred[:, -1]
        print(f"预测T+1: {predT1}, 预测T+2: {predT2}")
 
        # 绘制预测数据
        # 注意这里预测数据的起始x坐标是历史数据的最后一个点的x坐标
        plt.plot(prediction_x, pred[:, -1], marker='o', label='Prediction')
        plt.axvline(history_length - 1, color='red')  # 在图像的x位置处画一条红色竖线
        # 添加标题和图例
        plt.title("History and Prediction")
        plt.legend()
    return pred


def rolling_predict(model=None, args=None, device=None, scaler=None):
    # 滚动预测
    database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock'
    engine = sqlalchemy.create_engine(database_connection_string)
    history_data = pd.read_sql_query(f"SELECT trade_date, vol, high, low, open, close FROM historical_data_for_seq2seq WHERE ts_code = '{args.tsCode}'", engine)  # 从MySQL数据库读取特定股票代码的数据
    # history_data = pd.read_csv(args.data_path)[args.target][-args.window_size * 4:].reset_index(drop=True)
    
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
    LIMIT 500
    """
    pre_data = pd.read_sql_query(query, engine).iloc[::-1].reset_index(drop=True)  # 从MySQL数据库读取特定股票代码的数据，获取T-2交易日起前500个交易日的数据，并将数据顺序倒转
    print(pre_data)
    columns = pre_data.columns[1:]
    columns = ['forecast' + column for column in columns]
    dict_of_lists = {column: [] for column in columns}
    results = []
    for i in range(int(len(pre_data)/args.pre_len)):
        rolling_data = pre_data.iloc[:args.pre_len * i]  # 转换为nadarry
        pred = predict(model, args, device, scaler, rolling_data)
        if args.feature == 'MS' or args.feature == 'S':
            for i in range(args.pred_len):
                results.append(pred[i][0].detach().cpu().numpy())
        else:
            for j in range(args.output_size):
                for i in range(args.pre_len):
                    dict_of_lists[columns[j]].append(pred[i][j])
        print(pred)
    predDate = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    if args.feature == 'MS' or args.feature == 'S':
        df = pd.DataFrame({'predDate': predDate, 'date': pre_data['date'], '{}'.format(args.target): pre_data[args.target],
                           'forecast{}'.format(args.target): pre_data[args.target]})
    else:
        df = pd.DataFrame(dict_of_lists)
        df['predDate'] = predDate
        df = pd.concat((pre_data, df), axis=1)
    
    database_connection_string = 'mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock'
    engine = sqlalchemy.create_engine(database_connection_string)
    # Assuming 'engine' is the SQLAlchemy engine object already created for database connection
    df.to_sql('interval_historical_data_for_seq2seq', con=engine, if_exists='append', index=False)
    pre_len = len(dict_of_lists['forecast' + args.target])
    # 绘图
    plt.figure()
    if args.feature == 'MS' or args.feature == 'S':
        plt.plot(range(len(history_data)), history_data,
                 label='Past Actual Values')
        plt.plot(range(len(history_data), len(history_data) + pre_len), pre_data[args.target][:pre_len].tolist(), label='Predicted Actual Values')
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
 
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series forecast')
    parser.add_argument('-model', type=str, default='LSTM2LSTM', help="模型持续更新")
    parser.add_argument('-window_size', type=int, default=512, help="时间窗口大小, window_size > pre_len")
    parser.add_argument('-pre_len', type=int, default=2, help="预测未来数据长度")
    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help="是否打乱数据加载器中的数据顺序")
    # parser.add_argument('-data_path', type=str, default='ETTh1.csv', help="你的数据数据地址")
    parser.add_argument('-target', type=str, default='close', help='你需要预测的特征列，这个值会最后保存在csv文件里')
    parser.add_argument('-input_size', type=int, default=5, help='你的特征个数不算时间那一列')
    parser.add_argument('-feature', type=str, default='M', help='[M, S, MS],多元预测多元,单元预测单元,多元预测单元')
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
    # parser.add_argument('-roolling_data_path', type=str, default='ETTh1Test.csv', help="你滚动数据集的地址，此部分属于进阶功能")
    args = parser.parse_args()
 
    if isinstance(args.device, int) and args.use_gpu:
        device = torch.device("cuda:" + f'{args.device}')
    else:
        device = torch.device("cpu")
    print("使用设备:", device)
    train_loader, test_loader, valid_loader, scaler = create_dataloader(args, device)
 
    if args.feature == 'MS' or args.feature == 'S':
        args.output_size = 1
    else:
        args.output_size = args.input_size
 
    print(args)
    
    # 实例化模型
    try:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model = EncoderDecoderWrapper(args.input_size, args.output_size, args.hidden_size, args.laryer_num, args.pre_len, args.window_size).to(device)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型成功<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    except:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始初始化{args.model}模型失败<<<<<<<<<<<<<<<<<<<<<<<<<<<")
 
    # 训练模型
    if args.train:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型训练<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        train(model, args, scaler, device)
    if args.test:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始{args.model}模型测试<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        test(model, args, test_loader, scaler)
    if args.inspect_fit:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>开始检验{args.model}模型拟合情况<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        inspect_model_fit(model, args, train_loader, scaler)
    if args.predict:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        predict(model, args, device, scaler,show=True)
    # if args.predict:
    #     print(f">>>>>>>>>>>>>>>>>>>>>>>>>滚动预测未来{args.pre_len}条数据<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    #     rolling_predict(model, args, device, scaler)
    plt.show()
