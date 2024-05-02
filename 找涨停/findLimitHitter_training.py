# -*- coding: UTF-8 -*-
"""
@Project ：aaa
@File    ：findLimitHitter_training.py
@Author  ：mAgIcErIc
@Contact : 745339023@qq.com
@Date    ：2024/4/23 0023 14:47
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models
from torchvision.models.resnet import ResNet50_Weights
from torch.optim import Adam
from sqlalchemy import create_engine, text
import pandas as pd
import os
import sys

# Importing Neptune
import neptune

# Importing API key and project from config
sys.path.append('../')
from config import neptune_key, neptune_project

# Initialize Neptune
run = neptune.init_run(
    project=neptune_project,
    api_token=neptune_key
)
# 检查GPU是否可用，并选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{'cuda' if torch.cuda.is_available() else 'cpu'}")


# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['indicator_plots_result']
        image = datasets.folder.default_loader(image_path)  # 加载图像
        target = row['limit_up']
        if self.transform:
            image = self.transform(image)
        return image, target


# 数据预处理步骤
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 连接数据库并加载数据集
engine = create_engine('mysql+pymysql://stock:Abcd1234!!@192.168.3.7:3306/aistock')
query = """
SELECT indicator_plots_result, limit_up
FROM train_data
"""
dataframe = pd.read_sql_query(query, engine)
print("数据库连接成功，数据加载完成。")

# 从数据集中分离出标记为1和0的数据
dataframe_1 = dataframe[dataframe['limit_up'] == 1]
dataframe_0 = dataframe[dataframe['limit_up'] == 0]

# 重新采样标记为0的数据，使其数量与标记为1的数据相等
dataframe_0_sample = dataframe_0.sample(n=len(dataframe_1), random_state=42)

# 合并数据集并重新洗牌
balanced_dataframe = pd.concat([dataframe_1, dataframe_0_sample]).sample(frac=1, random_state=42)
print("平衡后的训练集和测试集划分完成。")

# 划分训练集和测试集
train_dataframe = balanced_dataframe.sample(frac=0.8, random_state=42)
test_dataframe = balanced_dataframe.drop(train_dataframe.index)

# 创建数据集
train_dataset = CustomDataset(train_dataframe, transform=transform)
test_dataset = CustomDataset(test_dataframe, transform=transform)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
print("数据加载器创建完成。")

# 定义模型
class FinancialIndicatorNormalization(nn.Module):
    def __init__(self):
        super(FinancialIndicatorNormalization, self).__init__()
        # 定义金融指标对应的颜色掩码
        self.masks = {
            'MACD-DEA': (255, 255, 11),
            'MACD-DIFF': (255, 255, 255),
            '成交量': ((255, 50, 50), (0, 230, 0)),  # 成交量红和绿
            '股价蜡烛图': ((255, 0, 0), (84, 251, 251)),  # 股价蜡烛图红和蓝
            '大盘蜡烛图': (0, 152, 225),
            '60天移动平均线': (2, 226, 244),
            '30天移动平均线': (0, 230, 0),
            '20天移动平均线': (255, 128, 255),
            '10天移动平均线': (255, 255, 11),
            '5天移动平均线': (255, 255, 250),
        }

    def forward(self, x):
        # x的形状为[批大小, 通道数, 高度, 宽度]
        channels = []
        for key, color in self.masks.items():
            if key == '成交量' or key == '股价蜡烛图':
                # 对于成交量和股价蜡烛图，处理两种颜色
                mask_red = torch.tensor(color[0], dtype=torch.float32, device=x.device).view(1, 3, 1, 1) / 255.0
                mask_green_or_blue = torch.tensor(color[1], dtype=torch.float32, device=x.device).view(1, 3, 1, 1) / 255.0
                # 应用掩码并求和以生成单通道
                channel = torch.sum(x * mask_red, dim=1, keepdim=True) + torch.sum(x * mask_green_or_blue, dim=1, keepdim=True)
            else:
                mask = torch.tensor(color, dtype=torch.float32, device=x.device).view(1, 3, 1, 1) / 255.0
                channel = torch.sum(x * mask, dim=1, keepdim=True)
            channels.append(channel)
        # 合并所有生成的通道
        return torch.cat(channels, dim=1)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 2, 4, 8]):
        super(PyramidPoolingModule, self).__init__()
        self.pool_sizes = pool_sizes
        self.pooling_layers = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size=(size, size))
            for size in self.pool_sizes
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // len(self.pool_sizes), kernel_size=1)
            for _ in self.pool_sizes
        ])
        self.out_channels = in_channels + in_channels // len(self.pool_sizes) * len(self.pool_sizes)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        pooled_features = [x]
        for pool, conv in zip(self.pooling_layers, self.conv_layers):
            pooled = pool(x)
            convolved = conv(pooled)
            upsampled = F.interpolate(convolved, size=(height, width), mode='bilinear', align_corners=True)
            pooled_features.append(upsampled)
        return torch.cat(pooled_features, dim=1)


class ModifiedResNet50(nn.Module):
    def __init__(self, num_features, pretrained=True, use_se=True):
        super(ModifiedResNet50, self).__init__()
        if pretrained:
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet50(pretrained=False)

        # 修改第一个卷积层以接受10个通道的输入
        self.model.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # 使用SEBlock
        self.use_se = use_se
        if self.use_se:
            self.se_block = SEBlock(2048)  # 2048是ResNet50的特征数量

        # 保留ResNet50直到全局平均池化层的部分
        self.features = nn.Sequential(*list(self.model.children())[:-2])

        # 自定义全连接层
        self.classifier = nn.Linear(num_features, 2)

        # 添加金字塔池化模块
        self.pyramid_pooling = PyramidPoolingModule(2048)
        # 由于PPM改变了特征图的通道数，我们需要根据PPM的输出调整全连接层的输入特征数
        ppm_output_channels = self.pyramid_pooling.out_channels
        self.classifier = nn.Linear(ppm_output_channels, 2)

    def forward(self, x):
        x = self.features(x)
        if self.use_se:
            x = self.se_block(x)
        x = self.pyramid_pooling(x)  # 在特征提取后和分类之前应用金字塔池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


import torch.nn.functional as F
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 使用修改后的ResNet50模型
class FinancialIndicatorModel(nn.Module):
    def __init__(self):
        super(FinancialIndicatorModel, self).__init__()
        self.financial_norm = FinancialIndicatorNormalization()
        self.feature_extractor = ModifiedResNet50(num_features=2048, use_se=True)

    def forward(self, x):
        x = self.financial_norm(x)  # 应用金融指标正则化
        x = self.feature_extractor(x)  # 特征提取
        return x


model = FinancialIndicatorModel().to(device)
print("模型定义完成。")

# 检查是否存在预训练模型
model_path = 't2LimitHitPredictor.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("加载预训练模型成功。")

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
print("损失函数和优化器定义完成。")

print("训练集中标记为1的数量:", len(train_dataframe[train_dataframe['limit_up'] == 1]))
print("训练集中标记为0的数量:", len(train_dataframe[train_dataframe['limit_up'] == 0]))
print("测试集中标记为1的数量:", len(test_dataframe[test_dataframe['limit_up'] == 1]))
print("测试集中标记为0的数量:", len(test_dataframe[test_dataframe['limit_up'] == 0]))
print("训练集总数量:", len(train_dataframe))
print("测试集总数量:", len(test_dataframe))

# 提供一个简单的文本菜单，等待用户输入
input_prompt = "按回车键继续，输入'q'退出程序："
user_input = input(input_prompt)
if user_input.lower() == 'q':
    print("程序已退出。")
    exit()


# 训练模型
train_model = True  # 条件开关
if train_model:
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log to Neptune
            run["train/loss"].log(loss.item())

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print("模型训练完成。")

    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至 {model_path}。")
    
# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    correct_label_1 = 0
    correct_label_0 = 0
    total_label_1 = 0
    total_label_0 = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 分别统计标记1和标记0的准确率
        correct_label_1 += ((predicted == 1) & (labels == 1)).sum().item()
        correct_label_0 += ((predicted == 0) & (labels == 0)).sum().item()
        total_label_1 += (labels == 1).sum().item()
        total_label_0 += (labels == 0).sum().item()

    # Log to Neptune
    run["test/accuracy"].log(100 * correct / total)
    if total_label_1 > 0:
        run["test/accuracy_label_1"].log(100 * correct_label_1 / total_label_1)
    if total_label_0 > 0:
        run["test/accuracy_label_0"].log(100 * correct_label_0 / total_label_0)

    print(f'测试集上的模型总体准确率: {100 * correct / total:.2f} %')
    if total_label_1 > 0:
        print(f'标记1的准确率: {100 * correct_label_1 / total_label_1:.2f} %')
    if total_label_0 > 0:
        print(f'标记0的准确率: {100 * correct_label_0 / total_label_0:.2f} %')

engine.dispose()



