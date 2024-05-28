import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sqlalchemy import create_engine
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time
from tqdm import tqdm  # Import tqdm for progress bar

# 强制使用GPU
device = torch.device("cuda")
print("使用设备：cuda")

# 数据集类
class StockDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        stock_code = row['stockCode']
        price_vol_image = Image.open(row['price_vol_plot_path']).convert('RGB')
        macd_image = Image.open(row['macd_plot_path']).convert('RGB')
        boll_image = Image.open(row['boll_plot_path']).convert('RGB')
        kdj_image = Image.open(row['kdj_plot_path']).convert('RGB')
        t1_change_rate = row['t1ChangeRate']

        if self.transform:
            price_vol_image = self.transform(price_vol_image)
            macd_image = self.transform(macd_image)
            boll_image = self.transform(boll_image)
            kdj_image = self.transform(kdj_image)

        # Convert t1_change_rate to float tensor
        t1_change_rate = torch.tensor(t1_change_rate, dtype=torch.float32)

        return stock_code, price_vol_image, macd_image, boll_image, kdj_image, t1_change_rate

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 添加归一化
])

# 从数据库加载数据
engine = create_engine('mysql+pymysql://stock:Abcd1234!!@192.168.3.17:3306/aistock')
df = pd.read_sql('SELECT * FROM train_data_v2', engine)

# 检查数据集中的异常值
print(df.describe())
print(df.isnull().sum())

# 处理缺失值或异常值
df = df.dropna()  # 删除缺失值
# 其他处理异常值的方法

# 构建数据集和数据加载器
dataset = StockDataset(df, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print("Train Dataset Size:", len(train_dataset))
print("Test Dataset Size:", len(test_dataset))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 将batch_size从32减少到16
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # 将batch_size从32减少到16

# 定义模型
class FinancialImageModel(nn.Module):
    def __init__(self):
        super(FinancialImageModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_reduce_dim = nn.Linear(128*28*28, 512)  # Reduce dimensionality
        self.attention_layer = nn.MultiheadAttention(embed_dim=512, num_heads=4)
        self.spp_layer = SpatialPyramidPooling()
        self.fc_layers = nn.Sequential(
            nn.Linear(1612, 1024),  # Adjust input size to match SPP output
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Tanh()
        )

    def forward(self, price_vol_image, macd_image, boll_image, kdj_image):
        x1 = self.conv_layers(price_vol_image)
        x2 = self.conv_layers(macd_image)
        x3 = self.conv_layers(boll_image)
        x4 = self.conv_layers(kdj_image)

        x1 = self.fc_reduce_dim(x1.view(x1.size(0), -1)).unsqueeze(1)
        x2 = self.fc_reduce_dim(x2.view(x2.size(0), -1)).unsqueeze(1)
        x3 = self.fc_reduce_dim(x3.view(x3.size(0), -1)).unsqueeze(1)
        x4 = self.fc_reduce_dim(x4.view(x4.size(0), -1)).unsqueeze(1)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x, _ = self.attention_layer(x, x, x)
        x = x.view(x.size(0), -1)
        x = self.spp_layer(x)
        print(f"Shape after SPP: {x.shape}")  # Debug print statement
        x = self.fc_layers(x)
        return x

class SpatialPyramidPooling(nn.Module):
    def __init__(self):
        super(SpatialPyramidPooling, self).__init__()

    def forward(self, x):
        # print(f"Shape of x before view: {x.shape}")  # Debug print statement
        # Reshape to [batch_size, 1, 2048] before pooling
        x = x.view(x.size(0), 1, 2048)
        output1 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        output2 = F.adaptive_avg_pool2d(x, (2, 2)).view(x.size(0), -1)
        output3 = F.adaptive_avg_pool2d(x, (4, 4)).view(x.size(0), -1)
        output4 = F.adaptive_avg_pool2d(x, (7, 7)).view(x.size(0), -1)
        output5 = F.adaptive_avg_pool2d(x, (11, 11)).view(x.size(0), -1)
        output6 = F.adaptive_avg_pool2d(x, (14, 14)).view(x.size(0), -1)
        output7 = F.adaptive_avg_pool2d(x, (21, 21)).view(x.size(0), -1)
        output8 = F.adaptive_avg_pool2d(x, (28, 28)).view(x.size(0), -1)
        x = torch.cat([output1, output2, output3, output4, output5, output6, output7, output8], dim=1)
        return x.view(x.size(0), -1)

# 定义权重初始化函数
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0)

# 实例化模型并移动到GPU
model = FinancialImageModel().to(device)

# 应用权重初始化
model.apply(weights_init)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)  # 使用AdamW优化器

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for i, (stock_code, price_vol_image, macd_image, boll_image, kdj_image, t1_change_rate) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
        price_vol_image = price_vol_image.to(device)
        macd_image = macd_image.to(device)
        boll_image = boll_image.to(device)
        kdj_image = kdj_image.to(device)
        t1_change_rate = t1_change_rate.to(device)

        optimizer.zero_grad()
        outputs = model(price_vol_image, macd_image, boll_image, kdj_image)
        loss = criterion(outputs.squeeze(), t1_change_rate)
        loss.backward()
        
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # 每10个批次打印一次
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

    end_time = time.time()
    print(f"Epoch {epoch + 1} completed in {end_time - start_time:.2f} seconds")

# 测试模型
model.eval()
test_loss = 0.0
with torch.no_grad():
    for stock_code, price_vol_image, macd_image, boll_image, kdj_image, t1_change_rate in test_loader:
        price_vol_image = price_vol_image.to(device)
        macd_image = macd_image.to(device)
        boll_image = boll_image.to(device)
        kdj_image = kdj_image.to(device)
        t1_change_rate = t1_change_rate.to(device)

        outputs = model(price_vol_image, macd_image, boll_image, kdj_image)
        
        # 打印模型输出和目标值
        print(f"Outputs: {outputs}")
        print(f"Targets: {t1_change_rate}")
        
        loss = criterion(outputs.squeeze(), t1_change_rate)
        
        # 检查损失函数输入
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Found NaN or Inf in loss at test step")
            continue
        
        test_loss += loss.item()

print(f"Test loss: {test_loss / len(test_loader):.3f}")

# 保存模型
torch.save(model.state_dict(), "financial_image_model.pth")
print("模型已保存到 financial_image_model.pth")