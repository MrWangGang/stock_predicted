import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import mplfinance as mpf
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt
import json


# 设置 pandas 显示所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# 设置随机种子
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
# ================================
# 配置超参数
# ================================
config = {
    'seq_length': 10,
    'input_size': 11,  # 增加了更多特征，输入维度变为11
    'd_model': 128,
    'nhead': 8,
    'output_size': 1,
    'batch_size': 32,
    'learning_rate': 0.0001,
    'weight_decay': 1e-5,
    'epochs': 100,
    'dropout': 0.1,
    'val_split': 0.2,
    'accuracy_threshold': 0.01,
    'train_path': './datasets/train_data.csv',
    'test_path': './datasets/test_data.csv',
    'best_model_path': './model/tf_best_model.pth',  # 用于保存最好的模型
    'json_path': './report/tf_train_report.json',
    'save_png': './report/tf_report_table.png',
    'save_k_png': './report/tf_report_table_k.png'
}


# ================================
# 计算 MSE 和 RMSE
# ================================
def calculate_mse_rmse(predictions, targets, scaler, close_index):
    # 反归一化
    predictions_real = predictions * (scaler.data_max_[close_index] - scaler.data_min_[close_index]) + scaler.data_min_[
        close_index]
    targets_real = targets * (scaler.data_max_[close_index] - scaler.data_min_[close_index]) + scaler.data_min_[
        close_index]

    # 计算均方误差（MSE）
    mse = torch.mean((predictions_real - targets_real) ** 2)

    # 计算均方根误差（RMSE）
    rmse = torch.sqrt(mse)

    return mse.item(), rmse.item()


# ================================
# 计算准确率
# ================================
def calculate_accuracy(predictions, targets, threshold, scaler, close_index):
    # 计算归一化前的实际阈值
    threshold_real = threshold * (scaler.data_max_[close_index] - scaler.data_min_[close_index])

    # 反归一化
    predictions_real = predictions * (scaler.data_max_[close_index] - scaler.data_min_[close_index]) + scaler.data_min_[
        close_index]
    targets_real = targets * (scaler.data_max_[close_index] - scaler.data_min_[close_index]) + scaler.data_min_[
        close_index]

    # 计算绝对误差
    abs_error = torch.abs(predictions_real - targets_real)
    accuracy = (abs_error <= threshold_real).float().mean()
    return accuracy.item()


# ================================
# 数据预处理函数
# ================================
def load_and_preprocess_data(train_path, test_path, seq_length):
    # 读取数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 删除NA
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # 计算简单移动平均线 (SMA)，这里以5日为例，你可以调整周期
    train_df['SMA_5'] = train_df['Close'].rolling(window=5).mean()
    test_df['SMA_5'] = test_df['Close'].rolling(window=5).mean()

    # 计算相对强弱指数 (RSI)，这里以14日为例，你可以调整周期
    def rsi(data, period=14):
        deltas = data['Close'].diff()
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down
        rsi = np.zeros_like(data['Close'])
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(data)):
            delta = deltas[i]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period

            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi

    train_df['RSI_14'] = rsi(train_df, period=14)
    test_df['RSI_14'] = rsi(test_df, period=14)

    # 计算布林带
    def bollinger_bands(data, window=20):
        data['BB_Middle'] = data['Close'].rolling(window=window).mean()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=window).std()
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=window).std()
        return data

    train_df = bollinger_bands(train_df)
    test_df = bollinger_bands(test_df)

    # 计算MACD
    def macd(data, short_window=12, long_window=26, signal_window=9):
        data['MACD_Short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
        data['MACD_Long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
        data['MACD'] = data['MACD_Short'] - data['MACD_Long']
        data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
        return data

    train_df = macd(train_df)
    test_df = macd(test_df)

    # 去掉一些滚动计算后产生的NA值
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # 特征列选择，仅保留原始特征和新添加的有用特征
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_5', 'RSI_14', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'MACD'
    ]
    close_index = feature_cols.index('Close')  # 记住预测目标在特征列表中的索引

    train_data = train_df[feature_cols].values
    test_data = test_df[feature_cols].values

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    def create_sequences(data):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length, close_index]  # 预测Close
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(train_data)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config['val_split'], random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 打印训练集和测试集的前10个样本
    print("训练集前10个样本：")
    print(train_df.head(10))
    print("\n测试集前10个样本：")
    print(test_df.head(10))

    return train_loader, val_loader, test_data, scaler


# ================================
# LearnedPositionalEncoding 类
# ================================
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(LearnedPositionalEncoding, self).__init__()
        self.positional_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_embeds = self.positional_embeddings(positions)
        x = x + pos_embeds
        return x


# ================================
# SimpleTransformer 类
# ================================
class SimpleTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, output_size, dropout=config['dropout'], max_len=10):  # max_len 根据你的序列长度调整
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        # 修改为传入 max_len
        self.pos_encoder = LearnedPositionalEncoding(max_len, d_model)
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_size)
        embedded = self.dropout(self.embedding(src))  # shape: (batch_size, seq_len, d_model)
        src_len = embedded.size(1)
        position = torch.arange(0, src_len).unsqueeze(0).to(src.device)
        embedded_with_pos = self.pos_encoder(embedded)  # shape: (batch_size, seq_len, d_model)

        attn_output1, _ = self.self_attn1(embedded_with_pos, embedded_with_pos, embedded_with_pos)  # shape: (batch_size, seq_len, d_model)
        attn_output1 = self.dropout(attn_output1)
        norm_output1 = embedded_with_pos + attn_output1  # Residual connection

        attn_output2, _ = self.self_attn2(norm_output1, norm_output1, norm_output1)  # shape: (batch_size, seq_len, d_model)
        attn_output2 = self.dropout(attn_output2)
        norm_output2 = norm_output1 + attn_output2  # Residual connection

        # Indexing "last"
        last_output = norm_output2[:, -1, :]  # shape: (batch_size, d_model)

        fc1_output = self.relu(self.fc1(last_output))  # shape: (batch_size, d_model)
        fc_output = self.fc_out(fc1_output)  # shape: (batch_size, output_size)

        return fc_output


def train_model(model, train_loader, val_loader, epochs, lr, threshold, best_model_path, scaler, close_index,
                json_path=config['json_path']):
    loss_fn = nn.MSELoss(reduction='mean')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config['weight_decay'])

    best_val_acc = -float('inf')  # 初始化最好的验证集准确率

    # 创建字典来存储每个epoch的指标
    metrics = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "train_mse": [],
        "train_rmse": [],
        "val_loss": [],
        "val_acc": [],
        "val_mse": [],
        "val_rmse": []
    }

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, train_mse, train_rmse = 0.0, 0.0, 0.0, 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(output.squeeze(), y_batch, threshold, scaler, close_index)
            mse, rmse = calculate_mse_rmse(output.squeeze(), y_batch, scaler, close_index)
            train_mse += mse
            train_rmse += rmse

        model.eval()
        val_loss, val_acc, val_mse, val_rmse = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                loss = loss_fn(output.squeeze(), y_batch)
                val_loss += loss.item()
                val_acc += calculate_accuracy(output.squeeze(), y_batch, threshold, scaler, close_index)
                mse, rmse = calculate_mse_rmse(output.squeeze(), y_batch, scaler, close_index)
                val_mse += mse
                val_rmse += rmse

        # 如果当前验证集准确率更好，保存模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())  # 保存当前模型
            torch.save(best_model, best_model_path)  # 保存最佳模型

        # 保存指标到字典
        metrics["epochs"].append(epoch + 1)
        metrics["train_loss"].append(train_loss / len(train_loader))
        metrics["train_acc"].append(train_acc / len(train_loader))
        metrics["train_mse"].append(train_mse / len(train_loader))
        metrics["train_rmse"].append(train_rmse / len(train_loader))
        metrics["val_loss"].append(val_loss / len(val_loader))
        metrics["val_acc"].append(val_acc / len(val_loader))
        metrics["val_mse"].append(val_mse / len(val_loader))
        metrics["val_rmse"].append(val_rmse / len(val_loader))

        print(f'Epoch [{epoch + 1}/{epochs}] - '
              f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_acc / len(train_loader):.4f}, '
              f'Train MSE: {train_mse / len(train_loader):.4f}, '
              f'Train RMSE: {train_rmse / len(train_loader):.4f}, '
              f'Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Val Acc: {val_acc / len(val_loader):.4f}, '
              f'Val MSE: {val_mse / len(val_loader):.4f}, '
              f'Val RMSE: {val_rmse / len(val_loader):.4f}')

    # 将 metrics 字典保存到 JSON 文件
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def test_model(model, test_data, scaler, seq_length, best_model_path, close_index):
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))

    test_sequences = []
    for i in range(len(test_data) - seq_length):
        test_sequences.append(test_data[i:i + seq_length])
    test_sequences = np.array(test_sequences)

    test_tensor = torch.tensor(test_sequences, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(test_tensor).squeeze().numpy()

    predictions_real = predictions * (scaler.data_max_[close_index] - scaler.data_min_[close_index]) + scaler.data_min_[
        close_index]

    # 计算 MSE 和 RMSE
    mse, rmse = calculate_mse_rmse(torch.tensor(predictions_real), torch.tensor(test_data[seq_length:, close_index]),
                                   scaler, close_index)

    print("Test Data (First 10 Samples) and Predictions:")
    for i in range(10):
        actual_close_norm = test_data[i + seq_length, close_index]
        actual_close = actual_close_norm * (scaler.data_max_[close_index] - scaler.data_min_[close_index]) + \
                       scaler.data_min_[close_index]
        print(f"Test Data {i + 1}: {actual_close:.4f} -> Prediction: {predictions_real[i]:.4f}")

    # 使用 calculate_accuracy 函数计算准确率
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    targets = torch.tensor(test_data[seq_length:, close_index], dtype=torch.float32)
    accuracy = calculate_accuracy(predictions_tensor, targets, config['accuracy_threshold'], scaler, close_index) * 100

    print(f"Test Accuracy: {accuracy:.2f}%")

    return predictions_real


# Function to plot metrics
def plot_metrics(json_path):
    with open(json_path, 'r') as f:
        metrics = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].plot(metrics["epochs"], metrics["train_loss"], label="Train Loss", color='b')
    axes[0, 0].plot(metrics["epochs"], metrics["val_loss"], label="Val Loss", color='r')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()

    axes[0, 1].plot(metrics["epochs"], metrics["train_acc"], label="Train Accuracy", color='b')
    axes[0, 1].plot(metrics["epochs"], metrics["val_acc"], label="Val Accuracy", color='r')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()

    axes[1, 0].plot(metrics["epochs"], metrics["train_mse"], label="Train MSE", color='b')
    axes[1, 0].plot(metrics["epochs"], metrics["val_mse"], label="Val MSE", color='r')
    axes[1, 0].set_title('MSE')
    axes[1, 0].legend()

    axes[1, 1].plot(metrics["epochs"], metrics["train_rmse"], label="Train RMSE", color='b')
    axes[1, 1].plot(metrics["epochs"], metrics["val_rmse"], label="Val RMSE", color='r')
    axes[1, 1].set_title('RMSE')
    axes[1, 1].legend()

    # Remove any unused axes if necessary
    for i in range(2):
        for j in range(2):
            if not axes[i, j].has_data():
                axes[i, j].axis('off')

    # Make sure the save path is valid
    plt.savefig(config['save_png'])  # Save the figure as PNG file


def plot_k_line(test_path):
    # 直接读取test_data.csv获取数据
    test_df = pd.read_csv(test_path)
    # 将日期列转换为datetime类型
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    # 设置日期列为索引
    test_df.set_index('Date', inplace=True)

    # 绘制K线图
    fig, ax = mpf.plot(test_df, type='candle', style='charles', volume=True, returnfig=True)

    # 保存K线图
    plt.savefig(config['save_k_png'])
    plt.close()


def plot_k_line_with_predictions(csv_path, predictions_real, seq_length, save_path):
    """
    绘制包含真实 K 线与预测收盘价的 K 线图

    :param csv_path: 原始 CSV 文件的路径
    :param predictions_real: 预测的收盘价数组
    :param seq_length: 序列长度
    :param save_path: 保存图片的路径
    """
    # 读取原始 CSV 文件
    test_df = pd.read_csv(csv_path)
    print(f"原始测试数据的长度: {len(test_df)}")

    # 调整截取逻辑，确保截取后的数据长度和预测数据长度一致
    end_index = seq_length + len(predictions_real)
    k_data = test_df[seq_length:end_index].copy()

    print(f"截取后数据的长度: {len(k_data)}")
    print(f"预测数据的长度: {len(predictions_real)}")

    # 确保预测数据长度和截取后的数据长度一致
    if len(predictions_real) != len(k_data):
        raise ValueError("预测数据长度与截取后的数据长度不一致，请检查数据处理逻辑。")

    # 添加预测的收盘价列
    k_data['Predicted_Close'] = predictions_real
    # 将日期列转换为 datetime 类型
    k_data['Date'] = pd.to_datetime(k_data['Date'])
    # 设置日期列为索引
    k_data.set_index('Date', inplace=True)

    # 配置额外的绘图元素，将预测的收盘价用红色虚线绘制
    apds = [mpf.make_addplot(k_data['Predicted_Close'], color='r', linestyle='--', label='Predicted Close')]
    # 绘制 K 线图，不显示成交量，同时添加预测收盘价的曲线
    fig, axes = mpf.plot(k_data, type='candle', style='charles', volume=False, addplot=apds, returnfig=True)
    # 保存 K 线图
    fig.savefig(save_path)
    # 关闭图形
    plt.close(fig)


if __name__ == "__main__":
    train_loader, val_loader, test_data, scaler = load_and_preprocess_data(
        config['train_path'], config['test_path'], config['seq_length'])

    close_index = 3  # 'Close' feature index
    model = SimpleTransformer(config['input_size'], config['d_model'], config['nhead'], config['output_size'])

    # Train the model
    train_model(model, train_loader, val_loader,
                epochs=config['epochs'],
                lr=config['learning_rate'],
                threshold=config['accuracy_threshold'],
                best_model_path=config['best_model_path'],
                scaler=scaler, close_index=close_index)

    # Plot the metrics after training
    plot_metrics(config['json_path'])
    # Test the model
    predictions_real = test_model(model, test_data, scaler, config['seq_length'], config['best_model_path'],
                                  close_index)
    # 绘制K线图，传入测试数据文件路径
    plot_k_line_with_predictions(config['test_path'], predictions_real, config['seq_length'], config['save_k_png'])
