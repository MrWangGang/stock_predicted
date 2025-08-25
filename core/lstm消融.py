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
    'input_size': 14,  # 增加特征后，输入维度变为14
    'hidden_layer_size': 20,
    'output_size': 1,
    'batch_size': 64,
    'learning_rate': 0.01,
    'epochs': 100,
    'val_split': 0.2,
    'accuracy_threshold': 0.01,
    'train_path': './datasets/train_data.csv',
    'test_path': './datasets/test_data.csv',
    'best_model_path': './model/lstm_best_model.pth',  # 用于保存最好的模型
    'json_path': './report/lstm_train_report.json',
    'save_png': './report/lstm_report_table.png',
    'save_k_png': './report/lstm_report_table_k.png'
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


# 计算简单移动平均线
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()


# 计算相对强弱指数
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=window).mean()
    avg_loss = down.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# 计算布林带
def calculate_bollinger_bands(data, window):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return upper_band, lower_band


# 计算动量
def calculate_momentum(data, window):
    return data['Close'].diff(window)


# 计算平均真实波动范围 (ATR)
def calculate_atr(data, window):
    high_low = data['High'] - data['Low']
    high_prev_close = np.abs(data['High'] - data['Close'].shift())
    low_prev_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()


# 计算异同移动平均线 (MACD)
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


# 计算随机指标 (KDJ)
def calculate_kdj(data, window=9):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    rsv = (data['Close'] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3 * k - 2 * d
    return k, d, j


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

    # 将日期列转换为datetime类型
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    # 提取时间序列特征
    train_df['Year'] = train_df['Date'].dt.year
    train_df['Month'] = train_df['Date'].dt.month
    train_df['Day'] = train_df['Date'].dt.day
    train_df['DayOfWeek'] = train_df['Date'].dt.dayofweek
    test_df['Year'] = test_df['Date'].dt.year
    test_df['Month'] = test_df['Date'].dt.month
    test_df['Day'] = test_df['Date'].dt.day
    test_df['DayOfWeek'] = test_df['Date'].dt.dayofweek

    # 计算额外的特征
    train_df['SMA_10'] = calculate_sma(train_df, 10)
    train_df['RSI_14'] = calculate_rsi(train_df, 14)
    train_df['Upper_Band'], train_df['Lower_Band'] = calculate_bollinger_bands(train_df, 20)
    train_df['Momentum_5'] = calculate_momentum(train_df, 5)
    train_df['ATR_14'] = calculate_atr(train_df, 14)
    train_df['MACD'], train_df['Signal'], train_df['Hist'] = calculate_macd(train_df)
    train_df['K'], train_df['D'], train_df['J'] = calculate_kdj(train_df)
    test_df['SMA_10'] = calculate_sma(test_df, 10)
    test_df['RSI_14'] = calculate_rsi(test_df, 14)
    test_df['Upper_Band'], test_df['Lower_Band'] = calculate_bollinger_bands(test_df, 20)
    test_df['Momentum_5'] = calculate_momentum(test_df, 5)
    test_df['ATR_14'] = calculate_atr(test_df, 14)
    test_df['MACD'], test_df['Signal'], test_df['Hist'] = calculate_macd(test_df)
    test_df['K'], test_df['D'], test_df['J'] = calculate_kdj(test_df)

    # 删除包含NaN的行
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    # 特征列选择，包含额外特征
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'RSI_14', 'Upper_Band', 'Year', 'Month', 'Day', 'DayOfWeek', 'MACD', 'K'
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
# LSTM模型定义
# ================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def train_model(model, train_loader, val_loader, epochs, lr, threshold, best_model_path, scaler, close_index,
                json_path=config['json_path']):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    model = LSTMModel(config['input_size'], config['hidden_layer_size'], config['output_size'])

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
