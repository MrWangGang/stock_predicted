import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lstm import LSTMModel, config

def load_and_preprocess_test_data(test_path):
    try:
        test_df = pd.read_csv(test_path)
        test_df.dropna(inplace=True)

        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume'
        ]
        close_index = feature_cols.index('Close')
        test_data = test_df[feature_cols].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        test_data = scaler.fit_transform(test_data)

        # 直接使用这 10 天的数据作为一个序列
        test_sequences = np.array([test_data])

        print("测试集前10个样本：")
        print(test_df.head(10))

        return test_sequences, test_data, scaler, close_index
    except FileNotFoundError:
        print("错误：文件未找到，请检查文件路径。")
        return None, None, None, None
    except Exception as e:
        print(f"读取文件时出现未知错误: {e}")
        return None, None, None, None

def predict_stock_price(csv_path):
    test_sequences, _, scaler, close_index = load_and_preprocess_test_data(
        csv_path)

    if test_sequences is not None:
        test_tensor = torch.tensor(test_sequences, dtype=torch.float32)
        print("test_sequences shape:", test_sequences.shape)
        print("test_tensor shape:", test_tensor.shape)

        model = LSTMModel(config['input_size'], config['hidden_layer_size'], config['output_size'])
        model.load_state_dict(torch.load(config['best_model_path']))
        model.eval()

        with torch.no_grad():
            predictions = model(test_tensor).squeeze().numpy()

        predictions_real = predictions * (scaler.data_max_[close_index] - scaler.data_min_[close_index]) + scaler.data_min_[
            close_index]

        # 检查 predictions_real 是否为标量
        if np.isscalar(predictions_real):
            price = predictions_real
        else:
            price = predictions_real[0]

        print(f"预测未来一天的价格: {price:.4f}")
        return price

    return None