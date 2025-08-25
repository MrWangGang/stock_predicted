import pandas as pd

# 读取数据集
file_path = './datasets/GOOGL_2006-01-01_to_2018-01-01.csv'  # 替换为你的数据集文件路径
data = pd.read_csv(file_path)

# 将Date列转换为日期类型
data['Date'] = pd.to_datetime(data['Date'])

# 划分训练集和测试集
train_data = data[(data['Date'] >= '2006-01-01') & (data['Date'] <= '2016-12-30')]
test_data = data[data['Date'] >= '2017-01-01']

# 保存为CSV文件
train_data.to_csv('./datasets/train_data.csv', index=False)
test_data.to_csv('./datasets/test_data.csv', index=False)
